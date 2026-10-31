import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

J2000 = datetime(2000, 1, 1, 12, tzinfo=timezone.utc)


@dataclass
class SunPosition:
    elevation_deg: float  # above the horizon, no refraction
    azimuth_deg: float    # compass, clockwise from north (0 = N, 90 = E)


def solar_declination_and_eot(when: datetime) -> tuple[float, float]:
    """Sun declination (radians) and equation of time (minutes) at `when`, NOAA algorithm."""
    t = (when - J2000).total_seconds() / 86400.0  # days since J2000
    jc = t / 36525.0

    l0 = (280.46646 + jc * (36000.76983 + jc * 0.0003032)) % 360
    m = math.radians(357.52911 + jc * (35999.05029 - 0.0001537 * jc))
    e = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc)
    c = (math.sin(m) * (1.914602 - jc * (0.004817 + 0.000014 * jc))
         + math.sin(2 * m) * (0.019993 - 0.000101 * jc)
         + math.sin(3 * m) * 0.000289)
    true_lon = l0 + c
    omega = 125.04 - 1934.136 * jc
    app_lon = math.radians(true_lon - 0.00569 - 0.00478 * math.sin(math.radians(omega)))

    mean_obliq = 23 + (26 + (21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))) / 60) / 60
    obliq = math.radians(mean_obliq + 0.00256 * math.cos(math.radians(omega)))
    decl = math.asin(math.sin(obliq) * math.sin(app_lon))

    y = math.tan(obliq / 2) ** 2
    l0r = math.radians(l0)
    eq_time = 4 * math.degrees(
        y * math.sin(2 * l0r)
        - 2 * e * math.sin(m)
        + 4 * e * y * math.sin(m) * math.cos(2 * l0r)
        - 0.5 * y * y * math.sin(4 * l0r)
        - 1.25 * e * e * math.sin(2 * m)
    )

    return decl, eq_time


def sun_position(lat: float, lon: float, when: datetime) -> SunPosition:
    """NOAA solar position algorithm. `when` must be timezone-aware."""
    decl, eq_time = solar_declination_and_eot(when)
    minutes_utc = when.hour * 60 + when.minute + when.second / 60
    true_solar_min = (minutes_utc + eq_time + 4 * lon) % 1440
    hour_angle = math.radians(true_solar_min / 4 - 180)

    lat_r = math.radians(lat)
    cos_zen = (math.sin(lat_r) * math.sin(decl)
               + math.cos(lat_r) * math.cos(decl) * math.cos(hour_angle))
    cos_zen = max(-1.0, min(1.0, cos_zen))
    zen = math.acos(cos_zen)
    elevation = 90 - math.degrees(zen)

    sin_zen = math.sin(zen)
    if sin_zen < 1e-9 or math.cos(lat_r) < 1e-9:
        azimuth = 180.0  # undefined at the zenith or the poles; any value works
    else:
        cos_az = (math.sin(lat_r) * cos_zen - math.sin(decl)) / (math.cos(lat_r) * sin_zen)
        az = math.degrees(math.acos(max(-1.0, min(1.0, cos_az))))
        azimuth = (az + 180) % 360 if hour_angle > 0 else (540 - az) % 360

    return SunPosition(elevation_deg=elevation, azimuth_deg=azimuth)


def _angular_error(a: SunPosition, elevation_deg: float, azimuth_deg: float) -> float:
    d_az = (a.azimuth_deg - azimuth_deg + 180) % 360 - 180
    d_az *= math.cos(math.radians(elevation_deg))  # azimuth degrees shrink near the zenith
    return math.hypot(a.elevation_deg - elevation_deg, d_az)


def find_matching_times(
    lat: float,
    lon: float,
    elevation_deg: float,
    azimuth_deg: float,
    year: int,
    tolerance_deg: float,
) -> list[tuple[datetime, SunPosition, float]]:
    """Every local minimum of the angular error over `year` that is within tolerance.

    Returns (time_utc, sun_position, error_deg) tuples, best match first.
    Scans every 10 minutes, then refines each hit to the minute.
    """
    step = timedelta(minutes=10)
    start = datetime(year, 1, 1, tzinfo=timezone.utc)
    end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)

    times, errors = [], []
    t = start
    while t < end:
        times.append(t)
        errors.append(_angular_error(sun_position(lat, lon, t), elevation_deg, azimuth_deg))
        t += step

    matches = []
    for i, err in enumerate(errors):
        if err > tolerance_deg:
            continue
        left = errors[i - 1] if i > 0 else math.inf
        right = errors[i + 1] if i + 1 < len(errors) else math.inf
        if not (err <= left and err < right):
            continue

        best_t, best_err = times[i], err
        for k in range(-10 * 2, 10 * 2 + 1):  # +-10 min in 30 s steps
            cand = times[i] + timedelta(seconds=30 * k)
            cand_err = _angular_error(sun_position(lat, lon, cand), elevation_deg, azimuth_deg)
            if cand_err < best_err:
                best_t, best_err = cand, cand_err
        best_t = best_t.replace(second=0, microsecond=0)
        pos = sun_position(lat, lon, best_t)
        matches.append((best_t, pos, _angular_error(pos, elevation_deg, azimuth_deg)))

    matches.sort(key=lambda m: m[2])
    return matches


def find_locations(
    when: datetime,
    elevation_deg: float,
    azimuth_deg: float,
) -> list[tuple[float, float]]:
    """(lat, lon) pairs where the sun is at the given elevation and azimuth at `when`.

    Usually two mirror-image solutions; both are returned. Empty if none exist.
    """
    decl, eq_time = solar_declination_and_eot(when)
    h, a = math.radians(elevation_deg), math.radians(azimuth_deg)
    minutes_utc = when.hour * 60 + when.minute + when.second / 60

    # sin(decl) = sin(lat) sin(h) + cos(lat) cos(h) cos(a) = r * sin(lat + gamma)
    p, q = math.sin(h), math.cos(h) * math.cos(a)
    r = math.hypot(p, q)
    s = math.sin(decl) / r
    if abs(s) > 1:
        return []
    gamma = math.atan2(q, p)
    base = math.asin(s)

    locations = []
    for lat_r in (base - gamma, math.pi - base - gamma):
        lat_r = math.atan2(math.sin(lat_r), math.cos(lat_r))  # wrap to (-pi, pi]
        if abs(lat_r) > math.pi / 2 + 1e-9:
            continue
        lat_r = max(-math.pi / 2, min(math.pi / 2, lat_r))
        # Hour angle from the sun's altitude and azimuth at this latitude
        hour_angle = math.degrees(math.atan2(
            -math.sin(a) * math.cos(h) * math.cos(lat_r),
            math.sin(h) - math.sin(lat_r) * math.sin(decl),
        ))
        lon = (hour_angle - (minutes_utc + eq_time) / 4) % 360 - 180  # lon = H + 180 - X, wrapped
        locations.append((math.degrees(lat_r), lon))
    return locations
