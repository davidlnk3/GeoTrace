"""Fit the "Blind Dates" year classifier on frozen OpenCLIP embeddings.

Reproduces arXiv:2310.06633 (Barancova, Wevers, van Noord, CHR 2023): a multinomial
logistic regression over the 50 year classes 1950-1999, on top of frozen
ViT-B-32-quickgelu/laion400m_e32 image embeddings. Hyperparameters and the train/test split
come from the authors' own code (CANAL-amsterdam/dating-images, 4.Finetuning.ipynb):

    LogisticRegression(random_state=0, max_iter=5000)
    split: stratified 80/20, frozen in their data/image_data.json

Extra DEW images are added to TRAIN ONLY, so the test set stays exactly the authors' De Boer
test split and the reported MAE remains comparable to the paper's 6.65.

The fitted model is exported as safetensors (a weight matrix and a bias vector), not as a
pickle: it keeps scikit-learn out of the serving environment and avoids executing code on load.
"""
import argparse
import json
import platform
import sys
from datetime import date

import numpy as np
import sklearn
from safetensors.numpy import load_file, save_file
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error

YEAR_MIN, YEAR_MAX = 1950, 1999


def load_split(labels_path, emb_path):
    with open(labels_path) as fh:
        records = json.load(fh)
    blob = np.load(emb_path)
    emb, found = blob["embeddings"], blob["found"]
    years = np.array([r[2] for r in records])
    is_train = np.array([r[5] == "train" for r in records])
    if not found.all():
        print(f"WARNING: {int((~found).sum())} De Boer images missing from the tarball; dropped")
    tr = found & is_train
    te = found & ~is_train
    return emb[tr], years[tr], emb[te], years[te]


def load_extra(folder_manifest, emb_path):
    """DEW: returns (X, y) restricted to the 1950-1999 label space."""
    with open(folder_manifest) as fh:
        rows = json.load(fh)
    blob = np.load(emb_path)
    emb, found = blob["embeddings"], blob["found"]
    years = np.array([r["year"] for r in rows])
    keep = found & (years >= YEAR_MIN) & (years <= YEAR_MAX)
    print(f"DEW: {int(keep.sum())} of {len(rows)} usable (embedded and within {YEAR_MIN}-{YEAR_MAX})")
    return emb[keep], years[keep]


def fit_and_score(X_tr, y_tr, X_te, y_te, label):
    clf = LogisticRegression(random_state=0, max_iter=5000)
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    acc = accuracy_score(y_te, pred)
    mae = mean_absolute_error(y_te, pred)
    # Probability-weighted expected year: reported as a secondary number, not the paper's rule.
    exp_year = clf.predict_proba(X_te) @ clf.classes_
    mae_soft = mean_absolute_error(y_te, exp_year)
    print(f"[{label}] n_train={len(y_tr)} n_test={len(y_te)} "
          f"acc={acc:.4f} MAE={mae:.3f}y MAE_expected={mae_soft:.3f}y")
    return clf, {"n_train": len(y_tr), "n_test": len(y_te),
                 "accuracy": round(float(acc), 4), "mae_years": round(float(mae), 3),
                 "mae_years_expected_value": round(float(mae_soft), 3)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--deboer-emb", required=True)
    ap.add_argument("--dew-manifest")
    ap.add_argument("--dew-emb")
    ap.add_argument("--dew-eval-manifest", help="held-out DEW manifest, for cross-domain scoring")
    ap.add_argument("--dew-eval-emb")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    X_tr, y_tr, X_te, y_te = load_split(args.labels, args.deboer_emb)
    print(f"De Boer: {len(y_tr)} train / {len(y_te)} test, dim={X_tr.shape[1]}")

    metrics = {}
    # Baseline first, so the effect of adding DEW is measurable rather than asserted.
    base_clf, metrics["deboer_only"] = fit_and_score(X_tr, y_tr, X_te, y_te, "De Boer only")

    dew_n = 0
    if args.dew_manifest and args.dew_emb:
        X_d, y_d = load_extra(args.dew_manifest, args.dew_emb)
        dew_n = len(y_d)
        X_tr = np.concatenate([X_tr, X_d])
        y_tr = np.concatenate([y_tr, y_d])
    else:
        print("no DEW embeddings supplied; training De Boer only")

    clf, metrics["final"] = fit_and_score(X_tr, y_tr, X_te, y_te, "De Boer + DEW" if dew_n else "final")

    # Cross-domain score. De Boer is Dutch press photography; DEW is worldwide amateur Flickr.
    # Adding DEW is expected to cost a little in-domain and gain out-of-domain, so both are shown.
    if args.dew_eval_manifest and args.dew_eval_emb:
        X_ev, y_ev = load_extra(args.dew_eval_manifest, args.dew_eval_emb)
        metrics["heldout_dew"] = {
            "n": len(y_ev),
            "deboer_only_mae_years": round(float(mean_absolute_error(y_ev, base_clf.predict(X_ev))), 3),
            "final_mae_years": round(float(mean_absolute_error(y_ev, clf.predict(X_ev))), 3),
        }
        print(f"[held-out DEW] n={len(y_ev)} "
              f"De Boer only MAE={metrics['heldout_dew']['deboer_only_mae_years']}y "
              f"-> final MAE={metrics['heldout_dew']['final_mae_years']}y")

    # Export as a plain linear layer: logits = x @ W.T + b, year = classes[argmax(logits)].
    #
    # ascontiguousarray is load-bearing, not defensive: scikit-learn leaves coef_ F-contiguous,
    # and safetensors serialises the raw buffer as if it were C-ordered. Saving the array as it
    # comes off the estimator writes a (n_classes, dim) matrix whose values are transposed in
    # memory -- same shape, no error, silently wrong predictions. .astype() preserves the layout,
    # so it does not save us here.
    W = np.ascontiguousarray(clf.coef_, dtype=np.float32)
    b = np.ascontiguousarray(clf.intercept_, dtype=np.float32)
    classes = np.ascontiguousarray(clf.classes_, dtype=np.int32)
    out_path = f"{args.out_dir}/model.safetensors"
    save_file({"weight": W, "bias": b, "classes": classes}, out_path)

    # Prove the exported file reproduces the estimator's own decisions before publishing it.
    t = load_file(out_path)
    exported = t["classes"][(X_te @ t["weight"].T + t["bias"]).argmax(axis=1)]
    reference = clf.predict(X_te)
    disagree = int((exported != reference).sum())
    if disagree:
        raise SystemExit(f"export is unfaithful: differs from clf.predict on {disagree} test images")
    print(f"export verified against clf.predict on {len(reference)} test images")

    config = {
        "task": "image-capture-year-estimation",
        "paper": "arXiv:2310.06633 (Blind Dates, CHR 2023)",
        "backbone": {"library": "open_clip", "model": "ViT-B-32-quickgelu",
                     "pretrained": "laion400m_e32", "embedding_dim": int(W.shape[1]),
                     "l2_normalize": False},
        "head": {"type": "multinomial_logistic_regression",
                 "sklearn_params": {"random_state": 0, "max_iter": 5000},
                 "num_classes": int(W.shape[0]),
                 "year_min": int(classes.min()), "year_max": int(classes.max())},
        "training_data": {"de_boer_scene_detection": {"train": metrics["final"]["n_train"] - dew_n,
                                                      "test": metrics["final"]["n_test"],
                                                      "zenodo": "10.5281/zenodo.7137452"},
                          "dew_images_added_to_train": dew_n},
        "metrics": metrics,
        "trained_on": date.today().isoformat(),
        "versions": {"scikit_learn": sklearn.__version__, "python": platform.python_version()},
    }
    with open(f"{args.out_dir}/config.json", "w") as fh:
        json.dump(config, fh, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"wrote {args.out_dir}/model.safetensors and config.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
