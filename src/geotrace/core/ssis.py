# geotrace/core/ssis.py
import asyncio
import os

from gradio_client import Client, handle_file

DEFAULT_SPACE = "davidlnk3/ssis-demo"


class SSISClient:
    """SSIS detection on a Hugging Face Space, called anonymously unless a token is given.

    HF_TOKEN, if set, is sent so the calls use the account's ZeroGPU quota instead of the small
    anonymous one. Same rule as the MegaLoc Space backend in core/retrieval.py.
    """

    def __init__(self, space_id: str = DEFAULT_SPACE, token: str | None = None):
        self._space_id = space_id
        self._token = token
        self._client: Client | None = None

    def _connect(self) -> Client:
        if self._client is None:
            token = self._token or os.environ.get("HF_TOKEN") or None
            self._client = Client(self._space_id, token=token, verbose=False)
        return self._client

    async def detect(self, image: str) -> dict:
        client = await asyncio.to_thread(self._connect)
        result = await asyncio.to_thread(
            client.predict, handle_file(image), api_name="/detect_pairs"
        )
        if isinstance(result, tuple):
            result = next(r for r in result if isinstance(r, dict))
        return result


ssis = SSISClient()
