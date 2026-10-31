# geotrace/core/ssis.py
import asyncio

from gradio_client import Client, handle_file
from huggingface_hub import get_token

DEFAULT_SPACE = "davidlnk3/ssis-demo"


class SSISClient:

    def __init__(self, space_id: str = DEFAULT_SPACE, token: str | None = None):
        self._space_id = space_id
        self._token = token
        self._client: Client | None = None

    def _connect(self) -> Client:
        if self._client is None:
            token = self._token or get_token()
            if not token:
                raise RuntimeError(
                    "No Hugging Face token found. Run 'hf auth login' "
                    "or set HF_TOKEN in the server configuration."
                )
            self._client = Client(self._space_id, token=token)
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