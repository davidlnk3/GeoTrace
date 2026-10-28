import os
import uvicorn
from mcp_server import mcp

app = mcp.streamable_http_app(host="0.0.0.0")
 
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
