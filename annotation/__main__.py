"""python -m annotation  →  local gold-annotation server."""

import uvicorn

if __name__ == "__main__":
    uvicorn.run("annotation.app:app", host="127.0.0.1", port=8765, reload=True)
