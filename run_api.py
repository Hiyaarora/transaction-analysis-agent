"""Entry point for the web API: python run_api.py

The CLI (`python run.py --data ...`) is unaffected; both front ends drive the
same Agent.
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run("app.api.main:app", host="127.0.0.1", port=8000, reload=False)
