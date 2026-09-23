"""Entry point for the web API: python run_api.py

Binds to 0.0.0.0 and to $PORT when the platform sets one. A hosted container
is reached from outside itself, so 127.0.0.1 would accept nothing, and the
port is assigned by the host rather than chosen here. Locally neither variable
is set, so this is still http://localhost:8000.

One worker on purpose: sessions and their datasets live in this process's
memory, so a second worker would answer half the requests with "no dataset
loaded". Scaling out would mean moving that state out of the process first.

The CLI (`python run.py --data ...`) is unaffected; both front ends drive the
same Agent.
"""

import os

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.api.main:app",
        host=os.getenv("HOST", "0.0.0.0"),  # noqa: S104 - see the module docstring
        port=int(os.getenv("PORT", "8000")),
        workers=1,
        reload=False,
    )
