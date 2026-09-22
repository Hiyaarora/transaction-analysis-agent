"""The FastAPI application.

`create_app()` is a factory so tests get a fresh app (and a fresh session
store) and can override the LLM dependency.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.deps import get_store
from app.api.routes import router

# The Vite dev server. Deliberately explicit rather than "*": this API holds a
# session's active dataset, so it should not be callable from any origin.
DEV_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]


def create_app() -> FastAPI:
    app = FastAPI(title="Transaction Analysis Agent", version="1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=DEV_ORIGINS,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-Session-Id"],
    )
    app.include_router(router)
    get_store.cache_clear()  # a new app gets its own session store
    return app


app = create_app()
