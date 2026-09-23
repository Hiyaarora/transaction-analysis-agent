"""The FastAPI application: the JSON API, and the built React app in front of it.

`create_app()` is a factory so tests get a fresh app (and a fresh session
store) and can override the LLM dependency.

One process serves two things, and the order they are registered in matters:

    /api/*            the router - registered first, so no static route can
                      ever shadow an endpoint
    everything else   the Vite production build in frontend/dist

The catch-all deliberately refuses to answer an unknown /api path with HTML: a
missing endpoint stays a JSON 404 the client can read, which is what keeps the
two halves separate even though they share a port.

Serving the UI from the same origin as the API is also why there is no CORS
configuration here. The browser fetches /api/... from the host it loaded the
page from, and in development the Vite dev server proxies /api to this
process, so that is same-origin too. CORS would only be needed if the UI were
ever hosted somewhere else, and then it should be added deliberately for that
origin rather than left behind as a default.
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from app.api.deps import get_store
from app.api.routes import router

#: Where `npm run build` puts the UI. Absent in a checkout that has never been
#: built - the app then serves /api/* only, which is exactly what the tests and
#: the `npm run dev` workflow need.
FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"

INDEX = "index.html"


def create_app(frontend_dist: Path | None = FRONTEND_DIST) -> FastAPI:
    app = FastAPI(title="Transaction Analysis Agent", version="1.0")
    app.include_router(router)
    if frontend_dist is not None and (frontend_dist / INDEX).is_file():
        _serve_frontend(app, frontend_dist.resolve())
    get_store.cache_clear()  # a new app gets its own session store
    return app


def _serve_frontend(app: FastAPI, dist: Path) -> None:
    """Serve the build, with every unmatched path falling back to index.html.

    A single-page app owns its own routing, so reloading the browser on any
    path has to return the same document rather than a 404. Files that really
    exist in the build - the hashed assets, the icon - are returned as
    themselves.
    """
    index = dist / INDEX

    @app.get("/{requested:path}", include_in_schema=False)
    def spa(requested: str) -> FileResponse:
        if requested == "api" or requested.startswith("api/"):
            raise HTTPException(
                status_code=404,
                detail={"code": "not_found", "message": "No such endpoint."},
            )
        return FileResponse(_build_file(dist, requested) or index)


def _build_file(dist: Path, requested: str) -> Path | None:
    """The requested file from the build, or None if it is not one.

    The path comes from the URL, so it is untrusted: the resolved location has
    to still be inside the build directory. `../../.env` is not a file this
    server will hand out - it falls through to index.html like any other
    unknown path.
    """
    if not requested:
        return None
    candidate = (dist / requested).resolve()
    if not candidate.is_relative_to(dist) or not candidate.is_file():
        return None
    return candidate


app = create_app()
