# api_server/main.py
# License: MIT
"""
FastAPI application factory and Uvicorn entry point.

Usage
-----
Start with Uvicorn::

    uvicorn api_server.main:create_app --factory --host 0.0.0.0 --port 8000

Or via the installed CLI entry point::

    lattice-api

The ``create_app`` factory pattern lets Uvicorn's ``--factory`` flag create a
fresh app instance per worker process, which is the correct pattern for
multi-worker deployments.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api_server import __version__
from api_server.errors import register_exception_handlers
from api_server.routers import augment, compare, design, power_curve, sensitivity

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("lattice-api")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Called by Uvicorn's ``--factory`` flag and by the test fixtures.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # type: ignore[type-arg]
        # Pre-import heavy dependencies on startup so the first request
        # is not penalised by module-load latency.
        import numpy  # noqa: F401
        import scipy  # noqa: F401
        import patsy  # noqa: F401
        logger.info("lattice-api v%s started.", __version__)
        yield
        logger.info("lattice-api shutting down.")

    app = FastAPI(
        title="lattice-doe API",
        version=__version__,
        description=(
            "I-optimal experimental designs with power assurance — REST API.\n\n"
            "## Authentication\n"
            "No authentication is required. For production deployments, place "
            "a reverse proxy (nginx, Traefik) in front to enforce TLS and "
            "access controls.\n\n"
            "## Constraint expressions\n"
            "Use ``constraint_expr`` (a string) in ``design_opts`` to apply "
            "row-level constraints to the candidate set. Python callables "
            "cannot travel over HTTP. Allowed functions: ``abs``, ``min``, "
            "``max``, ``round``, ``sqrt``, ``log``, ``exp``, ``floor``, ``ceil``.\n\n"
            "## Parallel starts\n"
            "The ``workers`` field in ``design_opts`` is accepted but ignored "
            "inside the ASGI server (ProcessPoolExecutor conflicts with Uvicorn "
            "event-loop state). Use Uvicorn's own ``--workers`` flag instead."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # --- Routers ---
    app.include_router(design.router, tags=["Design"])
    app.include_router(power_curve.router, tags=["Power Curve"])
    app.include_router(sensitivity.router, tags=["Sensitivity & MDE"])
    app.include_router(compare.router, tags=["Criteria Comparison"])
    app.include_router(augment.router, tags=["Design Augmentation"])

    # --- Exception handlers ---
    register_exception_handlers(app)

    # --- Health check ---
    @app.get("/health", tags=["Health"], summary="Health check")
    async def health() -> dict:
        """Returns ``{"status": "ok"}`` when the server is running."""
        return {"status": "ok", "version": __version__}

    return app


def run() -> None:
    """Entry point for the ``lattice-api`` CLI command."""
    try:
        import uvicorn
    except ImportError:
        raise SystemExit(
            "uvicorn is required to run the API server. "
            'Install with: pip install "lattice-doe[server]"'
        )
    uvicorn.run(
        "api_server.main:create_app",
        factory=True,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )


if __name__ == "__main__":
    run()
