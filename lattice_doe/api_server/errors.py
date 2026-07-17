# api_server/errors.py
# License: MIT
"""
Exception handlers for the FastAPI application.

Mapping strategy
----------------
* Pydantic ``RequestValidationError``  → 422  (FastAPI default, enhanced message)
* ``ValueError``                        → 422  (user input error from core library)
* ``RuntimeError``                      → 422  (design-generation failure, bad config)
* Any other ``Exception``               → 500  (unexpected server error, logged)
"""
from __future__ import annotations

import logging
import traceback

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger("lattice-api")


def register_exception_handlers(app: FastAPI) -> None:
    """Attach all custom exception handlers to *app*."""

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # Pydantic v2 may include Exception objects in error 'ctx' dicts which
        # are not JSON-serializable; convert them to strings.
        errors = []
        for e in exc.errors():
            entry = dict(e)
            if "ctx" in entry and isinstance(entry["ctx"], dict):
                entry["ctx"] = {
                    k: str(v) if isinstance(v, Exception) else v
                    for k, v in entry["ctx"].items()
                }
            entry.pop("url", None)  # strip Pydantic v2 doc URLs (optional)
            errors.append(entry)
        return JSONResponse(
            status_code=422,
            content={
                "error": "ValidationError",
                "detail": errors,
            },
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(
        request: Request, exc: ValueError
    ) -> JSONResponse:
        logger.warning("ValueError at %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=422,
            content={
                "error": "InvalidInput",
                "detail": str(exc),
            },
        )

    @app.exception_handler(RuntimeError)
    async def runtime_error_handler(
        request: Request, exc: RuntimeError
    ) -> JSONResponse:
        logger.warning("RuntimeError at %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=422,
            content={
                "error": "DesignError",
                "detail": str(exc),
            },
        )

    @app.exception_handler(Exception)
    async def generic_error_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        logger.error(
            "Unhandled exception at %s:\n%s",
            request.url.path,
            traceback.format_exc(),
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "InternalServerError",
                "detail": "An unexpected error occurred. See server logs for details.",
            },
        )
