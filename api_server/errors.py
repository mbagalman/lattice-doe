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

logger = logging.getLogger("iopt-api")


def register_exception_handlers(app: FastAPI) -> None:
    """Attach all custom exception handlers to *app*."""

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": "ValidationError",
                "detail": exc.errors(),
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
