# api_server/__init__.py
# License: MIT
"""
iopt-power-design REST API server
===================================
FastAPI application exposing I-optimal DOE with power assurance over HTTP.

Quick start::

    uvicorn api_server.main:create_app --factory --host 0.0.0.0 --port 8000
    # or:
    iopt-api

Docs available at http://localhost:8000/docs (Swagger) and /redoc.
"""
__version__ = "0.1.0"
