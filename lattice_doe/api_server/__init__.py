# api_server/__init__.py
# License: MIT
"""
Lattice DOE REST API server
===========================
FastAPI application exposing powered optimal design workflows over HTTP.

Quick start::

    uvicorn lattice_doe.api_server.main:create_app --factory --host 0.0.0.0 --port 8000
    # or:
    lattice-api

Docs available at http://localhost:8000/docs (Swagger) and /redoc.
"""
__version__ = "0.1.0"
