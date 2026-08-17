# tests/test_packaging.py
# License: MIT
"""Packaging/distribution markers that must ship with the installed package."""

from pathlib import Path

import lattice_doe


class TestPyTypedMarker:
    """Third-opinion review (2026-08-16): the package ships full annotations
    but had no PEP 561 marker, so downstream type checkers and IDEs treated
    it as untyped. The marker must live next to the package and be listed in
    [tool.setuptools.package-data] to reach wheels."""

    def test_py_typed_ships_next_to_package(self):
        assert (Path(lattice_doe.__file__).parent / "py.typed").is_file()
