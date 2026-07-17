# lattice_doe/app/__init__.py
# License: MIT
"""The packaged Streamlit web UI.

This subpackage ships the multi-page Streamlit app with the ``[app]``
extra, launched via the ``lattice-app`` console command (see
``_launcher``). It is intentionally import-light: importing
``lattice_doe`` (or even ``lattice_doe.app``) must not require
streamlit — the pages and components are executed by Streamlit's script
runner, not imported as package modules, and resolve their own
``components``/``state`` imports through the app directory the runner
puts on ``sys.path``.

``pages/`` deliberately has no ``__init__.py``: Streamlit surfaces every
``.py`` file in that directory as a sidebar page, so it ships as package
data instead (see ``[tool.setuptools.package-data]``).
"""
