# lattice_doe/app/_launcher.py
# License: MIT
"""Console entry point for the packaged Streamlit UI (``lattice-app``)."""
from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    """Launch the Streamlit UI that ships inside the installed package.

    Equivalent to ``streamlit run <site-packages>/lattice_doe/app/app.py``.
    Extra command-line arguments pass straight through to Streamlit, so
    e.g. ``lattice-app --server.port 8600`` works as expected.
    """
    try:
        from streamlit.web import cli as stcli
    except ImportError as exc:
        raise SystemExit(
            "The web UI requires the [app] extra. Install it with:\n"
            "    pip install 'lattice-doe[app]'"
        ) from exc

    app_path = Path(__file__).resolve().parent / "app.py"
    sys.argv = ["streamlit", "run", str(app_path), *sys.argv[1:]]
    sys.exit(stcli.main())
