# tests/test_matrix_wire_format.py
"""REST wire-format tests for matrices, split from test_contrasts.py."""
import pytest

class TestMatrixWireFormatPreservesOrder:
    """UX-64: JSON object member order is not contractual, so matrices must
    travel as {columns: [...], data: [[...]]} — arrays DO preserve order. A
    rows-as-records payload silently permutes coefficients under a valid
    sort_keys round trip, and a positionally applied contrast then tests a
    different coefficient."""

    def test_records_lose_order_split_keeps_it(self):
        """The negative case that motivates the format: a sort_keys round
        trip reorders record keys but leaves the split orientation intact."""
        import json

        import pandas as pd

        from lattice_doe.api_server.serialization import (
            df_to_records, df_to_split, records_to_df, split_to_df,
        )

        df = pd.DataFrame(
            [[1.0, 0.0, 0.5]], columns=["Intercept", "C(g)[T.b]", "x"]
        )
        mangle = lambda obj: json.loads(json.dumps(obj, sort_keys=True))

        rec_cols = list(records_to_df(mangle(df_to_records(df))).columns)
        assert rec_cols != list(df.columns)          # records: order lost

        back = split_to_df(mangle(df_to_split(df)))
        assert list(back.columns) == list(df.columns)  # split: order kept
        assert (back.values == df.values).all()

    def test_sensitivity_correct_after_sort_keys_round_trip(self):
        """End to end: a client that re-serializes the /design response with
        sort_keys=True and feeds the matrix back to /sensitivity must get
        the run's own power — the columns array carries the order."""
        import json

        import numpy as np

        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        from lattice_doe.api_server.main import create_app

        client = TestClient(create_app())
        req = {
            "formula": "~ 1 + C(g) + x",
            "factors": {"g": ["a", "b"],
                        "x": {"type": "continuous", "low": 0.0, "high": 1.0}},
            # L targets C(g)[T.b] — the coefficient a key-sort MOVES
            # (['Intercept','C(g)[T.b]','x'] -> ['C(g)[T.b]','Intercept','x'])
            "power_cfg": {"type": "contrast", "L": [[0.0, 1.0, 0.0]],
                          "delta": [0.5], "alpha": 0.05, "power": 0.8,
                          "sigma": 1.0, "max_n": 20},
            "design_opts": {"candidate_points": 80, "random_state": 3,
                            "starts": 1},
        }
        r = client.post("/design", json=req)
        assert r.status_code == 200, r.text[:300]
        body = json.loads(json.dumps(r.json(), sort_keys=True))  # adversarial

        mm = body["model_matrix"]
        assert set(mm) == {"columns", "data"}        # split orientation
        assert mm["columns"] == ["Intercept", "C(g)[T.b]", "x"]

        r2 = client.post("/sensitivity", json={
            "formula": req["formula"], "factors": req["factors"],
            "power_cfg": req["power_cfg"], "design_opts": req["design_opts"],
            "design_df": body["design_df"], "model_matrix": mm,
            "sigma_range": [0.5, 2.0], "sigma_points": 3,
        })
        assert r2.status_code == 200, r2.text[:300]
        assert np.isclose(r2.json()["nominal_power"],
                          body["report"]["achieved_power"], atol=1e-9)
