# Quick Start Guide (10 minutes)

This guide gets you from install to a working powered design quickly.

## 1) Install

From the project root:

```bash
# Core package
pip install -e .

# Optional: CLI YAML support
pip install -e ".[cli]"
```

## 2) Python quick start (contrast mode)

Create a small script and run it with `python your_script.py`.

```python
from iopt_power_design import (
    i_optimal_powered_design,
    PowerContrastConfig,
    DesignOptions,
)

formula = "~ 1 + A + B + A:B"
factors = {
    "A": ["low", "high"],
    "B": (0.0, 10.0),
}

# For this formula/factor setup, p=4 columns in X:
# [Intercept, A[T.high], B, A[T.high]:B]
power_cfg = PowerContrastConfig(
    L=[[0, 0, 1, 0]],  # test B main effect
    delta=[0.5],       # minimum detectable effect
    alpha=0.05,
    power=0.80,
    sigma=1.0,
    max_n=500,
)

opts = DesignOptions(
    auto_candidate=True,
    starts=5,
    criterion="I",      # or "D" / "A"
    random_state=42,    # must be an int
)

result = i_optimal_powered_design(formula, factors, power_cfg, opts)

print("n:", result["report"]["n"])
print("achieved_power:", round(result["report"]["achieved_power"], 4))
print(result["design_df"].head())
```

You should get:
- `result["design_df"]`: selected run table
- `result["buckets_df"]`: counts by factor-level bucket
- `result["report"]`: diagnostics and search metadata

## 3) CLI quick start

Generate a starter config and run:

```bash
# Print a commented template and save it to a file
iopt-design --template contrast > quickstart.yaml

# Edit quickstart.yaml as needed, then generate the design
iopt-design --config quickstart.yaml --out quickstart
```

Outputs:
- `quickstart_design.csv`
- `quickstart_buckets.csv`
- `quickstart_report.json`

## 4) Common next steps

1. Compare criteria (`"I"`, `"D"`, `"A"`): use `compare_criteria(...)`.
2. Add runs to an existing design: use `augment_design(...)`.
3. Sensitivity and MDE checks: use `power_sensitivity(...)` and `min_detectable_effect(...)`.

## 5) Streamlit web UI

The interactive web front-end lets you configure and run designs without writing code.

### Local run

```bash
# Install the package with Streamlit and Plotly
pip install -e ".[app]"

# Launch the app (opens in your browser at http://localhost:8501)
streamlit run app/app.py
```

### Streamlit Community Cloud (free hosting)

1. Push this repository to GitHub (already done if you're reading this there).
2. Go to [share.streamlit.io](https://share.streamlit.io) and click **New app**.
3. Select your repository and set **Main file path** to `app/app.py`.
4. Click **Deploy** — no secrets or environment variables required.

### Docker

Build and run with the included `Dockerfile`:

```bash
# Build the image (from the project root)
docker build -t iopt-doe .

# Run — app available at http://localhost:8501
docker run -p 8501:8501 iopt-doe
```

## 6) If something fails

- `ValueError: power_cfg.max_n must be greater than p`:
  increase `max_n` or simplify the formula.
- No convergence warning:
  raise `max_n`, increase `starts`, and/or enable `auto_candidate=True`.
- Parallel on macOS/Windows:
  put `workers > 1` calls under `if __name__ == "__main__":`.
