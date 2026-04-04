# Plotly Interactive Power Charts — Development Plan & Ticket Pack

Tracks all work for **Enhancement #13** (opt-in Plotly backend for power charts).

**Rules for contributors:**
1. Before starting a ticket, set its `Status` to `Claimed` and fill in `Claimed by`.
2. When done, check the box in the Dashboard and set `Status` to `Done`.
3. Never start work on a ticket marked `Claimed` by someone else — pick a different `Open` ticket or coordinate first.
4. If you hit a usage limit mid-ticket, leave a `Progress note` in the ticket card so the next session can continue without re-reading the whole codebase.

---

## Dashboard

| ID | Title | Epic | Status | Claimed by |
|----|-------|------|--------|------------|
| [A1](#a1-stub-plot_backendspy--update-pyprojecttoml) | Stub `plot_backends.py` + update `pyproject.toml` | Infrastructure | Done | Claude |
| [B1](#b1-power_curve_by_n-plotly-figure--wire-up) | `power_curve_by_n` Plotly figure + wire-up | curve_by_n | Done | Claude |
| [C1](#c1-power_curve_by_effect-plotly-figure--wire-up) | `power_curve_by_effect` Plotly figure + wire-up | curve_by_effect | Done | Claude |
| [D1](#d1-power_surface_2d-plotly-figure--wire-up) | `power_surface_2d` Plotly figure + wire-up | surface_2d | Done | Claude |
| [E1](#e1-power_sensitivity-plotly-figure--wire-up) | `power_sensitivity` Plotly figure + wire-up | sensitivity | Done | Claude |
| [F1](#f1-exports--public-api-wiring) | Exports & public API wiring | Wiring | Done | Claude |
| [G1](#g1-unit-tests) | Unit tests | Tests & Docs | Done | Claude |
| [G2](#g2-documentation-updates) | Documentation updates | Tests & Docs | Done | Claude |

**Progress:** 8 / 8 tickets done. ✅

---

## Design Decisions

### `plot_backend` parameter — opt-in, no breaking change

Add `plot_backend: Literal["matplotlib", "plotly"] = "matplotlib"` to the four
target functions. The existing `plot=False` default is unchanged, so all existing
callers are unaffected. The new parameter is ignored when `plot=False`.

```python
# Existing behaviour — no change
result = power_curve_by_n(formula, factors, power_cfg, design_opts=opts)

# Opt-in Plotly
result = power_curve_by_n(
    formula, factors, power_cfg, design_opts=opts,
    plot=True, plot_backend="plotly"
)
fig = result["figure"]   # plotly.graph_objects.Figure
fig.show()               # interactive in Jupyter / browser
```

### All Plotly helpers live in `plot_backends.py`

A new `lattice_doe/plot_backends.py` holds one Plotly builder per chart type.
This keeps `power_curves.py` and `api.py` clean and isolates the `plotly` import to
one module (soft dependency — guarded by a try/except at module level).

Public surface in `plot_backends.py`:

| Helper | Called by |
|--------|-----------|
| `plotly_curve_by_n(df, power_cfg, target_n)` | `power_curves.power_curve_by_n` |
| `plotly_curve_by_effect(df, power_cfg, min_detectable, n)` | `power_curves.power_curve_by_effect` |
| `plotly_surface_2d(power_grid, axis1, axis2, power_cfg, param1, param2)` | `power_curves.power_surface_2d` |
| `plotly_sensitivity(df, power_cfg, nominal_pwr, n)` | `api.power_sensitivity` |

`plot_backends.py` is an internal module — not exported from `__init__.py`.

### Plotly is already available in `[app]` extras

`plotly>=5.0` is already declared in `[app]`. For users who don't install the app
but want Plotly charts in Jupyter, add it to the existing `[viz]` extras group
(currently only matplotlib + seaborn). No new extras group is needed.

### Return type for `"figure"` key

The `"figure"` key in the result dict becomes
`Optional[Union[matplotlib.figure.Figure, plotly.graph_objects.Figure]]`.
Existing callers that check `if result["figure"] is not None` or call
`result["figure"].savefig(...)` continue to work on the matplotlib path.

### Chart specifications

**`plotly_curve_by_n`** — two-panel figure using `make_subplots(rows=2, cols=1, shared_xaxes=True)`:
- Row 1: `go.Scatter(x=n, y=power, mode="lines+markers")` + `add_hline(target_power)` + `add_vline(target_n)`. Y-range [0, 1.05]. Hover shows `n`, `power %`, `λ`.
- Row 2: `go.Scatter(x=n, y=i_criterion)` on left y-axis + `go.Scatter(x=n, y=d_efficiency)` on right y-axis (secondary_y=True). Hover adds `i_criterion` and `d_efficiency`.
- Template: `"plotly_white"`. Title reflects contrast vs R² mode.

**`plotly_curve_by_effect`** — single panel:
- `go.Scatter(x=effect_size, y=power, mode="lines+markers")` + `add_hline(target_power)` + `add_hline(0.80, dash="dot")` + `add_vline(mde)`.
- X-axis label: `"Effect Size Multiplier"` (contrast) or `"R² Effect Size"`.
- Hover shows effect size value, power %, λ.

**`plotly_surface_2d`** — heatmap with contour overlay:
- Base: `go.Heatmap(z=power_grid, x=axis2, y=axis1, colorscale="Viridis", zmin=0, zmax=1, colorbar=dict(title="Power"))`.
- Target contour: `go.Contour(z=power_grid, x=axis2, y=axis1, contours=dict(start=target_power, end=target_power, coloring="none"), line=dict(color="white", width=2, dash="dash"), showscale=False, name=f"Target={target_power}")`.
- Axis labels, title from `param1` / `param2` names.

**`plotly_sensitivity`** — single panel (contrast or R² mode):
- `go.Scatter(x=sweep_values, y=power, mode="lines+markers")`.
- `add_vline(x=nominal_value)` — nominal sigma or r2 reference.
- `add_hline(y=target_power)` — target.
- `add_hline(y=nominal_pwr, dash="dot")` — achieved power at nominal.
- Hover shows sweep value, power %, λ.

### `power_surface_2d` — export gap

`power_surface_2d` is currently implemented in `power_curves.py` but is **not**
exported from `__init__.py` or `api.py`. Ticket F1 adds it to `__init__.py`.

### `power_curve_by_n` / `power_curve_by_effect` — api.py wrappers

The api.py wrappers (`power_curve_by_n`, `power_curve_by_effect`) return a DataFrame
only (not the full dict). They exist for backward compat (TD-3 is the ticket to
remove them). For this enhancement:
- Thread `plot_backend` through the wrappers so the impl receives it.
- Since the wrappers discard the figure and return only `data`, `plot_backend` has no
  user-visible effect via the public API wrappers. Document in the docstring that
  users who want a Plotly figure should call
  `lattice_doe.power_curves.power_curve_by_n(...)` directly.

---

## Current plotting landscape (before this enhancement)

| Function | Location | Returns figure? | Backend |
|----------|----------|-----------------|---------|
| `power_curve_by_n` (impl) | `power_curves.py:46` | Yes, when `plot=True` | matplotlib only |
| `power_curve_by_effect` (impl) | `power_curves.py:264` | Yes, when `plot=True` | matplotlib only |
| `power_surface_2d` | `power_curves.py:450` | Yes, when `plot=True` | matplotlib only |
| `power_sensitivity` | `api.py:644` | Yes, when `plot=True` | matplotlib only (lazy import) |
| `power_curve_by_n` (wrapper) | `api.py:558` | No (returns DataFrame) | n/a |
| `power_curve_by_effect` (wrapper) | `api.py:580` | No (returns DataFrame) | n/a |

---

## Epic A — Infrastructure

---

### A1 Stub `plot_backends.py` + update `pyproject.toml`

**Status:** Done
**Claimed by:** Claude
**Est.:** 30 minutes
**Depends on:** nothing
**Progress note:** Complete. `plot_backends.py` created with try/except plotly guard, `_INSTALL_HINT` constant, and four stub functions that raise `ImportError` (when plotly absent) or `NotImplementedError` (when present, pending implementation). `pyproject.toml` `[viz]` updated to include `plotly>=5.0`. Import verified. 229 tests still pass.

**What to do:**

1. Create `lattice_doe/plot_backends.py` with:
   ```python
   """Plotly figure builders for power curve functions."""
   try:
       import plotly.graph_objects as go
       from plotly.subplots import make_subplots
       _HAS_PLOTLY = True
   except ImportError:
       _HAS_PLOTLY = False

   __all__ = [
       "plotly_curve_by_n",
       "plotly_curve_by_effect",
       "plotly_surface_2d",
       "plotly_sensitivity",
   ]
   ```
   Add four stub functions that each raise `ImportError` if `_HAS_PLOTLY` is False, or `NotImplementedError` otherwise. This gives later tickets a valid import target.

2. Add `plotly>=5.0` to the `[viz]` extras group in `pyproject.toml`:
   ```toml
   viz = [
     "matplotlib>=3.5,<4",
     "seaborn>=0.11,<1",
     "plotly>=5.0",
   ]
   ```
   (Plotly is already in `[app]`; this makes it available for non-Streamlit users too.)

3. Verify: `python -c "import lattice_doe.plot_backends"` exits without error.
4. Run `pytest tests/ -q` — all existing tests still pass.

**Acceptance criteria:**
- [ ] `lattice_doe/plot_backends.py` exists and is importable.
- [ ] `[viz]` in `pyproject.toml` includes `plotly>=5.0`.
- [ ] Existing tests unaffected.

---

## Epic B — `power_curve_by_n` Plotly

---

### B1 `power_curve_by_n` Plotly figure + wire-up

**Status:** Done
**Claimed by:** Claude
**Est.:** 2–3 hours
**Depends on:** A1
**Progress note:** Complete. `plotly_curve_by_n` implemented in `plot_backends.py` using `make_subplots(specs=[[{}],[{"secondary_y":True}]])`. Row 1: power line + target hline + target_n vline. Row 2: I-criterion (left y) + D-efficiency (secondary right y). `plot_backend` parameter added to `power_curves.power_curve_by_n`; matplotlib block re-indented under `elif _HAS_MATPLOTLIB:`. Smoke-tested: returns `go.Figure` with 3 traces. 229 tests pass.

**What to do:**

**Step 1 — implement `plotly_curve_by_n` in `plot_backends.py`:**

```python
def plotly_curve_by_n(df, power_cfg, target_n):
    """Two-panel Plotly figure: power vs n (top) + design metrics (bottom)."""
    if not _HAS_PLOTLY:
        raise ImportError("plotly is required. pip install \"lattice-doe[viz]\"")
    from .config import PowerContrastConfig
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=("Statistical Power vs n", "Design Quality Metrics"),
        vertical_spacing=0.08,
    )
    # Row 1: power curve
    fig.add_trace(go.Scatter(
        x=df["n"], y=df["power"], mode="lines+markers", name="Power",
        hovertemplate="n=%{x}<br>power=%{y:.3f}<extra></extra>",
    ), row=1, col=1)
    fig.add_hline(y=power_cfg.power, line_dash="dash", line_color="red",
                  annotation_text=f"Target {power_cfg.power:.0%}", row=1, col=1)
    if target_n is not None:
        fig.add_vline(x=target_n, line_dash="dash", line_color="green",
                      annotation_text=f"n={target_n}", row=1, col=1)
    # Row 2: I-criterion + D-efficiency (secondary y on same row)
    fig.add_trace(go.Scatter(
        x=df["n"], y=df["i_criterion"], mode="lines+markers",
        name="I-criterion", line=dict(color="green"),
        hovertemplate="n=%{x}<br>I-crit=%{y:.4f}<extra></extra>",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=df["n"], y=df["d_efficiency"], mode="lines+markers",
        name="D-efficiency", line=dict(color="orange"), yaxis="y3",
        hovertemplate="n=%{x}<br>D-eff=%{y:.4f}<extra></extra>",
    ), row=2, col=1)
    # Title
    from .config import PowerContrastConfig
    if isinstance(power_cfg, PowerContrastConfig):
        title = (f"Power vs n — Contrast Test "
                 f"(σ={power_cfg.sigma}, α={power_cfg.alpha})")
    else:
        title = (f"Power vs n — Global F-Test "
                 f"(R²={power_cfg.r2_target}, α={power_cfg.alpha})")
    fig.update_layout(template="plotly_white", title=title,
                      yaxis=dict(range=[0, 1.05], title="Power"),
                      xaxis2=dict(title="Sample Size (n)"),
                      yaxis2=dict(title="I-criterion", color="green"),
                      yaxis3=dict(title="D-efficiency", overlaying="y2",
                                  side="right", color="orange"))
    return fig
```

The secondary y-axis for D-efficiency on row 2 requires care: use `make_subplots(specs=[[{}], [{"secondary_y": True}]])` instead of the simpler form above if Plotly's `secondary_y` support is preferred.

**Step 2 — add `plot_backend` to `power_curves.py:power_curve_by_n`:**

Add parameter after `figsize`:
```python
plot_backend: Literal["matplotlib", "plotly"] = "matplotlib",
```

In the plotting block, replace:
```python
fig = None
if plot and _HAS_MATPLOTLIB:
    # ... existing matplotlib code ...
```
with:
```python
fig = None
if plot:
    if plot_backend == "plotly":
        from .plot_backends import plotly_curve_by_n
        fig = plotly_curve_by_n(df, power_cfg, target_n)
    elif _HAS_MATPLOTLIB:
        # ... existing matplotlib code unchanged ...
```

**Acceptance criteria:**
- [ ] `power_curves.power_curve_by_n(..., plot=True, plot_backend="plotly")` returns a `plotly.graph_objects.Figure`.
- [ ] `plot=False` (default) ignores `plot_backend` — returns `None` for `"figure"`.
- [ ] `plot=True, plot_backend="matplotlib"` (default) still returns a matplotlib Figure.
- [ ] `ImportError` with install hint when plotly is absent.
- [ ] Two-panel figure renders correctly in Jupyter (hover, zoom work).

---

## Epic C — `power_curve_by_effect` Plotly

---

### C1 `power_curve_by_effect` Plotly figure + wire-up

**Status:** Open
**Claimed by:**
**Est.:** 1–2 hours
**Depends on:** A1

**What to do:**

**Step 1 — implement `plotly_curve_by_effect` in `plot_backends.py`:**

Single-panel figure:
```python
def plotly_curve_by_effect(df, power_cfg, min_detectable, n):
    if not _HAS_PLOTLY:
        raise ImportError(...)
    from .config import PowerContrastConfig
    is_contrast = isinstance(power_cfg, PowerContrastConfig)
    x_col = "effect_size"
    x_label = "Effect Size Multiplier" if is_contrast else "R² Effect Size"
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df[x_col], y=df["power"], mode="lines+markers", name="Power",
        hovertemplate=f"{x_label}=%{{x:.3f}}<br>power=%{{y:.3f}}<extra></extra>",
    ))
    fig.add_hline(y=power_cfg.power, line_dash="dash", line_color="green",
                  annotation_text=f"Target {power_cfg.power:.0%}")
    fig.add_hline(y=0.80, line_dash="dot", line_color="gray",
                  annotation_text="80%")
    if min_detectable is not None:
        fig.add_vline(x=min_detectable, line_dash="dash", line_color="orange",
                      annotation_text=f"MDE={min_detectable:.3f}")
    if is_contrast:
        title = f"Power vs Effect Size at n={n} — Contrast Test (σ={power_cfg.sigma}, α={power_cfg.alpha})"
    else:
        title = f"Power vs Effect Size at n={n} — Global F-Test (α={power_cfg.alpha})"
    fig.update_layout(template="plotly_white", title=title,
                      xaxis_title=x_label, yaxis_title="Statistical Power",
                      yaxis=dict(range=[0, 1.05]))
    return fig
```

**Step 2 — add `plot_backend` to `power_curves.py:power_curve_by_effect`:**

Same pattern as B1: add `plot_backend="matplotlib"` parameter, dispatch in the plotting block.

**Acceptance criteria:**
- [ ] `power_curves.power_curve_by_effect(..., plot=True, plot_backend="plotly")` returns a `go.Figure`.
- [ ] MDE vertical line present when `min_detectable` is not None.
- [ ] Defaults unchanged.

---

## Epic D — `power_surface_2d` Plotly

---

### D1 `power_surface_2d` Plotly figure + wire-up

**Status:** Open
**Claimed by:**
**Est.:** 2–3 hours
**Depends on:** A1

**What to do:**

This is the most complex chart: a heatmap with an overlaid target-power contour.

**Step 1 — implement `plotly_surface_2d` in `plot_backends.py`:**

```python
def plotly_surface_2d(power_grid, axis1, axis2, power_cfg, param1, param2):
    if not _HAS_PLOTLY:
        raise ImportError(...)
    fig = go.Figure()
    # Base heatmap: rows = axis1 (param1), cols = axis2 (param2)
    fig.add_trace(go.Heatmap(
        z=power_grid,
        x=axis2,
        y=axis1,
        colorscale="Viridis",
        zmin=0, zmax=1,
        colorbar=dict(title="Power"),
        hovertemplate=f"{param1}=%{{y:.3g}}<br>{param2}=%{{x:.3g}}<br>power=%{{z:.3f}}<extra></extra>",
    ))
    # Target-power contour overlay
    fig.add_trace(go.Contour(
        z=power_grid,
        x=axis2,
        y=axis1,
        contours=dict(
            start=power_cfg.power,
            end=power_cfg.power,
            size=1e-6,        # effectively a single contour line
            coloring="none",
        ),
        line=dict(color="white", width=2, dash="dash"),
        showscale=False,
        name=f"Target power = {power_cfg.power:.2f}",
        hoverinfo="skip",
    ))
    fig.update_layout(
        template="plotly_white",
        title=f"Power Surface: {param1} × {param2}  (target={power_cfg.power:.2f}, white contour)",
        xaxis_title=param2,
        yaxis_title=param1,
    )
    return fig
```

Note: `power_grid` rows correspond to `axis1` (param1 values) and columns to `axis2` (param2 values), matching the existing `power_curves.power_surface_2d` layout where `G2, G1 = np.meshgrid(axis2, axis1)`.

**Step 2 — add `plot_backend` to `power_curves.py:power_surface_2d`:**

Same pattern: add `plot_backend="matplotlib"` parameter, dispatch in the plotting block.

**Acceptance criteria:**
- [ ] `power_curves.power_surface_2d(..., plot=True, plot_backend="plotly")` returns a `go.Figure`.
- [ ] Heatmap orientation matches matplotlib output (param1 on y-axis, param2 on x-axis).
- [ ] White dashed contour marks the target power level.
- [ ] Hover shows param1 value, param2 value, and power.

---

## Epic E — `power_sensitivity` Plotly

---

### E1 `power_sensitivity` Plotly figure + wire-up

**Status:** Open
**Claimed by:**
**Est.:** 1–2 hours
**Depends on:** A1

**What to do:**

`power_sensitivity` has two sub-modes (contrast: sweep sigma; R²: sweep r2_target).
Both produce the same chart shape — a single line with reference lines.

**Step 1 — implement `plotly_sensitivity` in `plot_backends.py`:**

```python
def plotly_sensitivity(df, power_cfg, nominal_pwr, n):
    if not _HAS_PLOTLY:
        raise ImportError(...)
    from .config import PowerContrastConfig
    is_contrast = isinstance(power_cfg, PowerContrastConfig)

    if is_contrast:
        x_col, x_label = "sigma", "σ (residual standard deviation)"
        nominal_x = power_cfg.sigma
        title = f"Power Sensitivity to σ  (n = {n})"
    else:
        x_col, x_label = "r2_target", "R² (population effect size)"
        nominal_x = power_cfg.r2_target
        title = f"Power Sensitivity to R²  (n = {n})"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df[x_col], y=df["power"], mode="lines+markers", name="Power",
        hovertemplate=f"{x_label}=%{{x:.3f}}<br>power=%{{y:.3f}}<extra></extra>",
    ))
    fig.add_vline(x=nominal_x, line_dash="dash", line_color="gray",
                  annotation_text=f"Nominal = {nominal_x}")
    fig.add_hline(y=power_cfg.power, line_dash="dash", line_color="red",
                  annotation_text=f"Target {power_cfg.power:.0%}")
    fig.add_hline(y=nominal_pwr, line_dash="dot", line_color="steelblue",
                  annotation_text=f"Power @ nominal: {nominal_pwr:.3f}")
    fig.update_layout(
        template="plotly_white", title=title,
        xaxis_title=x_label, yaxis_title="Statistical Power",
        yaxis=dict(range=[0, 1.05]),
    )
    return fig
```

**Step 2 — add `plot_backend` to `api.py:power_sensitivity`:**

Add `plot_backend: Literal["matplotlib", "plotly"] = "matplotlib"` after `figsize`.

Replace the two `if plot:` blocks (one in the R² branch, one in the contrast branch) with dispatchers:
```python
fig = None
if plot:
    if plot_backend == "plotly":
        from .plot_backends import plotly_sensitivity
        fig = plotly_sensitivity(df, power_cfg, float(nominal_pwr), n)
    else:
        try:
            import matplotlib.pyplot as plt
            # ... existing matplotlib code unchanged ...
        except ImportError:
            pass
```

Note: `power_sensitivity` currently uses a lazy `import matplotlib.pyplot as plt` inside the `if plot:` block. Keep that pattern for the matplotlib path; the plotly path adds a parallel branch.

Also add `Literal` to the `typing` import in `api.py` if not already present.

**Acceptance criteria:**
- [ ] `power_sensitivity(..., plot=True, plot_backend="plotly")` returns `{"figure": go.Figure, ...}`.
- [ ] Both contrast and R² modes produce a correct chart.
- [ ] Reference lines (nominal, target, nominal_pwr) are all present.
- [ ] `plot=False` still returns `{"figure": None, ...}` regardless of `plot_backend`.

---

## Epic F — Wiring & Exports

---

### F1 Exports & public API wiring

**Status:** Open
**Claimed by:**
**Est.:** 1 hour
**Depends on:** B1, C1, D1, E1

**What to do:**

1. **Export `power_surface_2d` from `__init__.py`:**
   ```python
   from .power_curves import power_surface_2d  # noqa: F401
   ```
   Add `"power_surface_2d"` to `__all__`.

2. **Thread `plot_backend` through api.py wrappers** (even though they discard the figure, for API consistency):
   - `power_curve_by_n` wrapper: add `plot_backend="matplotlib"` param, pass it to `_power_curve_by_n_impl`.
   - `power_curve_by_effect` wrapper: same.
   - `generate_power_curves`: add `plot_backend="matplotlib"` param, pass through to both underlying calls.

3. **Update `__init__.py` re-exports** for the wrappers to include `plot_backend` (no code change needed in `__init__.py` itself — the signature change in `api.py` is enough since it re-exports via `from .api import ...`).

4. **Docstring note on the api.py wrappers:** Add a note to `power_curve_by_n` and `power_curve_by_effect` docstrings:
   > Note: This wrapper returns the curve DataFrame only. To access the figure (including Plotly figures), call `lattice_doe.power_curves.power_curve_by_n(...)` directly.

**Acceptance criteria:**
- [ ] `from lattice_doe import power_surface_2d` works.
- [ ] `power_surface_2d` appears in `__all__` in `__init__.py`.
- [ ] `power_curve_by_n(..., plot_backend="plotly")` passes the parameter through without error (even though it's discarded).
- [ ] `generate_power_curves(..., plot_backend="plotly")` passes through.

---

## Epic G — Tests & Docs

---

### G1 Unit tests

**Status:** Open
**Claimed by:**
**Est.:** 2–3 hours
**Depends on:** F1

**What to do:**

Create `tests/test_plot_backends.py` (or add to an existing test file).

**`TestPlotlyBackendImport`:**
- `test_import_without_plotly`: mock `plotly` as absent; assert each helper raises `ImportError` with `"viz"` or `"plotly"` in the message.

**`TestPlotlyCurveByN`** (requires plotly — use `pytest.importorskip("plotly")`):
- `test_returns_plotly_figure`: call `power_curves.power_curve_by_n(..., plot=True, plot_backend="plotly")` — assert `result["figure"]` is a `plotly.graph_objects.Figure`.
- `test_matplotlib_default_unchanged`: call with `plot=True, plot_backend="matplotlib"` — assert `result["figure"]` is `None` (since matplotlib isn't installed in CI) or a `Figure`.
- `test_plot_false_ignores_backend`: call with `plot=False, plot_backend="plotly"` — assert `result["figure"] is None`.

**`TestPlotlyCurveByEffect`** (requires plotly):
- `test_returns_plotly_figure`: `power_curves.power_curve_by_effect(..., plot=True, plot_backend="plotly")` returns `go.Figure`.
- `test_plot_false_ignores_backend`: `plot=False` → `figure is None`.

**`TestPlotlySurface2d`** (requires plotly):
- `test_returns_plotly_figure`: `power_curves.power_surface_2d(..., plot=True, plot_backend="plotly")` returns `go.Figure`.
- `test_plot_false_ignores_backend`.

**`TestPlotlySensitivity`** (requires plotly):
- `test_contrast_returns_plotly_figure`: `power_sensitivity(..., plot=True, plot_backend="plotly")` with `PowerContrastConfig` returns `go.Figure`.
- `test_r2_returns_plotly_figure`: same with `PowerR2Config`.
- `test_plot_false_ignores_backend`.

**`TestPowerSurface2dExport`:**
- `test_power_surface_2d_importable`: `from lattice_doe import power_surface_2d` — no error.

All tests must use minimal `n_points`/`grid_points` (3–5) and `starts=1` to keep runtime short.

**Acceptance criteria:**
- [ ] All tests pass with `pip install -e ".[viz,dev]"`.
- [ ] Tests are skipped (not failed) when plotly is absent.
- [ ] No test runs for more than 30 seconds.

---

### G2 Documentation updates

**Status:** Open
**Claimed by:**
**Est.:** 1 hour
**Depends on:** G1

**What to do:**

**`README.md`:**
- Add `plotly>=5.0` to the `[viz]` install snippet in the Installation section.
- Add a "Interactive Charts (Plotly)" subsection under the Power Curves section:
  ```bash
  pip install -e ".[viz]"
  ```
  ```python
  result = power_curve_by_n(
      formula, factors, power_cfg, design_opts=opts,
      plot=True, plot_backend="plotly",
  )
  result["figure"].show()   # interactive in Jupyter / browser
  ```
  Mention: hover tooltips, zoom/pan, one-click PNG export (camera icon), and Streamlit compatibility (`st.plotly_chart(result["figure"])`).

**`docs/recipes.md`:**
- Add Recipe 8 "Interactive Plotly power charts":
  - `power_curve_by_n` with plotly backend.
  - `power_sensitivity` with plotly backend.
  - Note on direct use of `power_curves.power_curve_by_n` for figures.

**`ENHANCEMENTS.md`:**
- Move Enhancement #13 from the backlog to the Completed table.

**Acceptance criteria:**
- [ ] `README.md` has a Plotly install snippet and usage example.
- [ ] `docs/recipes.md` has Recipe 8.
- [ ] ENHANCEMENTS.md marks #13 as done.
- [ ] All code examples are consistent with the implemented API.

---

## Suggested Build Order

```
A1
 ├── B1
 ├── C1
 ├── D1
 └── E1
      ↓ (all four done)
      F1
       ↓
       G1 → G2
```

B1, C1, D1, and E1 are independent of each other — they can be worked in parallel
(or in any order) once A1 is done. F1 requires all four to be complete.

**Estimated total:** 2–3 days (8 tickets × 30 min – 3 hrs each).
