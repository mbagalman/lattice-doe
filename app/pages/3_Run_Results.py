"""
Page 3 — Run the design and display results (Epic E, tickets E1–E6).

E1  Config summary + "Generate design" button; builds power_cfg and design_opts
    from session state; calls find_optimal_design inside a spinner;
    captures errors and warnings.
E2  Report metrics card (6 st.metric columns) + "Full report" JSON expander.
E3  Design table + CSV download.
E4  Buckets table + CSV download.
E5  Power curve expander: analytical n-sweep using the result's noncentrality,
    rendered as an interactive Plotly chart with target-power and chosen-n lines.
E6  Excel workbook download (design, buckets, report sheets); degrades gracefully
    if xlsxwriter is not installed.
"""

from __future__ import annotations

import importlib.util
import io

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import f as scipy_f
from scipy.stats import ncf as scipy_ncf

from state import init_state, render_sidebar

st.set_page_config(page_title="Run & Results — Lattice DOE", layout="wide")
init_state()
render_sidebar()

try:
    from iopt_power_design import find_optimal_design, find_multiresponse_design
    from iopt_power_design.config import (
        DesignOptions,
        PowerContrastConfig,
        PowerGLMContrastConfig,
        PowerR2Config,
        MultiResponseOptions,
        ResponseSpec,
    )
    from iopt_power_design.contrasts import contrast_from_scenarios
    from iopt_power_design._request_builder import build_power_cfg, build_design_opts
    _HAS_IOPT = True
except ImportError:
    _HAS_IOPT = False

_HAS_PLOTLY = importlib.util.find_spec("plotly") is not None
_HAS_XLSXWRITER = importlib.util.find_spec("xlsxwriter") is not None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _factors_to_spec(factors: list[dict]) -> dict:
    spec: dict = {}
    for f in factors:
        spec[f["name"]] = (
            (f["low"], f["high"]) if f["type"] == "Continuous" else list(f["levels"])
        )
    return spec


def _parse_matrix(text: str) -> np.ndarray:
    rows = []
    for line in text.strip().splitlines():
        line = line.strip()
        if line:
            rows.append([float(x) for x in line.replace(",", " ").split()])
    return np.array(rows)


def _parse_vector(text: str) -> np.ndarray:
    return np.array([float(x) for x in text.replace(",", " ").split()])


def _build_power_cfg(ss: dict):
    """Build PowerContrastConfig, PowerR2Config, or PowerGLMContrastConfig from session state."""
    if ss["power_mode"] == "contrast":
        if ss["contrast_input_mode"] == "matrix":
            L = _parse_matrix(ss["L_text"])
            delta = _parse_vector(ss["delta_text"])
        else:
            factors = ss["factors"]
            factor_spec = _factors_to_spec(factors)
            scenario_a = {f["name"]: ss.get(f"scen_a_{f['name']}") for f in factors}
            scenario_b = {f["name"]: ss.get(f"scen_b_{f['name']}") for f in factors}
            L, delta = contrast_from_scenarios(
                formula=ss["formula"],
                factors=factor_spec,
                scenario_a=scenario_a,
                scenario_b=scenario_b,
                sesoi=float(ss["sesoi"]),
            )
        return build_power_cfg(dict(
            power_mode="contrast", L=L, delta=delta,
            alpha=float(ss["alpha"]), power=float(ss["power_target"]),
            sigma=float(ss["sigma"]), max_n=int(ss["max_n"]),
        ))
    elif ss["power_mode"] == "glm":
        L = _parse_matrix(ss["L_text"])
        delta = _parse_vector(ss["delta_text"])
        return build_power_cfg(dict(
            power_mode="glm", L=L, delta=delta,
            baseline=float(ss.get("glm_baseline", 0.20)),
            family=ss.get("glm_family", "binomial"),
            link=ss.get("glm_link", "").strip() or None,
            alpha=float(ss["alpha"]), power=float(ss["power_target"]),
            max_n=int(ss["max_n"]),
        ))
    else:
        return build_power_cfg(dict(
            power_mode="r2", r2_target=float(ss["r2_target"]),
            alpha=float(ss["alpha"]), power=float(ss["power_target"]),
            max_n=int(ss["max_n"]), lambda_mode=ss["lambda_mode"],
        ))


def _build_multi_response_cfg(ss: dict) -> "MultiResponseOptions":
    """Build MultiResponseOptions from session-state mr_responses list."""
    import numpy as np  # noqa: PLC0415
    specs = []
    for r in ss.get("mr_responses", []):
        r_alpha = float(r.get("alpha", ss.get("alpha", 0.05)))
        r_power = float(r.get("power", ss.get("power_target", 0.80)))
        r_sigma = float(r.get("sigma", 1.0))
        r_weight = float(r.get("weight", 1.0))
        r_formula = r.get("formula", "").strip() or None
        _r_mode = r.get("power_mode", "contrast")
        if _r_mode == "glm":
            L = _parse_matrix(r.get("L_text", ""))
            delta = _parse_vector(r.get("delta_text", ""))
            pcfg = build_power_cfg(dict(
                power_mode="glm", L=L, delta=delta,
                baseline=float(r.get("glm_baseline", 0.20)),
                family=r.get("glm_family", "binomial"),
                link=r.get("glm_link", "").strip() or None,
                alpha=r_alpha, power=r_power,
            ))
        elif _r_mode == "contrast":
            L = _parse_matrix(r.get("L_text", ""))
            delta = _parse_vector(r.get("delta_text", ""))
            pcfg = build_power_cfg(dict(
                power_mode="contrast", L=L, delta=delta,
                alpha=r_alpha, power=r_power, sigma=r_sigma,
            ))
        else:
            pcfg = build_power_cfg(dict(
                power_mode="r2", r2_target=float(r.get("r2_target", 0.15)),
                alpha=r_alpha, power=r_power,
            ))
        specs.append(ResponseSpec(
            name=str(r.get("name", f"R{len(specs) + 1}")),
            power_cfg=pcfg,
            weight=r_weight,
            formula=r_formula,
        ))
    # Parse optional sigma_joint matrix
    sigma_joint = None
    _sj_text = ss.get("mr_sigma_joint", "").strip()
    if _sj_text:
        try:
            _sj_rows = []
            for _line in _sj_text.splitlines():
                _line = _line.strip()
                if _line:
                    _sj_rows.append([float(x) for x in _line.replace(",", " ").split()])
            sigma_joint = np.array(_sj_rows)
        except (ValueError, Exception):
            sigma_joint = None
    return MultiResponseOptions(
        responses=specs,
        power_combination=ss.get("mr_combination", "min"),
        sigma_joint=sigma_joint,
    )


def _build_design_opts(ss: dict) -> DesignOptions:
    _do_d: dict = dict(
        criterion=ss["criterion"],
        starts=int(ss["starts"]),
        random_state=int(ss["random_state"]),
        auto_candidate=bool(ss["auto_candidate"]),
        n_blocks=int(ss.get("n_blocks", 0)),
        block_factor_name=ss.get("block_factor_name", "Block"),
        preallocate_categorical=bool(ss.get("preallocate_categorical", False)),
        alloc_min_per_cell=int(ss.get("alloc_min_per_cell", 1)),
        alloc_max_per_cell=int(ss.get("alloc_max_per_cell", 0)),
        constraint_expr=ss.get("constraint_expr", "").strip() or None,
    )
    if not ss["auto_candidate"]:
        _do_d["candidate_points"] = int(ss["candidate_points"])
    # Split-plot options
    if ss.get("split_plot_enabled", False):
        htc_names = ss.get("sp_htc_factors") or []
        if htc_names:
            _do_d["split_plot"] = dict(
                htc_factors=list(htc_names),
                n_whole_plots=int(ss.get("sp_n_whole_plots", 4)),
                eta=float(ss.get("sp_eta", 1.0)),
                subplots_per_wp=int(ss.get("sp_subplots_per_wp", 0)),
                df_method=str(ss.get("sp_df_method", "auto")),
            )
    return build_design_opts(_do_d)


def _jsonify(obj):
    """Recursively convert numpy scalars/arrays to JSON-safe Python types."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonify(v) for v in obj]
    return obj


def _approx_power_curve(
    n_vals: list[int],
    report: dict,
    alpha: float,
    power_mode: str,
    lambda_mode: str,
) -> list[float]:
    """
    Approximate power at each n by scaling the run's noncentrality
    and recomputing the noncentral F-test.

    Scaling convention:
      - contrast mode: λ ∝ n
      - R² mode + lambda_mode='n': λ ∝ n
      - R² mode + lambda_mode='n_minus_p': λ ∝ (n - p)

    and recomputing the noncentral F-test. Fast — no additional design builds.
    """
    p = int(report["p"])
    df_num = int(report["df_num"])
    n_result = int(report["n"])
    lambda_result = float(report["noncentrality_lambda"])

    powers = []
    for n in n_vals:
        if n <= p:
            powers.append(0.0)
            continue
        df_denom = n - p
        if power_mode == "r2" and lambda_mode == "n_minus_p":
            denom_ref = max(n_result - p, 1)
            lambda_n = lambda_result * (df_denom / denom_ref)
        else:
            lambda_n = lambda_result * (n / n_result)
        f_crit = scipy_f.ppf(1.0 - alpha, df_num, df_denom)
        powers.append(float(scipy_ncf.sf(f_crit, df_num, df_denom, lambda_n)))
    return powers


def _make_excel(design_df: pd.DataFrame, buckets_df: pd.DataFrame, report: dict) -> bytes:
    """Build an in-memory .xlsx workbook with design, buckets, and report sheets."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        design_df.to_excel(writer, sheet_name="Design", index=False)
        buckets_df.to_excel(writer, sheet_name="Buckets", index=False)
        report_rows = [
            {"Key": k, "Value": str(_jsonify(v))}
            for k, v in report.items()
        ]
        pd.DataFrame(report_rows).to_excel(writer, sheet_name="Report", index=False)
    return output.getvalue()


# ---------------------------------------------------------------------------
# Page content
# ---------------------------------------------------------------------------

st.title("Step 3 \u00b7 Run & Results")

ss = st.session_state
factors = ss.get("factors", [])
formula = ss.get("formula", "")
power_mode = ss.get("power_mode", "contrast")

# ---------------------------------------------------------------------------
# Config summary (E1)
# ---------------------------------------------------------------------------
with st.expander("Current configuration", expanded=not bool(ss.get("result"))):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Factors**")
        if factors:
            for f in factors:
                if f["type"] == "Continuous":
                    st.caption(f"`{f['name']}` \u2014 continuous [{f['low']}, {f['high']}]")
                else:
                    lvls = ", ".join(f["levels"]) if f["levels"] else "(none)"
                    st.caption(f"`{f['name']}` \u2014 categorical: {lvls}")
        else:
            st.caption("None defined \u2014 go to Page 1.")
        _formula_display = formula or "\u2014"
        st.markdown(f"**Formula:** `{_formula_display}`")
    with c2:
        _mode_labels = {"contrast": "Contrast-based", "r2": "Global R\u00b2", "glm": "GLM (logistic/Poisson)"}
        mode_label = _mode_labels.get(power_mode, power_mode)
        st.markdown(f"**Power mode:** {mode_label}")
        if power_mode == "contrast":
            input_mode = ss.get("contrast_input_mode", "matrix")
            st.caption(f"Input: {'L matrix' if input_mode == 'matrix' else 'Scenario builder'}")
            if input_mode == "matrix":
                _l_preview = (ss.get("L_text", "").strip() or "\u2014")[:40]
                _d_preview = (ss.get("delta_text", "").strip() or "\u2014")[:40]
                st.caption(f"L text: `{_l_preview}`")
                st.caption(f"\u03b4 text: `{_d_preview}`")
            else:
                st.caption(f"SESOI = {ss.get('sesoi', 1.0)}")
        elif power_mode == "glm":
            _glm_fam = ss.get("glm_family", "binomial")
            _glm_bl = ss.get("glm_baseline", 0.20)
            st.caption(f"Family: {_glm_fam}  |  baseline = {_glm_bl}")
            _l_preview = (ss.get("L_text", "").strip() or "\u2014")[:40]
            _d_preview = (ss.get("delta_text", "").strip() or "\u2014")[:40]
            st.caption(f"L text: `{_l_preview}`")
            st.caption(f"\u03b4 (LP scale): `{_d_preview}`")
        else:
            st.caption(f"R\u00b2 target = {ss.get('r2_target', 0.15)}")
            st.caption(f"\u03bb mode: {ss.get('lambda_mode', 'n')}")
    with c3:
        st.markdown("**Power parameters**")
        st.caption(f"\u03b1 = {ss.get('alpha', 0.05)}")
        st.caption(f"Target power = {ss.get('power_target', 0.80)}")
        if power_mode == "contrast":
            st.caption(f"\u03c3 = {ss.get('sigma', 1.0)}")
        st.caption(f"Max n = {ss.get('max_n', 500)}")
        st.caption(
            f"Criterion: {ss.get('criterion','I')}  |  "
            f"Starts: {ss.get('starts', 8)}  |  "
            f"Seed: {ss.get('random_state', 42)}"
        )

st.markdown("---")

# ---------------------------------------------------------------------------
# Readiness check + run button (E1)
# ---------------------------------------------------------------------------
_issues: list[str] = []
if not factors:
    _issues.append("No factors defined \u2014 go to **Page 1**.")
if not formula.strip():
    _issues.append("No formula entered \u2014 go to **Page 1**.")
_needs_contrast_matrix = (
    power_mode == "glm"
    or (power_mode == "contrast" and ss.get("contrast_input_mode") == "matrix")
)
if _needs_contrast_matrix:
    if not ss.get("L_text", "").strip():
        _issues.append("L matrix is empty \u2014 go to **Page 2**.")
    if not ss.get("delta_text", "").strip():
        _issues.append("\u03b4 (effect size) is empty \u2014 go to **Page 2**.")

for issue in _issues:
    st.warning(issue)

col_run, col_clear = st.columns([2, 1])
with col_run:
    run_clicked = st.button(
        "\U0001f50d  Generate design",
        type="primary",
        disabled=bool(_issues) or not _HAS_IOPT,
        use_container_width=True,
    )
with col_clear:
    if ss.get("result") is not None:
        if st.button("Clear result", use_container_width=True):
            ss["result"] = None
            ss["run_error"] = None
            st.rerun()

if not _HAS_IOPT:
    st.error("iopt_power_design is not installed. Run `pip install -e '.[app]'` first.")

if run_clicked and not _issues and _HAS_IOPT:
    try:
        design_opts = _build_design_opts(ss)
        factor_spec = _factors_to_spec(factors)
        with st.spinner("Searching for the optimal design\u2026"):
            if ss.get("mr_enabled", False):
                multi_cfg = _build_multi_response_cfg(ss)
                result = find_multiresponse_design(
                    formula=formula,
                    factors=factor_spec,
                    multi_cfg=multi_cfg,
                    design_opts=design_opts,
                )
                ss["_last_power_cfg"] = None
            else:
                power_cfg = _build_power_cfg(ss)
                result = find_optimal_design(
                    formula=formula,
                    factors=factor_spec,
                    power_cfg=power_cfg,
                    design_opts=design_opts,
                )
                ss["_last_power_cfg"] = power_cfg
        ss["result"] = result
        ss["run_error"] = None
        st.rerun()
    except Exception as exc:
        ss["run_error"] = str(exc)
        ss["result"] = None

if ss.get("run_error"):
    st.error(f"Run failed: {ss['run_error']}")
    with st.expander("Troubleshooting tips"):
        st.markdown(
            """
- **L columns \u2260 p**: check Page 1 \u2192 'Model matrix columns' for the correct p.
- **max_n too small**: raise it on Page 2 \u2192 'Max sample size'.
- **Power never reached**: increase starts, lower target power, or widen factor ranges.
- **Constraint too strict**: relax or remove the constraint expression on Page 2.
- **Empty L row**: every row of L must contain at least one non-zero value.
"""
        )

# ---------------------------------------------------------------------------
# Show results only after a successful run
# ---------------------------------------------------------------------------
result = ss.get("result")
if result is None:
    if not ss.get("run_error"):
        st.info("Click **Generate design** above to run.")
    st.stop()

# Detect multi-response result (flat dict) vs single-response (nested "report" dict).
_is_mr = "report" not in result
if _is_mr:
    design_df = result["design"]
    buckets_df = result["buckets"]
    _mr_target = max(
        (float(r.get("power", ss.get("power_target", 0.80))) for r in ss.get("mr_responses", [])),
        default=float(ss.get("power_target", 0.80)),
    )
    report = {
        "n": result["n"],
        "achieved_power": result["achieved_power"],
        "target_power": _mr_target,
        "elapsed_sec": result.get("elapsed_sec", 0.0),
        "criterion": "Compound" if result.get("compound_criterion") else ss.get("criterion", "I"),
        "search_strategy": result.get("search_strategy", "—"),
        "warnings": result.get("warnings", []),
        "combination_rule": result.get("combination_rule", "min"),
        "combined_power": result["achieved_power"],
        "p": result.get("p"),
        **{f"{r['name']}_power": r["power"] for r in result.get("responses", [])},
    }
    if "joint_power" in result:
        report["joint_power"] = result["joint_power"]
else:
    report = result["report"]
    design_df = result["design_df"]
    buckets_df = result["buckets_df"]

for w in report.get("warnings", []):
    st.warning(f"Run warning: {w}")

st.markdown("---")

# ---------------------------------------------------------------------------
# E2 — Report metrics card
# ---------------------------------------------------------------------------
st.subheader("Results")

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Sample size n", report["n"])
m2.metric("Achieved power", f"{report['achieved_power']:.1%}")
m3.metric("Target power", f"{report['target_power']:.1%}")
m4.metric("Elapsed", f"{report['elapsed_sec']:.1f} s")
m5.metric("Criterion", report["criterion"])
m6.metric("Strategy", report.get("search_strategy", "\u2014"))

with st.expander("Full report (JSON)"):
    st.json(_jsonify(report))

# Per-response powers (multi-response mode)
_mr_response_keys = [k for k in report if k.endswith("_power") and k != "achieved_power"]
if _mr_response_keys:
    with st.expander("Per-response powers", expanded=True):
        _pr_data = {k.replace("_power", ""): [f"{report[k]:.3f}"] for k in sorted(_mr_response_keys)}
        if "combined_power" in report:
            _pr_data["combined"] = [f"{report['combined_power']:.3f}"]
        st.dataframe(
            __import__("pandas").DataFrame(_pr_data, index=["power"]),
            use_container_width=True,
        )
        if "combination_rule" in report:
            st.caption(f"Combination rule: **{report['combination_rule']}**")

if report.get("test_type") == "wald_chi2":
    with st.expander("GLM summary", expanded=True):
        gc1, gc2, gc3, gc4 = st.columns(4)
        gc1.metric("Family", report.get("family", "\u2014"))
        gc2.metric("Baseline", f"{report.get('baseline', 0):.4g}")
        gc3.metric("Test type", "Wald \u03c7\u00b2")
        if report.get("glm_weight") is not None:
            gc4.metric("Fisher weight w", f"{report['glm_weight']:.4f}")
        if report.get("link"):
            st.caption(f"Link function: **{report['link']}**")

if "split_plot" in report:
    sp = report["split_plot"]
    with st.expander("Split-plot summary", expanded=True):
        spc1, spc2, spc3, spc4 = st.columns(4)
        spc1.metric("Whole plots", sp.get("n_whole_plots", "—"))
        spc2.metric("Sub-plots / WP", sp.get("subplots_per_wp", "—"))
        spc3.metric("η (variance ratio)", f"{sp.get('eta', 0.0):.2f}")
        spc4.metric("df method", sp.get("df_method", "—"))
        htc = ", ".join(sp.get("htc_factors", []))
        etc = ", ".join(sp.get("etc_factors", []))
        st.caption(f"HTC factors: {htc or '—'}  |  ETC factors: {etc or '—'}")

st.markdown("---")

# ---------------------------------------------------------------------------
# E3 — Design table + CSV
# ---------------------------------------------------------------------------
col_h, col_dl = st.columns([3, 1])
col_h.subheader(f"Design matrix  ({len(design_df)} runs)")
col_dl.download_button(
    "\u2b07 Download design CSV",
    data=design_df.to_csv(index=False),
    file_name="design.csv",
    mime="text/csv",
    use_container_width=True,
)
st.dataframe(design_df, use_container_width=True, height=280)

st.markdown("---")

# ---------------------------------------------------------------------------
# E4 — Buckets table + CSV
# ---------------------------------------------------------------------------
col_h2, col_dl2 = st.columns([3, 1])
col_h2.subheader(f"Unique run allocations  ({len(buckets_df)} unique)")
col_dl2.download_button(
    "\u2b07 Download buckets CSV",
    data=buckets_df.to_csv(index=False),
    file_name="buckets.csv",
    mime="text/csv",
    use_container_width=True,
)
st.dataframe(buckets_df, use_container_width=True, height=220)

st.markdown("---")

# ---------------------------------------------------------------------------
# E5 — Power curve (analytical approximation) — single-response OLS only
# ---------------------------------------------------------------------------
_is_glm_result = report.get("test_type") == "wald_chi2"
if _is_mr:
    with st.expander("Power curve (n sweep)"):
        st.info(
            "Analytical power curve approximation is not available for multi-response runs. "
            "Use `power_curve_by_n_multiresponse()` from the Python API instead."
        )
elif _is_glm_result:
    with st.expander("Power curve (n sweep)"):
        st.info(
            "Power curve by n is available for GLM designs \u2014 "
            "use the **Analysis** tab or `power_curve_by_n()` from the Python API. "
            "The noncentral-F approximation used here is not valid for Wald \u03c7\u00b2 tests."
        )
else:
    with st.expander("Power curve (n sweep)"):
        n_result = int(report["n"])
        max_n_cfg = int(ss.get("max_n", 500))
        p_model = int(report["p"])

        n_max_default = min(n_result * 2, max_n_cfg)
        n_max_slider = st.slider(
            "Upper n for sweep",
            min_value=min(n_result, p_model + 2),
            max_value=max_n_cfg,
            value=n_max_default,
            help="Right edge of the power curve. Drag to extend or narrow the range.",
        )

        if st.button("Plot power curve"):
            if not _HAS_PLOTLY:
                st.warning("Plotly is not installed. Run `pip install -e '.[app]'`.")
            else:
                import plotly.graph_objects as go

                n_vals = list(range(p_model + 1, n_max_slider + 1))
                alpha = float(ss.get("alpha", 0.05))
                power_mode_ss = ss.get("power_mode", "contrast")
                lambda_mode_ss = ss.get("lambda_mode", "n")
                powers = _approx_power_curve(
                    n_vals=n_vals,
                    report=report,
                    alpha=alpha,
                    power_mode=power_mode_ss,
                    lambda_mode=lambda_mode_ss,
                )
                target_power = float(report["target_power"])
                actual_power = float(report["achieved_power"])

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=n_vals, y=powers, mode="lines", name="Approx. power",
                    line=dict(color="#1f77b4", width=2),
                    hovertemplate="n=%{x}<br>power=%{y:.3f}<extra></extra>",
                ))
                fig.add_trace(go.Scatter(
                    x=[n_result], y=[actual_power], mode="markers",
                    name=f"Result (n={n_result}, power={actual_power:.3f})",
                    marker=dict(color="#d62728", size=10),
                    hovertemplate=f"n={n_result}<br>power={actual_power:.3f}<extra></extra>",
                ))
                fig.add_hline(
                    y=target_power, line_dash="dash", line_color="grey",
                    annotation_text=f"Target {target_power:.0%}",
                    annotation_position="bottom right",
                )
                fig.add_vline(
                    x=n_result, line_dash="dot", line_color="#d62728",
                    annotation_text=f"n={n_result}",
                    annotation_position="top right",
                )
                fig.update_layout(
                    xaxis_title="Sample size n", yaxis_title="Power",
                    yaxis=dict(range=[0, 1.05]),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    margin=dict(t=40, b=40), height=400,
                )
                st.plotly_chart(fig, use_container_width=True)
                if power_mode_ss == "r2" and lambda_mode_ss == "n_minus_p":
                    scale_note = "\u03bb scaled with (n - p) (R\u00b2 mode, n_minus_p)."
                else:
                    scale_note = "\u03bb scaled with n."
                st.caption(
                    f"Power approximated from the run result using noncentral-F scaling; {scale_note} "
                    "The red dot is the exact achieved power."
                )

st.markdown("---")

# ---------------------------------------------------------------------------
# E6 — Export (Excel + redundant CSV buttons)
# ---------------------------------------------------------------------------
_HAS_JINJA2 = importlib.util.find_spec("jinja2") is not None

st.subheader("Export")
exp_cols = st.columns([1, 1, 1, 1])

with exp_cols[0]:
    st.download_button(
        "\u2b07 Design CSV",
        data=design_df.to_csv(index=False),
        file_name="design.csv",
        mime="text/csv",
        use_container_width=True,
    )

with exp_cols[1]:
    if _HAS_XLSXWRITER:
        st.download_button(
            "\u2b07 Excel workbook",
            data=_make_excel(design_df, buckets_df, report),
            file_name="lattice_design.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    else:
        st.info("Install `lattice-doe[extras]` to enable Excel export.")

# G2 — JSON report download
with exp_cols[2]:
    import json as _json
    _report_json = _json.dumps(_jsonify(report), indent=2).encode()
    st.download_button(
        "\u2b07 Report JSON",
        data=_report_json,
        file_name="lattice_report.json",
        mime="application/json",
        use_container_width=True,
        help="Download the full run report as a JSON file.",
    )

# D3 — HTML report download
with exp_cols[3]:
    if _HAS_JINJA2:
        import os as _os
        import tempfile as _tempfile
        from iopt_power_design import generate_report as _generate_report
        _power_cfg_d3 = st.session_state.get("_last_power_cfg")
        _formula_d3 = st.session_state.get("formula", "")
        _factors_d3_raw = st.session_state.get("factors", [])
        _factors_d3 = {f["name"]: (f["low"], f["high"]) if f["type"] == "Continuous" else list(f["levels"]) for f in _factors_d3_raw}
        if _power_cfg_d3 is not None and _factors_d3:
            # Generate HTML once per result; cache bytes in session state so
            # repeated renders (slider drags, etc.) don't re-write to disk.
            _result_key = id(result)
            if ss.get("_html_report_result_id") != _result_key:
                _tmp_path = None
                try:
                    _fd, _tmp_path = _tempfile.mkstemp(suffix=".html")
                    _os.close(_fd)
                    _generate_report(
                        result=result,
                        formula=_formula_d3,
                        factors=_factors_d3,
                        power_cfg=_power_cfg_d3,
                        output_path=_tmp_path,
                        include_power_curve=False,
                    )
                    with open(_tmp_path, "rb") as _fh:
                        ss["_html_report_bytes"] = _fh.read()
                    ss["_html_report_result_id"] = _result_key
                except Exception as _rpt_err:
                    ss.pop("_html_report_bytes", None)
                    ss.pop("_html_report_result_id", None)
                    st.warning(f"Report generation failed: {_rpt_err}")
                finally:
                    if _tmp_path is not None and _os.path.exists(_tmp_path):
                        _os.unlink(_tmp_path)
            if ss.get("_html_report_bytes"):
                st.download_button(
                    "\u2b07 HTML report",
                    data=ss["_html_report_bytes"],
                    file_name="lattice_report.html",
                    mime="text/html",
                    use_container_width=True,
                    help="Download a self-contained HTML report (opens offline in any browser).",
                )
        else:
            st.button("\u2b07 HTML report", disabled=True, use_container_width=True,
                      help="Re-run the design to generate the report.")
    else:
        st.info('Install `lattice-doe[report]` to enable HTML report download.')
