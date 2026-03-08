"""
Page 4 — Advanced analysis and export (Epics F & G).

F1  Sensitivity analysis  — power vs sigma (contrast) or vs R² (R² mode)
F2  Minimum detectable effect (MDE) — bisect for smallest detectable effect
F3  Compare criteria — run I/D/A and display side-by-side summary + chart
G1  YAML config export — download current config as CLI-compatible YAML
G2  JSON report download — download full run report as JSON
"""

from __future__ import annotations

import importlib.util
import io
import json

import numpy as np
import streamlit as st

from state import init_state, render_sidebar

st.set_page_config(page_title="Analysis — I-Opt Power Design", layout="wide")
init_state()
render_sidebar()

try:
    from iopt_power_design import (
        compare_criteria,
        min_detectable_effect,
        power_sensitivity,
    )
    from iopt_power_design.config import (
        DesignOptions,
        PowerContrastConfig,
        PowerR2Config,
    )
    from iopt_power_design.contrasts import contrast_from_scenarios

    _HAS_IOPT = True
except ImportError:
    _HAS_IOPT = False

_HAS_PLOTLY = importlib.util.find_spec("plotly") is not None
_HAS_YAML = importlib.util.find_spec("yaml") is not None


# ---------------------------------------------------------------------------
# Helpers (mirrors those in 3_Run_Results.py)
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
    """Reconstruct PowerContrastConfig or PowerR2Config from session state."""
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
        return PowerContrastConfig(
            L=L,
            delta=delta,
            alpha=float(ss["alpha"]),
            power=float(ss["power_target"]),
            sigma=float(ss["sigma"]),
            max_n=int(ss["max_n"]),
        )
    else:
        return PowerR2Config(
            r2_target=float(ss["r2_target"]),
            alpha=float(ss["alpha"]),
            power=float(ss["power_target"]),
            max_n=int(ss["max_n"]),
            lambda_mode=ss["lambda_mode"],
        )


def _build_design_opts(ss: dict) -> DesignOptions:
    kwargs: dict = dict(
        criterion=ss["criterion"],
        starts=int(ss["starts"]),
        random_state=int(ss["random_state"]),
        auto_candidate=bool(ss["auto_candidate"]),
    )
    if not ss["auto_candidate"]:
        kwargs["candidate_points"] = int(ss["candidate_points"])
    expr = ss.get("constraint_expr", "").strip()
    if expr:
        kwargs["constraint_expr"] = expr
    return DesignOptions(**kwargs)


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


def _build_yaml(ss: dict) -> str:
    """Serialize current session state to a CLI-compatible YAML string."""
    if not _HAS_YAML:
        return ""
    import yaml  # type: ignore

    factors = ss.get("factors", [])

    def _default_scenarios() -> tuple[dict, dict]:
        """Build valid fallback scenarios from factor definitions."""
        scenario_a: dict = {}
        scenario_b: dict = {}
        for f in factors:
            name = f.get("name", "")
            if not name:
                continue
            if f.get("type") == "Continuous":
                low = float(f.get("low", 0.0))
                high = float(f.get("high", low))
                scenario_a[name] = low
                scenario_b[name] = high
            else:
                levels = list(f.get("levels", []))
                if levels:
                    scenario_a[name] = levels[0]
                    scenario_b[name] = levels[-1]
        return scenario_a, scenario_b
    factor_block: dict = {}
    for f in factors:
        if f["type"] == "Continuous":
            factor_block[f["name"]] = {"type": "continuous", "range": [f["low"], f["high"]]}
        else:
            factor_block[f["name"]] = {"type": "categorical", "levels": list(f["levels"])}

    cfg: dict = {
        "formula": ss.get("formula", "~ 1 + A + B"),
        "factors": factor_block,
        "alpha": float(ss.get("alpha", 0.05)),
        "power": float(ss.get("power_target", 0.80)),
    }

    if ss.get("power_mode") == "contrast":
        cfg["sigma"] = float(ss.get("sigma", 1.0))
        if ss.get("contrast_input_mode") == "scenario":
            scenario_a = {
                f["name"]: ss.get(f"scen_a_{f['name']}")
                for f in factors
            }
            scenario_b = {
                f["name"]: ss.get(f"scen_b_{f['name']}")
                for f in factors
            }
            cfg["contrast"] = {
                "scenario_a": scenario_a,
                "scenario_b": scenario_b,
                "sesoi": float(ss.get("sesoi", 1.0)),
            }
        else:
            L_text = ss.get("L_text", "").strip()
            delta_text = ss.get("delta_text", "").strip()
            used_matrix = False
            if L_text and delta_text:
                try:
                    L = _parse_matrix(L_text)
                    delta = _parse_vector(delta_text)
                    # Export explicit contrast only when dimensions are valid.
                    if L.ndim == 2 and L.shape[0] > 0 and delta.ndim == 1 and len(delta) == L.shape[0]:
                        cfg["contrast"] = {
                            "L": L.tolist(),
                            "delta": delta.tolist(),
                        }
                        used_matrix = True
                except Exception:
                    pass
            if not used_matrix:
                # Keep export CLI-compatible even when matrix input is empty/invalid.
                scenario_a, scenario_b = _default_scenarios()
                cfg["contrast"] = {
                    "scenario_a": scenario_a,
                    "scenario_b": scenario_b,
                    "sesoi": float(ss.get("sesoi", 1.0)),
                }
    else:
        cfg["r2_target"] = float(ss.get("r2_target", 0.15))

    design_block: dict = {
        "auto_candidate": bool(ss.get("auto_candidate", True)),
        "starts": int(ss.get("starts", 8)),
        "criterion": ss.get("criterion", "I"),
        "random_state": int(ss.get("random_state", 42)),
        "algo": "fedorov",
    }
    if not ss.get("auto_candidate", True):
        design_block["candidate_points"] = int(ss.get("candidate_points", 2000))
    expr = ss.get("constraint_expr", "").strip()
    if expr:
        design_block["constraint_expr"] = expr
    cfg["design"] = design_block

    return yaml.dump(cfg, default_flow_style=False, sort_keys=False, allow_unicode=True)


# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------

st.title("Step 4 \u00b7 Advanced Analysis & Export")

ss = st.session_state
result = ss.get("result")

# Guard: most analysis requires a run result
_HAS_RESULT = result is not None
_no_result_msg = (
    "\u26a0\ufe0f No design result yet. "
    "Generate a design on **Step 3 \u00b7 Run & Results** first."
)

# ---------------------------------------------------------------------------
# G1 — YAML config export (always available, no result needed)
# ---------------------------------------------------------------------------

st.subheader("Export Configuration")

col_yaml, col_json = st.columns(2)

with col_yaml:
    if _HAS_YAML:
        yaml_str = _build_yaml(ss)
        st.download_button(
            label="\u2b07\ufe0f Download YAML config",
            data=yaml_str,
            file_name="iopt_config.yaml",
            mime="text/yaml",
            help=(
                "Download the current factor, formula, and power settings as a "
                "CLI-compatible YAML file. Use with: "
                "`iopt-design --config iopt_config.yaml`"
            ),
        )
        with st.expander("Preview YAML"):
            st.code(yaml_str, language="yaml")
    else:
        st.info(
            "Install `pyyaml` to enable YAML export: `pip install pyyaml`"
        )

# G2 — JSON report download
with col_json:
    if _HAS_RESULT:
        report = result.get("report", {})
        json_bytes = json.dumps(_jsonify(report), indent=2).encode()
        st.download_button(
            label="\u2b07\ufe0f Download report JSON",
            data=json_bytes,
            file_name="iopt_report.json",
            mime="application/json",
            help="Download the full run report as a JSON file.",
        )
    else:
        st.markdown("*JSON report download available after a successful run (Step 3).*")

st.markdown("---")

# ---------------------------------------------------------------------------
# Analysis sections — require a result
# ---------------------------------------------------------------------------

if not _HAS_RESULT:
    st.info(_no_result_msg)
    st.stop()

if not _HAS_IOPT:
    st.error(
        "iopt_power_design is not importable. "
        "Run `pip install -e '.[app]'` from the project root."
    )
    st.stop()

factors = ss.get("factors", [])
formula = ss.get("formula", "")
factor_spec = _factors_to_spec(factors)

# Try to reconstruct the power config from session state.
try:
    power_cfg = _build_power_cfg(ss)
    _cfg_ok = True
except Exception as exc:
    st.warning(
        f"Could not reconstruct power configuration from current settings: {exc}\n\n"
        "Re-check the contrast/R\u00b2 inputs on **Step 2** and re-run on **Step 3**."
    )
    _cfg_ok = False

design_opts = _build_design_opts(ss)

# ===========================================================================
# F1 — Sensitivity analysis
# ===========================================================================

st.subheader("F1 \u00b7 Sensitivity Analysis")
st.markdown(
    "Evaluate how **achieved power** changes as a key assumption varies, "
    "using the fixed design from Step 3 (no new search required)."
)

if not _cfg_ok:
    st.warning("Fix power configuration on Step 2 to enable sensitivity analysis.")
else:
    is_contrast = ss["power_mode"] == "contrast"
    design_df = result["design_df"]

    if is_contrast:
        col1, col2, col3 = st.columns(3)
        sigma_nom = float(ss.get("sigma", 1.0))
        with col1:
            sigma_min = st.number_input(
                "\u03c3 min", min_value=1e-4, value=sigma_nom * 0.5, format="%.4g",
                key="sens_sigma_min",
            )
        with col2:
            sigma_max = st.number_input(
                "\u03c3 max", min_value=1e-4, value=sigma_nom * 2.0, format="%.4g",
                key="sens_sigma_max",
            )
        with col3:
            sigma_pts = st.slider("Points", 5, 50, 25, key="sens_sigma_pts")
        sweep_label = f"\u03c3 range [{sigma_min:.3g}, {sigma_max:.3g}], {sigma_pts} points"
    else:
        col1, col2, col3 = st.columns(3)
        r2_nom = float(ss.get("r2_target", 0.15))
        with col1:
            r2_min = st.number_input(
                "R\u00b2 min", min_value=0.001, max_value=0.99, value=max(0.01, r2_nom * 0.3),
                format="%.3f", key="sens_r2_min",
            )
        with col2:
            r2_max = st.number_input(
                "R\u00b2 max", min_value=0.001, max_value=0.99, value=min(0.99, r2_nom * 3.0),
                format="%.3f", key="sens_r2_max",
            )
        with col3:
            r2_pts = st.slider("Points", 5, 50, 25, key="sens_r2_pts")
        sweep_label = f"R\u00b2 range [{r2_min:.3f}, {r2_max:.3f}], {r2_pts} points"

    if st.button("Run sensitivity", key="btn_sensitivity"):
        try:
            with st.spinner("Running sensitivity sweep\u2026"):
                if is_contrast:
                    sens = power_sensitivity(
                        formula=formula,
                        factors=factor_spec,
                        power_cfg=power_cfg,
                        design_df=design_df,
                        sigma_range=(float(sigma_min), float(sigma_max)),
                        sigma_points=int(sigma_pts),
                        design_opts=design_opts,
                    )
                else:
                    sens = power_sensitivity(
                        formula=formula,
                        factors=factor_spec,
                        power_cfg=power_cfg,
                        design_df=design_df,
                        r2_range=(float(r2_min), float(r2_max)),
                        r2_points=int(r2_pts),
                        design_opts=design_opts,
                    )
            ss["_sensitivity_result"] = sens
            ss["_sensitivity_contrast"] = is_contrast
        except Exception as exc:
            st.error(f"Sensitivity sweep failed: {exc}")

    sens = ss.get("_sensitivity_result")
    if sens is not None:
        data = sens["data"]
        nominal_power = float(sens["nominal_power"])
        target_power = float(ss.get("power_target", 0.80))

        if _HAS_PLOTLY:
            import plotly.graph_objects as go

            if ss.get("_sensitivity_contrast", True):
                x_col, x_label = "sigma", "\u03c3 (residual std dev)"
                x_nominal = float(sens["sigma_nominal"])
            else:
                x_col, x_label = "r2_target", "R\u00b2 target"
                x_nominal = float(sens["r2_nominal"])

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=data[x_col],
                    y=data["power"],
                    mode="lines",
                    name="Achieved power",
                    line=dict(color="#1f77b4", width=2),
                )
            )
            fig.add_hline(
                y=target_power,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Target {target_power:.0%}",
                annotation_position="top right",
            )
            fig.add_hline(
                y=nominal_power,
                line_dash="dot",
                line_color="steelblue",
                annotation_text=f"Nominal {nominal_power:.1%}",
                annotation_position="bottom right",
            )
            fig.add_vline(
                x=x_nominal,
                line_dash="dash",
                line_color="gray",
                annotation_text="Nominal value",
                annotation_position="top left",
            )
            fig.update_layout(
                xaxis_title=x_label,
                yaxis_title="Statistical power",
                yaxis=dict(range=[0, 1.05]),
                height=380,
                margin=dict(t=30, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.dataframe(data)

        st.caption(f"Nominal power = **{nominal_power:.1%}** | Sweep: {sweep_label}")

st.markdown("---")

# ===========================================================================
# F2 — Minimum detectable effect
# ===========================================================================

st.subheader("F2 \u00b7 Minimum Detectable Effect")
st.markdown(
    "Find the **smallest effect** your fixed design can detect at a given power level "
    "(no new design search; analytical bisection on the fixed X matrix)."
)

if not _cfg_ok:
    st.warning("Fix power configuration on Step 2 to compute the MDE.")
else:
    mde_power = st.number_input(
        "Target power for MDE",
        min_value=0.50,
        max_value=0.999,
        value=float(ss.get("power_target", 0.80)),
        step=0.01,
        format="%.2f",
        key="mde_power_target",
        help="Power level at which to find the minimum detectable effect.",
    )

    if st.button("Compute MDE", key="btn_mde"):
        try:
            with st.spinner("Computing MDE\u2026"):
                mde_result = min_detectable_effect(
                    design_df=result["design_df"],
                    formula=formula,
                    factors=factor_spec,
                    power_cfg=power_cfg,
                    target_power=float(mde_power),
                    design_opts=design_opts,
                )
            ss["_mde_result"] = mde_result
        except Exception as exc:
            st.error(f"MDE computation failed: {exc}")

    mde = ss.get("_mde_result")
    if mde is not None:
        mode = mde.get("mode", "contrast")
        mde_val = float(mde["mde"])
        ach_pwr = float(mde["achieved_power"])
        n_used = int(mde["n"])

        col1, col2, col3 = st.columns(3)
        if mode == "contrast":
            delta_arr = np.asarray(power_cfg.delta, dtype=float).reshape(-1)
            col1.metric(
                "MDE (scale on \u03b4)",
                f"{mde_val:.4f}\u00d7",
                help="Multiply your configured \u03b4 by this factor to get the minimum detectable effect.",
            )
            if delta_arr.size == 1:
                abs_effect = mde_val * abs(float(delta_arr[0]))
                col2.metric(
                    "MDE absolute |\u03b4|",
                    f"{abs_effect:.4g}",
                    help="Absolute detectable effect for the single configured contrast.",
                )
            else:
                abs_effect_norm = mde_val * float(np.linalg.norm(delta_arr))
                col2.metric(
                    "MDE absolute \u2016\u03b4\u2016\u2082",
                    f"{abs_effect_norm:.4g}",
                    help=(
                        "For multi-contrast \u03b4 vectors, this reports a norm-based magnitude "
                        "to avoid misleading cancellation from signed means."
                    ),
                )
        else:
            col1.metric(
                "Min detectable R\u00b2",
                f"{mde_val:.4f}",
                help="Smallest R\u00b2 detectable at the specified power.",
            )
            col2.metric(
                "Cohen\u2019s f\u00b2",
                f"{mde_val / (1 - mde_val):.4f}" if mde_val < 1 else "\u221e",
            )

        col3.metric("Achieved power at MDE", f"{ach_pwr:.1%}")

        if mode == "contrast" and np.asarray(power_cfg.delta).size > 1:
            st.caption(
                "Multi-contrast mode: absolute MDE is reported as \u2016\u03b4\u2016\u2082 \u00d7 scale "
                "(vector magnitude), not a signed mean."
            )

        if mde_val == float("inf"):
            st.warning(
                "The design cannot achieve the target power even for very large effects. "
                "Consider increasing n (raise Max sample size on Step 2 and re-run)."
            )

st.markdown("---")

# ===========================================================================
# F3 — Compare criteria
# ===========================================================================

st.subheader("F3 \u00b7 Compare Optimality Criteria")
st.markdown(
    "Run the design search under **I**, **D**, and/or **A** optimality and "
    "compare the resulting sample sizes and achieved power."
)
st.warning(
    "\u23f1 This runs a complete design search for **each** selected criterion. "
    "It may take several minutes for complex problems. "
    "Reduce **Random starts** on Step 2 to speed it up."
)

if not _cfg_ok:
    st.warning("Fix power configuration on Step 2 to run criteria comparison.")
else:
    selected_criteria = st.multiselect(
        "Criteria to compare",
        options=["I", "D", "A"],
        default=["I", "D", "A"],
        key="compare_criteria_select",
        help=(
            "I: minimise average prediction variance (recommended). "
            "D: maximise det(X\u2019X). "
            "A: minimise tr((X\u2019X)\u207b\u00b9)."
        ),
    )

    if st.button(
        "Run comparison",
        key="btn_compare",
        disabled=len(selected_criteria) == 0,
        type="secondary",
    ):
        try:
            with st.spinner(
                f"Running design search for {', '.join(selected_criteria)}\u2026 "
                "(this may take a while)"
            ):
                comparison = compare_criteria(
                    formula=formula,
                    factors=factor_spec,
                    power_cfg=power_cfg,
                    design_opts=design_opts,
                    criteria=selected_criteria,
                )
            ss["_comparison_result"] = comparison
        except Exception as exc:
            st.error(f"Comparison failed: {exc}")

    comp = ss.get("_comparison_result")
    if comp is not None:
        summary = comp["summary"]

        st.subheader("Summary")
        st.dataframe(
            summary.style.format(
                {
                    "achieved_power": "{:.1%}",
                    "elapsed_sec": "{:.2f} s",
                    "condition_number": "{:.1f}",
                    "d_efficiency": "{:.3f}",
                }
            ),
            use_container_width=True,
        )

        if _HAS_PLOTLY:
            import plotly.graph_objects as go

            target_power = float(ss.get("power_target", 0.80))
            colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
            criteria_list = list(summary["criterion"])
            bar_colors = [colors[i % len(colors)] for i in range(len(criteria_list))]

            fig = go.Figure()

            # Achieved power bars
            fig.add_trace(
                go.Bar(
                    x=criteria_list,
                    y=list(summary["achieved_power"]),
                    name="Achieved power",
                    marker_color=bar_colors,
                    yaxis="y",
                )
            )

            # n markers on secondary axis
            fig.add_trace(
                go.Scatter(
                    x=criteria_list,
                    y=list(summary["n"]),
                    name="n (runs)",
                    mode="markers+text",
                    marker=dict(size=12, symbol="diamond", color="black"),
                    text=[str(v) for v in summary["n"]],
                    textposition="top center",
                    yaxis="y2",
                )
            )

            fig.add_hline(
                y=target_power,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Target {target_power:.0%}",
                annotation_position="top right",
            )

            fig.update_layout(
                barmode="group",
                xaxis_title="Criterion",
                yaxis=dict(title="Achieved power", range=[0, 1.05], tickformat=".0%"),
                yaxis2=dict(
                    title="n (runs)",
                    overlaying="y",
                    side="right",
                    showgrid=False,
                ),
                legend=dict(x=0.01, y=0.99),
                height=400,
                margin=dict(t=30, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

        # Offer design downloads for each criterion
        with st.expander("Download individual designs"):
            for crit, res in comp["results"].items():
                df = res["design_df"]
                csv = df.to_csv(index=False).encode()
                st.download_button(
                    label=f"\u2b07\ufe0f {crit}-optimal design (n={len(df)})",
                    data=csv,
                    file_name=f"design_{crit.lower()}_optimal.csv",
                    mime="text/csv",
                    key=f"dl_crit_{crit}",
                )
