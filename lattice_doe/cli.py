# cli.py
# License: MIT
"""
Command-line interface for Lattice DOE
=====================================================

Example
-------
$ lattice --config config.yml --out ./design --excel
$ lattice --config config.yml --dry-run
$ lattice --config config.yml -v

Config schema (YAML/JSON)
-------------------------
# Minimal contrast-powered example
formula: "~ 1 + A + B + A:B"
factors:
  A: [low, med, high]
  B: [0.0, 10.0]         # OR use {B: [0.0, 10.0]} as list; tuple in JSON isn't standard

# Contrast definition via scenarios (recommended)
contrast:
  scenario_a: {A: low, B: 5.0}
  scenario_b: {A: high, B: 5.0}
  sesoi: 1.0

alpha: 0.05
power: 0.9
sigma: 2.0

# Design options (all optional)
design:
  candidate_points: 4000
  starts: 8
  algo: fedorov   # or coordinate, detmax
  criterion: I
  max_iter: 200
  random_state: 42
  xtx_jitter: 1.0e-8

# Output (all optional)
output:
  basename: design
  excel: true     # also write design.xlsx; csv/json always written
"""
from __future__ import annotations

import argparse
import json
import sys
import os
import logging
from pathlib import Path
from typing import Any, Dict, Optional, List, Union

try:
    import yaml  # type: ignore
    _HAS_YAML = True
except Exception:  # pragma: no cover - optional dep
    _HAS_YAML = False

import pandas as pd

from .api import find_optimal_design, find_multiresponse_design
from .config import (
    PowerContrastConfig, PowerR2Config, PowerGLMContrastConfig,
    DesignOptions, MultiResponseOptions, ResponseSpec,
)
from .contrasts import contrast_from_scenarios
from ._request_builder import build_power_cfg, build_design_opts

logger = logging.getLogger("lattice")


def _ensure_utf8_output() -> None:
    """Reconfigure stdout/stderr to UTF-8 so help text and reports render correctly.

    Help text and design summaries contain non-ASCII glyphs (e.g. subscripts like
    n₀, Greek letters like σ). On Windows the console defaults to a legacy code page
    (cp1252) that cannot encode these, raising UnicodeEncodeError. Switching the
    streams to UTF-8 avoids that. Streams that do not support reconfigure (e.g. an
    io.StringIO substituted in tests) are left untouched.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8")
        except (ValueError, OSError):  # pragma: no cover - environment dependent
            pass


# -------------------------
# Parsing helpers
# -------------------------

def _load_config(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yml", ".yaml"}:
        if not _HAS_YAML:
            logger.error("PyYAML is required to read YAML configs. Install with: pip install pyyaml")
            raise RuntimeError("PyYAML is not installed.")
        try:
            return yaml.safe_load(text)  # type: ignore
        except Exception as e:
            raise ValueError(f"Failed to parse YAML file at {path}: {e}") from e
            
    # Fallback to JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON file at {path}: {e}") from e


def _validate_config_keys(cfg: Dict[str, Any]) -> None:
    """
    Check for presence of required top-level config keys before processing.
    Raises KeyError with actionable messages.
    """
    if not isinstance(cfg, dict):
        raise ValueError(
            "Config file is empty or not a valid YAML/JSON mapping. "
            "Expected a top-level key/value structure."
        )
    if "formula" not in cfg:
        raise KeyError("Config validation failed: 'formula' key is required.")
    if "factors" not in cfg:
        raise KeyError("Config validation failed: 'factors' key is required.")
        
    if "responses" not in cfg and "contrast" not in cfg and "r2_target" not in cfg and "family" not in cfg:
        raise KeyError(
            "Config validation failed: Must contain either a 'responses' list (multi-response), "
            "a 'contrast' block, an 'r2_target' key, or a 'family' key (GLM mode)."
        )
        
    if "contrast" in cfg:
        c = cfg.get("contrast")
        if not isinstance(c, dict):
            raise KeyError(
                f"Config validation failed: 'contrast' must be a mapping, "
                f"got {type(c).__name__!r}. "
                "Expected either {scenario_a, scenario_b, sesoi} for scenario mode "
                "or {L, delta} for explicit matrix mode."
            )
        is_scenario = {"scenario_a", "scenario_b", "sesoi"} <= c.keys()
        is_explicit = {"L", "delta"} <= c.keys()
        if not is_scenario and not is_explicit:
            raise KeyError(
                "Config 'contrast' block validation failed: "
                "Must contain [scenario_a, scenario_b, sesoi] OR explicit [L, delta]."
            )
            

def _as_factors(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize factor specs from config to the package FactorSpec.

    Accepted forms:
    - Explicit (recommended):
        A:
          type: continuous
          range: [0, 10]
        B:
          type: categorical
          levels: [low, med, high]
    - Shorthand:
        A: [0, 10]       # continuous (2-number list)
        B: [low, high]   # categorical (list of strings)
    """
    factors: Dict[str, Any] = {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"Config 'factors' must be a dictionary, not {type(raw).__name__}."
        )
        
    for k, v in raw.items():
        if isinstance(v, dict):
            t = (v.get("type") or "").lower()
            if t == "continuous":
                r = v.get("range")
                if not (isinstance(r, (list, tuple)) and len(r) == 2):
                    raise ValueError(f"Factor '{k}': continuous requires 'range: [low, high]'")
                factors[k] = (float(r[0]), float(r[1]))
            elif t == "categorical":
                levels = v.get("levels")
                if not isinstance(levels, (list, tuple)) or len(levels) == 0:
                    raise ValueError(f"Factor '{k}': categorical requires non-empty 'levels: [...]'")
                factors[k] = list(levels)
            else:
                raise ValueError(f"Factor '{k}': unknown type {t!r}; use 'continuous' or 'categorical'")
        elif isinstance(v, (list, tuple)):
            # Heuristic: list of strings => categorical; 2-number list => continuous; else categorical
            if len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
                factors[k] = (float(v[0]), float(v[1]))
            else:
                factors[k] = list(v)
        else:
            raise ValueError(f"Factor '{k}': unsupported spec {v!r}")
    return factors


def _make_power_cfg(cfg: Dict[str, Any], formula: str, factors: Dict[str, Any]):
    import numpy as np  # noqa: PLC0415
    alpha = float(cfg.get("alpha", 0.05))
    power = float(cfg.get("power", 0.8))
    sigma = float(cfg.get("sigma", 1.0))
    max_n = int(cfg.get("max_n", 500))

    # GLM path — triggered by presence of 'family' key
    family = cfg.get("family")
    if family is not None:
        baseline = cfg.get("baseline")
        if baseline is None:
            raise ValueError("'baseline' is required when 'family' is set.")
        link = cfg.get("link", None)
        c = cfg.get("contrast") or {}
        if {"scenario_a", "scenario_b", "sesoi"} <= c.keys():
            L, delta = contrast_from_scenarios(
                formula, factors, c["scenario_a"], c["scenario_b"], float(c["sesoi"]),
            )
        elif "L" in c and "delta" in c:
            L = np.asarray(c["L"], dtype=float)
            delta = np.asarray(c["delta"], dtype=float)
        else:
            raise ValueError(
                "GLM mode requires a 'contrast' block with "
                "[scenario_a, scenario_b, sesoi] or explicit [L, delta]."
            )
        return build_power_cfg(
            dict(power_mode="glm", L=L, delta=delta, baseline=float(baseline),
                 family=str(family), link=link, alpha=alpha, power=power, max_n=max_n),
        )

    if "contrast" in cfg:
        c = cfg["contrast"] or {}
        if {"scenario_a", "scenario_b", "sesoi"} <= c.keys():
            L, delta = contrast_from_scenarios(
                formula, factors, c["scenario_a"], c["scenario_b"], float(c["sesoi"]),
            )
        elif "L" in c and "delta" in c:
            L = np.asarray(c["L"], dtype=float)
            delta = np.asarray(c["delta"], dtype=float)
        else:
            raise ValueError("Internal error: Invalid contrast config structure.")
        return build_power_cfg(
            dict(power_mode="contrast", L=L, delta=delta,
                 alpha=alpha, power=power, sigma=sigma, max_n=max_n),
        )

    # Otherwise R^2 mode
    if "r2_target" not in cfg:
        raise ValueError("Internal error: Missing r2_target or contrast block.")
    return build_power_cfg(
        dict(power_mode="r2", r2_target=float(cfg["r2_target"]),
             alpha=alpha, power=power, sigma=sigma, max_n=max_n),
    )


def _make_multi_response_cfg(
    cfg: Dict[str, Any], formula: str, factors: Dict[str, Any]
) -> MultiResponseOptions:
    """Parse the ``responses:`` YAML block into a :class:`MultiResponseOptions`."""
    raw_responses = cfg.get("responses")
    if not isinstance(raw_responses, list) or len(raw_responses) < 2:
        raise ValueError(
            "Config 'responses' must be a list of at least 2 response entries."
        )

    global_alpha = float(cfg.get("alpha", 0.05))
    global_power = float(cfg.get("power", 0.8))

    specs: List[ResponseSpec] = []
    for r in raw_responses:
        if not isinstance(r, dict):
            raise ValueError(f"Each 'responses' entry must be a mapping; got {type(r).__name__!r}.")
        name = str(r.get("name", "")).strip()
        if not name:
            raise KeyError("Each response entry must have a non-empty 'name' key.")

        r_alpha = float(r.get("alpha", global_alpha))
        r_power = float(r.get("power", global_power))
        r_sigma = float(r.get("sigma", 1.0))
        r_weight = float(r.get("weight", 1.0))
        r_formula = r.get("formula", None)
        eff_formula = r_formula or formula

        r_max_n = int(r.get("max_n", cfg.get("max_n", 500)))

        if "contrast" in r:
            c = r["contrast"] or {}
            if {"scenario_a", "scenario_b", "sesoi"} <= c.keys():
                L, delta = contrast_from_scenarios(
                    eff_formula,
                    factors,
                    c["scenario_a"],
                    c["scenario_b"],
                    float(c["sesoi"]),
                )
            elif "L" in c and "delta" in c:
                import numpy as np  # local import
                L = np.asarray(c["L"], dtype=float)
                delta = np.asarray(c["delta"], dtype=float)
            else:
                raise KeyError(
                    f"Response '{name}' contrast block must contain "
                    "[scenario_a, scenario_b, sesoi] or explicit [L, delta]."
                )
            # GLM response: contrast block present AND family/baseline keys present.
            if "family" in r or "baseline" in r:
                r_family = str(r.get("family", "binomial"))
                r_link = r.get("link") or None
                r_baseline = r.get("baseline")
                if r_baseline is None:
                    raise KeyError(
                        f"Response '{name}' has GLM keys but is missing required 'baseline'."
                    )
                pcfg: Union[PowerContrastConfig, PowerR2Config, PowerGLMContrastConfig] = (
                    build_power_cfg(
                        dict(power_mode="glm", L=L, delta=delta,
                             baseline=float(r_baseline), family=r_family, link=r_link,
                             alpha=r_alpha, power=r_power, max_n=r_max_n),
                    )
                )
            else:
                pcfg = build_power_cfg(
                    dict(power_mode="contrast", L=L, delta=delta,
                         alpha=r_alpha, power=r_power, sigma=r_sigma, max_n=r_max_n),
                )
        elif "r2_target" in r:
            pcfg = build_power_cfg(
                dict(power_mode="r2", r2_target=float(r["r2_target"]),
                     alpha=r_alpha, power=r_power, sigma=r_sigma, max_n=r_max_n),
            )
        else:
            raise KeyError(
                f"Response '{name}' must contain either a 'contrast' block or an 'r2_target' key."
            )

        specs.append(ResponseSpec(name=name, power_cfg=pcfg, formula=r_formula, weight=r_weight))

    power_combination = str(cfg.get("power_combination", "min"))
    return MultiResponseOptions(responses=specs, power_combination=power_combination)


def _apply_sp_cli_args(cfg: Dict[str, Any], args) -> Dict[str, Any]:
    """Merge CLI split-plot flags into *cfg*'s ``split_plot`` block.

    Any of the five SP flags (``--htc-factors``, ``--n-whole-plots``,
    ``--eta``, ``--subplots-per-wp``, ``--df-method``) triggers the merge when
    provided.  Previously only ``--htc-factors`` / ``--n-whole-plots`` did,
    so ``--eta`` / ``--subplots-per-wp`` / ``--df-method`` were silently
    ignored when given alone (CR-26).

    Returns a shallow copy of *cfg* with the updated ``split_plot`` key;
    the original dict is never mutated.
    """
    if not any(x is not None for x in (
        args.htc_factors, args.n_whole_plots, args.eta,
        args.subplots_per_wp, args.df_method,
    )):
        return cfg  # nothing to do

    cfg = dict(cfg)  # shallow copy so original isn't mutated
    sp_block = dict(cfg.get("split_plot") or {})
    if args.htc_factors is not None:
        sp_block["htc_factors"] = [f.strip() for f in args.htc_factors.split(",") if f.strip()]
    if args.n_whole_plots is not None:
        sp_block["n_whole_plots"] = args.n_whole_plots
    if args.eta is not None:
        sp_block["eta"] = args.eta
    if args.subplots_per_wp is not None:
        sp_block["subplots_per_wp"] = args.subplots_per_wp if args.subplots_per_wp > 0 else None
    if args.df_method is not None:
        sp_block["df_method"] = args.df_method
    cfg["split_plot"] = sp_block
    return cfg


def _apply_glm_cli_args(cfg: Dict[str, Any], args) -> Dict[str, Any]:
    """Merge CLI GLM flags (``--family``, ``--link``, ``--baseline``) into *cfg*.

    Returns a shallow copy of *cfg* with updated keys; the original is never mutated.
    """
    if not any(x is not None for x in (args.family, args.link, args.baseline)):
        return cfg

    cfg = dict(cfg)
    if args.family is not None:
        cfg["family"] = args.family
    if args.link is not None:
        cfg["link"] = args.link
    if args.baseline is not None:
        cfg["baseline"] = args.baseline
    return cfg


def _make_design_opts(cfg: Dict[str, Any]) -> DesignOptions:
    d = cfg.get("design", {})
    # workers: None means serial; YAML can specify an int or omit/null it
    raw_workers = d.get("workers", None)
    workers = int(raw_workers) if raw_workers is not None else None

    # Split-plot options live at the top-level cfg key, not inside "design"
    _sp_d = None
    sp_block = cfg.get("split_plot")
    if sp_block and isinstance(sp_block, dict):
        htc_raw = sp_block.get("htc_factors", [])
        if isinstance(htc_raw, str):
            htc_factors = [f.strip() for f in htc_raw.split(",") if f.strip()]
        else:
            htc_factors = [str(f) for f in htc_raw]
        n_whole_plots = int(sp_block.get("n_whole_plots", 4))
        if htc_factors and n_whole_plots >= 2:
            _sp_d = dict(
                htc_factors=htc_factors,
                n_whole_plots=n_whole_plots,
                eta=float(sp_block.get("eta", 1.0)),
                subplots_per_wp=sp_block.get("subplots_per_wp"),
                df_method=str(sp_block.get("df_method", "auto")),
            )

    _do_d: Dict[str, Any] = dict(
        candidate_points=int(d.get("candidate_points", 2000)),
        auto_candidate=bool(d.get("auto_candidate", False)),
        cand_min=int(d.get("cand_min", 1000)),
        cand_max=int(d.get("cand_max", 10000)),
        cat_cells_cap=int(d.get("cat_cells_cap", 10000)),
        per_cell_alpha=float(d.get("per_cell_alpha", 1.5)),
        per_cell_min=int(d.get("per_cell_min", 5)),
        per_cell_max=int(d.get("per_cell_max", 20)),
        allow_candidate_growth=bool(d.get("allow_candidate_growth", False)),
        growth_factor=float(d.get("growth_factor", 2.0)),
        starts=int(d.get("starts", 5)),
        algo=str(d.get("algo", "fedorov")),
        criterion=str(d.get("criterion", "I")),
        max_iter=int(d.get("max_iter", 1000)),
        random_state=int(d.get("random_state", 123)),
        xtx_jitter=float(d.get("xtx_jitter", 1e-8)),
        constraint_expr=str(d["constraint_expr"]) if d.get("constraint_expr") is not None else None,
        workers=workers,
        parallel_seed_stride=int(d.get("parallel_seed_stride", 10000)),
    )
    if _sp_d is not None:
        _do_d["split_plot"] = _sp_d

    return build_design_opts(_do_d)


# -------------------------
# CLI entry point
# -------------------------

_TEMPLATE_CONTRAST = """\
# lattice config — contrast mode
# Generate with: lattice --template contrast > config.yml

formula: "~ 1 + A + B + A:B"

factors:
  A: [low, high]      # categorical: list of levels
  B: [0.0, 10.0]      # continuous:  [low, high] (two numeric values)

# Power mode: contrast test  (use r2_target instead for global F-test)
contrast:
  # Option 1 — scenario-based (recommended; automatically builds L and delta)
  scenario_a: {{A: low,  B: 5.0}}
  scenario_b: {{A: high, B: 5.0}}
  sesoi: 1.0           # smallest effect of interest (response units)

  # Option 2 — explicit L matrix and delta vector (advanced users)
  # L: [[0, 0, 1, 0]]  # one row per contrast; p columns (must match Patsy encoding)
  # delta: [0.5]        # one element per contrast row

alpha: 0.05
power: 0.80
sigma: 1.0             # assumed residual standard deviation

design:
  auto_candidate: true           # adaptive candidate sizing (recommended)
  candidate_points: 2000         # used only when auto_candidate: false
  cand_min: 1000                 # lower bound for adaptive sizing
  cand_max: 10000                # upper bound for adaptive sizing
  allow_candidate_growth: false  # grow candidate once if conditioning is poor
  starts: 5                      # number of random starts
  algo: fedorov                  # fedorov | coordinate
  criterion: I                   # I = I-optimal (avg prediction variance); D = D-optimal (det X'X); A = A-optimal (sum of coeff variances)
  max_iter: 1000                 # max iterations per start
  random_state: 123              # random seed for reproducibility
  xtx_jitter: 1.0e-8             # ridge for numerical stability
  workers: null                  # null = serial; integer > 1 for parallel starts
  # constraint_expr: "Temperature <= 2 * Pressure"  # optional; string alternative to constraint_func

output:
  basename: design               # output file prefix
  excel: false                   # also write an .xlsx workbook

# Split-plot / hard-to-change factors (optional)
# Uncomment and edit to activate split-plot mode.
# split_plot:
#   htc_factors: [A]             # factor names that are hard-to-change (whole-plot factors)
#   n_whole_plots: 6             # number of whole plots (≥ 2)
#   eta: 1.0                     # variance ratio σ²_wp / σ²_sp (≥ 0; 0 = OLS)
#   subplots_per_wp: 4           # sub-plots per WP; omit for auto
#   df_method: auto              # auto | conservative | sp_only

# Multi-response (optional)
# Replace the 'contrast:' block above with a 'responses:' list to optimise
# a single design for multiple response variables simultaneously.
# The top-level 'contrast:' / 'r2_target' keys must be removed when using this.
# power_combination: min         # min | product | weighted_mean
# responses:
#   - name: Yield
#     sigma: 2.0
#     contrast:
#       scenario_a: {{A: low,  B: 5.0}}
#       scenario_b: {{A: high, B: 5.0}}
#       sesoi: 1.0
#   - name: Purity
#     sigma: 0.5
#     r2_target: 0.20
"""

_TEMPLATE_R2 = """\
# lattice config — global R² mode
# Generate with: lattice --template r2 > config.yml

formula: "~ 1 + A + B + A:B"

factors:
  A: [low, high]      # categorical: list of levels
  B: [0.0, 10.0]      # continuous:  [low, high] (two numeric values)

# Power mode: omnibus F-test for global R²
r2_target: 0.15        # detect R² >= this value (Cohen's f² = r2 / (1 - r2))
alpha: 0.05
power: 0.80

design:
  auto_candidate: true
  starts: 5
  algo: fedorov
  criterion: I                   # I = I-optimal (avg prediction variance); D = D-optimal (det X'X); A = A-optimal (sum of coeff variances)
  random_state: 123
  # constraint_expr: "Temperature <= 2 * Pressure"  # optional; string constraint

output:
  basename: design
  excel: false

# Split-plot / hard-to-change factors (optional)
# Uncomment and edit to activate split-plot mode.
# split_plot:
#   htc_factors: [A]             # factor names that are hard-to-change (whole-plot factors)
#   n_whole_plots: 6             # number of whole plots (≥ 2)
#   eta: 1.0                     # variance ratio σ²_wp / σ²_sp (≥ 0; 0 = OLS)
#   subplots_per_wp: 4           # sub-plots per WP; omit for auto
#   df_method: auto              # auto | conservative | sp_only
"""


_TEMPLATE_GLM_BINOMIAL = """\
# lattice config — GLM binomial/logistic mode
# Generate with: lattice --template glm-binomial > config.yml

formula: "~ 1 + A + B + A:B"

factors:
  A: [low, high]      # categorical: list of levels
  B: [0.0, 10.0]      # continuous:  [low, high] (two numeric values)

# Power mode: Wald chi-square for a binomial (logistic) GLM contrast
family: binomial       # binomial | poisson
link: logit            # logit (default for binomial) | log
baseline: 0.30         # baseline response probability p₀  (0 < p₀ < 1)

contrast:
  # Option 1 — scenario-based (recommended)
  scenario_a: {{A: low,  B: 5.0}}
  scenario_b: {{A: high, B: 5.0}}
  sesoi: 0.15          # smallest effect of interest on the LP scale

  # Option 2 — explicit L matrix and delta vector (advanced users)
  # L: [[0, 0, 1, 0]]
  # delta: [0.15]

alpha: 0.05
power: 0.80

design:
  auto_candidate: true           # adaptive candidate sizing (recommended)
  candidate_points: 2000
  cand_min: 1000
  cand_max: 10000
  starts: 5
  algo: fedorov
  criterion: I
  max_iter: 1000
  random_state: 123
  xtx_jitter: 1.0e-8
  workers: null

output:
  basename: design
  excel: false
"""

_TEMPLATE_GLM_POISSON = """\
# lattice config — GLM Poisson/log mode
# Generate with: lattice --template glm-poisson > config.yml

formula: "~ 1 + A + B + A:B"

factors:
  A: [low, high]      # categorical: list of levels
  B: [0.0, 10.0]      # continuous:  [low, high] (two numeric values)

# Power mode: Wald chi-square for a Poisson (log-linear) GLM contrast
family: poisson        # binomial | poisson
link: log              # log (default for Poisson)
baseline: 2.0          # baseline expected count μ₀  (> 0)

contrast:
  # Option 1 — scenario-based (recommended)
  scenario_a: {{A: low,  B: 5.0}}
  scenario_b: {{A: high, B: 5.0}}
  sesoi: 0.30          # smallest effect of interest on the LP scale

  # Option 2 — explicit L matrix and delta vector (advanced users)
  # L: [[0, 0, 1, 0]]
  # delta: [0.30]

alpha: 0.05
power: 0.80

design:
  auto_candidate: true
  candidate_points: 2000
  cand_min: 1000
  cand_max: 10000
  starts: 5
  algo: fedorov
  criterion: I
  max_iter: 1000
  random_state: 123
  xtx_jitter: 1.0e-8
  workers: null

output:
  basename: design
  excel: false
"""


def _print_template(mode: str) -> None:
    """Print a fully-commented YAML config template to stdout."""
    if mode == "contrast":
        print(_TEMPLATE_CONTRAST)
    elif mode == "r2":
        print(_TEMPLATE_R2)
    elif mode == "glm-binomial":
        print(_TEMPLATE_GLM_BINOMIAL)
    elif mode == "glm-poisson":
        print(_TEMPLATE_GLM_POISSON)
    else:
        raise ValueError(f"Unknown template mode: {mode!r}")


def main(argv: Optional[List[str]] = None) -> int:
    _ensure_utf8_output()
    parser = argparse.ArgumentParser(
        prog="lattice",
        description="Lattice DOE — power-assured optimal design generator",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to YAML/JSON config file (required unless --template is used)",
    )
    parser.add_argument("--out", default="design", help="Output basename (no extension)")
    parser.add_argument("--excel", action="store_true", help="Also write an Excel workbook")
    parser.add_argument(
        "--html-report",
        action="store_true",
        help="Write a self-contained HTML report alongside the CSV outputs "
             "(requires lattice-doe[report])",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and output path, then exit without running design generation."
    )
    parser.add_argument(
        "--template",
        choices=["contrast", "r2", "glm-binomial", "glm-poisson"],
        metavar="MODE",
        help="Print a commented example config (contrast | r2 | glm-binomial | glm-poisson) to stdout and exit.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging output (DEBUG level)"
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        default=False,
        help=(
            "Stream live search progress (phase, iteration, n, power, elapsed) "
            "to stderr while the design search runs. Enabled automatically when "
            "stderr is an interactive terminal; use --no-progress to suppress."
        ),
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        default=False,
        help=(
            "Suppress the live progress stream even on an interactive terminal "
            "(progress is otherwise shown automatically when stderr is a TTY)."
        ),
    )
    parser.add_argument(
        "--sheets",
        metavar="URL_OR_ID",
        default=None,
        help=(
            "Google Spreadsheet URL or ID. Reads config from the 'Config' sheet "
            "and writes design results back to Results/Design/Buckets sheets. "
            "Requires: pip install 'lattice-doe[sheets]'"
        ),
    )
    parser.add_argument(
        "--sheets-credentials",
        metavar="PATH",
        default=None,
        help=(
            "Path to a service account JSON credentials file for Google Sheets. "
            "If omitted, falls back to the GOOGLE_APPLICATION_CREDENTIALS env var, "
            "then to OAuth2 browser flow."
        ),
    )
    parser.add_argument(
        "--excel-template",
        metavar="PATH",
        default=None,
        help=(
            "Create a pre-filled Excel workbook template at PATH (.xlsx) and exit. "
            "Use --template-mode to choose 'r2' (default) or 'contrast'. "
            "Requires: pip install 'lattice-doe[extras]'"
        ),
    )
    parser.add_argument(
        "--template-mode",
        choices=["r2", "contrast", "glm-binomial", "glm-poisson"],
        default="r2",
        help="Example mode for --excel-template: 'r2' (default), 'contrast', 'glm-binomial', or 'glm-poisson'.",
    )
    parser.add_argument(
        "--excel-run",
        metavar="PATH",
        default=None,
        help=(
            "Read config from the 'Config' sheet of an existing .xlsx workbook at PATH, "
            "run the design search, and write Results/Design/Buckets sheets back. "
            "Requires: pip install 'lattice-doe[extras]'"
        ),
    )
    parser.add_argument(
        "--multi-response",
        action="store_true",
        default=False,
        help=(
            "Treat the config as a multi-response design. "
            "Requires a 'responses:' list in the config (see --template contrast). "
            "When the config contains 'responses:', this flag is inferred automatically."
        ),
    )
    parser.add_argument(
        "--robustness-report",
        action="store_true",
        default=False,
        help=(
            "After running the design search, print a compact robustness summary "
            "showing how power changes across ranges of sigma, effect size, and alpha. "
            "No additional dependencies required."
        ),
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        default=False,
        help=(
            "Exit 0 even when the search finishes without reaching the target "
            "power. By default a partial result (report status 'partial') "
            "exits with code 3 so automation can detect it."
        ),
    )
    # GLM flags (override YAML family / link / baseline keys)
    parser.add_argument(
        "--family",
        choices=["binomial", "poisson"],
        default=None,
        metavar="FAMILY",
        help="GLM response family (binomial | poisson). Overrides 'family' in the config.",
    )
    parser.add_argument(
        "--link",
        choices=["logit", "log"],
        default=None,
        metavar="LINK",
        help="GLM link function (logit | log). Overrides 'link' in the config.",
    )
    parser.add_argument(
        "--baseline",
        type=float,
        default=None,
        metavar="B",
        help=(
            "Baseline response value for GLM power (p₀ for binomial, μ₀ for Poisson). "
            "Overrides 'baseline' in the config."
        ),
    )
    # Split-plot / hard-to-change factor flags (override YAML split_plot: block)
    parser.add_argument(
        "--htc-factors",
        metavar="A,B,...",
        default=None,
        help=(
            "Comma-separated list of hard-to-change (whole-plot) factor names. "
            "Activates split-plot mode. Overrides 'split_plot.htc_factors' in the config."
        ),
    )
    parser.add_argument(
        "--n-whole-plots",
        type=int,
        metavar="N",
        default=None,
        help=(
            "Number of whole plots (outer randomisation units). "
            "Required when --htc-factors is given. Overrides 'split_plot.n_whole_plots'."
        ),
    )
    parser.add_argument(
        "--eta",
        type=float,
        metavar="ETA",
        default=None,
        help=(
            "Variance ratio σ²_wp / σ²_sp (≥ 0). Default 1.0. "
            "Overrides 'split_plot.eta'."
        ),
    )
    parser.add_argument(
        "--subplots-per-wp",
        type=int,
        metavar="S",
        default=None,
        help=(
            "Sub-plots per whole plot (0 or omit = auto). "
            "Overrides 'split_plot.subplots_per_wp'."
        ),
    )
    parser.add_argument(
        "--df-method",
        choices=["auto", "conservative", "sp_only"],
        default=None,
        metavar="METHOD",
        help=(
            "Denominator df method for power: auto | conservative | sp_only. "
            "Overrides 'split_plot.df_method'."
        ),
    )
    args = parser.parse_args(argv)

    # Handle --template before anything else (no --config required)
    if args.template:
        _print_template(args.template)
        return 0

    # Handle --sheets before --config check (sheets path does not need --config)
    if args.sheets:
        try:
            from lattice_doe.sheets import sheets_run, SheetsError  # noqa: PLC0415
        except ImportError:
            print(
                "Error: Google Sheets support requires gspread.\n"
                "  pip install 'lattice-doe[sheets]'",
                file=sys.stderr,
            )
            return 1

        creds = (
            args.sheets_credentials
            or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        )
        try:
            result = sheets_run(args.sheets, credentials=creds)
        except SheetsError as e:
            print(f"Sheets error: {e}", file=sys.stderr)
            return 1

        r = result["report"]
        print(
            f"Design written to spreadsheet.\n"
            f"  n={r['n']}, p={r['p']}, "
            f"achieved_power={r['achieved_power']:.3f}, "
            f"elapsed={r.get('elapsed_sec', 0.0):.1f}s\n"
            f"  {result['spreadsheet_url']}"
        )
        return 0

    # Handle --excel-template (create a starter workbook and exit)
    if args.excel_template:
        try:
            from lattice_doe.excel_template import create_excel_template, ExcelError  # noqa: PLC0415
        except ImportError:
            print(
                "Error: Excel support requires openpyxl.\n"
                "  pip install 'lattice-doe[extras]'",
                file=sys.stderr,
            )
            return 1
        try:
            dest = create_excel_template(args.excel_template, example=args.template_mode)
            print(f"Excel template created: {dest}")
        except (ExcelError, Exception) as e:
            print(f"Error creating Excel template: {e}", file=sys.stderr)
            return 1
        return 0

    # Handle --excel-run (bidirectional Excel run and exit)
    if args.excel_run:
        try:
            from lattice_doe.excel_template import excel_run, ExcelError  # noqa: PLC0415
        except ImportError:
            print(
                "Error: Excel support requires openpyxl.\n"
                "  pip install 'lattice-doe[extras]'",
                file=sys.stderr,
            )
            return 1
        try:
            result = excel_run(args.excel_run)
        except ExcelError as e:
            print(f"Excel error: {e}", file=sys.stderr)
            return 1

        r = result["report"]
        print(
            f"Design written to workbook.\n"
            f"  n={r['n']}, p={r['p']}, "
            f"achieved_power={r['achieved_power']:.3f}, "
            f"elapsed={r.get('elapsed_sec', 0.0):.1f}s\n"
            f"  {result['excel_path']}"
        )
        return 0

    # --config is required for all other operations
    if args.config is None:
        parser.error(
            "--config is required. Alternatives: "
            "--template {r2,contrast,glm-binomial,glm-poisson} to print a YAML scaffold, "
            "--sheets URL to run from a Google Sheet, "
            "--excel-template PATH to create an Excel workbook, or "
            "--excel-run PATH to run from an existing Excel workbook."
        )

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)-7s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr  # Log to stderr to separate from stdout results
    )
    logger.debug("Verbose logging enabled.")

    cfg_path = Path(args.config)

    try:
        # 1. Load Config
        if not cfg_path.exists():
            # Use FileNotFoundError for specific I/O error
            raise FileNotFoundError(f"Config file not found: {cfg_path.resolve()}")
            
        logger.info(f"Loading config from: {cfg_path.resolve()}")
        cfg = _load_config(cfg_path)
        
        # 2. Validate Config Structure
        logger.debug("Validating config keys...")
        _validate_config_keys(cfg)
        
        # 3. Parse Config Sections (can raise ValueError)
        logger.debug("Parsing config sections...")
        formula = str(cfg["formula"])
        factors = _as_factors(cfg["factors"])

        # Merge CLI GLM flags into cfg before power config is built (CLI takes priority over YAML)
        cfg = _apply_glm_cli_args(cfg, args)

        _is_multiresponse = bool(args.multi_response) or "responses" in cfg
        if _is_multiresponse:
            power_cfg = None
            multi_cfg = _make_multi_response_cfg(cfg, formula, factors)
        else:
            power_cfg = _make_power_cfg(cfg, formula, factors)
            multi_cfg = None

        # Merge CLI split-plot flags into cfg["split_plot"] (CLI takes priority over YAML)
        cfg = _apply_sp_cli_args(cfg, args)

        design_opts = _make_design_opts(cfg)

        # 4. Validate Output Path
        out_cfg = cfg.get("output", {})
        basename = out_cfg.get("basename", args.out)
        out_path = Path(basename)
        out_dir = out_path.parent
        
        # Create parent directory
        logger.debug(f"Ensuring output directory exists: {out_dir.resolve()}")
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Check write permissions
        if not os.access(out_dir, os.W_OK):
            raise PermissionError(
                f"Output directory is not writable: {out_dir.resolve()}"
            )
        logger.debug("Output directory is writable.")

        # 5. Handle --dry-run
        if args.dry_run:
            logger.info("\n--- Dry Run Validation Successful ---")
            logger.info(f"  Formula: {formula}")
            logger.info(f"  Factors: {list(factors.keys())}")
            if _is_multiresponse:
                logger.info(f"  Mode: multi-response ({len(multi_cfg.responses)} responses)")
            else:
                logger.info(f"Power Config: {power_cfg.__class__.__name__}")
            logger.info(f"Design Algo: {design_opts.algo}")
            logger.info(f" Output Dir: {out_dir.resolve()} (writable)")
            logger.info("--- Exiting without design generation ---")
            return 0

        # --- End Validation, Start Main Task ---
        logger.info("Config validated. Running powered design generation...")

        # Live progress to stderr (UX-3). Explicit via --progress/--verbose, and
        # on by default on an interactive terminal so a long run never looks
        # hung; --no-progress opts out. Non-TTY (piped/CI) stays quiet.
        _on_progress = None
        _show_progress = (args.progress or args.verbose) or (
            not args.no_progress and sys.stderr.isatty()
        )
        if _show_progress:
            from lattice_doe.progress import ProgressReporter

            def _render_progress(ev) -> None:
                _bits = [f"[{ev.elapsed_sec:6.1f}s] {ev.phase}"]
                if ev.trial_n is not None:
                    _bits.append(f"n={ev.trial_n}")
                if ev.current_power is not None:
                    _bits.append(f"power={ev.current_power:.4f}")
                if ev.message:
                    _bits.append(f"— {ev.message}")
                print("  ".join(_bits), file=sys.stderr, flush=True)

            _on_progress = ProgressReporter(_render_progress, min_interval=0.3)

        if _is_multiresponse:
            result = find_multiresponse_design(
                formula=formula,
                factors=factors,
                multi_cfg=multi_cfg,
                design_opts=design_opts,
                on_progress=_on_progress,
            )
            # Unified envelope (UX-6): design_df / buckets_df / report already
            # present. Enrich the report with the extra fields the CLI summary
            # printout expects.
            design_df = result["design_df"]
            buckets_df = result["buckets_df"]
            report = result["report"]
            report["combined_power"] = report["achieved_power"]
            report["algo"] = design_opts.algo
            report["starts"] = design_opts.starts
            for _rd in report.get("responses", []):
                report[f"{_rd['name']}_power"] = _rd["power"]
        else:
            result = find_optimal_design(
                formula,
                factors,
                power_cfg,
                design_opts,
                on_progress=_on_progress,
            )
            design_df = result["design_df"]
            buckets_df = result["buckets_df"]
            report = result["report"]

        # Determine output basename (already done, just use `out_path`)
        logger.info(f"Writing output files with basename: {out_path}")

        # Always write CSV + JSON
        design_csv = out_path.with_name(out_path.name + "_design.csv")
        buckets_csv = out_path.with_name(out_path.name + "_buckets.csv")
        report_json = out_path.with_name(out_path.name + "_report.json")

        design_df.to_csv(design_csv, index=False)
        buckets_df.to_csv(buckets_csv, index=False)
        def _json_default(obj: Any) -> Any:
            if isinstance(obj, Path):
                return str(obj)
            raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

        with open(report_json, "w", encoding="utf-8") as _fh:
            json.dump(report, _fh, indent=2, default=_json_default)

        logger.info(f"Wrote: {design_csv}")
        logger.info(f"Wrote: {buckets_csv}")
        logger.info(f"Wrote: {report_json}")

        # Optional HTML report (CLI flag or config key; not supported for multi-response)
        html_report_flag = args.html_report or bool(out_cfg.get("html_report", False))
        if html_report_flag:
            if _is_multiresponse:
                logger.warning("HTML report is not yet supported for multi-response designs; skipped.")
                report_html = None
            else:
                report_html = out_path.with_name(out_path.name + "_report.html")
                try:
                    from .report import generate_report
                    generate_report(
                        result=result,
                        formula=formula,
                        factors=factors,
                        power_cfg=power_cfg,
                        output_path=report_html,
                        include_power_curve=False,
                    )
                    logger.info(f"Wrote: {report_html}")
                except ImportError:
                    logger.warning(
                        "HTML report skipped: jinja2 is not installed. "
                        'Install it with: pip install "lattice-doe[report]"'
                    )
                except Exception as _rpt_err:
                    logger.warning(f"HTML report failed: {_rpt_err}")
        else:
            report_html = None

        # Optional Excel (CLI flag has priority; config can also request excel)
        excel_flag = args.excel or bool(out_cfg.get("excel", False))
        if excel_flag:
            excel_path = out_path.with_suffix(".xlsx")
            logger.debug(f"Writing Excel file to: {excel_path}")
            with pd.ExcelWriter(excel_path, engine="xlsxwriter") as xw:
                design_df.to_excel(xw, index=False, sheet_name="design")
                buckets_df.to_excel(xw, index=False, sheet_name="buckets")
                pd.DataFrame([report]).to_excel(xw, index=False, sheet_name="report")
            logger.info(f"Wrote: {excel_path}")

        # Optional robustness report (single-response only)
        if args.robustness_report:
            if _is_multiresponse:
                logger.warning("--robustness-report is not supported for multi-response designs; skipped.")
            try:
                from lattice_doe.analysis import robustness_report  # noqa: PLC0415
                rob = robustness_report(
                    design_df=design_df,
                    formula=formula,
                    factors=factors,
                    power_cfg=power_cfg,
                )
                s = rob["summary"]
                t = rob["thresholds"]
                print("\n=== Robustness Report ===")
                print(f"{'mode':>22}: {rob['mode']}")
                print(f"{'nominal_power':>22}: {rob['nominal_power']:.3f}")
                print(f"{'target_power':>22}: {s['power_target']:.2f}")
                print(f"{'pct_scenarios_passing':>22}: {s['pct_scenarios_passing']:.1%}")
                print(f"{'worst_power':>22}: {s['worst_power']:.3f}")
                print(f"{'median_power':>22}: {s['median_power']:.3f}")
                print(f"{'best_power':>22}: {s['best_power']:.3f}")
                print("  Threshold crossings (where power = target):")
                if t["max_sigma_for_target"] is not None:
                    print(f"{'max_sigma':>22}: {t['max_sigma_for_target']:.4g}"
                          "  (σ must not exceed this)")
                if t["min_effect_for_target"] is not None:
                    effect_label = "min_effect_scale" if rob["mode"] == "contrast" else "min_r2_target"
                    print(f"{'':>22}  ({effect_label}: {t['min_effect_for_target']:.4g})")
                if t["min_alpha_for_target"] is not None:
                    print(f"{'min_alpha':>22}: {t['min_alpha_for_target']:.4g}"
                          "  (α must not be below this)")
            except Exception as _rob_err:
                logger.warning(f"Robustness report failed: {_rob_err}")

        # Pretty-print short summary (This stays as print() to stdout)
        print("\n=== Lattice DOE ===")
        for k in ("n", "p", "df_num", "df_denom", "alpha", "target_power", "achieved_power"):
            if k in report:
                print(f"{k:>15}: {report[k]}")
        # Multi-response per-response powers
        if _is_multiresponse:
            for rspec in multi_cfg.responses:
                key = f"{rspec.name}_power"
                if key in report:
                    print(f"{key:>15}: {report[key]:.3f}")
            if "combined_power" in report:
                print(f"{'combined_power':>15}: {report['combined_power']:.3f}")
        print(f"{'criterion':>15}: {report.get('criterion', 'N/A')}")
        print(f"{'algo':>15}: {report.get('algo', 'N/A')}")
        print(f"{'starts':>15}: {report.get('starts', 'N/A')}")
        # --- Enhancement 10: richer run metadata ---
        if html_report_flag and report_html is not None and report_html.exists():
            print(f"{'html_report':>15}: {report_html}")
        if "elapsed_sec" in report:
            print(f"{'elapsed_sec':>15}: {report['elapsed_sec']:.4f} s")
        if "search_strategy" in report:
            print(f"{'strategy':>15}: {report['search_strategy']}")
        if "random_state" in report:
            print(f"{'random_state':>15}: {report['random_state']}")
        if "split_plot" in report:
            sp = report["split_plot"]
            print(f"{'split_plot':>15}:")
            print(f"{'  n_whole_plots':>15}: {sp.get('n_whole_plots')}")
            print(f"{'  subplots_per_wp':>15}: {sp.get('subplots_per_wp')}")
            print(f"{'  eta':>15}: {sp.get('eta')}")
            htc = ", ".join(sp.get("htc_factors", []))
            print(f"{'  htc_factors':>15}: {htc}")
            print(f"{'  df_method':>15}: {sp.get('df_method')}")
        _warns = report.get("warnings", [])
        if _warns:
            print(f"{'warnings':>15}: {len(_warns)} issue(s)")
            for _w in _warns:
                print(f"{'':>17}! {_w}")
        else:
            print(f"{'warnings':>15}: none")

        # UX-7: partial completion is a distinct exit code so automation can
        # detect it without parsing warning strings.
        if report.get("status") == "partial":
            _reason = report.get("termination_reason")
            if args.allow_partial:
                # Explicitly accepted: informational, exit 0 (not an error).
                logger.warning(
                    "Search finished WITHOUT reaching the target power "
                    f"(termination_reason={_reason}). Partial outputs were "
                    "written; exiting 0 (--allow-partial)."
                )
            else:
                logger.error(
                    "Search finished WITHOUT reaching the target power "
                    f"(termination_reason={_reason}). Outputs were written; "
                    "exiting 3. Use --allow-partial to exit 0 instead."
                )
                return 3
        return 0

    # --- Specific Error Handling ---
    except FileNotFoundError as e:
        logger.error(f"[Config Error] {e}")
        return 2
        
    except (KeyError, ValueError) as e:
        # Catches validation errors, parsing errors
        logger.error(f"[Config Error] {e}")
        return 2
        
    except PermissionError as e:
        logger.error(f"[IO Error] {e}")
        return 2
        
    except IOError as e:
        logger.error(f"[IO Error] Failed to write output files: {e}")
        return 2

    except Exception as e:
        # Catch-all for unexpected runtime errors
        # exc_info=True will add traceback if logging level is DEBUG
        logger.error(f"[Unexpected Error] {type(e).__name__}: {e}", exc_info=args.verbose)
        return 2
    # --- End Error Handling ---


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
