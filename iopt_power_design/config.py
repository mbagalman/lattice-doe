# config.py
# License: MIT
"""
Configuration dataclasses for power-assured I-optimal DOE.

These classes define structured inputs for:
- Power calculation modes (contrast vs global R²)
- Design generation options (including parallel starts and adaptive candidate sizing)
- Validation and defaults

This file intentionally contains no procedural code — only structured
configuration objects used throughout the package.
"""
from __future__ import annotations

import ast as _ast
import math as _math
import warnings as _warnings
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Literal, Union, TYPE_CHECKING
import numpy as np  # core dependency

if TYPE_CHECKING:
    import pandas as pd


# ---------------------------------------------------------------------
# Declarative constraint expression compiler
# ---------------------------------------------------------------------

# Whitelisted function names callable inside constraint expressions.
_CONSTRAINT_ALLOWED_FUNCS: frozenset = frozenset({
    "abs", "max", "min", "round",
    "sqrt", "log", "log2", "log10", "exp",
    "floor", "ceil",
})

# Restricted globals supplied to eval().  __builtins__ is empty so that
# Python's built-in namespace is not reachable at runtime.
_CONSTRAINT_SAFE_GLOBALS: dict = {
    "__builtins__": {},
    "abs": abs, "max": max, "min": min, "round": round,
    "True": True, "False": False,
    "sqrt": _math.sqrt, "log": _math.log, "log2": _math.log2,
    "log10": _math.log10, "exp": _math.exp, "pi": _math.pi,
    "floor": _math.floor, "ceil": _math.ceil,
}

# AST node types that are permitted anywhere in the expression tree.
# Anything not in this set is rejected at compile time, before eval().
_CONSTRAINT_ALLOWED_NODES: frozenset = frozenset({
    _ast.Expression,
    # Boolean logic
    _ast.BoolOp, _ast.And, _ast.Or,
    # Arithmetic
    _ast.BinOp,
    _ast.Add, _ast.Sub, _ast.Mult, _ast.Div,
    _ast.Mod, _ast.Pow, _ast.FloorDiv,
    # Unary operators
    _ast.UnaryOp, _ast.Not, _ast.USub, _ast.UAdd,
    # Comparisons
    _ast.Compare,
    _ast.Eq, _ast.NotEq, _ast.Lt, _ast.LtE, _ast.Gt, _ast.GtE,
    _ast.In, _ast.NotIn,
    # Literals and name references
    _ast.Constant,
    _ast.Name, _ast.Load,
    # Whitelisted function calls (validated separately in visit_Call)
    _ast.Call,
})


class _ConstraintExprValidator(_ast.NodeVisitor):
    """AST visitor that rejects any construct not on the safe-list.

    Blocks attribute access (``x.__class__``), subscripts (``x[0]``),
    comprehensions, lambdas, imports, starred args, and any name that
    starts with an underscore — all of which can be used to escape a
    ``__builtins__: {}`` sandbox via subclass traversal.
    """

    def generic_visit(self, node: _ast.AST) -> None:
        if type(node) not in _CONSTRAINT_ALLOWED_NODES:
            raise ValueError(
                f"constraint_expr: forbidden construct "
                f"'{type(node).__name__}' in expression. "
                "Only arithmetic/comparison/boolean operators and "
                f"whitelisted functions ({sorted(_CONSTRAINT_ALLOWED_FUNCS)}) "
                "are allowed."
            )
        super().generic_visit(node)

    def visit_Name(self, node: _ast.Name) -> None:
        if node.id.startswith("_"):
            raise ValueError(
                f"constraint_expr: name {node.id!r} is not allowed. "
                "Names starting with '_' are forbidden."
            )
        # All other names (factor columns + whitelisted functions) are OK.
        # No child nodes to visit (Name is a leaf).

    def visit_Call(self, node: _ast.Call) -> None:
        # Only direct calls like sqrt(x) are permitted — no attribute calls.
        if not isinstance(node.func, _ast.Name):
            raise ValueError(
                "constraint_expr: only simple function calls are allowed "
                f"(e.g., 'sqrt(x)'). Found: {type(node.func).__name__!r}."
            )
        if node.func.id not in _CONSTRAINT_ALLOWED_FUNCS:
            raise ValueError(
                f"constraint_expr: function '{node.func.id}' is not allowed. "
                f"Allowed functions: {sorted(_CONSTRAINT_ALLOWED_FUNCS)}."
            )
        if node.keywords:
            raise ValueError(
                "constraint_expr: keyword arguments in function calls "
                "are not allowed."
            )
        # Validate each positional argument (starred args → ast.Starred,
        # which is not in _CONSTRAINT_ALLOWED_NODES and will be rejected).
        for arg in node.args:
            self.visit(arg)


def _compile_constraint_expr(expr: str) -> "Callable[[pd.Series], bool]":
    """Compile a string constraint expression into a row-level callable.

    The expression is **AST-validated** before compilation, blocking all
    constructs that could escape the restricted execution environment:
    attribute access, subscripts, comprehensions, lambdas, imports, dunder
    names, and any function call not in the explicit whitelist.

    Parameters
    ----------
    expr : str
        Boolean expression referencing factor column names.
        Examples::

            "Temperature <= 2 * Pressure"
            "Catalyst != 'C' or Time <= 3"
            "sqrt(Temperature) + Pressure <= 20"

    Returns
    -------
    callable
        A function ``f(row: pd.Series) -> bool`` suitable for use as
        ``DesignOptions.constraint_func``.

    Raises
    ------
    ValueError
        If *expr* contains a syntax error or a forbidden construct.
    """
    # 1. Parse to AST (catches syntax errors early).
    try:
        tree = _ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(
            f"constraint_expr has a syntax error: {expr!r}\n  {exc}"
        ) from exc

    # 2. Walk the AST and reject any unsafe node (attribute access, subscript,
    #    comprehensions, dunder names, non-whitelisted calls, …).
    _ConstraintExprValidator().visit(tree)

    # 3. Compile the validated AST to a code object (fast, reusable per row).
    code = compile(tree, "<constraint_expr>", "eval")

    def _constraint(row: "pd.Series") -> bool:
        try:
            return bool(eval(code, _CONSTRAINT_SAFE_GLOBALS, dict(row)))
        except NameError as exc:
            raise ValueError(
                f"constraint_expr references an undefined name: {exc}.\n"
                f"  Available factor columns: {list(row.index)}\n"
                f"  Expression: {expr!r}"
            ) from exc
        except Exception as exc:
            raise ValueError(
                f"constraint_expr evaluation failed for a candidate row: {exc}\n"
                f"  Expression: {expr!r}"
            ) from exc

    return _constraint


# ---------------------------------------------------------------------
# Power analysis configurations
# ---------------------------------------------------------------------
@dataclass
class PowerContrastConfig:
    """Power configuration for *contrast-based* tests.

    Parameters
    ----------
    L : ndarray (q x p)
        Contrast matrix (must match model matrix column count).
    delta : ndarray (q,)
        Minimum detectable effect vector (same length as L rows).
    alpha : float
        Significance level for the test (two-sided F).
    power : float
        Desired statistical power.
    sigma : float
        Residual standard deviation.
    tol_power : float, default 1e-3
        Tolerance around target power.
    max_iter : int, default 200
        Maximum iterations for n-search.
    max_n : int, default 2000
        Hard cap on sample size.
    verbose : bool, default False
        Print progress during search.
    """

    L: np.ndarray
    delta: np.ndarray
    alpha: float = 0.05
    power: float = 0.8
    sigma: float = 1.0
    tol_power: float = 1e-3
    max_iter: int = 200
    max_n: int = 2000
    verbose: bool = False

    def __post_init__(self):
        # --- Type and shape normalization ---
        self.L = np.atleast_2d(self.L)
        self.delta = np.atleast_1d(self.delta)

        # --- Shape validation ---
        if self.L.ndim != 2:
            raise ValueError(f"L must be a 2D array, but got ndim={self.L.ndim}")
        if self.delta.ndim != 1:
            raise ValueError(f"delta must be a 1D array, but got ndim={self.delta.ndim}")
        if self.L.shape[0] != len(self.delta):
            raise ValueError(
                f"L has {self.L.shape[0]} rows but delta has {len(self.delta)} elements"
            )

        # --- Range and content validation ---
        if not (0 < self.alpha < 1):
            raise ValueError(f"alpha must be in (0, 1), but got {self.alpha}")
        if not (0 < self.power < 1):
            raise ValueError(f"power must be in (0, 1), but got {self.power}")
        if self.sigma <= 0:
            raise ValueError(f"sigma must be > 0, but got {self.sigma}")
        if self.tol_power <= 0:
            raise ValueError(f"tol_power must be > 0, but got {self.tol_power}")
        if self.max_iter <= 0:
            raise ValueError(f"max_iter must be > 0, but got {self.max_iter}")
        if self.max_n <= 0:
            raise ValueError(f"max_n must be > 0, but got {self.max_n}")

        # --- ADDED: Contrast content validation ---
        if np.any(np.all(self.L == 0, axis=1)):
            raise ValueError("L matrix contains at least one all-zero row.")
        if np.any(np.isclose(self.delta, 0)):
            raise ValueError(
                "delta vector contains zero or near-zero values. "
                "The 'sesoi' (delta) must be non-zero."
            )

    def __str__(self) -> str:
        # ADDED: String representation for debugging
        l_shape = self.L.shape
        d_shape = self.delta.shape
        return (
            f"PowerContrastConfig(L.shape={l_shape}, delta.shape={d_shape}, "
            f"alpha={self.alpha}, power={self.power}, sigma={self.sigma})"
        )


@dataclass
class PowerR2Config:
    """Power configuration for *global R²* tests (full model F-test).

    Parameters
    ----------
    r2_target : float
        Target population R² effect size.
    alpha : float
        Significance level for the test (F-test).
    power : float
        Desired statistical power.
    tol_power : float, default 1e-3
        Tolerance around target power.
    max_iter : int, default 200
        Maximum iterations for n-search.
    max_n : int, default 2000
        Hard cap on sample size.
    verbose : bool, default False
        Print progress during search.
    lambda_mode : {'n', 'n_minus_p'}, default 'n'
        Convention for noncentrality parameter:
        - 'n' (common in statsmodels, G*Power)
        - 'n_minus_p' (more conservative, previous default)
    """

    r2_target: float
    alpha: float = 0.05
    power: float = 0.8
    sigma: float = 1.0  # ADDED: Include sigma for consistency, though not used in R2 calc
    tol_power: float = 1e-3
    max_iter: int = 200
    max_n: int = 2000
    verbose: bool = False
    lambda_mode: Literal["n", "n_minus_p"] = "n"

    def __post_init__(self):
        if not (0 < self.r2_target < 1):
            raise ValueError(
                f"r2_target must be between (0, 1), but got {self.r2_target}"
            )
        if not (0 < self.alpha < 1):
            raise ValueError(f"alpha must be in (0, 1), but got {self.alpha}")
        if not (0 < self.power < 1):
            raise ValueError(f"power must be in (0, 1), but got {self.power}")
        if self.lambda_mode not in ("n", "n_minus_p"):
            raise ValueError(f"lambda_mode must be 'n' or 'n_minus_p'")
        if self.tol_power <= 0:
            raise ValueError(f"tol_power must be > 0, but got {self.tol_power}")
        if self.max_iter <= 0:
            raise ValueError(f"max_iter must be > 0, but got {self.max_iter}")
        if self.max_n <= 0:
            raise ValueError(f"max_n must be > 0, but got {self.max_n}")

    def __str__(self) -> str:
        # ADDED: String representation for debugging
        return (
            f"PowerR2Config(r2_target={self.r2_target}, "
            f"alpha={self.alpha}, power={self.power}, lambda_mode='{self.lambda_mode}')"
        )


# ---------------------------------------------------------------------
# Split-plot design configuration
# ---------------------------------------------------------------------
@dataclass
class SplitPlotOptions:
    """Configuration for split-plot (hard-to-change factor) designs.

    A split-plot experiment groups runs into **whole plots** (WPs).  All
    runs within a whole plot share the same setting of every hard-to-change
    (HTC) factor; only the easy-to-change (ETC) sub-plot factors vary
    freely within a WP.

    The variance model is two-stratum::

        y_ij = Xβ + τ_i + ε_ij
        τ_i ~ N(0, σ²_wp)     (whole-plot error, shared within WP i)
        ε_ij ~ N(0, σ²_sp)    (sub-plot error, independent)
        η = σ²_wp / σ²_sp     (variance ratio)

    Parameters
    ----------
    htc_factors : list of str
        Names of the hard-to-change (whole-plot) factors.  Must be a
        non-empty subset of the factor names passed to the API.
    n_whole_plots : int
        Number of whole plots (outer randomisation units).  Must be ≥ 2.
    eta : float, default 1.0
        Variance ratio ``σ²_wp / σ²_sp``.  Must be ≥ 0.
        ``eta=0`` is equivalent to the standard OLS (single-stratum) model.
    subplots_per_wp : int or None, default None
        Number of sub-plots per whole plot.  ``None`` lets the API
        auto-compute a reasonable value: ``max(2, ceil(p / n_whole_plots) + 1)``.
        Must be ≥ 1 when provided.
    df_method : {"auto", "conservative", "sp_only"}, default "auto"
        How to assign denominator degrees of freedom for power calculations.

        * ``"auto"`` — classify each contrast row as whole-plot (WP) or
          sub-plot (SP) based on which factors it involves; use WP df for
          pure-WP contrasts and SP df for all others.
        * ``"conservative"`` — always use WP df (never anti-conservative;
          recommended when in doubt).
        * ``"sp_only"`` — always use SP df (may be anti-conservative for
          pure WP-factor contrasts).
    criterion_ignore_vr : bool, default False
        If ``True``, use the standard OLS optimality criterion during
        design search and ignore the variance ratio.  Useful for
        comparison studies; not recommended for production use.
    """

    htc_factors: List[str]
    n_whole_plots: int
    eta: float = 1.0
    subplots_per_wp: Optional[int] = None
    df_method: Literal["auto", "conservative", "sp_only"] = "auto"
    criterion_ignore_vr: bool = False

    def __post_init__(self) -> None:
        if not self.htc_factors:
            raise ValueError(
                "SplitPlotOptions.htc_factors must be a non-empty list of factor names."
            )
        if not all(isinstance(f, str) and f for f in self.htc_factors):
            raise ValueError(
                "SplitPlotOptions.htc_factors must contain non-empty strings."
            )
        if len(self.htc_factors) != len(set(self.htc_factors)):
            raise ValueError(
                "SplitPlotOptions.htc_factors contains duplicate factor names."
            )
        if not isinstance(self.n_whole_plots, int) or isinstance(self.n_whole_plots, bool):
            raise ValueError("n_whole_plots must be an integer.")
        if self.n_whole_plots < 2:
            raise ValueError(
                f"n_whole_plots must be ≥ 2, got {self.n_whole_plots}."
            )
        if self.eta < 0:
            raise ValueError(
                f"eta (variance ratio σ²_wp/σ²_sp) must be ≥ 0, got {self.eta}."
            )
        if self.subplots_per_wp is not None:
            if not isinstance(self.subplots_per_wp, int) or isinstance(self.subplots_per_wp, bool):
                raise ValueError("subplots_per_wp must be an integer or None.")
            if self.subplots_per_wp < 1:
                raise ValueError(
                    f"subplots_per_wp must be ≥ 1, got {self.subplots_per_wp}."
                )
        if self.df_method not in ("auto", "conservative", "sp_only"):
            raise ValueError(
                f"df_method must be 'auto', 'conservative', or 'sp_only'; "
                f"got {self.df_method!r}."
            )

    def __str__(self) -> str:
        sp = (
            f"subplots_per_wp={self.subplots_per_wp}"
            if self.subplots_per_wp is not None
            else "subplots_per_wp=auto"
        )
        return (
            f"SplitPlotOptions(htc_factors={self.htc_factors}, "
            f"n_whole_plots={self.n_whole_plots}, eta={self.eta}, "
            f"{sp}, df_method='{self.df_method}')"
        )


# ---------------------------------------------------------------------
# Design generation options
# ---------------------------------------------------------------------
@dataclass
class DesignOptions:
    """Options controlling design generation, numerical stability, parallel starts,
    and adaptive candidate sizing.

    Parameters
    ----------
    candidate_points : int, default 2000
        Number of candidate points to sample for continuous factors when
        auto_candidate=False. Ignored when auto_candidate=True.
    auto_candidate : bool, default False
        If True, automatically determine candidate size based on factor complexity:
        - Pure continuous: uses cand_min
        - Pure categorical: counts cells (up to cap) × per_cell_alpha
        - Mixed: hybrid approach based on categorical cells and continuous dimensions
    cand_min : int, default 1000
        Minimum candidate points when auto_candidate=True. Ensures sufficient
        coverage even for simple factor spaces.
    cand_max : int, default 10000
        Maximum candidate points when auto_candidate=True. Prevents excessive
        memory usage for complex factor spaces.
    cat_cells_cap : int, default 10000
        Cap on categorical cell enumeration to avoid combinatorial explosion.
        When the Cartesian product of categorical levels exceeds this, we sample
        a subset rather than enumerate all combinations.
    per_cell_alpha : float, default 1.5
        Multiplier for categorical cells in adaptive sizing. For purely categorical
        designs, candidate_points = min(cells × per_cell_alpha, cand_max).
    per_cell_min : int, default 5
        Minimum points per categorical cell for mixed designs. Ensures adequate
        continuous sampling within each categorical combination.
    per_cell_max : int, default 20
        Maximum points per categorical cell for mixed designs. Prevents oversampling
        in high-dimensional continuous spaces.
    allow_candidate_growth : bool, default False
        If True, adaptively grow candidate set once if the first iteration shows
        poor numerical conditioning (condition number > 1e6). Helps recover from
        underspecified initial candidate regions.
    growth_factor : float, default 2.0
        Factor to multiply candidate size by when growing (if enabled). Applied
        once when poor conditioning detected, capped at cand_max.
    random_state : int, default 123
        Random seed for reproducibility across candidate generation, design search,
        and parallel starts.
    criterion : {"I", "D", "A"}, default 'I'
        Optimality criterion for design search.

        * ``"I"`` (I-optimal) — minimises average prediction variance over the
          candidate region.  Preferred when prediction accuracy across the
          factor space is the primary goal.
        * ``"D"`` (D-optimal) — maximises ``det(X'X)``.  Preferred when
          precise estimation of all model coefficients is the primary goal.
        * ``"A"`` (A-optimal) — minimises ``trace((X'X)^-1)``, the sum of
          coefficient-estimate variances.  Preferred when all coefficients
          should be estimated with equal precision.
    algo : {'fedorov', 'coordinate'}, default 'fedorov'
        Algorithm for optimal design search:
        - 'fedorov': Classic Fedorov exchange algorithm
        - 'coordinate': Coordinate exchange (may be faster for large problems)
    starts : int, default 5
        Number of random starts to avoid local optima. In serial mode, this is
        the number of independent Fedorov-exchange starts run sequentially, with
        the best result returned. In parallel mode (workers > 1), this is the
        total number of single-start trials run independently.
    max_iter : int, default 1000
        Maximum iterations per start for design search convergence. Increase for
        difficult optimization landscapes or tight convergence tolerance.
    xtx_jitter : float, default 1e-8
        Diagonal jitter for (X'X)^-1 numerical stability. Added to diagonal of
        X'X before inversion to handle near-singular cases. Must be > 0.
    constraint_func : callable, optional
        Python callable applied to candidate rows to filter infeasible points.
        Must accept a pandas Series (single candidate row) and return bool.
        Example: ``lambda row: row['Temperature'] <= row['Pressure'] * 2``
        Cannot be set at the same time as *constraint_expr*; if both are
        provided, *constraint_expr* takes precedence.
    constraint_expr : str, optional
        String alternative to *constraint_func* for use in YAML/JSON configs.
        The expression is evaluated with factor column names as local variables.
        A restricted set of math helpers is available (``sqrt``, ``log``,
        ``log10``, ``log2``, ``exp``, ``floor``, ``ceil``, ``pi``, ``abs``,
        ``min``, ``max``, ``round``).  No imports are permitted.
        Example: ``"Temperature <= 2 * Pressure"``
        If both *constraint_expr* and *constraint_func* are provided,
        *constraint_expr* takes precedence and *constraint_func* is overwritten.
    workers : int, optional
        Number of parallel workers (processes) for random starts. If None or <=1,
        runs `starts` independent Fedorov-exchange starts serially and selects
        the best. If > 1, launches independent single-start optimizations in
        parallel processes and selects best.
    parallel_seed_stride : int, default 10000
        Offset added between per-start seeds to decorrelate parallel trials.
        Each worker i gets seed = random_state + i * parallel_seed_stride.
    
    Notes
    -----
    Adaptive candidate sizing (auto_candidate=True) is recommended when:
    - Factor space complexity is unknown or varies between problems
    - Mixing categorical and continuous factors with many levels
    - Memory constraints exist but you want optimal coverage

    The allow_candidate_growth option provides a safety net for cases where
    the initial candidate estimate proves insufficient, detected via poor
    numerical conditioning of the design matrix.

    Feasibility constraints can be specified as either a Python callable
    (*constraint_func*) for programmatic use, or as a string expression
    (*constraint_expr*) for YAML/JSON configs.  When *constraint_expr* is
    set it is compiled to a callable and overwrites *constraint_func*.
    """

    # Core candidate generation
    candidate_points: int = 2000
    auto_candidate: bool = False
    cand_min: int = 1000
    cand_max: int = 10000
    cat_cells_cap: int = 10000
    per_cell_alpha: float = 1.5
    per_cell_min: int = 5
    per_cell_max: int = 20

    # Adaptive refinement
    allow_candidate_growth: bool = False
    growth_factor: float = 2.0

    # Search configuration
    random_state: int = 123
    criterion: str = "I"
    algo: Literal["fedorov", "coordinate"] = "fedorov"
    starts: int = 5
    max_iter: int = 1000
    xtx_jitter: float = 1e-8

    # Categorical pre-allocation (Enhancement 26)
    preallocate_categorical: bool = False
    alloc_min_per_cell: int = 1
    alloc_max_per_cell: Optional[int] = None
    alloc_wynn_max_iter: int = 500
    alloc_wynn_tol: float = 1e-6

    # Blocked design options (Enhancement 20)
    n_blocks: Optional[int] = None
    block_sizes: Optional[List[int]] = field(default=None, repr=False)
    block_factor_name: str = "Block"

    # Split-plot design options (Enhancement 22)
    split_plot: Optional[SplitPlotOptions] = field(default=None, repr=False)

    # Advanced options
    constraint_func: Optional[Callable[["pd.Series"], bool]] = field(
        default=None, repr=False
    )
    constraint_expr: Optional[str] = field(default=None, repr=False)
    workers: Optional[int] = None
    parallel_seed_stride: int = 10_000

    def __post_init__(self):
        # ADDED: Range checks for numerical parameters
        if not isinstance(self.random_state, int) or isinstance(self.random_state, bool):
            raise ValueError(
                f"random_state must be an int, got {type(self.random_state).__name__!r} "
                f"({self.random_state!r}). "
                "Pass an integer seed, e.g. DesignOptions(random_state=42)."
            )
        if self.candidate_points <= 0:
            raise ValueError("candidate_points must be > 0")
        if self.cand_min <= 0:
            raise ValueError("cand_min must be > 0")
        if self.cand_max < self.cand_min:
            raise ValueError("cand_max must be >= cand_min")
        if self.cat_cells_cap <= 0:
            raise ValueError("cat_cells_cap must be > 0")
        if self.per_cell_alpha <= 0:
            raise ValueError("per_cell_alpha must be > 0")
        if self.per_cell_min <= 0:
            raise ValueError("per_cell_min must be > 0")
        if self.per_cell_max < self.per_cell_min:
            raise ValueError("per_cell_max must be >= per_cell_min")
        if self.growth_factor <= 1.0:
            raise ValueError("growth_factor must be > 1.0")
        if self.starts <= 0:
            raise ValueError("starts must be > 0")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be > 0")
        if self.xtx_jitter <= 0:
            raise ValueError("xtx_jitter must be > 0 for numerical stability")
        if self.parallel_seed_stride <= 0:
            raise ValueError("parallel_seed_stride must be > 0")
        if self.algo not in ("fedorov", "coordinate"):
            raise ValueError("algo must be 'fedorov' or 'coordinate'")
        if self.criterion not in ("I", "D", "A"):
            raise ValueError(
                f"criterion must be 'I' (I-optimal), 'D' (D-optimal), or 'A' (A-optimal); "
                f"got {self.criterion!r}."
            )
        if self.workers is not None and self.workers <= 0:
            self.workers = None  # Treat 0 or negative as serial
        if self.alloc_min_per_cell < 0:
            raise ValueError("alloc_min_per_cell must be >= 0")
        if self.alloc_max_per_cell is not None and self.alloc_max_per_cell < 1:
            raise ValueError("alloc_max_per_cell must be >= 1 (or None for unconstrained)")
        if (
            self.alloc_max_per_cell is not None
            and self.alloc_max_per_cell < self.alloc_min_per_cell
        ):
            raise ValueError("alloc_max_per_cell must be >= alloc_min_per_cell")
        if self.alloc_wynn_max_iter < 1:
            raise ValueError("alloc_wynn_max_iter must be >= 1")
        if self.alloc_wynn_tol <= 0:
            raise ValueError("alloc_wynn_tol must be > 0")

        # --- Blocked design validation ---
        if self.n_blocks is not None:
            if not isinstance(self.n_blocks, int) or isinstance(self.n_blocks, bool):
                raise ValueError("n_blocks must be an integer or None")
            if self.n_blocks < 2:
                raise ValueError(
                    "n_blocks must be >= 2; use None for unblocked designs."
                )
            if self.block_sizes is not None:
                if len(self.block_sizes) != self.n_blocks:
                    raise ValueError(
                        f"len(block_sizes)={len(self.block_sizes)} != "
                        f"n_blocks={self.n_blocks}. Provide one size per block."
                    )
                if any(s < 1 for s in self.block_sizes):
                    raise ValueError("All block_sizes must be >= 1.")
        elif self.block_sizes is not None:
            raise ValueError(
                "block_sizes requires n_blocks to be set. "
                "Set n_blocks >= 2 or leave block_sizes=None."
            )
        if not self.block_factor_name:
            raise ValueError("block_factor_name must be a non-empty string.")

        # --- Split-plot / blocked combination guard ---
        if self.split_plot is not None and self.n_blocks is not None:
            _warnings.warn(
                "Both split_plot and n_blocks are set. Blocked split-plot designs "
                "(three-stratum variance models) are not yet supported. "
                "n_blocks will be ignored when split_plot is active.",
                UserWarning,
                stacklevel=3,
            )

        # --- Declarative constraint expression ---
        # If constraint_expr is set, compile it to a callable and store in
        # constraint_func.  constraint_expr takes precedence if both are
        # provided (also handles dataclasses.replace copying both fields).
        if self.constraint_expr is not None:
            self.constraint_func = _compile_constraint_expr(self.constraint_expr)

    def __str__(self) -> str:
        # ADDED: String representation for debugging
        has_constraint = self.constraint_func is not None
        expr_preview = (
            f", expr={self.constraint_expr!r}" if self.constraint_expr else ""
        )
        return (
            f"DesignOptions(auto_candidate={self.auto_candidate}, "
            f"algo='{self.algo}', starts={self.starts}, workers={self.workers}, "
            f"xtx_jitter={self.xtx_jitter}, constraint={has_constraint}{expr_preview})"
        )



# ---------------------------------------------------------------------
# GLM (logistic / Poisson) power configuration
# ---------------------------------------------------------------------

@dataclass
class PowerGLMContrastConfig:
    """Power configuration for GLM contrast tests using the Wald chi-square test.

    Supports binomial (logistic) and Poisson families with their canonical
    link functions (logit and log respectively).

    The design search uses a **null-based locally optimal** information matrix:

        M = w · X′X    where  w = p₀(1 − p₀)  [binomial]
                               w = μ₀          [Poisson]

    Because ``w`` is a positive scalar it cancels from I/D/A criteria, so
    the Fedorov exchange is structurally identical to OLS.  Only the power
    calculation differs — the test statistic follows a noncentral chi-square
    distribution (df = number of contrast rows) rather than a noncentral F.

    .. note:: **Approximation scope.**
        The scalar weight ``w`` is evaluated once at the null baseline and
        applied uniformly to every design point.  This is exact when all
        predictors are at their reference levels (i.e., the true operating
        point equals the null), and is a reasonable approximation for
        moderate effects.  For realistic designs with substantial covariate
        ranges and nonzero slopes the true Fisher information weight varies
        per design point (``wᵢ = p(xᵢ)(1−p(xᵢ))`` for binomial), so the
        constant-weight approximation may overstate or understate power.
        Full point-wise GLM-optimal design (where each candidate point
        carries its own Fisher weight derived from a nominal parameter
        vector) is a planned future enhancement.

    Parameters
    ----------
    L : ndarray, shape (q, p)
        Contrast matrix.  Same semantics as ``PowerContrastConfig.L``.
        Must be 2-D; all-zero rows are rejected.
    delta : ndarray, shape (q,)
        Effect sizes on the **linear predictor (LP) scale**.

        * Binomial (logit link): difference in log-odds, e.g. ``0.5`` means a
          0.5-nat increase in logit P(Y=1).  Equivalent to log(OR) where OR is
          the odds ratio; ``delta = log(1.65) ≈ 0.5``.
        * Poisson (log link): difference in log-rate, e.g. ``0.3`` means a
          ≈ 30 % increase in the expected count.

        Zero or near-zero values are rejected.
    baseline : float
        Baseline mean on the **response scale** at the null / reference point.

        * Binomial: event probability ∈ (0, 1), e.g. ``0.20`` for 20 % baseline.
        * Poisson: expected count > 0, e.g. ``2.5`` events per unit.

        The Fisher information weight is derived from this value:
        ``w = baseline*(1 − baseline)`` (binomial) or ``w = baseline`` (Poisson).
    family : {'binomial', 'poisson'}, default 'binomial'
        Distributional family.
    link : {'logit', 'log'} or None, default None
        Link function.  ``None`` selects the canonical link for the family
        (``'logit'`` for binomial, ``'log'`` for Poisson).
    alpha : float, default 0.05
        Significance level for the Wald chi-square test.
    power : float, default 0.80
        Desired statistical power.
    tol_power : float, default 1e-3
        Convergence tolerance for the n-search.
    max_iter : int, default 200
        Maximum n-search iterations.
    max_n : int, default 2000
        Hard cap on sample size.
    """

    L: np.ndarray
    delta: np.ndarray
    baseline: float
    family: Literal["binomial", "poisson"] = "binomial"
    link: Optional[Literal["logit", "log"]] = None
    alpha: float = 0.05
    power: float = 0.80
    tol_power: float = 1e-3
    max_iter: int = 200
    max_n: int = 2000

    def __post_init__(self) -> None:
        # --- Type and shape normalisation ---
        self.L = np.atleast_2d(self.L)
        self.delta = np.atleast_1d(self.delta)

        # --- Shape validation ---
        if self.L.ndim != 2:
            raise ValueError(f"L must be a 2-D array, got ndim={self.L.ndim}")
        if self.delta.ndim != 1:
            raise ValueError(f"delta must be a 1-D array, got ndim={self.delta.ndim}")
        if self.L.shape[0] != len(self.delta):
            raise ValueError(
                f"L has {self.L.shape[0]} rows but delta has {len(self.delta)} elements"
            )

        # --- Family and link validation ---
        if self.family not in ("binomial", "poisson"):
            raise ValueError(
                f"family must be 'binomial' or 'poisson', got {self.family!r}"
            )
        _canonical = {"binomial": "logit", "poisson": "log"}
        if self.link is None:
            # Store the canonical link explicitly so callers never see None.
            object.__setattr__(self, "link", _canonical[self.family])
        else:
            _allowed = {
                "binomial": ("logit",),
                "poisson":  ("log",),
            }
            if self.link not in _allowed[self.family]:
                raise ValueError(
                    f"link {self.link!r} is not valid for family {self.family!r}. "
                    f"Allowed: {_allowed[self.family]}"
                )

        # --- Baseline validation ---
        if self.family == "binomial":
            if not (0.0 < self.baseline < 1.0):
                raise ValueError(
                    f"baseline must be in (0, 1) for binomial family, "
                    f"got {self.baseline}"
                )
            if min(self.baseline, 1.0 - self.baseline) < 0.05:
                _warnings.warn(
                    f"GLM baseline {self.baseline:.3f} is near a boundary; "
                    "required sample size may be very large.",
                    RuntimeWarning,
                    stacklevel=3,
                )
        else:  # poisson
            if self.baseline <= 0.0:
                raise ValueError(
                    f"baseline must be > 0 for Poisson family, got {self.baseline}"
                )

        # --- Numeric range validation ---
        if not (0.0 < self.alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha}")
        if not (0.0 < self.power < 1.0):
            raise ValueError(f"power must be in (0, 1), got {self.power}")
        if self.tol_power <= 0:
            raise ValueError(f"tol_power must be > 0, got {self.tol_power}")
        if self.max_iter <= 0:
            raise ValueError(f"max_iter must be > 0, got {self.max_iter}")
        if self.max_n <= 0:
            raise ValueError(f"max_n must be > 0, got {self.max_n}")

        # --- Contrast content validation ---
        if np.any(np.all(self.L == 0, axis=1)):
            raise ValueError("L matrix contains at least one all-zero row.")
        if np.any(np.isclose(self.delta, 0)):
            raise ValueError(
                "delta vector contains zero or near-zero values on the LP scale."
            )

    def __str__(self) -> str:
        return (
            f"PowerGLMContrastConfig(family={self.family!r}, link={self.link!r}, "
            f"baseline={self.baseline}, L.shape={self.L.shape}, "
            f"alpha={self.alpha}, power={self.power})"
        )


def glm_fisher_weight(cfg: "PowerGLMContrastConfig") -> float:
    """Return the scalar Fisher information weight at the null-model baseline.

    This is the per-observation variance-function value evaluated at the
    baseline mean:

    * Binomial: ``w = p₀ · (1 − p₀)``
    * Poisson:  ``w = μ₀``

    The locally optimal information matrix under the null is ``M = w · X′X``.
    Because ``w`` is a positive scalar it does not affect I/D/A design
    criteria, but it does scale the Wald chi-square noncentrality parameter:

        λ = w · δᵀ [L · (X′X)⁻¹ · Lᵀ]⁻¹ δ

    Parameters
    ----------
    cfg : PowerGLMContrastConfig

    Returns
    -------
    float
        Strictly positive Fisher weight.
    """
    if cfg.family == "binomial":
        return cfg.baseline * (1.0 - cfg.baseline)
    return float(cfg.baseline)  # Poisson: w = μ₀


# ---------------------------------------------------------------------
# Multi-response design configuration
# ---------------------------------------------------------------------

#: Type alias for the per-response power configuration discriminated union.
PowerCfg = Union["PowerContrastConfig", "PowerR2Config", "PowerGLMContrastConfig"]


@dataclass
class ResponseSpec:
    """Specification of one response variable's power requirements.

    Parameters
    ----------
    name : str
        Label for this response (used in reporting and result dict keys).
    power_cfg : PowerContrastConfig | PowerR2Config
        Power requirements for this response.  Each response may use a
        different mode (contrast or R²), have its own sigma, L, delta, etc.
    formula : str or None, default None
        Patsy formula for this response.  If None, the global formula
        passed to i_optimal_multiresponse_design() is used.  Setting a
        different formula per-response activates the compound criterion path.
    weight : float, default 1.0
        Relative importance weight used when power_combination="weighted_mean".
        Weights are normalised internally (they do not need to sum to 1).
    """

    name: str
    power_cfg: PowerCfg
    formula: Optional[str] = None
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ResponseSpec.name must be a non-empty string.")
        if not isinstance(self.power_cfg, (PowerContrastConfig, PowerR2Config, PowerGLMContrastConfig)):
            raise TypeError(
                "ResponseSpec.power_cfg must be a PowerContrastConfig, PowerR2Config, "
                "or PowerGLMContrastConfig."
            )
        if self.weight <= 0:
            raise ValueError(f"ResponseSpec.weight must be > 0, got {self.weight}.")


@dataclass
class MultiResponseOptions:
    """Options for multi-response powered design.

    Parameters
    ----------
    responses : list of ResponseSpec
        One entry per response variable.  Must contain at least two entries
        (use i_optimal_powered_design for single-response problems).
    power_combination : {"min", "product", "weighted_mean"}, default "min"
        Rule for aggregating per-response powers into a single scalar used
        by the binary n-search.

        * "min"           — design adequate when *all* responses reach target
                            (conservative; recommended default).
        * "product"       — combined = ∏ p_i, interpreted as the joint
                            probability all responses pass simultaneously.
                            **Statistically valid only when responses are
                            independent.**  For correlated responses, use
                            ``sigma_joint`` (Hotelling T²) instead.
        * "weighted_mean" — weighted average; tolerates weaker minor responses.
    sigma_joint : ndarray (k x k) or None, default None
        Inter-response error covariance matrix for Hotelling T² joint power.
        Must be symmetric positive definite with k = len(responses).
        When None, per-response independence combination is used.
        Only valid when all responses share the same formula and use contrast mode.
    """

    responses: List[ResponseSpec]
    power_combination: Literal["min", "product", "weighted_mean"] = "min"
    sigma_joint: Optional[np.ndarray] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if len(self.responses) < 2:
            raise ValueError(
                "MultiResponseOptions requires at least 2 ResponseSpec entries. "
                "Use i_optimal_powered_design() for single-response problems."
            )
        names = [r.name for r in self.responses]
        if len(names) != len(set(names)):
            raise ValueError("ResponseSpec names must be unique.")
        if self.power_combination not in ("min", "product", "weighted_mean"):
            raise ValueError(
                "power_combination must be 'min', 'product', or 'weighted_mean'."
            )
        if self.sigma_joint is not None:
            k = len(self.responses)
            arr = np.asarray(self.sigma_joint)
            if arr.shape != (k, k):
                raise ValueError(
                    f"sigma_joint must be ({k},{k}) for {k} responses, "
                    f"got {arr.shape}."
                )


__all__ = [
    "PowerContrastConfig",
    "PowerR2Config",
    "PowerGLMContrastConfig",
    "glm_fisher_weight",
    "DesignOptions",
    "SplitPlotOptions",
    "ResponseSpec",
    "MultiResponseOptions",
]
