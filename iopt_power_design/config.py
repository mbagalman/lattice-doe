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
from dataclasses import dataclass, field
from typing import Callable, Optional, Literal, TYPE_CHECKING
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


__all__ = ["PowerContrastConfig", "PowerR2Config", "DesignOptions"]
