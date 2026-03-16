# sheets.py
# License: MIT
"""
Google Sheets bidirectional connector for iopt-power-design
============================================================

Reads a structured configuration from a Google Spreadsheet and writes the
resulting optimal design back to the same spreadsheet.

Sheet layout (Config sheet)
---------------------------
The ``Config`` sheet uses sentinel headers in column A to delimit three
sections.  The parser scans column A for each sentinel and reads the rows
between them:

  ``[SETTINGS]``  — key/value pairs for formula, power parameters, and
                    design options.
  ``[CONTRAST]``  — optional; required only when ``power_mode = contrast``.
                    One ``L_row`` entry per contrast (comma-separated floats);
                    one ``delta`` entry (comma-separated floats).
  ``[FACTORS]``   — factor table: name | type | value1 | value2 | …
                    type = ``continuous`` → (low, high) tuple
                    type = ``categorical`` → list of levels

Output sheets
-------------
``sheets_run()`` writes three sheets after a successful design run:

  ``Results``  — summary key/value table (n, power, elapsed, …)
  ``Design``   — full design DataFrame
  ``Buckets``  — run-frequency buckets

Authentication
--------------
Two auth modes are supported via the ``credentials`` parameter:

  ``"path/to/service_account.json"``
      Service-account auth — suitable for automation and CI.
      The spreadsheet must be shared with the service-account email.
  ``None``
      OAuth2 browser flow via ``gspread.oauth()`` — opens a browser tab on
      first use and caches the token in ``~/.config/gspread/``.

Install
-------
  pip install "iopt-power-design[sheets]"
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .config import (
    PowerContrastConfig, PowerR2Config, PowerGLMContrastConfig,
    DesignOptions, SplitPlotOptions, MultiResponseOptions, ResponseSpec,
)
from ._request_builder import build_power_cfg, build_design_opts

# ---------------------------------------------------------------------------
# Soft dependency guard — same pattern as plot_backends.py (Plotly)
# ---------------------------------------------------------------------------
try:
    import gspread  # type: ignore
    _HAS_GSPREAD = True
except ImportError:
    _HAS_GSPREAD = False

_INSTALL_HINT = 'pip install "iopt-power-design[sheets]"'


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------
class SheetsError(RuntimeError):
    """Raised for all Google Sheets integration failures.

    Covers authentication errors, missing or malformed sheet sections,
    API errors, and write failures.  Underlying exceptions (gspread API
    errors, network errors) are attached as the ``__cause__`` so callers
    can inspect them with ``except SheetsError as e: e.__cause__``.
    """


# ---------------------------------------------------------------------------
# Section sentinel constants
# ---------------------------------------------------------------------------
_SENTINEL_SETTINGS   = "[SETTINGS]"
_SENTINEL_CONTRAST   = "[CONTRAST]"
_SENTINEL_FACTORS    = "[FACTORS]"
_SENTINEL_RESPONSES  = "[RESPONSES]"


# ---------------------------------------------------------------------------
# Authentication helper
# ---------------------------------------------------------------------------
def _get_client(credentials: Optional[str]) -> "gspread.Client":
    """Return an authenticated gspread Client.

    Parameters
    ----------
    credentials : str or None
        Path to a service account JSON file.  Pass ``None`` to use the
        OAuth2 browser flow (``gspread.oauth()``), which opens a browser
        tab on first use and caches the token in ``~/.config/gspread/``.

    Returns
    -------
    gspread.Client

    Raises
    ------
    ImportError
        If gspread / google-auth are not installed.
    """
    if not _HAS_GSPREAD:
        raise ImportError(
            f"gspread is required for Google Sheets integration. {_INSTALL_HINT}"
        )
    if credentials is not None:
        return gspread.service_account(filename=credentials)
    return gspread.oauth()


# ---------------------------------------------------------------------------
# Config sheet row reader
# ---------------------------------------------------------------------------
def _read_all_rows(worksheet: "gspread.Worksheet") -> List[List[str]]:
    """Return all cell values from *worksheet* as a list of string rows.

    Delegates directly to ``worksheet.get_all_values()``, which returns
    every row as a list of cell values (already coerced to strings by
    gspread).  Trailing empty strings within a row are included; completely
    blank rows are preserved as empty lists.
    """
    return worksheet.get_all_values()


# ---------------------------------------------------------------------------
# Config sheet parser
# ---------------------------------------------------------------------------
def _parse_config_sheet(
    worksheet: "gspread.Worksheet",
) -> Tuple[str, Dict[str, Any], Union[PowerContrastConfig, PowerR2Config, None], DesignOptions, Optional[MultiResponseOptions]]:
    """Parse the Config worksheet and return ``(formula, factors, power_cfg, design_opts, multi_cfg)``.

    The worksheet must contain three sentinel-delimited sections in column A:

    ``[SETTINGS]``
        Key/value pairs for formula, power parameters, and design options.
    ``[CONTRAST]``
        Optional; required when ``power_mode = contrast``.  Contains ``L_row``
        entries (one per contrast) and a single ``delta`` entry.
    ``[FACTORS]``
        Factor table: name | type | value1 | value2 | …
    ``[RESPONSES]``
        Optional; when present activates multi-response mode.
        Header row: name | power_mode | sigma | alpha | power | weight | L_row | delta | r2_target
        One data row per response; a ``power_combination`` key-value row may follow.

    Parameters
    ----------
    worksheet : gspread.Worksheet
        The Config worksheet to parse.

    Returns
    -------
    (formula, factors, power_cfg, design_opts, multi_cfg)
        ``power_cfg`` is ``None`` and ``multi_cfg`` is a :class:`MultiResponseOptions`
        when ``[RESPONSES]`` is present; otherwise ``multi_cfg`` is ``None``.

    Raises
    ------
    SheetsError
        For any missing sentinel, missing required key, unrecognised
        ``power_mode``, or malformed factor / contrast data.
    """
    rows = _read_all_rows(worksheet)

    # ------------------------------------------------------------------
    # 1. Locate sentinel row indices
    # ------------------------------------------------------------------
    idx_settings:  Optional[int] = None
    idx_contrast:  Optional[int] = None
    idx_factors:   Optional[int] = None
    idx_responses: Optional[int] = None

    for i, row in enumerate(rows):
        col_a = row[0].strip() if row else ""
        if col_a == _SENTINEL_SETTINGS:
            idx_settings = i
        elif col_a == _SENTINEL_CONTRAST:
            idx_contrast = i
        elif col_a == _SENTINEL_FACTORS:
            idx_factors = i
        elif col_a == _SENTINEL_RESPONSES:
            idx_responses = i

    if idx_settings is None:
        raise SheetsError(
            f"Config sheet is missing the '{_SENTINEL_SETTINGS}' sentinel in column A. "
            "Add a row with '[SETTINGS]' in the first column."
        )
    if idx_factors is None:
        raise SheetsError(
            f"Config sheet is missing the '{_SENTINEL_FACTORS}' sentinel in column A. "
            "Add a row with '[FACTORS]' in the first column."
        )

    # ------------------------------------------------------------------
    # 2. Parse [SETTINGS] section
    # ------------------------------------------------------------------
    # Rows between idx_settings+1 and the next sentinel (whichever comes first)
    sentinels_after = [
        j for j in (idx_contrast, idx_factors, idx_responses)
        if j is not None and j > idx_settings
    ]
    settings_end = min(sentinels_after) if sentinels_after else len(rows)

    settings: Dict[str, str] = {}
    for row in rows[idx_settings + 1 : settings_end]:
        col_a = row[0].strip() if len(row) > 0 else ""
        col_b = row[1].strip() if len(row) > 1 else ""
        if col_a:  # skip blank-key rows (separator lines)
            settings[col_a] = col_b

    # Required keys
    if "formula" not in settings or not settings["formula"]:
        raise SheetsError(
            "Config sheet [SETTINGS] is missing the required 'formula' key."
        )
    # power_mode is required unless a [RESPONSES] section is present
    if idx_responses is None and ("power_mode" not in settings or not settings["power_mode"]):
        raise SheetsError(
            "Config sheet [SETTINGS] is missing the required 'power_mode' key."
        )

    formula    = settings["formula"]
    power_mode = settings.get("power_mode", "").strip().lower()

    if power_mode and power_mode not in ("r2", "contrast", "glm"):
        raise SheetsError(
            f"Config sheet [SETTINGS] power_mode must be 'r2', 'contrast', or 'glm'; "
            f"got {settings['power_mode']!r}."
        )

    # Optional numeric keys with defaults
    def _float(key: str, default: float) -> float:
        raw = settings.get(key, "").strip()
        if not raw:
            return default
        try:
            return float(raw)
        except ValueError:
            raise SheetsError(
                f"Config sheet [SETTINGS] key '{key}' must be a number; got {raw!r}."
            ) from None

    def _int(key: str, default: int) -> int:
        raw = settings.get(key, "").strip()
        if not raw:
            return default
        try:
            return int(raw)
        except ValueError:
            raise SheetsError(
                f"Config sheet [SETTINGS] key '{key}' must be an integer; got {raw!r}."
            ) from None

    def _bool(key: str, default: bool) -> bool:
        raw = settings.get(key, "").strip().lower()
        if not raw:
            return default
        if raw in ("true", "yes", "1"):
            return True
        if raw in ("false", "no", "0"):
            return False
        raise SheetsError(
            f"Config sheet [SETTINGS] key '{key}' must be true/false; got {raw!r}."
        )

    alpha        = _float("alpha",        0.05)
    power_target = _float("power",        0.80)
    sigma        = _float("sigma",        1.0)
    r2_target    = _float("r2_target",    0.25)
    max_n        = _int("max_n",          500)
    criterion    = settings.get("criterion",    "I").strip() or "I"
    starts       = _int("starts",         5)
    max_iter_do  = _int("max_iter",       1000)
    random_state = _int("random_state",   123)
    # Blocked design options (Enhancement 20)
    n_blocks_raw          = _int("n_blocks",          0)
    block_factor_name_raw = settings.get("block_factor_name", "Block").strip() or "Block"
    # Categorical pre-allocation options (Enhancement 26)
    preallocate_categorical = _bool("preallocate_categorical", False)
    alloc_min_per_cell      = _int("alloc_min_per_cell",  1)
    alloc_max_per_cell_raw  = _int("alloc_max_per_cell",  0)  # 0 → None (no limit)
    # Split-plot options (Enhancement 22)
    htc_factors_raw    = settings.get("htc_factors", "").strip()
    n_whole_plots_raw  = _int("n_whole_plots", 0)
    sp_eta             = _float("eta",            1.0)
    subplots_per_wp_raw = _int("subplots_per_wp", 0)   # 0 → auto
    df_method_sp       = settings.get("df_method", "auto").strip() or "auto"
    # GLM options (GL-9)
    glm_family   = settings.get("family", "binomial").strip() or "binomial"
    glm_link     = settings.get("link", "").strip() or None
    glm_baseline = _float("baseline", 0.0)

    # ------------------------------------------------------------------
    # 3. Build DesignOptions
    # ------------------------------------------------------------------
    _do_d: Dict[str, Any] = dict(
        criterion=criterion,
        starts=starts,
        max_iter=max_iter_do,
        random_state=random_state,
        n_blocks=n_blocks_raw,
        block_factor_name=block_factor_name_raw,
        preallocate_categorical=preallocate_categorical,
        alloc_min_per_cell=alloc_min_per_cell,
        alloc_max_per_cell=alloc_max_per_cell_raw,
    )
    if htc_factors_raw and n_whole_plots_raw >= 2:
        htc_list = [f.strip() for f in htc_factors_raw.split(",") if f.strip()]
        if htc_list:
            _do_d["split_plot"] = dict(
                htc_factors=htc_list,
                n_whole_plots=n_whole_plots_raw,
                eta=sp_eta,
                subplots_per_wp=subplots_per_wp_raw,
                df_method=df_method_sp,
            )
    design_opts = build_design_opts(_do_d, error_cls=SheetsError, context="Config sheet")

    # ------------------------------------------------------------------
    # 4. Build power config  (skipped when [RESPONSES] is present)
    # ------------------------------------------------------------------
    power_cfg: Optional[Union[PowerContrastConfig, PowerR2Config, PowerGLMContrastConfig]] = None

    if idx_responses is None:
        if power_mode == "r2":
            power_cfg = build_power_cfg(
                dict(power_mode="r2", r2_target=r2_target, alpha=alpha,
                     power=power_target, sigma=sigma, max_n=max_n),
                error_cls=SheetsError, context="Config sheet",
            )

        else:  # contrast or glm — both require [CONTRAST] for L and delta
            if idx_contrast is None:
                raise SheetsError(
                    f"power_mode is {power_mode!r} but the '{_SENTINEL_CONTRAST}' sentinel "
                    "is missing from the Config sheet."
                )

            # Rows between idx_contrast+1 and idx_factors
            sentinels_after_contrast = [
                j for j in (idx_factors,) if j > idx_contrast
            ]
            contrast_end = min(sentinels_after_contrast) if sentinels_after_contrast else len(rows)

            l_rows: List[List[float]] = []
            delta_values: Optional[List[float]] = None

            for row in rows[idx_contrast + 1 : contrast_end]:
                col_a = row[0].strip() if len(row) > 0 else ""
                col_b = row[1].strip() if len(row) > 1 else ""
                if not col_a:
                    continue
                if col_a == "L_row":
                    try:
                        l_rows.append([float(v.strip()) for v in col_b.split(",") if v.strip()])
                    except ValueError as e:
                        raise SheetsError(
                            f"Config sheet [CONTRAST] L_row contains non-numeric value: {col_b!r}. "
                            f"Expected comma-separated floats."
                        ) from e
                elif col_a == "delta":
                    try:
                        delta_values = [float(v.strip()) for v in col_b.split(",") if v.strip()]
                    except ValueError as e:
                        raise SheetsError(
                            f"Config sheet [CONTRAST] delta contains non-numeric value: {col_b!r}. "
                            f"Expected comma-separated floats."
                        ) from e

            if not l_rows:
                raise SheetsError(
                    "Config sheet [CONTRAST] has no 'L_row' entries. "
                    "Add at least one row with 'L_row' in column A and "
                    "comma-separated contrast coefficients in column B."
                )
            if delta_values is None:
                raise SheetsError(
                    "Config sheet [CONTRAST] is missing the 'delta' row. "
                    "Add a row with 'delta' in column A and the effect size(s) in column B."
                )
            if len(delta_values) != len(l_rows):
                raise SheetsError(
                    f"Config sheet [CONTRAST] delta has {len(delta_values)} value(s) but "
                    f"there are {len(l_rows)} L_row(s). They must have the same length."
                )

            if power_mode == "glm":
                if not glm_baseline:
                    raise SheetsError(
                        "Config sheet [SETTINGS] 'baseline' is required when power_mode is 'glm'."
                    )
                power_cfg = build_power_cfg(
                    dict(power_mode="glm", L=l_rows, delta=delta_values,
                         baseline=glm_baseline, family=glm_family, link=glm_link,
                         alpha=alpha, power=power_target, max_n=max_n),
                    error_cls=SheetsError, context="Config sheet",
                )
            else:
                power_cfg = build_power_cfg(
                    dict(power_mode="contrast", L=l_rows, delta=delta_values,
                         alpha=alpha, power=power_target, sigma=sigma, max_n=max_n),
                    error_cls=SheetsError, context="Config sheet",
                )

    # ------------------------------------------------------------------
    # 5. Parse [FACTORS] section
    # ------------------------------------------------------------------
    factors: Dict[str, Any] = {}

    # Factors end at [RESPONSES] if present, otherwise end of sheet
    factors_end = idx_responses if idx_responses is not None and idx_responses > idx_factors else len(rows)
    # First non-sentinel row after [FACTORS] is the header — skip it.
    factors_data_start = idx_factors + 2   # +1 = first row after sentinel, +1 = skip header

    for row in rows[factors_data_start : factors_end]:
        col_a = row[0].strip() if len(row) > 0 else ""
        col_b = row[1].strip() if len(row) > 1 else ""
        values = [row[c].strip() for c in range(2, len(row)) if row[c].strip()]

        if not col_a:
            continue  # blank row — skip

        factor_name = col_a
        factor_type = col_b.lower()

        if factor_type == "continuous":
            if len(values) < 2:
                raise SheetsError(
                    f"Factor '{factor_name}' is continuous but has fewer than 2 values "
                    f"(low, high). Got: {values!r}."
                )
            try:
                low  = float(values[0])
                high = float(values[1])
            except ValueError as e:
                raise SheetsError(
                    f"Factor '{factor_name}' continuous bounds must be numeric; "
                    f"got {values[0]!r}, {values[1]!r}."
                ) from e
            factors[factor_name] = (low, high)

        elif factor_type == "categorical":
            levels = values
            if len(levels) < 2:
                raise SheetsError(
                    f"Factor '{factor_name}' is categorical but has fewer than 2 levels. "
                    f"Got: {levels!r}."
                )
            factors[factor_name] = levels

        else:
            raise SheetsError(
                f"Factor '{factor_name}' has unrecognised type {col_b!r}. "
                "Must be 'continuous' or 'categorical'."
            )

    if not factors:
        raise SheetsError(
            "Config sheet [FACTORS] section contains no factor definitions. "
            "Add at least one factor row after the header."
        )

    # ------------------------------------------------------------------
    # 6. Parse [RESPONSES] section (optional — multi-response mode)
    # ------------------------------------------------------------------
    multi_cfg: Optional[MultiResponseOptions] = None

    if idx_responses is not None:
        # Header row at idx_responses + 1:
        #   name|power_mode|sigma|alpha|power|weight|L_row|delta|r2_target|formula|
        #   lambda_mode|max_n|max_iter|tol_power
        # Special key rows (before/after response data): power_combination, sigma_joint
        # Data rows start at idx_responses + 2
        responses_data_start = idx_responses + 2
        specs: List[ResponseSpec] = []
        power_combination = "min"
        sigma_joint_arr: Optional[np.ndarray] = None

        for row in rows[responses_data_start:]:
            col_a = row[0].strip() if len(row) > 0 else ""
            col_b = row[1].strip() if len(row) > 1 else ""
            if not col_a:
                continue
            if col_a == "power_combination":
                power_combination = col_b or "min"
                continue
            if col_a == "sigma_joint":
                # Format: semicolon-separated matrix rows, comma-separated values.
                # E.g. "1.0,0.3; 0.3,1.0"
                if col_b:
                    try:
                        _sj_rows = []
                        for _line in col_b.split(";"):
                            _line = _line.strip()
                            if _line:
                                _sj_rows.append([float(x) for x in _line.replace(",", " ").split()])
                        sigma_joint_arr = np.array(_sj_rows, dtype=float)
                    except (ValueError, TypeError) as e:
                        raise SheetsError(
                            f"[RESPONSES] sigma_joint: invalid matrix format: {e}"
                        ) from e
                continue

            # Data columns: name|power_mode|sigma|alpha|power|weight|L_row|delta|r2_target|
            #               formula|lambda_mode|max_n|max_iter|tol_power
            def _rcell(idx: int) -> str:
                return row[idx].strip() if len(row) > idx else ""

            r_name = col_a
            r_mode = col_b.lower()
            r_sigma = float(_rcell(2)) if _rcell(2) else sigma
            r_alpha = float(_rcell(3)) if _rcell(3) else alpha
            r_power_v = float(_rcell(4)) if _rcell(4) else power_target
            r_weight = float(_rcell(5)) if _rcell(5) else 1.0
            r_L_str  = _rcell(6)
            r_d_str  = _rcell(7)
            r_r2_str = _rcell(8)
            r_formula_str = _rcell(9) or None
            r_lambda_mode = _rcell(10) or "n"
            r_max_n   = int(_rcell(11)) if _rcell(11) else 2000
            r_max_iter = int(_rcell(12)) if _rcell(12) else 200
            r_tol_power = float(_rcell(13)) if _rcell(13) else 1e-3
            # cols 14/15: GLM family and baseline (optional — blank for non-GLM responses)
            r_glm_family   = _rcell(14).strip() or None
            r_glm_baseline_str = _rcell(15).strip()

            _resp_ctx = f"[RESPONSES] row '{r_name}'"
            if r_mode == "contrast":
                if not r_L_str or not r_d_str:
                    raise SheetsError(
                        f"{_resp_ctx}: contrast mode requires "
                        "L_row (col 7) and delta (col 8) values."
                    )
                try:
                    L_vals = [float(v.strip()) for v in r_L_str.split(",") if v.strip()]
                    d_vals = [float(v.strip()) for v in r_d_str.split(",") if v.strip()]
                except ValueError as e:
                    raise SheetsError(
                        f"{_resp_ctx}: non-numeric L_row or delta: {e}"
                    ) from e
                pcfg: Union[PowerContrastConfig, PowerR2Config, PowerGLMContrastConfig] = (
                    build_power_cfg(
                        dict(power_mode="contrast", L=[L_vals], delta=d_vals,
                             alpha=r_alpha, power=r_power_v, sigma=r_sigma,
                             max_n=r_max_n, max_iter=r_max_iter, tol_power=r_tol_power),
                        error_cls=SheetsError, context=_resp_ctx,
                    )
                )
            elif r_mode == "r2":
                if not r_r2_str:
                    raise SheetsError(
                        f"{_resp_ctx}: r2 mode requires r2_target (col 9)."
                    )
                pcfg = build_power_cfg(
                    dict(power_mode="r2", r2_target=float(r_r2_str),
                         alpha=r_alpha, power=r_power_v, sigma=r_sigma,
                         lambda_mode=r_lambda_mode,
                         max_n=r_max_n, max_iter=r_max_iter, tol_power=r_tol_power),
                    error_cls=SheetsError, context=_resp_ctx,
                )
            elif r_mode == "glm":
                if not r_L_str or not r_d_str:
                    raise SheetsError(
                        f"{_resp_ctx}: glm mode requires "
                        "L_row (col 7) and delta (col 8) values."
                    )
                if not r_glm_baseline_str:
                    raise SheetsError(
                        f"{_resp_ctx}: glm mode requires baseline (col 16)."
                    )
                try:
                    L_vals = [float(v.strip()) for v in r_L_str.split(",") if v.strip()]
                    d_vals = [float(v.strip()) for v in r_d_str.split(",") if v.strip()]
                    r_glm_baseline = float(r_glm_baseline_str)
                except ValueError as e:
                    raise SheetsError(
                        f"{_resp_ctx}: non-numeric value: {e}"
                    ) from e
                pcfg = build_power_cfg(
                    dict(power_mode="glm", L=[L_vals], delta=d_vals,
                         baseline=r_glm_baseline, family=r_glm_family or "binomial",
                         alpha=r_alpha, power=r_power_v, max_n=r_max_n),
                    error_cls=SheetsError, context=_resp_ctx,
                )
            else:
                raise SheetsError(
                    f"{_resp_ctx}: power_mode must be 'contrast', 'r2', or 'glm'; "
                    f"got {col_b!r}."
                )

            try:
                specs.append(ResponseSpec(
                    name=r_name, power_cfg=pcfg,
                    formula=r_formula_str, weight=r_weight,
                ))
            except (ValueError, TypeError) as e:
                raise SheetsError(f"{_resp_ctx}: {e}") from e

        if len(specs) < 2:
            raise SheetsError(
                f"[RESPONSES] section must define at least 2 responses; "
                f"found {len(specs)}."
            )
        try:
            multi_cfg = MultiResponseOptions(
                responses=specs,
                power_combination=power_combination,
                sigma_joint=sigma_joint_arr,
            )
        except (ValueError, TypeError) as e:
            raise SheetsError(f"Config sheet produced invalid MultiResponseOptions: {e}") from e

    return formula, factors, power_cfg, design_opts, multi_cfg


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def _get_or_create_sheet(
    spreadsheet: "gspread.Spreadsheet",
    title: str,
) -> "gspread.Worksheet":
    """Return the named worksheet, creating it (1000 rows × 20 cols) if absent."""
    try:
        return spreadsheet.worksheet(title)
    except gspread.exceptions.WorksheetNotFound:
        return spreadsheet.add_worksheet(title=title, rows=1000, cols=20)


def _df_to_rows(df: "pd.DataFrame") -> List[List[Any]]:
    """Convert a DataFrame to a list-of-lists with Python-native scalar types.

    All numpy scalars are coerced to ``float`` or ``int`` so that gspread
    can JSON-serialise the values without error.
    """
    headers: List[Any] = list(df.columns)
    data_rows: List[List[Any]] = []
    for row in df.itertuples(index=False, name=None):
        coerced = []
        for v in row:
            if isinstance(v, (np.integer,)):
                coerced.append(int(v))
            elif isinstance(v, (np.floating,)):
                coerced.append(float(v))
            elif isinstance(v, float) and np.isnan(v):
                coerced.append("")
            else:
                coerced.append(v)
        data_rows.append(coerced)
    return [headers] + data_rows


def _write_results(
    spreadsheet: "gspread.Spreadsheet",
    result: Dict[str, Any],
    results_sheet: str = "Results",
    design_sheet: str = "Design",
    buckets_sheet: str = "Buckets",
    clear_results: bool = True,
) -> None:
    """Write design results back to the spreadsheet.

    Writes three sheets:

    * **Results** — summary key/value table (n, power, diagnostics, …).
    * **Design**  — full design DataFrame with column headers.
    * **Buckets** — run-frequency buckets DataFrame.

    Missing sheets are created automatically.  Existing sheets are cleared
    before writing when *clear_results* is ``True`` (the default).

    Parameters
    ----------
    spreadsheet : gspread.Spreadsheet
        Open spreadsheet object returned by ``gspread.Client.open*()``.
    result : dict
        Return value of ``i_optimal_powered_design()``.  Expected keys:
        ``"report"``, ``"design_df"``, ``"buckets_df"``.
    results_sheet : str, default "Results"
        Name of the summary sheet.
    design_sheet : str, default "Design"
        Name of the design table sheet.
    buckets_sheet : str, default "Buckets"
        Name of the buckets sheet.
    clear_results : bool, default True
        Clear existing content in each output sheet before writing.
    """
    _is_mr = "report" not in result

    # ------------------------------------------------------------------
    # 1. Results sheet — key/value summary
    # ------------------------------------------------------------------
    ws_results = _get_or_create_sheet(spreadsheet, results_sheet)
    if clear_results:
        ws_results.clear()

    if _is_mr:
        summary_rows: List[List[Any]] = [
            ["n",                int(result["n"])],
            ["achieved_power",   float(result["achieved_power"])],
            ["combination_rule", str(result.get("combination_rule", "min"))],
            ["compound_criterion", str(result.get("compound_criterion", False))],
            ["elapsed_sec",      float(result.get("elapsed_sec", 0.0))],
            ["search_strategy",  str(result.get("search_strategy", ""))],
            ["generated_at",     datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")],
            ["warnings",         "\n".join(result.get("warnings", []))],
        ]
        for _r in result.get("responses", []):
            summary_rows.append([f"{_r['name']}_power", float(_r["power"])])
        if "joint_power" in result:
            summary_rows.append(["joint_power", float(result["joint_power"])])
        design_df_val = result["design"]
        buckets_df_val = result["buckets"]
    else:
        report = result["report"]
        diags  = report.get("diagnostics", {})
        summary_rows = [
            ["n",               int(report["n"])],
            ["p",               int(report["p"])],
            ["df_num",          int(report["df_num"])],
            ["df_denom",        int(report["df_denom"])],
            ["alpha",           float(report["alpha"])],
            ["target_power",    float(report["target_power"])],
            ["achieved_power",  float(report["achieved_power"])],
            ["noncentrality_λ", float(report["noncentrality_lambda"])],
            ["i_criterion",     float(diags["i_criterion"])   if "i_criterion"   in diags else ""],
            ["d_efficiency",    float(diags["d_efficiency"])  if "d_efficiency"  in diags else ""],
            ["condition_number",float(diags["condition_number"]) if "condition_number" in diags else ""],
            ["criterion",       str(report["criterion"])],
            ["elapsed_sec",     float(report.get("elapsed_sec", 0.0))],
            ["generated_at",    datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")],
            ["warnings",        "\n".join(report.get("warnings", []))],
        ]
        design_df_val = result["design_df"]
        buckets_df_val = result["buckets_df"]

    ws_results.update("A1", summary_rows)

    # ------------------------------------------------------------------
    # 2. Design sheet — full design DataFrame
    # ------------------------------------------------------------------
    ws_design = _get_or_create_sheet(spreadsheet, design_sheet)
    if clear_results:
        ws_design.clear()

    ws_design.update("A1", _df_to_rows(design_df_val))

    # ------------------------------------------------------------------
    # 3. Buckets sheet — run-frequency buckets
    # ------------------------------------------------------------------
    ws_buckets = _get_or_create_sheet(spreadsheet, buckets_sheet)
    if clear_results:
        ws_buckets.clear()

    ws_buckets.update("A1", _df_to_rows(buckets_df_val))


# ---------------------------------------------------------------------------
# Template creator
# ---------------------------------------------------------------------------

# Pre-built Config sheet rows for each example.
# Each inner list is [col_A, col_B] — exactly what ws.update("A1", rows) expects.
_TEMPLATE_ROWS: Dict[str, List[List[str]]] = {
    "r2": [
        [_SENTINEL_SETTINGS, ""],
        ["formula",      "x1 + x2"],
        ["power_mode",   "r2"],
        ["alpha",        "0.05"],
        ["power",        "0.80"],
        ["sigma",        "1.0"],
        ["r2_target",    "0.30"],
        ["max_n",        "500"],
        ["criterion",    "I"],
        ["starts",       "5"],
        ["max_iter",     "1000"],
        ["random_state", "123"],
        # Blocked design: set n_blocks >= 2 to enable; 0 = unblocked (default).
        ["n_blocks",                "0"],
        ["block_factor_name",       "Block"],
        # Categorical pre-allocation: set to true to enable Wynn algorithm.
        ["preallocate_categorical", "false"],
        ["alloc_min_per_cell",      "1"],
        ["alloc_max_per_cell",      "0"],   # 0 = no upper limit
        # Split-plot: set htc_factors and n_whole_plots >= 2 to enable.
        ["htc_factors",             ""],    # comma-separated HTC factor names (leave blank = disabled)
        ["n_whole_plots",           "0"],   # 0 = disabled; >= 2 to enable split-plot
        ["eta",                     "1.0"], # variance ratio sigma2_wp / sigma2_sp
        ["subplots_per_wp",         "0"],   # 0 = auto
        ["df_method",               "auto"],# auto | conservative | sp_only
        ["", ""],
        [_SENTINEL_FACTORS, ""],
        ["factor_name", "type",       "value1", "value2"],
        ["x1",          "continuous", "-1.0",   "1.0"],
        ["x2",          "continuous", "-1.0",   "1.0"],
    ],
    "contrast": [
        [_SENTINEL_SETTINGS, ""],
        ["formula",      "x1 + x2"],
        ["power_mode",   "contrast"],
        ["alpha",        "0.05"],
        ["power",        "0.80"],
        ["sigma",        "1.0"],
        ["max_n",        "500"],
        ["criterion",    "I"],
        ["starts",       "5"],
        ["max_iter",     "1000"],
        ["random_state", "123"],
        # Blocked design: set n_blocks >= 2 to enable; 0 = unblocked (default).
        ["n_blocks",                "0"],
        ["block_factor_name",       "Block"],
        # Categorical pre-allocation: set to true to enable Wynn algorithm.
        ["preallocate_categorical", "false"],
        ["alloc_min_per_cell",      "1"],
        ["alloc_max_per_cell",      "0"],   # 0 = no upper limit
        # Split-plot: set htc_factors and n_whole_plots >= 2 to enable.
        ["htc_factors",             ""],    # comma-separated HTC factor names (leave blank = disabled)
        ["n_whole_plots",           "0"],   # 0 = disabled; >= 2 to enable split-plot
        ["eta",                     "1.0"], # variance ratio sigma2_wp / sigma2_sp
        ["subplots_per_wp",         "0"],   # 0 = auto
        ["df_method",               "auto"],# auto | conservative | sp_only
        ["", ""],
        [_SENTINEL_CONTRAST, ""],
        ["L_row",  "0,1,0"],
        ["delta",  "1.0"],
        ["", ""],
        [_SENTINEL_FACTORS, ""],
        ["factor_name", "type",       "value1", "value2"],
        ["x1",          "continuous", "-1.0",   "1.0"],
        ["x2",          "continuous", "-1.0",   "1.0"],
    ],
    # GLM binomial example: single contrast, logistic regression power.
    # family=binomial, baseline=event probability at H₀.
    "glm-binomial": [
        [_SENTINEL_SETTINGS, ""],
        ["formula",      "x1 + x2"],
        ["power_mode",   "glm"],
        ["alpha",        "0.05"],
        ["power",        "0.80"],
        ["family",       "binomial"],
        ["link",         ""],          # blank = canonical logit link
        ["baseline",     "0.20"],      # P(event) under H₀
        ["max_n",        "500"],
        ["criterion",    "I"],
        ["starts",       "5"],
        ["max_iter",     "1000"],
        ["random_state", "123"],
        ["n_blocks",                "0"],
        ["block_factor_name",       "Block"],
        ["preallocate_categorical", "false"],
        ["alloc_min_per_cell",      "1"],
        ["alloc_max_per_cell",      "0"],
        ["htc_factors",             ""],
        ["n_whole_plots",           "0"],
        ["eta",                     "1.0"],
        ["subplots_per_wp",         "0"],
        ["df_method",               "auto"],
        ["", ""],
        [_SENTINEL_CONTRAST, ""],
        # Model: Intercept + x1 + x2 → contrast x1 only: L = [0, 1, 0]
        ["L_row",  "0,1,0"],
        ["delta",  "0.5"],
        ["", ""],
        [_SENTINEL_FACTORS, ""],
        ["factor_name", "type",       "value1", "value2"],
        ["x1",          "continuous", "-1.0",   "1.0"],
        ["x2",          "continuous", "-1.0",   "1.0"],
    ],
    # GLM Poisson example: single contrast, log-linear power.
    # family=poisson, baseline=rate at H₀.
    "glm-poisson": [
        [_SENTINEL_SETTINGS, ""],
        ["formula",      "x1 + x2"],
        ["power_mode",   "glm"],
        ["alpha",        "0.05"],
        ["power",        "0.80"],
        ["family",       "poisson"],
        ["link",         ""],          # blank = canonical log link
        ["baseline",     "2.0"],       # expected count (rate) under H₀
        ["max_n",        "500"],
        ["criterion",    "I"],
        ["starts",       "5"],
        ["max_iter",     "1000"],
        ["random_state", "123"],
        ["n_blocks",                "0"],
        ["block_factor_name",       "Block"],
        ["preallocate_categorical", "false"],
        ["alloc_min_per_cell",      "1"],
        ["alloc_max_per_cell",      "0"],
        ["htc_factors",             ""],
        ["n_whole_plots",           "0"],
        ["eta",                     "1.0"],
        ["subplots_per_wp",         "0"],
        ["df_method",               "auto"],
        ["", ""],
        [_SENTINEL_CONTRAST, ""],
        ["L_row",  "0,1,0"],
        ["delta",  "0.3"],
        ["", ""],
        [_SENTINEL_FACTORS, ""],
        ["factor_name", "type",       "value1", "value2"],
        ["x1",          "continuous", "-1.0",   "1.0"],
        ["x2",          "continuous", "-1.0",   "1.0"],
    ],
    # Multi-response example: two R² responses, joint optimisation.
    # [RESPONSES] replaces [power_mode/r2_target/sigma] from [SETTINGS].
    # Column order: name|power_mode|sigma|alpha|power|weight|L_row|delta|r2_target|
    #               formula|lambda_mode|max_n|max_iter|tol_power|family|baseline
    # Special rows before/after data: power_combination, sigma_joint
    "multiresponse": [
        [_SENTINEL_SETTINGS, ""],
        ["formula",      "x1 + x2"],
        ["criterion",    "I"],
        ["starts",       "5"],
        ["max_iter",     "1000"],
        ["random_state", "123"],
        ["n_blocks",                "0"],
        ["block_factor_name",       "Block"],
        ["preallocate_categorical", "false"],
        ["alloc_min_per_cell",      "1"],
        ["alloc_max_per_cell",      "0"],
        ["htc_factors",             ""],
        ["n_whole_plots",           "0"],
        ["eta",                     "1.0"],
        ["subplots_per_wp",         "0"],
        ["df_method",               "auto"],
        ["", ""],
        [_SENTINEL_FACTORS, ""],
        ["factor_name", "type",       "value1", "value2"],
        ["x1",          "continuous", "-1.0",   "1.0"],
        ["x2",          "continuous", "-1.0",   "1.0"],
        ["", ""],
        [_SENTINEL_RESPONSES, ""],
        # Header (informational — not parsed):
        ["name", "power_mode", "sigma", "alpha", "power", "weight",
         "L_row", "delta", "r2_target", "formula",
         "lambda_mode", "max_n", "max_iter", "tol_power"],
        # Special rows:
        ["power_combination", "min"],
        # sigma_joint: semicolon-separated rows, comma-separated values.
        # Leave blank to disable Hotelling T² joint-power calculation.
        ["sigma_joint", ""],
        # Per-response data rows:
        ["Y1", "r2", "", "", "", "1.0", "", "", "0.15", "", "n", "", "", ""],
        ["Y2", "r2", "", "", "", "1.0", "", "", "0.20", "", "n", "", "", ""],
    ],
}


def create_sheet_template(
    title: str = "iopt-power-design template",
    credentials: Optional[str] = None,
    example: str = "r2",
    share_anyone: bool = False,
) -> str:
    """Create a new Google Spreadsheet pre-populated with a working example.

    The new spreadsheet contains four sheets:

    * **Config**  — pre-filled with a runnable example (ready for ``sheets_run()``).
    * **Results** — empty; populated by ``sheets_run()``.
    * **Design**  — empty; populated by ``sheets_run()``.
    * **Buckets** — empty; populated by ``sheets_run()``.

    By default the spreadsheet is **not** shared publicly.  Pass
    ``share_anyone=True`` to grant link-based write access to anyone — useful
    for quick prototyping but **not recommended** for private data.

    Parameters
    ----------
    title : str, default "iopt-power-design template"
        Title of the new Google Spreadsheet.
    credentials : str or None
        Path to a service account JSON credentials file, or ``None`` to use
        the OAuth2 browser flow (opens a browser tab on first use).
    example : {"r2", "contrast", "multiresponse", "glm-binomial", "glm-poisson"}, default "r2"
        Which working example to pre-populate in the Config sheet.

        * ``"r2"``            — global R² power config with two continuous factors.
        * ``"contrast"``      — single contrast (L matrix + delta) with two
          continuous factors.
        * ``"multiresponse"`` — two-response R² joint design with a ``[RESPONSES]``
          section demonstrating all per-response and ``sigma_joint`` fields.
        * ``"glm-binomial"``  — GLM logistic power with Wald χ² test (binomial family).
        * ``"glm-poisson"``   — GLM log-linear power with Wald χ² test (Poisson family).
    share_anyone : bool, default False
        If ``True``, share the new spreadsheet with anyone who has the link
        (writer access).  Disabled by default to avoid accidental data exposure.

    Returns
    -------
    str
        URL of the newly created Google Spreadsheet.

    Raises
    ------
    ImportError
        If ``gspread`` / ``google-auth`` are not installed.
    ValueError
        If *example* is not ``"r2"``, ``"contrast"``, or ``"multiresponse"``.
    SheetsError
        If spreadsheet creation or population fails.
    """
    if not _HAS_GSPREAD:
        raise ImportError(
            f"gspread is required for Google Sheets integration. {_INSTALL_HINT}"
        )
    if example not in _TEMPLATE_ROWS:
        raise ValueError(
            f"Unknown example {example!r}. Supported values: "
            f"{sorted(_TEMPLATE_ROWS.keys())}."
        )

    try:
        client = _get_client(credentials)
    except Exception as e:
        raise SheetsError(f"Authentication failed: {e}") from e

    try:
        sh = client.create(title)
    except Exception as e:
        raise SheetsError(f"Failed to create spreadsheet {title!r}: {e}") from e

    if share_anyone:
        try:
            sh.share(None, perm_type="anyone", role="writer")
        except Exception:
            # Sharing is best-effort; don't abort template creation if it fails
            # (e.g. organisation policies that disallow public sharing).
            pass

    try:
        # Rename the default first sheet to "Config"
        config_ws = sh.sheet1
        config_ws.update_title("Config")

        # Populate with the example rows
        config_ws.update("A1", _TEMPLATE_ROWS[example])

        # Add empty output sheets
        for sheet_name in ("Results", "Design", "Buckets"):
            sh.add_worksheet(title=sheet_name, rows=200, cols=20)

    except Exception as e:
        raise SheetsError(
            f"Failed to populate template spreadsheet: {e}"
        ) from e

    return sh.url


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def sheets_run(
    spreadsheet_url_or_id: str,
    *,
    credentials: Optional[str] = None,
    config_sheet: str = "Config",
    results_sheet: str = "Results",
    design_sheet: str = "Design",
    buckets_sheet: str = "Buckets",
    clear_results: bool = True,
) -> Dict[str, Any]:
    """Read a Config sheet, run the optimal design, and write results back.

    This is the primary entry point for the Google Sheets integration.  It
    combines authentication, config parsing, design optimisation, and result
    writing into a single call.

    Parameters
    ----------
    spreadsheet_url_or_id : str
        Full Google Sheets URL (``https://docs.google.com/spreadsheets/d/…``)
        or bare spreadsheet ID string.
    credentials : str or None, default None
        Path to a service account JSON file for automation/CI, or ``None`` to
        use the OAuth2 browser flow (opens a browser tab on first use and
        caches the token in ``~/.config/gspread/``).
    config_sheet : str, default "Config"
        Name of the worksheet that holds the structured configuration.
    results_sheet : str, default "Results"
        Name of the worksheet to write the summary key/value table to.
    design_sheet : str, default "Design"
        Name of the worksheet to write the full design DataFrame to.
    buckets_sheet : str, default "Buckets"
        Name of the worksheet to write the run-frequency buckets to.
    clear_results : bool, default True
        Clear existing content in each output sheet before writing.

    Returns
    -------
    dict
        The return value of ``i_optimal_powered_design()``, with one extra
        key added:

        ``"spreadsheet_url"``
            The URL of the spreadsheet that results were written to.

    Raises
    ------
    ImportError
        If ``gspread`` / ``google-auth`` are not installed.
    SheetsError
        For authentication failures, missing or malformed config sections,
        design errors, or write failures.
    """
    if not _HAS_GSPREAD:
        raise ImportError(
            f"gspread is required for Google Sheets integration. {_INSTALL_HINT}"
        )

    # ------------------------------------------------------------------
    # 1. Authenticate
    # ------------------------------------------------------------------
    try:
        client = _get_client(credentials)
    except ImportError:
        raise
    except Exception as e:
        raise SheetsError(f"Authentication failed: {e}") from e

    # ------------------------------------------------------------------
    # 2. Open the spreadsheet (URL or bare key)
    # ------------------------------------------------------------------
    try:
        try:
            sh = client.open_by_url(spreadsheet_url_or_id)
        except gspread.exceptions.NoValidUrlKeyFound:
            # Input is not a URL — treat it as a bare spreadsheet key
            sh = client.open_by_key(spreadsheet_url_or_id)
    except Exception as e:
        raise SheetsError(
            f"Could not open spreadsheet {spreadsheet_url_or_id!r}: {e}\n"
            "Check that the spreadsheet exists and is shared with your account."
        ) from e

    # ------------------------------------------------------------------
    # 3. Get the Config worksheet
    # ------------------------------------------------------------------
    try:
        config_ws = sh.worksheet(config_sheet)
    except Exception as e:
        raise SheetsError(
            f"Worksheet {config_sheet!r} not found in spreadsheet. "
            f"Create it or use the 'config_sheet' parameter. Error: {e}"
        ) from e

    # ------------------------------------------------------------------
    # 4. Parse config
    # ------------------------------------------------------------------
    try:
        formula, factors, power_cfg, design_opts, multi_cfg = _parse_config_sheet(config_ws)
    except SheetsError:
        raise
    except Exception as e:
        raise SheetsError(f"Unexpected error while parsing config sheet: {e}") from e

    # ------------------------------------------------------------------
    # 5. Run the design (lazy import to avoid circular imports)
    # ------------------------------------------------------------------
    try:
        if multi_cfg is not None:
            from .api import i_optimal_multiresponse_design  # noqa: PLC0415
            result: Dict[str, Any] = i_optimal_multiresponse_design(
                formula=formula,
                factors=factors,
                multi_cfg=multi_cfg,
                design_opts=design_opts,
            )
        else:
            from .api import i_optimal_powered_design  # noqa: PLC0415
            result = i_optimal_powered_design(
                formula=formula,
                factors=factors,
                power_cfg=power_cfg,
                design_opts=design_opts,
            )
    except Exception as e:
        raise SheetsError(
            f"Design optimisation failed: {e}"
        ) from e

    # ------------------------------------------------------------------
    # 6. Write results back to the spreadsheet
    # ------------------------------------------------------------------
    try:
        _write_results(
            sh,
            result,
            results_sheet=results_sheet,
            design_sheet=design_sheet,
            buckets_sheet=buckets_sheet,
            clear_results=clear_results,
        )
    except SheetsError:
        raise
    except Exception as e:
        raise SheetsError(f"Failed to write results to spreadsheet: {e}") from e

    # ------------------------------------------------------------------
    # 7. Annotate and return
    # ------------------------------------------------------------------
    result["spreadsheet_url"] = sh.url
    return result


__all__ = ["SheetsError", "sheets_run", "create_sheet_template"]
