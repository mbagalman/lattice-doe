"""
Shared multi-response configuration builder (UX-65).

Reconstructs :class:`~lattice_doe.config.MultiResponseOptions` from the
session-state ``mr_responses`` list. Used by the Run page (to launch the
search) and by the Analysis page (to bind sensitivity/MDE to ONE selected
response's formula, power configuration and authoritative model matrix) —
one builder, so the two pages can never disagree about what was configured.

lattice_doe is imported lazily: the pages import this module unconditionally
and must still render their "not installed" notice when the package is
absent.
"""

from typing import Any


def parse_matrix(text: str):
    import numpy as np

    rows = []
    for line in text.strip().splitlines():
        line = line.strip()
        if line:
            rows.append([float(x) for x in line.replace(",", " ").split()])
    return np.array(rows)


def parse_vector(text: str):
    import numpy as np

    return np.array([float(x) for x in text.replace(",", " ").split()])


def build_multi_response_cfg(ss: Any) -> "Any":
    """Build MultiResponseOptions from the session-state mr_responses list."""
    import numpy as np

    from lattice_doe._request_builder import build_power_cfg
    from lattice_doe.config import MultiResponseOptions, ResponseSpec

    specs = []
    for r in ss.get("mr_responses", []):
        r_alpha = float(r.get("alpha", ss.get("alpha", 0.05)))
        r_power = float(r.get("power", ss.get("power_target", 0.80)))
        r_sigma = float(r.get("sigma", 1.0))
        r_weight = float(r.get("weight", 1.0))
        r_formula = r.get("formula", "").strip() or None
        _r_mode = r.get("power_mode", "contrast")
        if _r_mode == "glm":
            L = parse_matrix(r.get("L_text", ""))
            delta = parse_vector(r.get("delta_text", ""))
            pcfg = build_power_cfg(dict(
                power_mode="glm", L=L, delta=delta,
                baseline=float(r.get("glm_baseline", 0.20)),
                family=r.get("glm_family", "binomial"),
                link=r.get("glm_link", "").strip() or None,
                alpha=r_alpha, power=r_power,
            ))
        elif _r_mode == "contrast":
            L = parse_matrix(r.get("L_text", ""))
            delta = parse_vector(r.get("delta_text", ""))
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
                    _sj_rows.append(
                        [float(x) for x in _line.replace(",", " ").split()]
                    )
            sigma_joint = np.array(_sj_rows)
        except (ValueError, Exception):
            sigma_joint = None
    return MultiResponseOptions(
        responses=specs,
        power_combination=ss.get("mr_combination", "min"),
        sigma_joint=sigma_joint,
    )
