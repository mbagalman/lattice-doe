# Statistics Review

Date: 2026-03-16

Reviewed against Git commit: `bb8bc94`

## Overall assessment

Most of the linear-model, R2, blocking, and split-plot machinery looks statistically coherent under the assumptions the package states. The main unresolved correctness issue is in the target semantics for multi-response search when the combination rule is not `min`.

The largest methodological gaps are:

1. General GLM optimal-design support with design-point-specific Fisher weights.
2. Stronger finite-sample mixed-model inference for split-plot designs.
3. Broader correlated multi-response power support beyond shared-formula OLS contrasts.

## Findings

### High

#### 1. Multi-response `product` and `weighted_mean` search use the wrong target scale

The binary sample-size search always uses the hardest individual response target:

- `target = max(r.power_cfg.power for r in multi_cfg.responses)`

That is appropriate for `power_combination="min"`, but not for `product` or `weighted_mean`, where adequacy should be evaluated on the combined scale rather than the hardest marginal scale. As implemented, the package can overstate the required sample size, emit misleading convergence warnings, and make the non-`min` rules mean something different from what the configuration documentation says.

Relevant code:

- `/Users/michaelbagalman/Documents/GitHub Projects/DOE Idea/iopt_power_design/api.py:1028`
- `/Users/michaelbagalman/Documents/GitHub Projects/DOE Idea/iopt_power_design/config.py:948`
- `/Users/michaelbagalman/Documents/GitHub Projects/DOE Idea/iopt_power_design/power.py:663`

Notes:

- In a direct local repro with two responses each targeting 0.8, the `product` rule still searched against 0.8 rather than a joint target such as 0.64.

### Medium

#### 2. GLM support is a local constant-weight approximation, not a general GLM DOE implementation

The GLM power routine uses a scalar Fisher weight derived from a single baseline value and applies it uniformly across the design:

- binomial: `w = p0 (1 - p0)`
- Poisson: `w = mu0`

That is a valid local approximation for the narrow case the code documents, but it is not the general GLM information structure used in optimal design, where the weight matrix typically depends on the design point and a nominal parameter vector. For realistic logistic or Poisson DOE problems with nonzero slopes, that missing layer is important.

Relevant code:

- `/Users/michaelbagalman/Documents/GitHub Projects/DOE Idea/iopt_power_design/power.py:257`
- `/Users/michaelbagalman/Documents/GitHub Projects/DOE Idea/iopt_power_design/power.py:304`
- `/Users/michaelbagalman/Documents/GitHub Projects/DOE Idea/iopt_power_design/config.py:797`

### Medium

#### 3. Split-plot finite-sample inference still relies on heuristic denominator degrees of freedom

The split-plot power path uses GLS for the information matrix, which is the right structural move, but denominator degrees of freedom are assigned using a WP-versus-SP classification heuristic. That is directionally reasonable and much better than ignoring split-plot structure, but it is not a full mixed-model small-sample inference method such as Satterthwaite or Kenward-Roger. In unbalanced or nearly singular settings, this can misstate finite-sample power.

Relevant code:

- `/Users/michaelbagalman/Documents/GitHub Projects/DOE Idea/iopt_power_design/power.py:447`
- `/Users/michaelbagalman/Documents/GitHub Projects/DOE Idea/iopt_power_design/split_plot.py:324`

### Medium

#### 4. Joint multi-response modeling is still narrow

Correlated multi-response support is only implemented for shared-formula OLS contrast mode through `sigma_joint`. The API explicitly rejects correlated GLM responses and split-plot multi-response cases. That limitation is statistically important because the independence-based `product` rule is only a clean probability statement when dependence is either negligible or modeled explicitly.

Relevant code:

- `/Users/michaelbagalman/Documents/GitHub Projects/DOE Idea/iopt_power_design/api.py:1046`
- `/Users/michaelbagalman/Documents/GitHub Projects/DOE Idea/iopt_power_design/api.py:1087`

## Review scope

- Code inspection of the core statistical modules and recent GLM / split-plot / multi-response paths.
- Direct local repro of the multi-response aggregation issue.
- No code changes were made as part of this review.
