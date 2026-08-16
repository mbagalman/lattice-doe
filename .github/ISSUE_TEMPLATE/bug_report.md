---
name: Bug report
about: Something produced a wrong result, crashed, or behaved unexpectedly
title: ""
labels: bug
assignees: ""
---

## What happened

A clear description of the problem — wrong numbers, a crash, an unexpected
warning, a design that violates its constraints.

## Minimal reproduction

The smallest complete snippet (or YAML config) that shows the problem:

```python
from lattice_doe import find_optimal_design, PowerContrastConfig, DesignOptions

factors = {...}
cfg = PowerContrastConfig(...)
opts = DesignOptions(random_state=..., ...)   # please include random_state
result = find_optimal_design("...", factors, cfg, opts)
```

## Expected vs actual

- **Expected:**
- **Actual:** (include the full traceback if it crashed)

## Environment

- lattice-doe version:
- Python version:
- OS:
- Installed extras (e.g. `[cli]`, `[server]`, `[report]`):
