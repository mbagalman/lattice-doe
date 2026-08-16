## Summary

What this changes and **why** (link the issue or the
`docs/planning/ENHANCEMENTS.md` ticket if there is one).

## Checklist

- [ ] Fast suite green locally: `pytest -m "not slow"`
- [ ] `black --check --line-length 100 lattice_doe/` clean
- [ ] `ruff check lattice_doe/` clean
- [ ] mypy count did not grow (and if it shrank, the `ci.yml` baseline is
      ratcheted down in this PR)
- [ ] Bug fixes include a regression test that fails on the pre-fix code
- [ ] New/changed behavior is pinned by exact-value tests, not smoke tests
- [ ] Ledger row updated in `docs/planning/ENHANCEMENTS.md` if this closes
      a tracked finding
