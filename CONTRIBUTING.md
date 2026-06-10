# Contributing

3DP-DDS is being developed as research software for robotic additive
manufacturing. Changes should preserve numerical clarity, explicit modeling
assumptions, and reproducible tests.

## Development setup

```bash
python -m pip install -e ".[all]"
python -m pip install "mypy>=1.10" "pytest>=8.0" "pytest-qt>=4.4" "ruff>=0.6"
```

## Validation

Run the same checks used by continuous integration:

```bash
pytest -q
ruff check src tests examples
mypy
```

The current mypy gate covers the shared value, validation, CLI, and occupancy
modules. Expand that list as additional modules reach a clean typing baseline.

## Change guidelines

- Keep geometry, process state, provenance, field storage, and analysis as
  separate concerns.
- Add focused tests for numerical behavior and invalid inputs.
- Document whether a field is geometric, diagnostic, or physically modeled.
- Preserve snapshot isolation for public result objects.
- Update checkpoint schema handling when serialized structures change.
- Avoid unrelated refactors in feature or bug-fix changes.

Use small commits with one coherent purpose. Release tags and publishing are
outside the current project scope.
