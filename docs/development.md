# Development Guide

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
make install       # base install with dev deps
make install-dev   # same as above (explicit alias)
```

Optional extras:

```bash
make install-mavlink      # PX4 SITL (mavsdk)
make install-perception   # YOLO (ultralytics)
```

## Workflow

```bash
make test                 # run pytest
make pipeline-demo        # run integrated pipeline
make quality              # validate artifacts with quality gate
make lint                 # ruff lint
make format               # ruff format
make typecheck            # mypy (soft mode)
make all-checks           # test + lint + typecheck
```

Recommended before commits:

```bash
make format && make lint && make test
```

For full validation:

```bash
make pipeline-quality-demo
```

## Repository structure

```
sentinel/
  common/               # Shared types, events, event bus, logging, geo utils
  config/               # YAML loader + Pydantic config schema
  recorder/             # JSONL event recorder
  mission_planner/      # Patterns, validation, metrics, planner
  autonomy_supervisor/  # State machine for mission lifecycle
  mavlink_bridge/       # Backend protocol, mock, MAVSDK, safety, bridge
  perception/           # Video source, mock/YOLO backends, drawing, metrics
  tracker/              # IoU tracker, track state, metrics, runner
  geolocalizer/         # Flat-ground pinhole geolocalizer, camera model
  tak_bridge/           # CoT XML builder, mapper, transport, bridge
  pipeline/             # Integrated mission pipeline, summary, report
  quality/              # Artifact validation, JSONL checks, smoke tests
apps/
  tools/                # CLI tools
tests/                  # pytest suite
configs/                # YAML config files
missions/               # Mission definition JSON files
docs/                   # Documentation
```

## Adding a new module

1. Create package under `sentinel/<module>/`
2. Add Pydantic models to `sentinel/common/types.py` if needed
3. Add config schema to `sentinel/config/schema.py`
4. Add YAML config to `configs/sim.yaml`
5. Implement module with clear public API
6. Publish events via EventBus
7. Write artifacts (JSONL, JSON) via run_dir
8. Add tests under `tests/`
9. Add CLI tool under `apps/tools/` if applicable
10. Add Makefile target
11. Update README.md and docs/architecture.md

## Rules

- No monolithic scripts — use packages and classes.
- Never break safe defaults (mock, dry_run, simulation).
- No heavy required dependencies — keep optional deps in extras.
- Every artifact must be parseable (valid JSON/JSONL).
- Every module must have tests.
- Use EventBus for cross-module communication, not direct calls.
- All state transitions must publish SystemEvents.
