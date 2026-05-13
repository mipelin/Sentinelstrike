# Quality Gate

The quality gate validates that a pipeline run produced correct, parseable artifacts.

## What it validates

### Smoke checks

- Required project files exist (configs, README, pyproject.toml).
- Core module imports succeed (types, planner, perception, tracker, geolocalizer, TAK, pipeline).

### Artifact validation

For a pipeline run directory:

- All expected artifacts exist.
- JSON files are parseable.
- JSONL files are valid (each line is parseable JSON).
- `report.md` exists and is non-empty.
- `pipeline_summary.json` contains required keys (`mission_id`, `planner`, `artifacts`).

### Expected pipeline artifacts

```
metadata.json
events.jsonl
mission_plan.json
detections.jsonl
tracks.jsonl
geo_observations.jsonl
tak_messages.jsonl
perception_metrics.json
tracker_metrics.json
geolocalizer_metrics.json
tak_metrics.json
pipeline_summary.json
report.md
```

## Usage

```bash
# Run pipeline then validate
make pipeline-quality-demo

# Validate latest run
make quality

# Validate without strict mode (warnings only)
make smoke
```

## Strict vs non-strict

- **Strict** (default): exits with code 1 if any validation fails. Use in CI.
- **Non-strict**: prints warnings but always exits 0. Use in development.

## Limitations

- No JSON schema validation — only checks parseability.
- No artifact content correctness checks beyond structure.
- No performance or timing validation.
- No coverage reporting yet.

## Future work

- **CI GitHub Actions** — automated pipeline + quality gate on push.
- **Coverage** — pytest-cov with threshold enforcement.
- **Artifact schema validation** — JSON Schema for each artifact type.
- **Performance benchmarks** — detect regressions in pipeline duration.
- **Jetson runtime checks** — validate on target hardware.
