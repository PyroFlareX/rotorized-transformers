# Experiment runs

Each experiment is published as a uniquely named directory:

`reports/experiments/<model>__<method>__<target>__<UTCstamp>/`

The UTC timestamp uses `YYYYMMDDTHHMMSSZ`. A run contains `report.md`, `metrics.json`, `config.yaml`, `environment.json`, `figures/`, and `logs/run.log`; figures are present when produced. `metrics.json` keeps measured metrics and formula-derived estimates in separate fields, and the report labels them separately.

`rotor.reporting.ExperimentRun` creates these directories atomically and renders the report from `reports/templates/experiment_report.md`. Failed runs are published with `status: failed`, including failure details, before the exception is re-raised.

## Rules

- Never edit metrics or other published run contents after publication; create a new run instead.
- Never delete failed runs. Failed and negative results are part of the experiment record.
- Never invent or back-fill results. Any value computed from a formula rather than measured must be labeled `estimate`.
- Compare compression methods at equal stored bytes and report compute separately.
- Interpretation must not claim rotor gains from SVD truncation; distinguish the rotor representation's contribution from ordinary low-rank truncation.
