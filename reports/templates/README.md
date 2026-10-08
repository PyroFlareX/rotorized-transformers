# Experiment report template

`experiment_report.md` is the Markdown template rendered by `rotor.reporting.ExperimentRun`. The helper creates one atomically published directory per run:

`reports/experiments/<model>__<method>__<target>__<UTCstamp>/`

The UTC timestamp format is `YYYYMMDDTHHMMSSZ`. Each directory contains:

- `report.md` — rendered report;
- `metrics.json` — run identity/status, measured `metrics`, formula-derived `estimates`, and failure details when applicable;
- `config.yaml` — resolved run configuration;
- `environment.json` — captured host and software environment;
- `figures/` — generated figures (when supplied);
- `logs/run.log` — run log.

Use `ExperimentRun` as a context manager to collect output: add measured values to `.metrics`, formula-derived values to `.estimates`, figures with `.add_figure()`, and optional sections with `.add_section()`. Its report includes the run identity, status, summary, measurements, estimates, environment, failures, interpretation, figures, and artifact references. The `$...` placeholders in `experiment_report.md` are populated by `ExperimentRun.render_report()`; added sections and figures are inserted into their corresponding report sections.

A failed run is still published with `status: failed` and its traceback, then the original exception is re-raised. Never silently discard failed runs. Keep measured metrics and estimates separate in both data and interpretation; formula-derived values are not measurements.