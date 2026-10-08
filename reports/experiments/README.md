# Experiment records

JSON run records written by scripts and by `rotor.evaluation.records.write_run_record`.

## File naming

`<UTC>-<kind>-<name>.json`, for example `20261008T213754Z-convert-qwen3.5-0.8b-ffn-b025-butterfly-weight.json`.

- `<UTC>`: UTC timestamp of the run start.
- `<kind>`: run type (`convert`, `eval`, `baseline`, `bench`, `recovery`, ...).
- `<name>`: experiment name from the config.

## Contents

Each record includes:

- the config used, or its hash;
- the environment: Python, torch, transformers and package versions, device, and host details;
- base checkpoint provenance (HF id, revision, shard SHA-256);
- metrics and loss histories as measured;
- byte accounting (dense vs structured bytes).

Records are written atomically. Do not edit them by hand.

## Rules

- Records are never invented or back-filled. Failed and negative runs are kept.
- Any value computed from a formula rather than measured is labeled `estimate`.
