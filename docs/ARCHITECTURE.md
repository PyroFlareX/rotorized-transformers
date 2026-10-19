# Architecture

## Package layout

```
src/rotor/
  telemetry.py    asynchronous process, host, and device resource sampling (JSONL)
  environment.py   measured host/software capability detection and resource monitoring
  reporting.py     atomic experiment run directories, metrics/estimates, logs and figures
  data.py          tokenization, deterministic token windows, window fingerprints
  math/            topology, Givens stages, RotorCascade/RotorLinear, low-rank residual, SVD→Givens decomposition
    butterfly.py   classic learned butterfly factorization
    sparse.py      magnitude-pruned sparse baseline
    ga.py          Euclidean GA plane rotors and batched SVD→rotor elimination (see ALGEBRAIC_GA_CONVERSION.md)
    clifford.py    Cl(2,0)/Cl(3,0) blade multiplication matrices and block projection
  formats/         rotor-scaler derived checkpoint format: spec constants, Manifest, write/read; ga.py: ga-rotor-svd format
  models/          checkpoint discovery (header-only), Qwen3.5 adapter, module replacement, loading, tiny test checkpoints
  converters/      budget allocation, projection-local and joint FFN fitting, calibration capture, conversion pipeline,
                   algebraic GA conversion, Clifford pilot runner
  evaluation/      reconstruction, LM, long-context, performance, baselines, generation, tasks, retrieval, collapse diagnostics
  training/        optimizer groups, layerwise fit schedules, recovery distillation
  runtime/         optional fused Triton inference backend for rotor-scaler cascades
scripts/           CLI entry points (table below)
runtime/           native runtime (interfaces/ header mirrors of the format and topology, kernels/, fork/); independent of Python
tests/             pytest suite (CPU-only; markers: slow, network, nas, cxx)
configs/           storage.yaml, models.yaml, experiments/*.yaml, evaluation/
reports/           templates/, experiments/ (published run directories), prompt-evaluations/, papers/
docs/              architecture, research plan, decisions, roadmap, algebraic GA conversion
```

### Dependency rules

- `rotor.math` imports only `torch`/stdlib: **no** `transformers`, no model or checkpoint concepts.
- `rotor.models` is the **only** place that knows checkpoint tensor names and module paths
  (`model.language_model.layers.N.mlp.up_proj.weight` on disk, `model.layers.N.mlp.up_proj` in the loaded model).
- `rotor.formats` depends on `math`, not on `models`; model-specific names enter only as opaque module strings.
- `rotor.converters`, `rotor.evaluation`, `rotor.training` may use `math`, `formats`, `models`; `math` and `formats` never import them.
- `runtime/` implements the format in C++ and does not depend on Python; the normative semantics live in
  `src/rotor/formats/spec.py` (mirrored by `runtime/interfaces/rotor_format.h`).
- Paths come only from `rotor.storage.load_storage()`; scripts never hardcode locations.

## Data flow

```mermaid
flowchart LR
  B[Base checkpoint on NAS<br/>base_root, read-only] --> D[Discovery<br/>models.discovery / qwen3_5]
  D --> A[Allocation<br/>converters.budget]
  A --> F[Fit<br/>converters.fit<br/>weight, activation, joint FFN output]
  C[Calibration and held-out teacher FFN rows<br/>disjoint fixed inputs] -.-> F
  F --> M[Manifest + safetensors<br/>formats.write_structured]
  M -->|atomic_dir| X[Derived checkpoint under NAS temp/ (rotor_derived_root)]
  X --> L[load_converted<br/>models.loading]
  B --> L
  L --> E[Evaluation<br/>LM, generation, tasks, retrieval, long-context, performance]
  E --> R[ExperimentRun<br/>reports/experiments/<model>__<method>__<target>__<UTCstamp>/]
  H[detect_environment / ResourceMonitor] --> R
  R --> O[report.md, metrics.json, config.yaml, environment.json, figures/, logs/run.log]
```

Each run directory is published by `rotor.reporting.ExperimentRun`; `ROTOR_REPORTS_DIR` overrides the reports root.
Failed runs are published with status `failed` before exceptions propagate. Formula-derived values are estimates and remain
separate from measured metrics.

The standalone Phase 4B runner is `scripts/fit_joint_ffn.py`, documented with its distinct joint-fit YAML schema and
CLI in `configs/experiments/README.md`; the checked-in template is
`configs/experiments/qwen3.5-0.8b-joint-ffn.yaml` (its calibration and held-out paths are intentionally null until
approved text is configured). The runner captures teacher MLP inputs from disjoint real-text windows, fits gate/up/down
together with a normalized squared composed FFN output objective, and writes the three modules in the existing
`rotor-scaler` format. It accounts for one combined payload budget, including biases; its rotor allocator spreads
remaining budget by least-filled structured-weight/dense-weight ratio. Joint snapshots are atomically stored beside the
checkpoint in `<name>.snapshots/`, bind the config/base/projection/data identity, and retain all three FP32 modules plus
the shared optimizer state. Inference checkpoints use guarded atomic publication; every run publishes config, provenance,
input fingerprints, metrics, and environment through `ExperimentRun`. The runner does not change format semantics or
the per-projection converter schema.

## Runtime telemetry and performance measurement

`rotor.telemetry.RuntimeTelemetry(path: Path, interval_seconds: float = 1.0)` records process/runtime resource use to a
separate JSONL listing, without changing `ResourceMonitor` or its existing report fields. Use it around conversion,
model loading, and/or evaluation where a time series is needed. The supplied path is opened exclusively (`x` mode):
choose a new path for each run; a pre-existing listing is never overwritten. Parent directories are created when
needed. Every sample and phase/error record is JSON-serializable, schema version 1, UTC-stamped, and includes a
monotonic elapsed offset in seconds. The sampler emits immediately on entry, then at the configured interval (default
1 Hz), and once on exit. Each line is flushed immediately, and the sampler thread is stopped and joined even when
the measured operation raises. A `run_exception` record preserves the exception type/message. Sampling or file
errors are retained in `as_dict()` and do not mask an exception from measured work.

Call `set_phase(name, request_id=None)` as work moves between loading, conversion, prefill/generation, or individual
prompts. A `phase` record marks each change, and subsequent samples carry that phase/request id. Samples before the
first explicit attribution use `phase: "unattributed"` and `request_id: null`. `as_dict()` is intended after context
exit; it reports sample count, requested interval, measured duration, total process CPU time and utilization, current and
sampled-peak RSS, per-device sampled memory peaks, host available RAM/swap, and process I/O counters, plus sampling
errors when applicable. Device snapshots in each sample contain current values.

### JSONL sample field definitions

All names carry units explicitly. Every record has integer `schema_version` (currently 1), `record_type` (`sample`,
`phase`, `run_exception`, or `telemetry_error`), `timestamp_utc` (ISO-8601 UTC), `elapsed_seconds` (monotonic time
since context entry), `phase` (string), and `request_id` (string or null). Samples additionally have a zero-based
`sample_index` and `sampling_interval_seconds` (requested interval, seconds), then these fields:
| Field | Meaning |
|---|---|
| `process_cpu_time_seconds` | Cumulative user+system CPU seconds of this process at sample time. |
| `process_cpu_utilization_percent` | CPU seconds divided by monotonic seconds since previous sample × 100. 100% means one fully occupied logical CPU; threaded processes may exceed 100%. First sample is null. |
| `process_rss_bytes` | Current resident set size, not a process-lifetime high-water mark. |
| `host_memory.available_ram_bytes`, `available_swap_bytes`, `total_swap_bytes`, `used_swap_bytes` | Host MemAvailable and swap capacity/free/used bytes. |
| `process_io.read_bytes`, `write_bytes` | Process block-storage bytes read/written where procfs provides them; these are kernel I/O accounting, not model-file byte totals. |
| `devices[]` (`backend: amd_sysfs`) | `card`, optional `pci_address`/`name`/`driver`, `gpu_busy_percent`, current/total `vram_*_bytes` and `gtt_*_bytes`, exposed `temperature_c` map, and `power_w`. GPU busy is percent, temperatures °C, power watts. |
| `devices[]` (`backend: cuda`) | `device_index`, `name`, current allocator `allocated_bytes` and `reserved_bytes`, only when an already-loaded PyTorch CUDA build reports CUDA available. These are allocator counters, not total device use. |
| `device_sensors` | Per-backend observation booleans; false means no supported sensor/backend was observable (not that the device was idle). |

Unavailable or unsupported measurements are JSON null (or an empty device list when no device backend is observable),
never fabricated zero values. Per-device fields may be null independently. AMD and process/host metrics use procfs/sysfs
direct reads; there is no per-sample `nvidia-smi` subprocess. CUDA allocator calls are conditional, but may involve a
runtime API call. The sampler wakes at the requested low rate and does cheap reads; OS scheduling, device API calls,
and filesystem flush latency still add overhead and are not guaranteed zero. For short operations, initial/final
samples dominate: treat the samples as a coarse timeline rather than cycle-level profiling. RSS/device peaks are
sampled high-water marks and can miss transient spikes. Host RAM/swap is system-wide. AMD VRAM/GTT are device-global,
not process-attributed. For CUDA, use allocator fields for this process and do not conflate them with total board use.

### Generation timing standard

Performance comparisons MUST report the input and generated token counts along with TTFT, prefill and decode rates
from the existing `rotor.evaluation.generation.measure_generation` result. It defines TTFT as whole-prompt prefill
plus first-token projection/argmax; prefill tokens/s is prompt tokens divided by TTFT. Decode tokens/s covers only
steps after the first generated token (one cached single-token forward plus argmax each), with rate
`(generated_tokens - 1) / decode_seconds`; it is null for a one-token output. Preserve the returned token IDs/text and
counts to establish which prompt/result was measured. Its resource subrecord is `ResourceMonitor` (peak RSS and
CUDA allocation), while RuntimeTelemetry provides the separate sampled time series; retain both rather than
substituting one for the other.

`measure_generation` performs no warmup. Label the first call cold/startup-inclusive; to report steady state, run a
distinct warmup generation and report a subsequent measured call as warm (do not silently discard or combine the cold
measurement). For base-versus-rotor comparisons, hold exact prompt token IDs, generated-token limit, device, dtype,
and runtime settings fixed and record both models' prompt/input and output token counts. GPU performance claims MUST
be backed by an actual GPU execution: if PyTorch falls back to CPU, report CPU measurements only, not GPU speed or
GPU throughput. Record device/runtime identity and relevant host memory/swap conditions; sampling evidence can expose
resource pressure but is not itself a substitute for generation latency metrics.

Example around full conversion (use a new, non-existing JSONL path for each run):

```sh
.venv/bin/python scripts/convert_model.py --config configs/experiments/qwen3.5-0.8b-ffn-weight.yaml --name <unique> --telemetry <new-jsonl-path>
```

The joint FFN output fitter is run separately from the projection-local converter:

```sh
.venv/bin/python scripts/fit_joint_ffn.py --config <joint-ffn-config.yaml>
```

See `configs/experiments/README.md` for its schema, input split requirements, exact resume flags, and output accounting.
