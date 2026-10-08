# Kernels

Reserved for CUDA kernels implementing the rotor cascade forward pass. No kernels exist yet.

## Requirements

- Output must match the Python reference (`rotor.math` / `RotorCascade.forward`) within a stated tolerance, for each supported dtype.
- Benchmarks use `rotor.evaluation.performance.time_callable` (warmup, iters, median and p10/p90 reported) and `linear_microbench` for comparison against dense `nn.Linear` at matched batch sizes.
- Benchmark results go into `reports/experiments/` as run records with the environment included. Label any estimate as `estimate`.
