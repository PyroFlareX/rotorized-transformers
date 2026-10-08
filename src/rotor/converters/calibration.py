"""Capture the inputs of selected linear modules on calibration windows (forward pre-hooks)."""

from __future__ import annotations

import torch


@torch.no_grad()
def collect_inputs(
    model: torch.nn.Module,
    module_paths: list[str],
    windows: torch.Tensor,
    max_rows: int,
    batch_size: int = 1,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Inputs of each module in ``module_paths`` as float32 CPU tensors ``(rows, in_features)``.

    ``windows`` is a ``(num_windows, seq_len)`` LongTensor of token ids, run in batches of ``batch_size``. The
    ``(batch * seq)`` rows are concatenated in window order and, if more than ``max_rows``, deterministically
    subsampled (seeded ``randperm``, kept in original order).
    """
    if windows.ndim != 2:
        raise ValueError(f"windows must be 2-D (num_windows, seq_len), got shape {tuple(windows.shape)}")
    if max_rows < 1 or batch_size < 1:
        raise ValueError("max_rows and batch_size must be >= 1")
    paths = list(dict.fromkeys(module_paths))
    captured: dict[str, list[torch.Tensor]] = {p: [] for p in paths}
    handles = []

    def make_hook(path: str):
        def hook(_module, args):
            x = args[0]
            captured[path].append(x.detach().reshape(-1, x.shape[-1]).to(device="cpu", dtype=torch.float32))

        return hook

    try:
        for path in paths:
            handles.append(model.get_submodule(path).register_forward_pre_hook(make_hook(path)))
        was_training = model.training
        model.eval()
        try:
            device = next(model.parameters()).device
            for start in range(0, windows.shape[0], batch_size):
                model(input_ids=windows[start : start + batch_size].to(device))
        finally:
            model.train(was_training)
    finally:
        for h in handles:
            h.remove()

    out: dict[str, torch.Tensor] = {}
    for path in paths:
        if not captured[path]:
            raise RuntimeError(f"module {path} received no input; is it on the forward path?")
        rows = torch.cat(captured[path], dim=0)
        if rows.shape[0] > max_rows:
            gen = torch.Generator().manual_seed(seed)
            idx = torch.randperm(rows.shape[0], generator=gen)[:max_rows].sort().values
            rows = rows[idx]
        out[path] = rows.contiguous()
    return out
