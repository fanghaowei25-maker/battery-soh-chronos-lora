#!/usr/bin/env python3
"""A differentiable replacement for ``Chronos2Pipeline.embed``.

Why this is needed
------------------
``Chronos2Pipeline.embed`` is decorated with ``@torch.no_grad()`` (pipeline.py
line 1066), so **no gradient can ever flow back into the backbone through it**.
Any "partial fine-tuning" experiment built on ``p.embed()`` silently trains
nothing — the frozen parameters and the "unfrozen" ones behave identically.

This module replicates exactly what ``embed`` does, minus the decorator:

    Chronos2Dataset(mode=TEST)  ->  model.encode()  ->  per-series tensors

so the same tensors come out but the autograd graph is intact.

Verified below by ``self_test()``: after one backward pass the unfrozen block
must have non-zero gradients and the frozen blocks must have none.
"""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from chronos.chronos2.dataset import Chronos2Dataset, DatasetMode


def encode_differentiable(model, inputs, context_length: int, batch_size: int,
                          output_patch_size: int, device: torch.device):
    """Yield (encoder_outputs_list, locs, scales) one batch at a time, with grad.

    Mirrors Chronos2Pipeline.embed but keeps the graph alive and does not move
    tensors to CPU.
    """
    ds = Chronos2Dataset(
        inputs, context_length=context_length, prediction_length=0,
        batch_size=batch_size, output_patch_size=output_patch_size,
        mode=DatasetMode.TEST,
    )
    loader = DataLoader(ds, batch_size=None, num_workers=0, shuffle=False,
                        drop_last=False)
    for batch in loader:
        assert batch["future_target"] is None
        ctx = batch["context"].to(device=device, dtype=torch.float32)
        gids = batch["group_ids"].to(device)
        encoder_outputs, (locs, scales), *_ = model.encode(context=ctx, group_ids=gids)
        ranges = batch["target_idx_ranges"]
        embeds = [encoder_outputs[0][s:e] for (s, e) in ranges]      # no .cpu()
        ls = [(locs[s:e], scales[s:e]) for (s, e) in ranges]
        yield embeds, ls


def pooled_embedding(model, inputs, *, context_length, batch_size, output_patch_size,
                     device, pool: str = "mean"):
    """Mean-pooled, concatenated embedding over the patch dimension.

    Returns a single tensor of shape (n_series_total, n_variates * d_model) with
    the autograd graph intact.
    """
    chunks = []
    for embeds, _ in encode_differentiable(model, inputs, context_length,
                                           batch_size, output_patch_size, device):
        for e in embeds:                                  # e: (nvar, P, d)
            a = e                                          # keep on device
            if pool == "mean":
                v = a.mean(dim=1)                          # (nvar, d)
            elif pool == "last":
                v = a[:, -1]
            else:
                raise ValueError(pool)
            chunks.append(v.reshape(-1))                   # (nvar*d,)
    return torch.stack(chunks)


def self_test(model, device="cpu", seq_len: int = 1024) -> bool:
    """Assert that fine-tuning actually receives gradients."""
    ctx = model.chronos_config.context_length if hasattr(model, "chronos_config") else 2048
    x = torch.randn(4, 1, seq_len, device=device)
    head = torch.nn.Linear(768, 1).to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    blocks = model.encoder.block
    targets = list(blocks[-1].parameters())
    for p in targets:
        p.requires_grad_(True)

    z = pooled_embedding(model, x, context_length=min(ctx, 2048), batch_size=4,
                         output_patch_size=16, device=torch.device(device))
    loss = ((head(z).squeeze(-1) - torch.zeros(z.shape[0], device=device)) ** 2).mean()
    loss.backward()

    got = sum(1 for p in targets if p.grad is not None and p.grad.abs().sum() > 0)
    leaked = sum(1 for p in blocks[0].parameters()
                 if p.grad is not None and p.grad.abs().sum() > 0)
    print(f"  末 block 有梯度的参数张量: {got}/{len(targets)}")
    print(f"  首 block（应冻结）有梯度: {leaked}  （应为 0）")
    print(f"  梯度范数: {max((p.grad.norm().item() for p in targets if p.grad is not None), default=0):.3e}")
    return got > 0 and leaked == 0


if __name__ == "__main__":
    import os
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from chronos import Chronos2Pipeline

    pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")
    ok = self_test(pipe.model, device="cpu")
    print("自检:", "通过" if ok else "失败")
