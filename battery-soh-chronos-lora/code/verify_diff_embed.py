#!/usr/bin/env python3
"""Verification suite for the backbone-adaptation experiment.

Checks, in order of importance:

  1. NUMERICAL EQUIVALENCE  — does ``diff_embed`` reproduce the official
     ``Chronos2Pipeline.embed`` output bit-for-bit when the model is frozen?
     If not, every comparison between the frozen and adapted arms is invalid.
  2. GRADIENT ROUTING       — do LoRA adapters at the first, middle and last
     layer all receive gradients, and do frozen blocks stay gradient-free?

Usage:
    python verify_diff_embed.py --model ./chronos-2 --data data/sit_v2_discharge.npz
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def pool_like_cached(embeds):
    """Same pooling as cache_sit_embeddings.py: per-variate mean over patches."""
    out = []
    for e in embeds:                       # e: (nvar, P, d)
        a = np.asarray(e, dtype=np.float32)
        out.append(a.mean(axis=1).reshape(-1))
    return np.stack(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="amazon/chronos-2")
    ap.add_argument("--data", type=Path, default=Path("data/sit_v2_discharge.npz"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--n", type=int, default=32)
    args = ap.parse_args()

    from device_utils import pick_device, resolve_model_id
    dev = pick_device(args.device)
    print("=" * 78)
    print("验证 1：diff_embed 与官方 embed 的数值一致性（冻结状态）")
    print("=" * 78)
    print(f"设备 {dev}")

    from chronos import Chronos2Pipeline
    from diff_embed import pooled_embedding

    pipe = Chronos2Pipeline.from_pretrained(resolve_model_id(args.model), device_map=dev)
    model = pipe.model
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    d = np.load(args.data, allow_pickle=True)
    X = d["V"][:args.n].astype(np.float32)[:, None, :]      # (n, 1, L)
    CL = min(pipe.model_context_length, 2048)
    ops = pipe.model_output_patch_size
    print(f"输入 {X.shape} | context_length={CL} | output_patch_size={ops}\n")

    # --- official path (has @torch.no_grad) --------------------------------
    with torch.no_grad():
        official, _ = pipe.embed(X, batch_size=8)
    A = pool_like_cached([o.detach().cpu().numpy() for o in official])

    # --- diff path, gradients disabled -> should be identical ---------------
    with torch.no_grad():
        B = pooled_embedding(model, X, context_length=CL, batch_size=8,
                             output_patch_size=ops,
                             device=torch.device(dev)).detach().cpu().numpy()

    print(f"官方 embed    输出 {A.shape}  dtype={A.dtype}")
    print(f"diff_embed   输出 {B.shape}  dtype={B.dtype}")
    if A.shape != B.shape:
        print("  [FAIL] 形状不一致，无法比较")
        return 1

    diff = np.abs(A - B)
    scale = np.abs(A).mean()
    print(f"\n最大绝对差  {diff.max():.3e}")
    print(f"平均绝对差  {diff.mean():.3e}")
    print(f"相对量级    {diff.max()/max(scale,1e-12):.3e}  (以 |A| 均值为参照)")
    ok = diff.max() < 1e-4 * max(scale, 1.0)
    print(f"\n判定: {'[OK] 两条路径数值一致' if ok else '[FAIL] 两条路径不一致'}")

    # --- gradient routing ---------------------------------------------------
    print("\n" + "=" * 78)
    print("验证 2：LoRA 梯度覆盖（首/中/末 adapter）")
    print("=" * 78)
    import torch.nn as nn
    from backbone_adapt import attach_lora, LoRALinear

    pipe2 = Chronos2Pipeline.from_pretrained(resolve_model_id(args.model), device_map=dev)
    m2 = pipe2.model
    for p in m2.parameters():
        p.requires_grad_(False)
    n_wrapped = attach_lora(m2, r=8, alpha=16)
    print(f"已挂载 LoRA 到 {n_wrapped} 个层")

    xg = torch.randn(4, 1, 1024, device=dev)
    head = nn.Linear(768, 1).to(dev)

    def one_step():
        head.zero_grad(); m2.zero_grad(set_to_none=True)
        z = pooled_embedding(m2, xg, context_length=CL, batch_size=4,
                             output_patch_size=ops, device=torch.device(dev))
        ((head(z).squeeze(-1) - torch.zeros(z.shape[0], device=dev)) ** 2).mean().backward()

    # LoRA standard init: B = 0, so on step 1 only B receives gradient
    # (d(delta)/dA is proportional to B). A starts moving on step 2.
    print("\n  [第 1 步] B 初始为零 -> 此时只有 B 应有梯度")
    one_step()

    n_blocks = len(m2.encoder.block)
    def report(tag):
        for lab, idx in [("首个 block (0)", 0), ("中间 block", n_blocks // 2),
                         ("末尾 block", n_blocks - 1)]:
            ads = [m for m in m2.encoder.block[idx].modules() if isinstance(m, LoRALinear)]
            gA = sum(1 for a in ads if a.A.grad is not None and a.A.grad.abs().sum() > 0)
            gB = sum(1 for a in ads if a.B.grad is not None and a.B.grad.abs().sum() > 0)
            nA = max((a.A.grad.norm().item() for a in ads if a.A.grad is not None), default=0.0)
            nB = max((a.B.grad.norm().item() for a in ads if a.B.grad is not None), default=0.0)
            print(f"  {tag} {lab:16} adapter {len(ads):2d} | A 有梯度 {gA:2d} (‖{nA:.2e}) | "
                  f"B 有梯度 {gB:2d} (‖{nB:.2e})")

    report("[第 1 步]")

    # apply the gradients so B becomes non-zero, then A should light up too
    for blk in m2.encoder.block:
        for mod in blk.modules():
            if isinstance(mod, LoRALinear):
                with torch.no_grad():
                    if mod.A.grad is not None: mod.A -= 1e-2 * mod.A.grad
                    if mod.B.grad is not None: mod.B -= 1e-2 * mod.B.grad
    print()
    print("  [第 2 步] B 已非零 -> A 与 B 都应有梯度")
    one_step()
    report("[第 2 步]")

    # frozen blocks must stay clean
    leaked = sum(1 for blk in m2.encoder.block for p in blk.parameters()
                 if not p.requires_grad and p.grad is not None
                 and p.grad.abs().sum() > 0)
    print(f"\n  未做 LoRA 的冻结参数中出现梯度的数量: {leaked}  （应为 0）")

    print("\n" + "=" * 78)
    print("结论:", "全部通过" if ok else "存在失败项")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
