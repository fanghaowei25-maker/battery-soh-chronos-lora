#!/usr/bin/env python3
"""Cache Frozen Chronos-2 embeddings for the SIT discharge-segment dataset.

Chronos-2 is frozen, so ``embed(X)`` never changes during head training.
Embedding once and reusing the cache is what makes the hyper-parameter and
ablation sweeps cheap (spec section 12).

Channel sets (spec section 13 ablation)
    V     -> [V]
    VC    -> [V, C_rate]
    VCT   -> [V, C_rate, T]      (the main configuration)

Pooling (spec section 6)
    Chronos-2 returns, per series, a tensor of shape
    ``(n_variates, n_patches, d_model)``.  We take the **mean over the patch
    dimension** for each variate and concatenate:

        z = concat(z_V, z_Crate, z_T)          shape (n_variates * 768,)

Usage
-----
    python3 cache_sit_embeddings.py --data data/sit_v2_discharge.npz \
        --channels VCT --out runs/emb_vct
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")   # the Xet CDN path fails here

CHANNEL_KEYS = {"V": ["V"], "VC": ["V", "C"], "VCT": ["V", "C", "T"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--channels", default="VCT", choices=list(CHANNEL_KEYS))
    ap.add_argument("--model-id", default="amazon/chronos-2")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument("--pool", default="mean", choices=["mean", "meanstd", "last"])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="smoke test only")
    args = ap.parse_args()

    from device_utils import pick_device, describe_device, resolve_model_id
    print(describe_device(args.device), flush=True)
    args.device = pick_device(args.device)
    args.model_id = resolve_model_id(args.model_id)

    out = args.out.resolve()
    if out.exists():
        raise SystemExit(f"output exists: {out}")
    out.mkdir(parents=True)

    d = np.load(args.data, allow_pickle=True)
    keys = CHANNEL_KEYS[args.channels]
    n = d["V"].shape[0]
    if args.limit:
        n = min(n, args.limit)
    X = np.stack([d[k][:n] for k in keys], axis=1).astype(np.float32)   # (n, nvar, L)
    print(f"input {X.shape}  channels={keys}  pool={args.pool}  device={args.device}", flush=True)

    import torch
    from chronos import Chronos2Pipeline

    t0 = time.monotonic()
    pipe = Chronos2Pipeline.from_pretrained(args.model_id, device_map=args.device)
    for p in pipe.model.parameters():
        p.requires_grad_(False)
    n_frozen = sum(p.numel() for p in pipe.model.parameters())
    print(f"backbone loaded in {time.monotonic()-t0:.0f}s | frozen params "
          f"{n_frozen/1e6:.1f}M | context {pipe.model_context_length}", flush=True)

    feats = None
    t0 = time.monotonic()
    done = 0
    bs = args.batch_size
    for s in range(0, n, bs):
        block = X[s:s + bs]
        emb = pipe.embed(block, batch_size=bs)[0]          # list, one per series
        vecs = []
        for e in emb:
            a = np.asarray(e, dtype=np.float32)            # (nvar, P, d)
            nvar, P, dmod = a.shape
            parts = []
            for v in range(nvar):
                av = a[v]
                if args.pool == "mean":
                    parts.append(av.mean(axis=0))
                elif args.pool == "last":
                    parts.append(av[-1])
                else:
                    parts.append(np.concatenate([av.mean(axis=0), av.std(axis=0)]))
            vecs.append(np.concatenate(parts))
        v = np.asarray(vecs, dtype=np.float32)
        if feats is None:
            feats = np.empty((n, v.shape[1]), dtype=np.float32)
        feats[s:s + len(v)] = v
        done += len(v)
        el = time.monotonic() - t0
        if done % (bs * 20) < bs or done == n:
            print(f"  {done:6d}/{n}  {el:6.0f}s  eta {el/done*(n-done)/60:.1f} min", flush=True)

    n_patches = np.asarray(emb[0]).shape[1] if hasattr(emb[0], "shape") else None
    np.savez_compressed(
        out / "embeddings.npz",
        Z=feats, channels=np.asarray(keys),
        soh_rated=d["soh_rated"][:n], soh_init=d["soh_init"][:n],
        cell_id=d["cell_id"][:n], group_id=d["group_id"][:n],
        c_rate_label=d["c_rate_label"][:n], temp_label=d["temp_label"][:n],
        cycle=d["cycle"][:n], rpt=d["rpt"][:n], split=d["split"][:n],
        c_init=d["c_init"],
    )
    (out / "config.json").write_text(json.dumps({
        "data": str(args.data), "model_id": args.model_id, "device": args.device,
        "channels": keys, "pool": args.pool, "batch_size": bs,
        "input_shape": list(X.shape), "embedding_dim": int(feats.shape[1]),
        "n_patches_per_variate": n_patches, "d_model": 768,
        "frozen_params_millions": n_frozen / 1e6,
        "model_context_length": pipe.model_context_length,
        "wall_clock_seconds": round(time.monotonic() - t0, 1),
        "note": "Chronos-2 frozen; mean pooling over patches; variates concatenated",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"embeddings {feats.shape} -> {out/'embeddings.npz'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
