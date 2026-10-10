#!/usr/bin/env python3
"""Step 1 — fix the sparse low-SOH coverage of the training split.

Diagnosis (see report appendix C)
---------------------------------
The training split is not *missing* deep-aged samples: cell 101-3 runs down to
SOH 0.114.  The problem is that the tail is extremely thin:

    SOH < 0.4 :  88 training samples  (0.85 %), all from ONE cell (101-3)
    SOH < 0.2 :  25 training samples  (0.24 %)
    test cell 001-8 alone has 174 samples below 0.4 and 66 below 0.2

So the model sees the deep-aged regime roughly 1 sample in 120, and only from a
single cell.  This script tests three cheap fixes that do not change the split:

    baseline        no reweighting
    invfreq         per-sample weight = 1 / (bin count), SOH bins of 0.1
    oversample      duplicate low-SOH samples up to a target share
    focal           weight emphasises large residuals as well as rare bins

Every variant is run with several seeds because single-run noise is ~0.5 pp
(report appendix B.0), and the split is never changed, so test cells stay unseen.

    python3 lowsoh_coverage.py --emb runs/emb_V --out results/lowsoh
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from train_sit_soh import full_metrics

SEEDS = (20260917, 1, 2, 3)


def make_weights(y, mode, n_bins=10, strength=1.0):
    """Per-sample training weights."""
    if mode == "baseline":
        return np.ones_like(y)
    idx = np.clip((y * n_bins).astype(int), 0, n_bins - 1)
    counts = np.bincount(idx, minlength=n_bins).astype(float)
    freq = counts[idx] / counts.sum()
    w = (1.0 / np.maximum(freq, 1e-9)) ** strength
    return w / w.mean()


def oversample(Z, y, target_share=0.15, thr=0.4):
    """Duplicate low-SOH rows so they reach ``target_share`` of the set."""
    low = y < thr
    n_low, n_tot = int(low.sum()), len(y)
    want = int(target_share * n_tot)
    if n_low == 0 or want <= n_low:
        return Z, y, np.ones(len(y)), 0
    reps = want // n_low
    extra = want % n_low
    Zx = np.concatenate([Z] + [Z[low]] * reps + [Z[low][:extra]], axis=0)
    yx = np.concatenate([y] + [y[low]] * reps + [y[low][:extra]], axis=0)
    return Zx, yx, np.ones(len(yx)), reps * n_low + extra


def fit(Ztr, ytr, Zva, yva, Zte, w, seed, epochs=200, patience=25,
        hidden=(256, 64), device="mps"):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device(device)
    mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-8
    d = Ztr.shape[1]
    layers = []
    for h in hidden:
        layers += [nn.Linear(d, h), nn.GELU(), nn.Dropout(0.1)]; d = h
    layers += [nn.Linear(d, 1)]

    class Net(nn.Module):
        def __init__(self):
            super().__init__(); self.net = nn.Sequential(nn.LayerNorm(Ztr.shape[1]), *layers)
        def forward(self, x): return self.net(x).squeeze(-1)

    Xtr = torch.tensor((Ztr - mu) / sd, dtype=torch.float32).to(dev)
    Xva = torch.tensor((Zva - mu) / sd, dtype=torch.float32).to(dev)
    ytr_t = torch.tensor(ytr, dtype=torch.float32).to(dev)
    yva_t = torch.tensor(yva, dtype=torch.float32).to(dev)
    wt = torch.tensor(w, dtype=torch.float32).to(dev)

    net = Net().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    n = Xtr.shape[0]
    best, best_val, bad = None, np.inf, 0
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n, device=dev)
        for s in range(0, n, 256):
            b = perm[s:s + 256]
            opt.zero_grad()
            loss = (wt[b] * (net(Xtr[b]) - ytr_t[b]) ** 2).mean()
            loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            v = float(torch.abs(net(Xva) - yva_t).mean())
        if v < best_val - 1e-6:
            best_val, bad = v, 0
            best = {k: t.clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best); net.eval()
    with torch.no_grad():
        pred = net(torch.tensor((Zte - mu) / sd, dtype=torch.float32).to(dev)).cpu().numpy()
    return pred, {"epochs": ep + 1, "val_mae_pp": best_val * 100}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", type=Path, default=Path("runs/emb_V"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    from device_utils import pick_device
    args.device = pick_device(args.device)
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)

    e = np.load(args.emb / "embeddings.npz", allow_pickle=True)
    Z, y, cells = e["Z"].astype(np.float64), e["soh_rated"].astype(float), e["cell_id"]
    sp = e["split"].astype(str)
    tr, va, te = sp == "train", sp == "validation", sp == "test"

    variants = {
        "baseline":         dict(mode="baseline"),
        "invfreq":          dict(mode="invfreq", strength=1.0),
        "invfreq_sqrt":     dict(mode="invfreq", strength=0.5),
        "oversample_15pct": dict(oversample=0.15),
        "oversample_30pct": dict(oversample=0.30),
    }

    rows, preds = [], []
    for name, cfg in variants.items():
        for seed in SEEDS:
            if "oversample" in cfg:
                Zt, yt, w, added = oversample(Z[tr], y[tr],
                                              target_share=cfg["oversample"])
            else:
                Zt, yt = Z[tr], y[tr]
                w = make_weights(yt, cfg["mode"], strength=cfg.get("strength", 1.0))
                added = 0
            pred, info = fit(Zt, yt, Z[va], y[va], Z[te], w, seed, device=args.device)
            m = full_metrics(y[te], pred)
            # per-cell, especially the deep-aged test cell
            err = np.abs(pred - y[te]) * 100
            per = {c: float(err[cells[te] == c].mean()) for c in sorted(set(cells[te]))}
            lowmask = y[te] < 0.4
            rows.append({"variant": name, "seed": seed, "added_samples": added,
                         "MAE_pp": m["MAE_pp"], "pm3pp": m["pm3pp"], "R2": m["R2"],
                         "MAE_lowSOH_pp": float(err[lowmask].mean()) if lowmask.any() else np.nan,
                         "lowSOH_pm3pp": float((err[lowmask] <= 3).mean()) if lowmask.any() else np.nan,
                         **{f"cell_{k}": v for k, v in per.items()}, **info})
            preds.append(pd.DataFrame({"variant": name, "seed": seed,
                                       "cell_id": cells[te], "cycle": e["cycle"][te],
                                       "soh_true": y[te], "soh_pred": pred}))
            print(f"  {name:18} seed={seed}  MAE={m['MAE_pp']:6.3f}  "
                  f"±3pp={m['pm3pp']:5.1%}  001-8={per.get('001-8', float('nan')):6.3f}  "
                  f"低SOH={rows[-1]['MAE_lowSOH_pp']:6.3f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / "lowsoh_variants.csv", index=False)
    pd.concat(preds, ignore_index=True).to_csv(out / "predictions.csv", index=False)

    agg = (df.groupby("variant")
             .agg(MAE_mean=("MAE_pp", "mean"), MAE_std=("MAE_pp", "std"),
                  pm3pp=("pm3pp", "mean"), R2=("R2", "mean"),
                  lowSOH_MAE=("MAE_lowSOH_pp", "mean"),
                  lowSOH_pm3pp=("lowSOH_pm3pp", "mean"),
                  cell_001_8=("cell_001-8", "mean"))
             .sort_values("MAE_mean"))
    print("\n=== 汇总（4 个种子的均值）===")
    print(agg.round(4).to_string())
    agg.to_csv(out / "lowsoh_summary.csv")
    (out / "config.json").write_text(json.dumps(
        {"emb": str(args.emb), "seeds": list(SEEDS), "variants": list(variants)},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nwrote ->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
