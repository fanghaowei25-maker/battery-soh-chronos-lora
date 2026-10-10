#!/usr/bin/env python3
"""Step 1 — a fixed, defensible evaluation protocol.

Fixes three weaknesses of the earlier evaluation:

1. **Only 3 test cells.**  A single group-stratified split gives one number per
   configuration, and with a 20-cell dataset that is far too few.  Here the
   primary protocol is **Leave-One-Cell-Out (LOCO)**: every cell is the test
   cell exactly once, so all 20 cells contribute and the result is a
   distribution rather than a point estimate.

2. **Confounded domain-transfer experiments.**  The earlier cross-load split
   used ``c_rate == "1C"`` for training, but the 1C cells include *both* room
   temperature and 40 °C — temperature was moving at the same time as C-rate.
   Likewise the earlier cross-temperature split tested on all 40 °C cells,
   which include both 1C and 2C.  Now the pairs are held strictly:

       跨负载:  1C@40°C (G2)  <->  2C@40°C (G3)     only C-rate changes
       跨温度:  1C@常温 (G1)  <->  1C@40°C (G2)     only temperature changes

3. **One seed.**  Every protocol runs ``--seeds`` times (default 5) and reports
   mean and standard deviation.

Metrics are reported for the official lifetime range (SOH >= 0.7) as the
primary figure, plus the 0.7-0.8 EOL band, 0.5-0.7 (second life), the full
range, and the anomalous cell separately.

    python3 evaluate_protocol.py --emb runs/emb_V --out results/protocol_v1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from train_sit_soh import full_metrics
from lowsoh_coverage import fit as fit_mlp

BANDS = [("full", 0.0, 9.9), ("ge0.9", 0.9, 9.9), ("0.8-0.9", 0.8, 0.9),
         ("0.7-0.8", 0.7, 0.8), ("0.5-0.7", 0.5, 0.7), ("lt0.5", 0.0, 0.5),
         ("official(>=0.7)", 0.7, 9.9)]


def band_metrics(y, pred, cells):
    err = np.abs(np.asarray(pred) - np.asarray(y)) * 100
    out = {}
    for name, lo, hi in BANDS:
        m = (y >= lo) & (y < hi)
        out[f"{name}_n"] = int(m.sum())
        out[f"{name}_MAE"] = float(err[m].mean()) if m.any() else np.nan
        out[f"{name}_pm3"] = float((err[m] <= 3).mean()) if m.any() else np.nan
    m = cells == "001-8"
    out["anomalous_MAE"] = float(err[m].mean()) if m.any() else np.nan
    return out


def protocols(cells, group):
    """Yield (name, kind, train_mask, val_mask, test_mask) for every fold.

    The validation cell is always a *training-side* cell, never the test cell,
    and it is excluded from training so early stopping stays honest.
    """
    uniq = sorted(set(cells.tolist()))
    # --- LOCO: every cell is test exactly once -------------------------------
    for i, c in enumerate(uniq):
        te = cells == c
        vc = uniq[(i + 1) % len(uniq)]
        if vc == c:
            vc = uniq[(i + 2) % len(uniq)]
        va = cells == vc
        yield f"LOCO:{c}", "loco", ~te & ~va, va, te

    # --- strict cross-load: 1C@40C <-> 2C@40C (only C-rate moves) -----------
    G2 = np.isin(cells, ["002-1", "002-2", "002-3", "002-4"])          # 1C, 40C
    G3 = np.isin(cells, ["002-5", "002-7", "003-1", "003-3",
                         "003-5", "003-7"])                            # 2C, 40C
    for tr_cells, te_cells, name in (
            (G2, G3, "cross_load: 1C@40C -> 2C@40C"),
            (G3, G2, "cross_load: 2C@40C -> 1C@40C")):
        for c in sorted(set(cells[tr_cells].tolist()))[:3]:            # 3 folds
            va = cells == c
            yield f"{name} [val={c}]", "cross_load", tr_cells & ~va, va, te_cells

    # --- strict cross-temperature: 1C@RT <-> 1C@40C (only T moves) ----------
    G1 = np.isin(cells, ["001-1", "001-2", "001-3", "001-4", "001-5",
                         "001-6", "001-7", "001-8", "101-1", "101-3"])  # 1C, RT
    for tr_cells, te_cells, name in (
            (G1, G2, "cross_temp: 1C@RT -> 1C@40C"),
            (G2, G1, "cross_temp: 1C@40C -> 1C@RT")):
        for c in sorted(set(cells[tr_cells].tolist()))[:3]:
            va = cells == c
            yield f"{name} [val={c}]", "cross_temp", tr_cells & ~va, va, te_cells


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", type=Path, default=Path("runs/emb_V"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--oversample", type=float, default=0.0,
                    help="低 SOH 过采样比例（0 = 关闭，保持与旧实验可比）")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--protocols", default="all",
                    choices=["all", "loco", "cross_load", "cross_temp"])
    args = ap.parse_args()
    from device_utils import pick_device
    args.device = pick_device(args.device)
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)

    e = np.load(args.emb / "embeddings.npz", allow_pickle=True)
    Z, y, cells = e["Z"].astype(np.float64), e["soh_rated"].astype(float), e["cell_id"]
    seeds = [20260917 + 101 * i for i in range(args.seeds)]

    from lowsoh_coverage import oversample as oversample_fn

    rows = []
    for name, kind, tr, va, te in protocols(cells, None):
        if args.protocols != "all" and kind != args.protocols:
            continue
        assert not (tr & va).any() and not (tr & te).any() and not (va & te).any(), \
            f"{name}: split overlap"
        for seed in seeds:
            Zt, yt = Z[tr], y[tr]
            if args.oversample > 0:
                Zt, yt, _, _ = oversample_fn(Zt, yt, target_share=args.oversample)
            pred, info = fit_mlp(Zt, yt, Z[va], y[va], Z[te], np.ones(len(yt)),
                                 seed, device=args.device)
            m = full_metrics(y[te], pred)
            rec = {"protocol": name, "kind": kind, "seed": seed,
                   "n_train_cells": int(len(set(cells[tr]))),
                   "val_cell": ",".join(sorted(set(cells[va].tolist()))),
                   "n_test": int(te.sum()),
                   "MAE_pp": m["MAE_pp"], "pm3pp": m["pm3pp"], "R2": m["R2"],
                   **band_metrics(y[te], pred, cells[te]), **info}
            rows.append(rec)
        print(f"  {name:<46} n_test={te.sum():5d}  "
              f"官方口径 MAE={np.mean([r['official(>=0.7)_MAE'] for r in rows if r['protocol']==name]):6.3f}  "
              f"全范围={np.mean([r['MAE_pp'] for r in rows if r['protocol']==name]):6.3f}",
              flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / "protocol_runs.csv", index=False)

    agg = (df.groupby(["kind", "protocol"])
             .agg(seeds=("seed", "nunique"),
                  n_test=("n_test", "first"),
                  full_MAE=("full_MAE", "mean"), full_pm3=("full_pm3", "mean"),
                  official_MAE=("official(>=0.7)_MAE", "mean"),
                  official_pm3=("official(>=0.7)_pm3", "mean"),
                  b07_08_MAE=("0.7-0.8_MAE", "mean"),
                  b05_07_MAE=("0.5-0.7_MAE", "mean"),
                  anomalous_MAE=("anomalous_MAE", "mean"),
                  R2=("R2", "mean"))
             .round(4))
    agg.to_csv(out / "protocol_summary.csv")

    print("\n=== 按实验类型汇总（对 folds × seeds 取平均）===")
    kind_mean = (df.groupby("kind")
                   .agg(MAPE_full_MAE=("full_MAE", "mean"),
                        official_MAE=("official(>=0.7)_MAE", "mean"),
                        official_pm3=("official(>=0.7)_pm3", "mean"),
                        b07_08_MAE=("0.7-0.8_MAE", "mean"),
                        b05_07_MAE=("0.5-0.7_MAE", "mean"),
                        anomalous_MAE=("anomalous_MAE", "mean"),
                        n_runs=("seed", "size")).round(4))
    print(kind_mean.to_string())

    (out / "config.json").write_text(json.dumps({
        "emb": str(args.emb), "seeds": seeds, "n_seeds": len(seeds),
        "oversample": args.oversample,
        "protocols": {"LOCO": "20 折，每只电芯各做一次测试集",
                      "cross_load": "1C@40°C (G2) <-> 2C@40°C (G3)，只变倍率",
                      "cross_temp": "1C@常温 (G1) <-> 1C@40°C (G2)，只变温度"},
        "metrics": {"primary": "官方口径 SOH>=0.7",
                    "secondary": ["0.7-0.8", "0.5-0.7", "全范围", "异常电芯 001-8"]},
        "model": "V-only frozen Chronos-2 embedding + MLP(256,64)，固定不变",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nwrote ->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
