#!/usr/bin/env python3
"""Frozen Chronos-2 embeddings + trainable SOH head on SIT: all experiments.

Heads (spec section 13)
    linear : Ridge, alpha by GroupKFold CV inside the training cells only
    mlp    : the architecture from spec section 7
             LayerNorm -> Linear(->256) -> GELU -> Dropout(0.1)
                       -> Linear(->64)  -> GELU -> Linear(->1)
             early stopping on held-out *cells*

Experiments
    main_crosscell   train on seen cells, test on unseen cells (primary)
    cross_load       1C -> 2C and 2C -> 1C
    cross_temp       RT -> 40C and 40C -> RT

Leakage discipline
    * cells never span train/val/test
    * the feature scaler is fitted on training cells only
    * test cells are never used for early stopping or model selection

Usage
-----
    python3 train_sit_soh.py --emb-root runs --out results/sit
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ------------------------------------------------------------------ metrics (spec section 7)
def full_metrics(y, pred) -> dict:
    y = np.asarray(y, float)
    pred = np.asarray(pred, float)
    e = np.abs(pred - y)
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "n": int(len(e)),
        "MAE": float(e.mean()),
        "MAE_pp": float(e.mean() * 100),
        "RMSE": rmse,
        "RMSE_pp": rmse * 100,
        "MAPE": float(np.mean(e / np.maximum(np.abs(y), 1e-6))),
        "R2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "pm3pp": float((e <= 0.03).mean()),
    }


def per_cell(y, pred, cells) -> pd.DataFrame:
    df = pd.DataFrame({"cell": cells, "y": y, "p": pred})
    df["err"] = (df.p - df.y).abs()
    g = df.groupby("cell").agg(n=("err", "size"), MAE_pp=("err", lambda s: s.mean() * 100),
                               pm3pp=("err", lambda s: (s <= 0.03).mean()),
                               bias_pp=("err", lambda s: 0.0))
    bias = df.assign(b=df.p - df.y).groupby("cell").b.mean() * 100
    g["bias_pp"] = bias
    return g.round(4)


def soh_bins(y, pred) -> pd.DataFrame:
    df = pd.DataFrame({"y": y, "p": pred})
    df["err"] = (df.p - df.y).abs()
    df["bin"] = pd.cut(df.y, [-np.inf, 0.8, 0.9, np.inf],
                       labels=["SOH<0.8", "0.8<=SOH<0.9", "SOH>=0.9"])
    return (df.groupby("bin", observed=True)
              .agg(n=("err", "size"), MAE_pp=("err", lambda s: s.mean() * 100),
                   pm3pp=("err", lambda s: (s <= 0.03).mean())).round(4))


def train_linear(Ztr, ytr, groups, Zte):
    """Ridge with alpha chosen by GroupKFold CV **over training cells**."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(Ztr)
    A = sc.transform(Ztr)
    groups = np.asarray(groups)
    n_splits = int(min(5, len(np.unique(groups))))
    best, best_err = None, np.inf
    for alpha in (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0):
        errs = []
        for tr, te in GroupKFold(n_splits=n_splits).split(A, ytr, groups):
            m = Ridge(alpha=alpha).fit(A[tr], ytr[tr])
            errs.append(np.abs(m.predict(A[te]) - ytr[te]).mean())
        e = float(np.mean(errs))
        if e < best_err:
            best, best_err = alpha, e
    model = Ridge(alpha=best).fit(A, ytr)
    return model.predict(sc.transform(Zte)), {"alpha": best, "cv_mae_pp": best_err * 100}


def train_mlp(Ztr, ytr, Zva, yva, seed=0, epochs=150, patience=20,
              batch_size=256, device="mps"):
    """Spec section 7 head, in torch, with mini-batch SGD."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    np.random.seed(seed)
    d_in = Ztr.shape[1]
    try:
        dev = torch.device(device)
    except Exception:
        dev = torch.device("cpu")

    class Head(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.net = nn.Sequential(
                nn.LayerNorm(d),
                nn.Linear(d, 256), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(256, 64), nn.GELU(),
                nn.Linear(64, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

    # normalise with training statistics only (spec section 16)
    mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-8
    Xtr = torch.tensor((Ztr - mu) / sd, dtype=torch.float32).to(dev)
    Xva = torch.tensor((Zva - mu) / sd, dtype=torch.float32).to(dev)
    ytr_t = torch.tensor(ytr, dtype=torch.float32).to(dev)
    yva_t = torch.tensor(yva, dtype=torch.float32).to(dev)

    model = Head(d_in).to(dev)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    lossf = nn.MSELoss()
    n = Xtr.shape[0]
    best_state, best_val, bad = None, np.inf, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=dev)
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            opt.zero_grad()
            loss = lossf(model(Xtr[idx]), ytr_t[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = float(torch.abs(model(Xva) - yva_t).mean())
        if v < best_val - 1e-6:
            best_val, bad = v, 0
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()

    def predict(Z):
        with torch.no_grad():
            return model(torch.tensor((Z - mu) / sd, dtype=torch.float32).to(dev)).cpu().numpy()

    return predict, {"n_trainable_params": int(n_trainable),
                     "epochs_run": ep + 1, "best_val_MAE_pp": float(best_val) * 100,
                     "device": str(dev)}


def fit_and_predict(trainer, Ztr, ytr, gtr, Zva, yva, Zte, seed=0, device="mps"):
    """Fit one head on the training cells; return predictions for val and test.

    The head is fitted **once** and applied to both targets, so the wall-clock
    cost does not depend on how many evaluation splits we ask for.
    """
    if trainer == "linear":
        pv, info = train_linear(Ztr, ytr, gtr, Zva)
        pt, _ = train_linear(Ztr, ytr, gtr, Zte)
        return pv, pt, info
    predict, info = train_mlp(Ztr, ytr, Zva, yva, seed=seed, device=device)
    return predict(Zva), predict(Zte), info


# ------------------------------------------------------------------ experiments
def split_sets(d):
    cells = d["cell_id"]
    cr, tp = d["c_rate_label"], d["temp_label"]
    return {
        "main_crosscell": (d["split"] == "train", d["split"] == "validation",
                           d["split"] == "test"),
        "cross_load_1C_to_2C": ((cr == "1C"), None, (cr == "2C")),
        "cross_load_2C_to_1C": ((cr == "2C"), None, (cr == "1C")),
        "cross_temp_RT_to_40C": ((tp == "RT"), None, (tp == "40C")),
        "cross_temp_40C_to_RT": ((tp == "40C"), None, (tp == "RT")),
    }


def carve_validation(tr_mask, cells, rng):
    """For domain-transfer splits there is no pre-made validation set; hold out
    two training-side cells (never test-side cells)."""
    tr_cells = sorted(set(cells[tr_mask].tolist()))
    if len(tr_cells) <= 3:
        return tr_mask, np.zeros_like(tr_mask)
    pick = list(rng.permutation(tr_cells)[:2])
    va = np.isin(cells, pick)
    return tr_mask & ~va, va


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-root", type=Path, required=True,
                    help="dir containing emb_V / emb_VC / emb_VCT subdirs")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--label", default="soh_rated", choices=["soh_rated", "soh_init"])
    ap.add_argument("--seed", type=int, default=20260917)
    ap.add_argument("--head-device", default="auto", choices=["auto", "mps", "cpu", "cuda"])
    args = ap.parse_args()

    from device_utils import pick_device
    args.head_device = pick_device(args.head_device)
    print(f"head device: {args.head_device}", flush=True)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    variants = [("V", "emb_V"), ("VC", "emb_VC"), ("VCT", "emb_VCT"),
                ("HAND", "emb_HAND")]
    rows, percell_rows, bin_rows, all_preds = [], [], [], []

    for chan, sub in variants:
        f = args.emb_root / sub / "embeddings.npz"
        if not f.exists():
            print(f"[skip] {sub} not found"); continue
        d = np.load(f, allow_pickle=True)
        Z, y = d["Z"].astype(np.float64), d[args.label].astype(float)
        cells = d["cell_id"]
        cfg = json.loads((args.emb_root / sub / "config.json").read_text())
        print(f"\n=== channels={chan} emb={Z.shape} dim={Z.shape[1]}", flush=True)

        for exp, (tr, va, te) in split_sets(d).items():
            if va is None:
                tr, va = carve_validation(tr, cells, rng)
            tr = np.asarray(tr); va = np.asarray(va); te = np.asarray(te)
            if tr.sum() == 0 or te.sum() == 0 or va.sum() == 0:
                print(f"  [{exp}] skipped (empty split)"); continue
            # leakage guard
            assert set(cells[tr]).isdisjoint(set(cells[te])), f"{exp}: train/test cell overlap"
            assert set(cells[tr]).isdisjoint(set(cells[va])), f"{exp}: train/val cell overlap"
            assert set(cells[va]).isdisjoint(set(cells[te])), f"{exp}: val/test cell overlap"

            for head in ("linear", "mlp"):
                pred_va, pred_te, info = fit_and_predict(
                    head, Z[tr], y[tr], cells[tr], Z[va], y[va], Z[te],
                    seed=args.seed, device=args.head_device)

                for part, yy, pp in (("validation", y[va], pred_va), ("test", y[te], pred_te)):
                    mm = full_metrics(yy, pp)
                    rows.append({"channels": chan, "embedding_dim": int(Z.shape[1]),
                                 "experiment": exp, "head": head, "split": part, **mm, **info})
                    if part == "test":
                        pc = per_cell(yy, pp, cells[te])
                        pc.insert(0, "experiment", exp); pc.insert(0, "head", head)
                        pc.insert(0, "channels", chan)
                        percell_rows.append(pc.reset_index())
                        sb = soh_bins(yy, pp).reset_index()
                        sb.insert(0, "experiment", exp); sb.insert(0, "head", head)
                        sb.insert(0, "channels", chan)
                        bin_rows.append(sb)
                        all_preds.append(pd.DataFrame({
                            "channels": chan, "experiment": exp, "head": head,
                            "cell_id": cells[te], "cycle": d["cycle"][te],
                            "group_id": d["group_id"][te],
                            "c_rate_label": d["c_rate_label"][te],
                            "temp_label": d["temp_label"][te],
                            "soh_true": yy, "soh_pred": pp}))
                        print(f"  [{exp:22}] {head:6} MAE {mm['MAE_pp']:6.3f} pp  "
                              f"±3pp {mm['pm3pp']:5.1%}  R2 {mm['R2']:6.3f}", flush=True)

    m = pd.DataFrame(rows)
    m.to_csv(out / "metrics.csv", index=False)
    pd.concat(percell_rows, ignore_index=True).to_csv(out / "per_cell.csv", index=False)
    pd.concat(bin_rows, ignore_index=True).to_csv(out / "soh_bins.csv", index=False)
    pd.concat(all_preds, ignore_index=True).to_csv(out / "predictions.csv", index=False)
    (out / "run_config.json").write_text(json.dumps({
        "emb_root": str(args.emb_root), "label": args.label, "seed": args.seed,
        "heads": ["linear (Ridge alpha by GroupKFold CV on train cells)",
                  "mlp (LayerNorm-256-GELU-Dropout0.1-64-GELU-1, early stop on val cells)"],
        "note": "Chronos-2 frozen; scaler fitted on training cells only",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    show = m[(m.split == "test")][["channels", "experiment", "head", "n", "MAE_pp",
                                   "RMSE_pp", "MAPE", "R2", "pm3pp"]]
    print("\n" + show.round(4).to_string(index=False))
    print("\nwrote ->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
