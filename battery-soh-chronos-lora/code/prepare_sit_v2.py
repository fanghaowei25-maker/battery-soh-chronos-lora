#!/usr/bin/env python3
"""Spec-compliant SIT dataset: one complete discharge segment = one sample.

Differences vs ``build_sit_dataset.py`` (the first pass):

* input is **[V, C_rate, T]** instead of voltage only
* the sample is the **complete discharge segment**, not a 600-point charge curve
* the segment is downsampled to a fixed length ``--length`` (default 1024) over
  normalised time, so the *whole* segment is used, no truncation

Why downsampling is legitimate here
-----------------------------------
Chronos-2 accepts up to 8192 points (``model_context_length``), so a full
~3776-point discharge would fit.  Measured embed cost on this machine for
``nvar=3``:

    L=1024 ->  75 ms/window    L=2048 -> 154 ms/window    L=3776 -> ~290 ms/window

A discharge at constant current is a smooth curve; 1024 points over ~3776 s is
one sample per 3.7 s and retains the shape information.  We therefore downsample
rather than pay 4x the wall-clock for three ablation passes.

Sign convention (recorded here and in the report)
-------------------------------------------------
SIT documents ``Current(A)`` as **positive = charge, negative = discharge**.
We keep that sign, so every discharge sample has a negative C-rate.  The
*magnitude* still separates the load domains (1C = -1.0, 2C = -2.0), which is
what the cross-load experiment needs.

Usage
-----
    python3 prepare_sit_v2.py --zip 00_data/SIT/Data.zip --length 1024 \
        --out 01_sit/data/sit_v2_discharge.npz
"""
from __future__ import annotations

import argparse
import io
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

RATED_AH = 50.0
FIRST_CYCLES_FOR_INIT = 5

CELL_GROUPS: dict[str, list[str]] = {
    "G1": ["001-1", "001-2", "001-3", "001-4", "001-5",
           "001-6", "001-7", "001-8", "101-1", "101-3"],
    "G2": ["002-1", "002-2", "002-3", "002-4"],
    "G3": ["002-5", "002-7", "003-1", "003-3", "003-5", "003-7"],
}
CELL_TESTER: dict[str, str] = {}
for _c in CELL_GROUPS["G1"]:
    CELL_TESTER[_c] = "Chroma_101" if _c.startswith("101") else "Repower_001"
for _c in CELL_GROUPS["G2"]:
    CELL_TESTER[_c] = "Repower_002"
for _c in CELL_GROUPS["G3"]:
    CELL_TESTER[_c] = "Repower_003" if _c.startswith("003") else "Repower_002"
CELL_GROUP: dict[str, str] = {c: g for g, cs in CELL_GROUPS.items() for c in cs}

# group -> (C-rate label, temperature label)
GROUP_CONDITION = {
    "G1": ("1C", "RT"),
    "G2": ("1C", "40C"),
    "G3": ("2C", "40C"),
}


def resample(y: np.ndarray, n: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return np.full(n, np.nan, dtype=np.float32)
    if y.size == 1:
        return np.full(n, y[0], dtype=np.float32)
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, y.size), y).astype(np.float32)


def pick(df, *needles):
    for c in df.columns:
        cl = str(c).lower()
        if any(nd in cl for nd in needles):
            return c
    return None


def process_cell(job):
    zip_path, cell_id, L, limit = job
    tester, group = CELL_TESTER[cell_id], CELL_GROUP[cell_id]
    crate_lab, temp_lab = GROUP_CONDITION[group]
    try:
        zf = zipfile.ZipFile(zip_path)
    except Exception as exc:
        return cell_id, None, f"open zip: {exc}"
    try:
        try:
            summary = pd.read_csv(io.BytesIO(zf.read(f"Data/Cycle_Summary/{cell_id}.csv")))
        except KeyError:
            return cell_id, None, "no Cycle_Summary"

        work = summary[summary["Type"].astype(str).str.lower() == "discharge"].copy()
        work = work.sort_values("Cycle").reset_index(drop=True)
        if limit:
            work = work.head(limit)
        if work.empty:
            return cell_id, None, "no discharge rows"

        q_dis = pd.to_numeric(work["Capacity_Ah"], errors="coerce").to_numpy(float)
        n_first = min(FIRST_CYCLES_FOR_INIT, int(np.isfinite(q_dis).sum()))
        c_init = float(np.nanmax(q_dis[:n_first])) if n_first else np.nan
        if not np.isfinite(c_init) or c_init <= 0:
            return cell_id, None, "bad c_init"
        q_smooth = (pd.Series(q_dis).rolling(5, center=True, min_periods=1)
                    .median().to_numpy())

        V, C, T, q, cyc_kept, skipped = [], [], [], [], [], 0
        for cycle, fname, q_raw in zip(work["Cycle"].to_numpy(int),
                                       work["Filename"].astype(str).to_numpy(), q_dis):
            member = f"Data/{tester}/Cycle_Data/{cell_id}/{fname}"
            try:
                df = pd.read_excel(io.BytesIO(zf.read(member)), sheet_name="Discharge")
            except Exception:
                skipped += 1
                continue
            vcol, icol = pick(df, "voltage"), pick(df, "current")
            tcol = pick(df, "celsius", "temperature")
            if vcol is None or icol is None:
                skipped += 1
                continue
            v = pd.to_numeric(df[vcol], errors="coerce").to_numpy(float)
            i = pd.to_numeric(df[icol], errors="coerce").to_numpy(float)
            t = (pd.to_numeric(df[tcol], errors="coerce").to_numpy(float)
                 if tcol else np.full(len(df), np.nan))
            V.append(resample(v, L))
            C.append(resample(i / RATED_AH, L))       # keep SIT's sign convention
            T.append(resample(t, L))
            q.append(q_raw)
            cyc_kept.append(int(cycle))

        if not V:
            return cell_id, None, f"nothing parsed (skipped {skipped})"
        keep = np.isin(work["Cycle"].to_numpy(int), cyc_kept)
        out = {
            "V": np.asarray(V, np.float32),
            "C": np.asarray(C, np.float32),
            "T": np.asarray(T, np.float32),
            "q_dis": np.asarray(q, np.float32),
            "q_smooth": q_smooth[keep].astype(np.float32),
            "soh_rated": (q_smooth[keep] / RATED_AH).astype(np.float32),
            "soh_init": (q_smooth[keep] / c_init).astype(np.float32),
            "cell_id": np.asarray([cell_id] * len(cyc_kept)),
            "group_id": np.asarray([group] * len(cyc_kept)),
            "c_rate_label": np.asarray([crate_lab] * len(cyc_kept)),
            "temp_label": np.asarray([temp_lab] * len(cyc_kept)),
            "cycle": np.asarray(cyc_kept, np.int32),
            "c_init": np.asarray([c_init], np.float32),
            "skipped": np.asarray([skipped], np.int32),
        }
        return cell_id, out, None
    except Exception as exc:
        return cell_id, None, f"{type(exc).__name__}: {exc}"
    finally:
        zf.close()


def make_split(cells, groups, rng_seed=20260917):
    """Cell-disjoint split that keeps every (C-rate, temperature) condition in
    every split, so the main experiment is not confounded by domain imbalance."""
    rng = np.random.default_rng(rng_seed)
    split = np.full(cells.shape, "train", dtype=object)
    for g in sorted(set(groups.tolist())):
        cs = sorted(set(cells[groups == g].tolist()))
        if len(cs) < 4:
            print(f"  WARNING: group {g} has {len(cs)} cell(s) -> all train")
            continue
        # deterministic but seed-controlled: shuffle, then take the tail two
        order = list(cs)
        rng.shuffle(order)
        split[cells == order[-2]] = "validation"
        split[cells == order[-1]] = "test"
    return split.astype("U10")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--length", type=int, default=1024)
    ap.add_argument("--cells", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260917)
    args = ap.parse_args()

    cells = args.cells or list(CELL_TESTER)
    import os
    workers = args.workers or min(8, os.cpu_count() or 4)
    jobs = [(str(args.zip), c, args.length, args.limit) for c in cells]
    print(f"parsing {len(cells)} cells, L={args.length}, {workers} workers ...", flush=True)
    t0 = time.time()
    parts = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(process_cell, j): j[1] for j in jobs}
        for fut in as_completed(futs):
            cid, out, err = fut.result()
            if err:
                print(f"  {cid:>7}: SKIP {err}", flush=True)
            else:
                parts[cid] = out
                print(f"  {cid:>7}: {out['cycle'].size:5d} cycles  C_init={float(out['c_init'][0]):.2f}  "
                      f"rated {out['soh_rated'].min():.3f}~{out['soh_rated'].max():.3f}", flush=True)
    print(f"parsed in {time.time()-t0:.0f}s", flush=True)
    if not parts:
        raise SystemExit("nothing parsed")

    keys = ["V", "C", "T", "q_dis", "q_smooth", "soh_rated", "soh_init",
            "cell_id", "group_id", "c_rate_label", "temp_label", "cycle"]
    merged = {k: np.concatenate([parts[c][k] for c in parts]) for k in keys}
    merged["rpt"] = merged["cycle"].astype(np.int64)
    merged["split"] = make_split(merged["cell_id"], merged["group_id"], args.seed)
    merged["c_init"] = np.asarray([float(parts[c]["c_init"][0]) for c in parts])
    merged["c_init_cells"] = np.asarray(list(parts))
    merged["length"] = np.asarray([args.length])
    merged["seed"] = np.asarray([args.seed])
    merged["sign_convention"] = np.asarray(["current>0 = charge (SIT convention)"])

    # leakage assertions (spec section 16)
    tr = set(merged["cell_id"][merged["split"] == "train"])
    va = set(merged["cell_id"][merged["split"] == "validation"])
    te = set(merged["cell_id"][merged["split"] == "test"])
    assert tr.isdisjoint(va), "train/val cell overlap"
    assert tr.isdisjoint(te), "train/test cell overlap"
    assert va.isdisjoint(te), "val/test cell overlap"
    dup = len(merged["cell_id"]) - len({(c, k) for c, k in
                                        zip(merged["cell_id"], merged["cycle"])})
    assert dup == 0, f"{dup} duplicate (cell, cycle) pairs"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **merged)

    print()
    print(f"rows {merged['soh_rated'].size} | V {merged['V'].shape} C {merged['C'].shape} T {merged['T'].shape}")
    print(f"cells train {len(tr)} / val {len(va)} / test {len(te)}")
    print(f"SOH(rated/=50)  {merged['soh_rated'].min():.3f} ~ {merged['soh_rated'].max():.3f}")
    print(f"SOH(init/=C_init) {merged['soh_init'].min():.3f} ~ {merged['soh_init'].max():.3f}")
    print(f"leakage assertions passed (0 duplicate (cell,cycle) pairs)")
    print(f"saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
