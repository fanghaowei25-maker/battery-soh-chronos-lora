#!/usr/bin/env python3
"""Analysis for the backbone-adaptation runs (plan steps 1 and 3).

Two things the raw summary does not give you:

1. **Learning-rate selection by validation only.**  The plan is explicit that
   the held-out test cell must never drive hyper-parameter choice, so the best
   ``lr_back`` is picked on ``val_mae_pp`` and only then read out on test.

2. **Paired per-cell comparison.**  A lower mean MAE can hide the fact that
   half the cells got worse.  This reports, cell by cell,
   ``ΔMAE = MAE_arm − MAE_baseline`` and counts how many improved.

    python3 analyze_adapt.py --runs results/lr_sweep --baseline results/adapt_quick
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.width", 220)


def load(path: Path) -> pd.DataFrame:
    if path is None:
        raise FileNotFoundError("基线路径为空")
    f = path / "adapt_runs.csv" if path.is_dir() else path
    if not Path(f).exists():
        raise FileNotFoundError(
            f"找不到 {f}\n"
            f"  说明：主实验一次性跑了两条臂（frozen + unfreeze1），"
            f"配对比较应当直接用【本次运行】的结果，"
            f"即 --baseline 指向和 --runs 同一个目录。")
    d = pd.read_csv(f)
    # protocol v2 renamed the selection metric; fall back for older runs.
    if "val_official_macro_pp" not in d.columns and "val_mae_pp" in d.columns:
        d["val_official_macro_pp"] = d["val_mae_pp"]
    return d


def select_lr(df: pd.DataFrame) -> pd.DataFrame:
    """Pick the backbone LR by validation MAE only, then report its test numbers."""
    t = (df.groupby(["arm", "lr_back"])
           .agg(val_official_macro=("val_official_macro_pp", "mean"),
                official=("official(>=0.7)_MAE", "mean"),
                band0708=("0.7-0.8_MAE", "mean"),
                full=("full_MAE", "mean"),
                cell_R2_median=("cell_R2", "median"),
                cell_R2_neg=("cell_R2", lambda s: float((s < 0).mean())),
                mean_epochs=("epochs_run", "mean"),
                runs=("seed", "size"))
           .round(4))
    print("=" * 104)
    print("Step 1：按 backbone 学习率分组")
    print("  选择依据 = 验证集的【每电芯 macro 官方口径 MAE】，不是测试集")
    print("=" * 104)
    print(t.to_string())
    for arm in t.index.get_level_values(0).unique():
        sub = t.loc[arm]
        if len(sub) > 1:
            best = sub.val_official_macro.idxmin()
            print(f"\n  [{arm}] 按验证集选出的最佳 lr_back = {best:g}")
            print(f"       该配置测试表现: official {sub.loc[best,'official']:.3f} pp | "
                  f"band0708 {sub.loc[best,'band0708']:.3f} | "
                  f"cell_R2 中位 {sub.loc[best,'cell_R2_median']:.3f}")
            # guard against picking a LR whose test result is wildly different
            rank_val = sub.val_official_macro.rank()
            rank_test = sub.official.rank()
            print(f"       验证集排名 {rank_val[best]:.0f} / 测试集排名 {rank_test[best]:.0f} "
                  f"（两者越接近，说明选择越稳）")
    return t


def paired(df_arm: pd.DataFrame, df_base: pd.DataFrame, arm_label: str,
           base_label: str, step: str = "Step 3") -> pd.DataFrame:
    """Per-cell ΔMAE for one arm against a baseline."""
    a = df_arm.groupby("test_cell")["official(>=0.7)_MAE"].mean().rename("arm")
    b = df_base.groupby("test_cell")["official(>=0.7)_MAE"].mean().rename("base")
    m = pd.concat([b, a], axis=1).dropna()
    m["ΔMAE"] = m.arm - m.base
    m = m.sort_values("ΔMAE")
    print("\n" + "=" * 104)
    print(f"{step}：逐电芯配对比较  [{arm_label}]  −  [{base_label}]")
    print("  （ΔMAE < 0 表示前者的 MAE 更低、更好）")
    print("=" * 104)
    print(m.round(3).to_string())
    imp = int((m["ΔMAE"] < 0).sum())
    wor = int((m["ΔMAE"] > 0).sum())
    print(f"\n  改善 {imp} 只 / 恶化 {wor} 只 / 共 {len(m)} 只")
    print(f"  mean ΔMAE   {m['ΔMAE'].mean():+.3f} pp")
    print(f"  median ΔMAE {m['ΔMAE'].median():+.3f} pp")
    print(f"  最差恶化    {m['ΔMAE'].max():+.3f} pp ({m['ΔMAE'].idxmax()})")
    print(f"  最好改善    {m['ΔMAE'].min():+.3f} pp ({m['ΔMAE'].idxmin()})")
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True, help="含 adapt_runs.csv 的目录")
    ap.add_argument("--baseline", type=Path, default=None,
                    help="配对比较的基线结果目录；**默认用 --runs 自身**"
                         "（因为主实验同一次运行里就含 frozen 和 unfreeze1 两臂）")
    ap.add_argument("--baseline-arm", default="frozen",
                    help="基线目录中用作参照的臂名")
    ap.add_argument("--arm", default=None, help="要比较的臂；默认自动挑最优")
    args = ap.parse_args()

    df = load(args.runs)
    print(f"载入 {len(df)} 条运行，臂: {sorted(df.arm.unique())}")
    t = select_lr(df)

    # Best configuration by *validation* only — the test column is read out
    # afterwards.  Sorting on the test metric here would silently turn this
    # into test-set model selection.
    cand = t.reset_index()
    best_row = cand.sort_values(["val_official_macro", "official"]).iloc[0]
    print(f"\n按验证集选出的最佳配置: {best_row.arm} / lr={best_row.lr_back:g} "
          f"-> 验证 {best_row.val_official_macro:.3f} pp "
          f"| 测试 official {best_row.official:.3f} pp")

    # ---- paired comparison ------------------------------------------------
    # Default: compare arms *inside the same run*.  The main experiment trains
    # ``frozen`` and ``unfreeze1`` under identical protocol/seeds/folds, so the
    # only clean baseline is the frozen arm from the same run — an external
    # baseline usually covers a different (smaller) set of test cells.
    base_src = args.baseline if args.baseline is not None else args.runs
    try:
        b = load(base_src)
    except FileNotFoundError as exc:
        print(f"\n[基线缺失] {exc}")
        print("  -> 自动改用 --runs 自身作为基线来源")
        b = df
        base_src = args.runs

    base = b[b.arm == args.baseline_arm]
    if len(base) == 0:
        print(f"\n[跳过配对比较] {base_src} 里没有 '{args.baseline_arm}' 臂"
              f"（该文件里只有 {sorted(b.arm.unique())}）")
        print("  主实验请确认 --arms 同时包含 frozen 和 unfreeze1")
    else:
        arm_name = args.arm or best_row.arm
        if arm_name == args.baseline_arm:
            print(f"\n[跳过配对比较] 最优臂就是基线臂 '{arm_name}'")
        else:
            a = df[df.arm == arm_name]
            if "lr_back" in a and a.lr_back.nunique() > 1:
                best_lr = a.groupby("lr_back").val_official_macro_pp.mean().idxmin()
                a = a[a.lr_back == best_lr]
                print(f"  （已按验证集把 {arm_name} 固定到 lr_back={best_lr:g}）")
            m = paired(a, base, arm_name, args.baseline_arm)
            # coverage check: make sure the baseline really covers the test cells
            cov = len(m) / max(len(set(a.test_cell)), 1)
            print(f"\n  基线覆盖 {len(m)} / {len(set(a.test_cell))} 只测试电芯"
                  f"（{cov:.0%}）")
            if cov < 1.0:
                print(f"  [注意] 有 {len(set(a.test_cell)) - len(m)} 只电芯没有基线，"
                      f"配对比较只反映有基线的那部分。")

    # ---- 同一个臂内部，不同 backbone 学习率的配对比较 ---------------------
    # 补跑 3e-5 之后 adapt_runs.csv 里会有 (unfreeze1, 1e-4) 和 (unfreeze1, 3e-5)
    # 两组。这两组用的是同一批折、同一批种子，唯一差别是学习率，所以可以直接
    # 逐电芯配对。选择哪个学习率仍然只看验证集那一列。
    for arm_i, sub in df.groupby("arm"):
        lrs = sorted(sub.lr_back.unique())
        if len(lrs) < 2:
            continue
        v = sub.groupby("lr_back").val_official_macro_pp.mean().sort_values()
        chosen = v.index[0]
        print("\n" + "=" * 104)
        print(f"[{arm_i}] 这个臂有 {len(lrs)} 个 backbone 学习率，"
              f"按验证集选出的是 {chosen:g}")
        print("=" * 104)
        print("  验证集 macro 官方口径（越小越好）:")
        print(v.round(3).to_string())
        for lr in lrs:
            if lr == chosen:
                continue
            paired(sub[sub.lr_back == lr], sub[sub.lr_back == chosen],
                   f"{arm_i} lr={lr:g}", f"{arm_i} lr={chosen:g}", step="Step 4")
            print(f"\n  解读：上面的 ΔMAE 是 [{arm_i} lr={lr:g}] 相对 "
                  f"[{arm_i} lr={chosen:g}] 的差值。\n"
                  f"  正值 = {chosen:g} 更好，负值 = {lr:g} 更好。")

    print("\n提示：改进需 > 1.5 pp 才算数（种子噪声 ~1.2 pp/电芯）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
