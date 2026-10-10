"""Backbone adaptation vs fully frozen — the decisive experiment.

Everything is held fixed except *which parameters may move*:

    frozen       backbone untouched, only the MLP head trains
                 (uses the cached embeddings — same numbers as protocol_v1)
    lora         LoRA (r=8, alpha=16) on q/v of every attention layer
    unfreeze1    last 1 encoder block trainable          (9.4 M)
    unfreeze2    last 2 encoder blocks trainable         (18.9 M)
    unfreeze3    last 3 encoder blocks trainable         (28.3 M)

Same SIT data, V-only input, same LOCO protocol, same MLP(256,64) head, same
loss, same seeds.  Only the trainable set changes.

**Critical implementation note.**  ``Chronos2Pipeline.embed`` carries a
``@torch.no_grad()`` decorator, so fine-tuning through it trains *nothing* —
an earlier timing run of mine was invalid for exactly this reason.  The
trainable arms therefore go through ``diff_embed.pooled_embedding``, which
reproduces the same tensors with the autograd graph intact (verified by
``diff_embed.self_test``).

Reported metrics (the last two are the ones that matter):

    MAE on SOH >= 0.7          primary
    MAE on the 0.7-0.8 band     EOL region, RUL-critical
    median per-cell R^2         is one cell's degradation tracked?
    fraction of cells with R^2 < 0

    python3 backbone_adapt.py --out results/adapt_v1 --folds 6 --seeds 1
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from train_sit_soh import full_metrics
from evaluate_protocol import band_metrics


# ------------------------------------------------------------------ LoRA
class LoRALinear(nn.Module):
    """y = W x + (alpha/r) * B (A x), with W frozen."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        # A/B must be created on the same device/dtype as the wrapped weight,
        # otherwise LoRA is silently unusable on GPU.
        dev, dt = base.weight.device, base.weight.dtype
        self.A = nn.Parameter(torch.zeros(r, base.in_features, device=dev, dtype=dt))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=dev, dtype=dt))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scaling = alpha / r

    def forward(self, x):
        return self.base(x) + self.scaling * (x @ self.A.T @ self.B.T)


# 默认只挂 q/v（原版 LoRA 论文的选择）。但在 Chronos-2 上，layer.1（group
# attention）的 q/k 梯度约 1e-8、基本不参与前向，挂上去等于浪费容量；而 o 和
# MLP 恰恰是梯度最大的地方。想覆盖全部线性层用
#   --lora-targets q k v o --lora-mlp
LORA_TARGET_PRESETS = {
    "qv": (("q", "v"), False),           # 48 个适配器，589,824 参数
    "qkvo": (("q", "k", "v", "o"), False),   # 96 个，1,179,648
    "all": (("q", "k", "v", "o"), True),     # 120 个，1,916,928
}


def attach_lora(model, r=8, alpha=16, targets=("q", "v"), include_mlp=False):
    """Wrap the chosen linear submodules of every encoder block."""
    n = 0
    for blk in model.encoder.block:
        for layer in blk.layer:
            sa = getattr(layer, "self_attention", None)
            if sa is not None:
                for name in targets:
                    mod = getattr(sa, name, None)
                    if isinstance(mod, nn.Linear):
                        setattr(sa, name, LoRALinear(mod, r, alpha))
                        n += 1
            if include_mlp:
                mlp = getattr(layer, "mlp", None)
                if mlp is not None:
                    for name in ("wi", "wo"):
                        mod = getattr(mlp, name, None)
                        if isinstance(mod, nn.Linear):
                            setattr(mlp, name, LoRALinear(mod, r, alpha))
                            n += 1
    return n


# ------------------------------------------------------------------ head
def make_head(d_in, hidden=(256, 64)):
    layers = []
    for h in hidden:
        layers += [nn.Linear(d_in, h), nn.GELU(), nn.Dropout(0.1)]
        d_in = h
    layers += [nn.Linear(d_in, 1)]
    return nn.Sequential(nn.LayerNorm(768), *layers)   # input is always 768 for V-only


def set_run_seed(seed: int) -> None:
    """Seed every RNG before constructing any trainable module."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def standardize_embedding(z, mu, sd):
    """Apply the training-fold embedding transform to cached or live features."""
    return (z - mu) / sd


def r2_per_cell(y, pred, cells):
    out = {}
    for c in np.unique(cells):
        m = cells == c
        st = ((y[m] - y[m].mean()) ** 2).sum()
        out[str(c)] = float(1 - ((y[m] - pred[m]) ** 2).sum() / st) if st > 0 else np.nan
    return out


def snapshot_trainable(model):
    """Snapshot only trainable parameters to keep LoRA checkpoint memory small."""
    return {n: p.detach().clone() for n, p in model.named_parameters()
            if p.requires_grad}


def restore_trainable(model, snap):
    """Restore a snapshot produced by :func:`snapshot_trainable`."""
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in snap:
                p.copy_(snap[n])


# --------------------------------------------------------------- validation protocol
SIT_GROUPS = {
    "1C/常温": ["001-1", "001-2", "001-3", "001-4", "001-5",
                "001-6", "001-7", "001-8", "101-1", "101-3"],
    "1C/40C": ["002-1", "002-2", "002-3", "002-4"],
    "2C/40C": ["002-5", "002-7", "003-1", "003-3", "003-5", "003-7"],
}


def pick_validation_cells(test_cell: str, all_cells: list[str], k: int = 3) -> list[str]:
    """Deterministic, metadata-balanced validation cells.

    Rule (fixed in advance, never informed by any model result):

      * one validation cell per operating condition, so the validation signal
        cannot end up representing a single domain;
      * chosen by rotating inside each condition group with an offset derived
        from the test cell's position in the sorted cell list.

    The rule uses only (a) the cell ordering and (b) the published condition
    metadata — never the test cell's difficulty or any prediction.
    """
    groups = [g for g in SIT_GROUPS.values()]
    i = all_cells.index(test_cell)
    picked: list[str] = []
    for gi, group in enumerate(groups):
        cand = sorted(set(group) - {test_cell} - set(picked))
        if not cand:
            continue
        picked.append(cand[(i * k + gi) % len(cand)])
    # top up from any remaining cell if a group was too small
    if len(picked) < k:
        rest = [c for c in all_cells if c != test_cell and c not in picked]
        for j, c in enumerate(rest):
            if len(picked) >= k:
                break
            picked.append(rest[(i + j) % len(rest)])
    return picked[:k]


def macro_official_val(model_head_forward, y, cells, val_cells, device, chunk=32):
    """Per-cell MAE over SOH>=0.7, then macro-average across validation cells.

    Deliberately *not* a pooled sample-weighted MAE: pooling lets a long-cycling
    cell dominate, which is exactly the kind of single-cell influence this
    protocol is meant to remove.
    """
    import torch
    per_cell = {}
    for c in val_cells:
        idx = np.where(cells == c)[0]
        keep = idx[y[idx] >= 0.7]
        if len(keep) == 0:
            continue
        preds = []
        for s in range(0, len(keep), chunk):
            preds.append(model_head_forward(keep[s:s + chunk]))
        pred = np.concatenate(preds)
        per_cell[c] = float(np.abs(pred - y[keep]).mean() * 100)
    if not per_cell:
        return float("nan"), {}
    return float(np.mean(list(per_cell.values()))), per_cell


# ------------------------------------------------------------------ training
def train_arm(pipe, head, Zcached, Vraw, y, cells, tr, va, te, *, arm, device,
              seed, epochs, lr_head, lr_back, batch=8, patience=4, context_length=2048,
              output_patch_size=16, rank=8, alpha=16, pristine_model=None,
              lora_targets=("q", "v"), lora_mlp=False):
    """Train one arm.  ``arm='frozen'`` uses cached embeddings; others fine-tune.

    ``pristine_model`` 传入后，本次训练会在一份**初始权重的全新副本**上进行。
    这是必需的：``pipe`` 在整个运行里只加载一次，如果不重置，第 N 折会从第
    N−1 折适配后的权重开始（而第 N−1 折的训练集里含第 N 折的测试电芯），
    既破坏了各折独立性，也让 LoRA 挂在一个已经被改动过的底座上。
    """
    from diff_embed import pooled_embedding

    set_run_seed(seed)
    dev = torch.device(device)
    if pristine_model is not None and pipe is not None and arm != "frozen":
        import copy as _copy
        pipe.model = _copy.deepcopy(pristine_model)
    # The frozen arm never touches the backbone, so no model is loaded for it.
    model = pipe.model if pipe is not None else None

    # --- configure trainable set -------------------------------------------
    if model is not None:
        for p in model.parameters():
            p.requires_grad_(False)
    # `frozen` uses the cached embeddings (fast).  `frozen_diff` runs the same
    # frozen backbone through diff_embed instead — a control that isolates any
    # difference coming from the forward path itself rather than from training.
    if arm in ("frozen", "frozen_diff"):
        back_params = []
        blocks_to_restore = []
    elif arm == "lora":
        n_lora = attach_lora(model, rank, alpha, lora_targets, lora_mlp)
        back_params = [p for p in model.parameters() if p.requires_grad]
        # LoRA adapters are submodules inside every block, so the snapshot must
        # cover all blocks (their state_dict includes the adapter weights).
        blocks_to_restore = list(model.encoder.block)
        print(f"      已挂载 LoRA 到 {n_lora} 个层（r={rank}, alpha={alpha}, "
              f"targets={'/'.join(lora_targets)}{'+mlp' if lora_mlp else ''}）",
              flush=True)
    elif arm.startswith("unfreeze"):
        k = int(arm[-1])
        blocks = model.encoder.block[-k:]
        for blk in blocks:
            for p in blk.parameters():
                p.requires_grad_(True)
        back_params = [p for p in model.parameters() if p.requires_grad]
        blocks_to_restore = list(blocks)
    else:
        raise ValueError(f"unknown arm {arm!r}")
    n_back = sum(p.numel() for p in back_params)

    # --- standardise the frozen embedding (or the raw input) ---------------
    mu_z = Zcached[tr].mean(0)
    sd_z = Zcached[tr].std(0) + 1e-8

    Ztr_t = torch.tensor((Zcached[tr] - mu_z) / sd_z, dtype=torch.float32)
    ytr = torch.tensor(y[tr], dtype=torch.float32).to(dev)

    # ---- validation: 3 cells, per-cell macro MAE over SOH >= 0.7 ----------
    # Checkpoint selection uses this macro figure, so the criterion that picks
    # "which epoch is best" is the same quantity we ultimately report, and it is
    # no longer decided by a single idiosyncratic cell.
    va_idx = np.where(va)[0]
    y_va_np = y[va_idx]
    cells_va = cells[va_idx]
    Zva_t = torch.tensor((Zcached[va_idx] - mu_z) / sd_z, dtype=torch.float32)
    mu_z_t = torch.tensor(mu_z, dtype=torch.float32, device=dev)
    sd_z_t = torch.tensor(sd_z, dtype=torch.float32, device=dev)

    head = head.to(dev)
    params = list(head.parameters()) + back_params
    opt = torch.optim.AdamW([
        {"params": list(head.parameters()), "lr": lr_head},
        {"params": back_params, "lr": lr_back},
    ], weight_decay=1e-5)
    # 自证：不打印传进来的变量，而是把**优化器里每个参数组真正的 lr** 读出来。
    # 变量和优化器不一致（上次就是这种）在终端里立刻能看出来。
    _labels = ["head", "backbone"]
    print("      [优化器] " + " | ".join(
        f"{_labels[i] if i < len(_labels) else f'group{i}'}: "
        f"lr={g['lr']:g}, {sum(p.numel() for p in g['params'])} 参数"
        for i, g in enumerate(opt.param_groups)), flush=True)
    lossf = nn.MSELoss()

    raw_tr = torch.tensor(Vraw[tr][:, None, :], dtype=torch.float32)
    raw_va = torch.tensor(Vraw[va_idx][:, None, :], dtype=torch.float32)
    n = int(tr.sum())
    best, best_back, best_val, bad, ep = None, None, np.inf, 0, 0
    nan_batches, nan_epochs = 0, 0
    val_cells = sorted(set(cells_va.tolist()))

    def forward_batch(x_raw, z_frozen):
        """One forward pass; returns shape (b,)."""
        if arm == "frozen":
            return head(z_frozen.to(dev)).squeeze(-1)
        z = pooled_embedding(model, x_raw.to(dev), context_length=context_length,
                             batch_size=x_raw.shape[0], output_patch_size=output_patch_size,
                             device=dev)
        z = standardize_embedding(z, mu_z_t, sd_z_t)
        return head(z).squeeze(-1)

    def eval_val():
        """Returns (predictions on the validation cells, macro official MAE in pp)."""
        head.eval()
        preds = []
        with torch.no_grad():
            for s in range(0, len(y_va_np), 32):
                xs = raw_va[s:s + 32] if arm != "frozen" else None
                preds.append(forward_batch(xs, Zva_t[s:s + 32]))
        pred = torch.cat(preds).cpu().numpy()
        per_cell = {}
        for c in val_cells:
            m = (cells_va == c) & (y_va_np >= 0.7)
            if m.sum() == 0:
                continue
            per_cell[c] = float(np.abs(pred[m] - y_va_np[m]).mean() * 100)
        macro = float(np.mean(list(per_cell.values()))) if per_cell else float("nan")
        return pred, macro, per_cell

    t0 = time.time()
    for ep in range(epochs):
        head.train()
        perm = torch.randperm(n)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            opt.zero_grad()
            pred = forward_batch(raw_tr[idx], Ztr_t[idx])
            loss = lossf(pred, ytr[idx])
            # A single NaN/Inf loss would poison every subsequent weight and the
            # run would then die later on `load_state_dict(None)`.  Skip it.
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 5.0)
                opt.step()
            else:
                nan_batches += 1
        val_pred, v, val_per_cell = eval_val()
        if not np.isfinite(v):
            nan_epochs += 1
        if np.isfinite(v) and v < best_val - 1e-5:
            best_val, bad = v, 0
            best = {k: t.clone() for k, t in head.state_dict().items()}
            # Snapshot only the *trainable* backbone part.  Cloning the whole
            # 119.5 M-parameter model on every improvement wastes ~478 MB per
            # copy and fragments the GPU; the frozen blocks never change.
            if back_params:
                best_back = snapshot_trainable(model)
            else:
                best_back = None
        else:
            bad += 1
        diag = " ".join(f"{c}={m:.2f}" for c, m in sorted(val_per_cell.items()))
        flag = "  [非有限值，本轮跳过]" if not np.isfinite(v) else ""
        print(f"      epoch {ep+1}/{epochs}  val官方口径(macro) "
              f"{v if np.isfinite(v) else float('nan'):6.3f} pp  "
              f"[{diag}]  ({time.time()-t0:.0f}s){flag}", flush=True)
        if bad >= patience:
            break

    # Never crash here: if no epoch produced a usable validation score, fall
    # back to whatever the head currently holds.
    if best is None:
        print("      [警告] 没有任何一轮得到有效验证分数，使用最后一轮权重", flush=True)
        best = {k: t.clone() for k, t in head.state_dict().items()}
        best_back = snapshot_trainable(model) if back_params else None
    head.load_state_dict(best)
    if best_back is not None:
        restore_trainable(model, best_back)
    head.eval()
    if model is not None:
        model.eval()

    te_idx = np.where(te)[0]
    preds = []
    with torch.no_grad():
        for s in range(0, len(te_idx), 32):
            chunk = te_idx[s:s + 32]
            zs = torch.tensor((Zcached[chunk] - mu_z) / sd_z, dtype=torch.float32)
            if arm == "frozen":
                preds.append(head(zs.to(dev)).squeeze(-1).cpu().numpy())
            else:
                xr = torch.tensor(Vraw[chunk][:, None, :], dtype=torch.float32)
                z = pooled_embedding(model, xr.to(dev), context_length=context_length,
                                     batch_size=xr.shape[0],
                                     output_patch_size=output_patch_size, device=dev)
                z = standardize_embedding(z, mu_z_t, sd_z_t)
                preds.append(head(z).squeeze(-1).cpu().numpy())
    # diagnostics only — never used for checkpoint selection
    full_val = float(np.abs(val_pred - y_va_np).mean() * 100)
    band = (y_va_np >= 0.7) & (y_va_np < 0.8)
    band_val = float(np.abs(val_pred[band] - y_va_np[band]).mean() * 100) if band.any() else np.nan
    test_pred = np.concatenate(preds)
    status = "ok"
    if not np.all(np.isfinite(test_pred)):
        status = "nonfinite_pred"
    if nan_epochs:
        status = (status + f";nan_epochs={nan_epochs}").lstrip(";")
    if nan_batches:
        status = (status + f";nan_batches={nan_batches}").lstrip(";")
    return test_pred, {
        "epochs_run": ep + 1, "n_backbone_params": n_back,
        "val_official_macro_pp": best_val,          # ← 用于 checkpoint 选择
        "val_full_macro_pp": full_val,              # ↓ 仅诊断
        "val_band0708_macro_pp": band_val,
        "n_val_cells": len(val_cells),
        "val_cells": "|".join(val_cells),
        "status": status,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/sit_v2_discharge.npz"))
    ap.add_argument("--emb", type=Path, default=Path("runs/emb_V"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arms", nargs="*",
                    default=["frozen", "frozen_diff", "lora", "unfreeze1", "unfreeze2"])
    ap.add_argument("--folds", type=int, default=0, help="0 = 全部 20 折")
    ap.add_argument("--test-cells", nargs="*", default=None,
                    help="显式指定测试电芯；指定后覆盖 --folds，推荐筛查时使用")
    ap.add_argument("--val-cells", type=int, default=3,
                    help="每折的验证电芯数（按固定的工况平衡规则挑选）")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=20,
                    help="适配臂的轮次上限；实际由早停决定")
    ap.add_argument("--lr-back-list", nargs="*", type=float, default=None,
                    help="指定多个 backbone 学习率做扫描，例如 1e-5 3e-5 1e-4")
    ap.add_argument("--epochs-frozen", type=int, default=150,
                    help="仅头训练收敛慢，给冻结臂更多轮次（走缓存嵌入，很快）")
    ap.add_argument("--patience-frozen", type=int, default=25)
    ap.add_argument("--patience-adapt", type=int, default=3)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-back", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lora-preset", choices=list(LORA_TARGET_PRESETS), default="qv",
                    help="LoRA 目标模块：qv=原版 LoRA（48 个适配器/589,824 参数）；"
                         "qkvo=加 k 和 o（96 个/1,179,648）；"
                         "all=qkvo 再加 MLP 的 wi/wo（120 个/1,916,928）")
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=float, default=16)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--restart", action="store_true",
                    help="忽略已有 adapt_runs.csv，从头开始（默认是断点续跑）")
    ap.add_argument("--dry-run", action="store_true",
                    help="只校验参数和折划分，不加载模型或训练")
    args = ap.parse_args()
    from device_utils import pick_device, resolve_model_id
    args.device = pick_device(args.device)
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)

    # ---- fail fast on bad arguments -------------------------------------
    # Arm names used to be validated inside train_arm, i.e. once per fold, so a
    # typo (or a stray copy-paste artefact such as "<br/>") only surfaced after
    # a whole arm had finished.  Validate everything up front instead.
    VALID_ARMS = {"frozen", "frozen_diff", "lora", "unfreeze1", "unfreeze2", "unfreeze3"}
    bad = [a for a in args.arms if a not in VALID_ARMS]
    if bad:
        raise SystemExit(
            f"无效的臂名: {bad}\n"
            f"可选: {sorted(VALID_ARMS)}\n"
            f"提示：如果是从聊天记录里复制的命令，注意换行处的 <br/>、反引号等"
            f"会被 PowerShell 当成参数。请手动清理命令后重试。")
    suspicious = [a for a in args.arms if any(t in a for t in ("<", ">", "&", "`"))]
    if suspicious:
        raise SystemExit(f"臂名里含疑似 HTML/转义字符: {suspicious}")
    for lr in ([args.lr_back] if args.lr_back_list is None else args.lr_back_list):
        if not (0 < lr < 1):
            raise SystemExit(f"可疑的 backbone 学习率: {lr}")
    print(f"参数检查通过 | 臂 {args.arms} | 学习率 "
          f"{args.lr_back_list or [args.lr_back]} | {len(args.arms)} 个臂")
    if "lora" in args.arms:
        _t, _m = LORA_TARGET_PRESETS[args.lora_preset]
        print(f"LoRA 配置 | preset={args.lora_preset} "
              f"targets={'/'.join(_t)}{'+mlp' if _m else ''} | "
              f"r={args.lora_r} | alpha={args.lora_alpha:g} | "
              f"scaling={args.lora_alpha/args.lora_r:g}")

    d = np.load(args.data, allow_pickle=True)
    Vraw = d["V"].astype(np.float32)
    y = d["soh_rated"].astype(float)
    cells = d["cell_id"]
    e = np.load(args.emb / "embeddings.npz", allow_pickle=True)
    Zcached = e["Z"].astype(np.float32)

    seeds = [20260917 + 101 * i for i in range(args.seeds)]
    all_cells = sorted(set(cells.tolist()))
    if args.test_cells:
        if len(set(args.test_cells)) != len(args.test_cells):
            raise SystemExit(f"--test-cells 含重复值: {args.test_cells}")
        missing = [c for c in args.test_cells if c not in all_cells]
        if missing:
            raise SystemExit(f"--test-cells 中存在未知电芯: {missing}")
        test_cells = list(args.test_cells)
    elif args.folds:
        step = max(1, len(all_cells) // args.folds)
        test_cells = all_cells[::step][:args.folds]
    else:
        test_cells = all_cells
    print(f"测试电芯 {len(test_cells)} 只: {test_cells}")

    # ---- pre-flight: verify every fold's split before spending hours -------
    for tc in test_cells:
        vcs = pick_validation_cells(tc, all_cells, k=args.val_cells)
        if len(vcs) != args.val_cells:
            raise SystemExit(f"折 test={tc} 只挑到 {len(vcs)} 只验证电芯")
        if tc in vcs:
            raise SystemExit(f"折 test={tc} 的验证集里包含测试电芯")
        if len(set(vcs)) != len(vcs):
            raise SystemExit(f"折 test={tc} 的验证电芯有重复: {vcs}")
        n_tr = len([c for c in all_cells if c != tc and c not in vcs])
        if n_tr < 2:
            raise SystemExit(f"折 test={tc} 训练电芯只剩 {n_tr} 只")
    print(f"[预检] {len(test_cells)} 折的划分全部合法"
          f"（每折 {args.val_cells} 只验证电芯，与测试电芯不重叠）")
    if args.dry_run:
        print("[dry-run] 参数与折划分检查通过，未加载模型、未训练")
        return 0

    need_model = any(a != "frozen" for a in args.arms)
    pipe = None
    pristine_model = None
    if need_model:
        from chronos import Chronos2Pipeline
        import os
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        pipe = Chronos2Pipeline.from_pretrained(resolve_model_id(), device_map=args.device)
        from diff_embed import self_test
        print("可微路径自检:", "通过" if self_test(pipe.model, device=args.device) else "失败")
        # 保留一份未被改动的模型引用：每次训练前 deepcopy 一份出来用。
        # 不这么做的话，第 N 折的起点是第 N−1 折适配后的权重，而第 N−1 折的
        # 训练集里含第 N 折的测试电芯 —— 跨折状态累积，既是泄漏也会让 LoRA
        # 挂在一个动过的底座上。
        pristine_model = pipe.model
        print("[重置] 每次训练前恢复初始权重（各折互相独立）", flush=True)

    # ---- resume: skip runs already present in adapt_runs.csv ----------------
    runs_csv = out / "adapt_runs.csv"
    pending_csv = out / "adapt_runs.pending.csv"
    done = set()
    rows = []
    if not args.restart:
        previous_parts = []
        for previous_path in (runs_csv, pending_csv):
            if not previous_path.exists():
                continue
            try:
                part = pd.read_csv(previous_path)
                if len(part):
                    previous_parts.append(part)
            except Exception as exc:
                print(f"[警告] 读取 {previous_path.name} 失败({exc})，忽略该文件",
                      flush=True)
        if previous_parts:
            prev = pd.concat(previous_parts, ignore_index=True)
            prev = prev.drop_duplicates(
                subset=["arm", "lr_back", "test_cell", "seed"], keep="last")
            rows = prev.to_dict("records")
            for r in prev.itertuples():
                done.add((r.arm, round(float(r.lr_back), 12),
                          str(r.test_cell), int(r.seed)))
            print(f"[续跑] 已恢复 {len(done)} 条结果，将跳过它们", flush=True)

    def flush():
        """Atomically persist progress without truncating a locked CSV on Windows.

        Excel/WPS can hold ``adapt_runs.csv`` open.  Writing directly to that
        path first truncates it and can then fail on close.  Write a complete
        temporary file and atomically replace the destination instead.  If the
        destination remains locked, retain the complete snapshot as
        ``adapt_runs.pending.csv``; the resume loader above reads both files.
        """
        frame = pd.DataFrame(rows)
        tmp_csv = out / "adapt_runs.tmp.csv"
        frame.to_csv(tmp_csv, index=False)
        for attempt in range(5):
            try:
                os.replace(tmp_csv, runs_csv)
                if pending_csv.exists():
                    try:
                        pending_csv.unlink()
                    except OSError:
                        pass
                return True
            except PermissionError:
                if attempt < 4:
                    time.sleep(1.0)
        os.replace(tmp_csv, pending_csv)
        print("[Windows文件锁] adapt_runs.csv 正被 Excel/WPS/预览程序占用；"
              "本次完整进度已安全保存到 adapt_runs.pending.csv。"
              "关闭占用程序后重跑即可续跑。", flush=True)
        return False

    lrs = args.lr_back_list or [args.lr_back]
    for arm in args.arms:
        if arm == "frozen":
            arm_lrs = [args.lr_back]
        else:
            arm_lrs = lrs
        for lr_back in arm_lrs:
            for tc in test_cells:
                te = cells == tc
                # Fixed, metadata-balanced rule: 3 validation cells covering the
                # three operating conditions.  Chosen from cell ordering and the
                # published condition metadata only — never from test difficulty.
                vcs = pick_validation_cells(tc, all_cells, k=args.val_cells)
                tr = np.isin(cells, [c for c in all_cells if c != tc and c not in vcs])
                va = np.isin(cells, vcs)
                for seed in seeds:
                    key = (arm, round(float(lr_back), 12), tc, int(seed))
                    if key in done:
                        print(f"\n--- {arm} | lr={lr_back:g} | test={tc} | seed={seed} "
                              f"-> 已完成，跳过", flush=True)
                        continue
                    print(f"\n--- {arm} | lr_back={lr_back:g} | test={tc} | "
                          f"val={vcs} | seed={seed}", flush=True)
                    ep_this = args.epochs_frozen if arm == "frozen" else args.epochs
                    pat_this = args.patience_frozen if arm == "frozen" else args.patience_adapt
                    try:
                        set_run_seed(seed)
                        head = make_head(768)
                        pred, info = train_arm(
                            pipe, head, Zcached, Vraw, y, cells, tr, va, te,
                            arm=arm, device=args.device, seed=seed, epochs=ep_this,
                            patience=pat_this, lr_head=args.lr_head, lr_back=lr_back,
                            batch=args.batch, pristine_model=pristine_model,
                            rank=args.lora_r, alpha=args.lora_alpha,
                            lora_targets=LORA_TARGET_PRESETS[args.lora_preset][0],
                            lora_mlp=LORA_TARGET_PRESETS[args.lora_preset][1])
                    except KeyboardInterrupt:
                        flush()
                        print("\n[中断] 已保存当前进度，重跑本命令会自动续跑。", flush=True)
                        raise
                    except Exception as exc:
                        # One bad fold must not kill a 12-16 h run.
                        import traceback
                        traceback.print_exc()
                        print(f"    [错误] 该次运行失败，跳过并继续："
                              f"{type(exc).__name__}: {str(exc)[:160]}", flush=True)
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                        continue
                    m = full_metrics(y[te], pred)
                    percell = r2_per_cell(y[te], pred, cells[te])
                    te_idx = np.where(te)[0]
                    pred_dir = out / "predictions"
                    pred_dir.mkdir(exist_ok=True)
                    pred_name = (f"{arm}_lr{float(lr_back):.0e}_cell{tc}_seed{seed}.csv"
                                 .replace("+", ""))
                    pred_path = pred_dir / pred_name
                    pd.DataFrame({
                        "sample_index": te_idx,
                        "cell_id": cells[te_idx],
                        "cycle": d["cycle"][te_idx],
                        "rpt": d["rpt"][te_idx],
                        "soh_true": y[te_idx],
                        "soh_pred": pred,
                    }).to_csv(pred_path, index=False)
                    rows.append({"arm": arm, "lr_back": lr_back, "test_cell": tc,
                                 "seed": seed, "MAE_pp": m["MAE_pp"], "pm3pp": m["pm3pp"],
                                 "R2": m["R2"], **band_metrics(y[te], pred, cells[te]),
                                 "cell_R2": percell.get(tc, np.nan),
                                 "predictions_file": str(pred_path.relative_to(out)), **info})
                    print(f"    -> 测试官方口径 {rows[-1]['official(>=0.7)_MAE']:6.3f} pp | "
                          f"0.7-0.8 带 {rows[-1]['0.7-0.8_MAE']:6.3f} | "
                          f"单电芯 R² {rows[-1]['cell_R2']:6.3f} | "
                          f"跑了 {info['epochs_run']} 轮 | "
                          f"验证(macro) {info['val_official_macro_pp']:.3f} pp", flush=True)
                    # write after every single run, not every arm, so an
                    # interruption never costs more than the current fold
                    flush()
        # report per-arm aggregate right away
        sub = pd.DataFrame([r for r in rows if r["arm"] == arm])
        if len(sub):
            for lr, g in sub.groupby("lr_back"):
                print(f"\n  [{arm} lr={lr:g}] official {g['official(>=0.7)_MAE'].mean():.3f} | "
                      f"band0708 {g['0.7-0.8_MAE'].mean():.3f} | "
                      f"cell_R2 中位 {g.cell_R2.median():.3f} | "
                      f"平均轮数 {g.epochs_run.mean():.1f}", flush=True)

    df = pd.DataFrame(rows)
    flush()
    if df.empty:
        print("[警告] 没有任何结果（全部被跳过或全部失败）")
        return 1
    # The raw CSV is already on disk; never let a reporting error lose it.
    try:
        agg = (df.groupby(["arm", "lr_back"])
             .agg(official=("official(>=0.7)_MAE", "mean"),
                  band0708=("0.7-0.8_MAE", "mean"),
                  full=("full_MAE", "mean"),
                  cell_R2_median=("cell_R2", "median"),
                  cell_R2_neg=("cell_R2", lambda s: float((s < 0).mean())),
                  mean_epochs=("epochs_run", "mean"),
                  back_params=("n_backbone_params", "first"),
                  runs=("seed", "size")).round(4))
        print("\n" + "=" * 96)
        print("=== 汇总：冻结 vs 适配（同协议、同头、同损失、同种子）===")
        print("=" * 96)
        print(agg.to_string())
        agg.to_csv(out / "adapt_summary.csv")
    except Exception as exc:
        import traceback; traceback.print_exc()
        print(f"[警告] 汇总失败({type(exc).__name__})，但 adapt_runs.csv 已保存")
    (out / "config.json").write_text(json.dumps({
        "protocol_version": "clean_v2_seeded_standardized",
        "data": str(args.data), "emb": str(args.emb), "arms": args.arms,
        "lr_back_list": lrs, "test_cells": test_cells, "seeds": seeds,
        "epochs": args.epochs, "epochs_frozen": args.epochs_frozen,
        "patience_adapt": args.patience_adapt, "patience_frozen": args.patience_frozen,
        "lr_head": args.lr_head, "batch": args.batch,
        "head_seeded_before_init": True,
        "live_embedding_standardization": "training-fold cached mean/std",
        "note": "唯一变量是可训练参数集合与 backbone 学习率；"
                "frozen 走缓存嵌入，frozen_diff 走可微前向（控制组），"
                "所有适配臂靠早停决定轮数",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nwrote ->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
