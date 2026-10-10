#!/usr/bin/env python3
"""Pre-flight check for the backbone-adaptation package.

Includes the one check that matters most here: **does the differentiable
embedding path actually deliver gradients?**  Without it every "fine-tuned"
arm silently trains nothing, because ``Chronos2Pipeline.embed`` is wrapped in
``@torch.no_grad()``.
"""
from __future__ import annotations

import sys
from pathlib import Path

OK, BAD = "  [OK]  ", "  [FAIL]"
problems = []


def note(ok, good, bad=None):
    if ok:
        print(OK, good)
    else:
        msg = bad or good
        print(BAD, msg)
        problems.append(msg)


print("=" * 72)
print("Backbone 适配实验包 · 环境自检")
print("=" * 72)
print(f"\n1) Python {sys.version.split()[0]}   {sys.executable}")

print("\n2) 计算设备")
try:
    import torch
    print(f"       torch {torch.__version__} | cuda {torch.version.cuda} "
          f"| 可用={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"       GPU {torch.cuda.get_device_name(0)}")
        a = torch.randn(256, 256, device="cuda")
        (a @ a).sum().item()
        print("       GPU 矩阵运算实际执行成功")
    else:
        note(False, "CUDA 不可用——微调在 CPU 上会慢到不可行",
             "CUDA 不可用；本实验需要 GPU（CPU 上每折要跑几小时）")
except ImportError as exc:
    note(False, f"torch 未安装: {exc}")

print("\n3) 其它依赖")
for mod, why in [("chronos", "Chronos-2 推理库"), ("sklearn", "指标"),
                 ("pandas", "数据处理"), ("numpy", "数值计算")]:
    try:
        m = __import__(mod)
        print(f"       {mod:10} {getattr(m, '__version__', 'ok')}  ({why})")
    except ImportError:
        note(False, f"{mod} 未安装（{why}）")

here = Path(__file__).resolve().parent

print("\n4) 模型与数据")
mdl = here / "chronos-2" / "model.safetensors"
note(mdl.exists(), f"模型 {mdl.stat().st_size/1e6:.0f} MB" if mdl.exists() else "",
     f"模型缺失: {mdl}")
ds = here / "data" / "sit_v2_discharge.npz"
if ds.exists():
    import numpy as np
    d = np.load(ds, allow_pickle=True)
    note(d["V"].shape == (14265, 1024),
         f"数据集 {d['V'].shape}，{len(set(d['cell_id'].tolist()))} 只电芯",
         f"数据集形状异常: {d['V'].shape}")
else:
    note(False, f"数据集缺失: {ds}")
emb = here / "runs" / "emb_V" / "embeddings.npz"
note(emb.exists(), "缓存嵌入 emb_V 存在（冻结臂用）", f"缓存嵌入缺失: {emb}")

print("\n5) ★ 可微路径梯度验证（最关键的一项）")
try:
    import torch  # noqa: F811  第 2 项若已失败，这里会重新抛 ImportError
    have_torch = True
except ImportError:
    have_torch = False
if not have_torch:
    note(False, "", "torch 未安装，跳过梯度验证")
else:
    try:
        import os
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        from chronos import Chronos2Pipeline
        from diff_embed import self_test
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = Chronos2Pipeline.from_pretrained(str(here / "chronos-2"), device_map=dev)
        ok = self_test(pipe.model, device=dev)
        note(ok, "梯度可正常回传到解冻层，且未泄漏到冻结层",
             "梯度验证失败——微调实验会训练不到任何东西，先别往下跑")
    except Exception as exc:
        note(False, f"可微路径验证出错: {type(exc).__name__}: {str(exc)[:120]}")

# 6) 历史基线是纯参照物，缺了照跑。主实验的基线来自它自己跑出来的 frozen 臂，
#    所以这里绝不能计入 problems —— 否则自检会在完全健康的环境上退出码 1。
print("\n6) 基线结果（对照用，可选）")
base = here / "results" / "protocol_v1_frozen6" / "adapt_runs.csv"
if base.exists():
    print(OK, "附带的 6 折参照基线存在（protocol_v1_frozen6）")
else:
    print("  [提示] 未找到历史基线文件——不影响运行")
    print("         主实验一次跑 frozen + unfreeze1 两条臂，配对比较用的基线")
    print("         就是同一次运行里的 frozen 臂，无需外部基线目录。")

print("\n" + "=" * 72)
if problems:
    print(f"发现 {len(problems)} 个问题：")
    for p in problems:
        print("   -", p)
    sys.exit(1)
print("自检全部通过。接下来：")
print("  1) python run_main_experiment.py --smoke   # 2 折 × 1 种子，约 20 分钟")
print("  2) python run_main_experiment.py           # 全量 120 次训练，12–16 小时")
sys.exit(0)
