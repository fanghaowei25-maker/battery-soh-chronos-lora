#!/usr/bin/env python3
"""Pick the best available torch device, so the same scripts run on any machine.

Order of preference when ``pref="auto"``: cuda -> mps -> cpu.

Notes for the RTX 5060 (Blackwell, sm_120)
------------------------------------------
Blackwell needs PyTorch >= 2.7.0 built against CUDA 12.8.  PyTorch 2.6 and
earlier have no sm_120 kernels and fail with
"no kernel image is available for execution on the device".
Check with:

    python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_capability())"

Expected on a 5060: a torch version >= 2.7, CUDA 12.8, capability (12, 0).
"""
from __future__ import annotations


def pick_device(pref: str = "auto") -> str:
    import torch

    if pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def describe_device(pref: str = "auto") -> str:
    """Human-readable one-liner; also warns about the Blackwell/pytorch mismatch."""
    import torch

    dev = pick_device(pref)
    bits = [f"requested={pref}", f"using={dev}", f"torch={torch.__version__}"]
    if dev == "cuda":
        cap = torch.cuda.get_device_capability()
        bits += [f"gpu={torch.cuda.get_device_name(0)}",
                 f"capability=sm_{cap[0]}{cap[1]}",
                 f"cuda={torch.version.cuda}"]
        if cap[0] >= 12 and torch.version.cuda and torch.version.cuda < "12.8":
            bits.append("WARNING: Blackwell needs CUDA 12.8 wheels (torch>=2.7)")
    return " | ".join(bits)


def resolve_model_id(model_id: str = "amazon/chronos-2", local_dir: str = "chronos-2") -> str:
    """Prefer a bundled local copy of the model over downloading it.

    If ``./chronos-2`` exists next to the scripts it is used, which makes the
    package runnable without any network access to HuggingFace.  Otherwise the
    HuggingFace repo id is returned and the weights are downloaded on first use.
    """
    from pathlib import Path

    if model_id != "amazon/chronos-2":
        return model_id
    here = Path(__file__).resolve().parent
    for cand in (here / local_dir, Path.cwd() / local_dir):
        if (cand / "model.safetensors").exists() and (cand / "config.json").exists():
            print(f"[device_utils] 使用本地模型: {cand}", flush=True)
            return str(cand)
    print(f"[device_utils] 未找到本地 {local_dir}/，将从 HuggingFace 下载 "
          f"{model_id}（首次约 456 MB）", flush=True)
    return model_id
