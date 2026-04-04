"""
Post-training dynamic int8 quantization (Linear layers) for MAPF-GPT checkpoints.

PyTorch dynamic quantization runs on CPU at inference time. Saved checkpoints set
`quantized: true`; mapf_gpt.inference loads them on CPU automatically.

Usage (from repository root):
  python extras/quantize_checkpoint.py -i weights/model-2M.pt -o weights/model-2M-int8.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

import torch
import torch.nn as nn

from mapf_gpt.model import GPT, GPTConfig

try:
    from torch.ao.quantization import quantize_dynamic
except ImportError:
    from torch.quantization import quantize_dynamic


def strip_prefix(state_dict: dict, prefix: str = "_orig_mod.") -> dict:
    out = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            out[k[len(prefix) :]] = v
        else:
            out[k] = v
    return out


def break_wte_lm_head_tie(model: GPT) -> None:
    w = model.transformer.wte.weight.data.clone()
    model.transformer.wte.weight = nn.Parameter(w.clone())
    model.lm_head.weight = nn.Parameter(w.clone())


def quantize_checkpoint(input_path: Path, output_path: Path) -> None:
    ckpt = torch.load(input_path, map_location="cpu")
    if "model_args" not in ckpt or "model" not in ckpt:
        raise ValueError("Checkpoint must contain 'model_args' and 'model'")

    model_args = dict(ckpt["model_args"])
    model = GPT(GPTConfig(**model_args))
    sd = strip_prefix(ckpt["model"])
    model.load_state_dict(sd, strict=False)
    model.eval()

    break_wte_lm_head_tie(model)
    model = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "model_args": model_args,
            "quantized": True,
            "quantization": "dynamic_int8",
            "source_checkpoint": str(input_path.resolve()),
        },
        output_path,
    )
    n_bytes = output_path.stat().st_size
    print(f"Saved {output_path} ({n_bytes / 1e6:.2f} MB on disk), dynamic int8 Linear")


def main():
    p = argparse.ArgumentParser(description="Dynamic int8 quantize MAPF-GPT checkpoint")
    p.add_argument("-i", "--input", type=Path, required=True)
    p.add_argument("-o", "--output", type=Path, required=True)
    args = p.parse_args()
    quantize_checkpoint(args.input, args.output)


if __name__ == "__main__":
    main()
