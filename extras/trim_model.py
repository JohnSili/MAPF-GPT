"""
Trim MAPF-GPT depth: keep the first n_layer transformer blocks from a checkpoint.

Usage (from repository root):
  python extras/trim_model.py --input weights/model-2M.pt --output weights/model-2M-L3.pt --n_layer 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

import torch

from mapf_gpt.model import GPT, GPTConfig


def trim_checkpoint(
    input_path: Path,
    output_path: Path,
    new_n_layer: int,
) -> None:
    ckpt = torch.load(input_path, map_location="cpu")
    if "model_args" not in ckpt or "model" not in ckpt:
        raise ValueError("Checkpoint must contain 'model_args' and 'model'")

    old_args = dict(ckpt["model_args"])
    old_n = old_args["n_layer"]
    if new_n_layer < 1 or new_n_layer > old_n:
        raise ValueError(f"n_layer must be in [1, {old_n}], got {new_n_layer}")

    new_args = {**old_args, "n_layer": new_n_layer}
    gptconf = GPTConfig(**new_args)
    model = GPT(gptconf)

    old_sd = ckpt["model"]
    unwanted = "_orig_mod."
    old_sd = {
        (k[len(unwanted) :] if k.startswith(unwanted) else k): v
        for k, v in old_sd.items()
    }

    new_sd = model.state_dict()
    for k in new_sd:
        if k not in old_sd:
            raise KeyError(f"Old checkpoint has no key {k} (needed for trimmed model)")
        new_sd[k] = old_sd[k].clone()

    model.load_state_dict(new_sd)

    out = {
        "model": model.state_dict(),
        "model_args": new_args,
        "trimmed_from": str(input_path.resolve()),
        "original_n_layer": old_n,
    }
    if "iter_num" in ckpt:
        out["iter_num"] = ckpt["iter_num"]
    if "best_val_loss" in ckpt:
        out["best_val_loss"] = ckpt["best_val_loss"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, output_path)
    n_params = model.get_num_params() / 1e6
    print(f"Saved {output_path} ({n_params:.3f}M params, n_layer={new_n_layer})")


def main():
    p = argparse.ArgumentParser(description="Trim MAPF-GPT transformer depth")
    p.add_argument("--input", "-i", type=Path, required=True)
    p.add_argument("--output", "-o", type=Path, required=True)
    p.add_argument(
        "--n_layer",
        type=int,
        required=True,
        help="Number of transformer blocks to keep (from the start of the stack)",
    )
    args = p.parse_args()
    trim_checkpoint(args.input, args.output, args.n_layer)


if __name__ == "__main__":
    main()
