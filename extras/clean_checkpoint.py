# Convert a full training checkpoint (with optimizer state, config, etc.)
# to a clean inference-only checkpoint containing only model weights and args.
# This reduces file size and avoids optimizer-related warnings during inference.

import torch
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', required=True)
parser.add_argument('-o', '--output', default=None)
args = parser.parse_args()

ckpt = torch.load(args.input, map_location='cpu')

clean_ckpt = {
    'model': ckpt['model'],
    'model_args': ckpt['model_args']
}

output_path = args.output if args.output else args.input
torch.save(clean_ckpt, output_path)

print(f"Saved to {output_path}")
print(f"Original: {os.path.getsize(args.input) / 1024:.1f} KB")
print(f"New: {os.path.getsize(output_path) / 1024:.1f} KB")