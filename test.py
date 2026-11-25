"""Evaluate a checkpoint on the COD test sets: saves the prediction maps and the scores.

  python test.py --config configs/edgecod_t.yaml --weights weights/edgecod_t_s2026.pth --out results/edgecod_t_s2026
  python test.py --config configs/edgecod_b.yaml --weights weights/edgecod_b_s2026.pth --sets CAMO COD10K --no_maps

Writes <out>/results_last.json (S-measure, E-measure, weighted F-measure, MAE, F-measure) and
<out>/maps/last/<set>/<name>.png (8-bit probability maps at the ground-truth size).
"""
import argparse

import torch

from edgecod.config import load_config
from edgecod.data import test_pairs
from edgecod.evaluate import evaluate_sets
from edgecod.model import EdgeCOD, load_weights


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--weights', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--data_root', help='overrides data.root of the config')
    p.add_argument('--sets', nargs='*', help='subset of the test sets (default: all in the config)')
    p.add_argument('--no_maps', action='store_true')
    args = p.parse_args()

    cfg = load_config(args.config)
    root = args.data_root or cfg['data']['root']
    model = load_weights(EdgeCOD.from_config(cfg['model'], pretrained=False), args.weights)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    sets = {name: test_pairs(root, name) for name in (args.sets or cfg['data']['test'])}
    evaluate_sets(model, sets, cfg['data']['test_size'], device, args.out, 'last', not args.no_maps)


if __name__ == '__main__':
    main()
