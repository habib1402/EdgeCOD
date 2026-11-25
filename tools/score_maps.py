"""Score a folder of prediction maps with the evaluation code of the paper (used for all re-scored methods).

  python tools/score_maps.py --pred results/edgecod_t_s2026/maps/last --gt_root data/TestDataset \
      --sets CAMO COD10K NC4K CHAMELEON --per_image per_image/edgecod_t_s2026

<pred>/<set>/<name>.png is matched to <gt_root>/<set>/GT/<name>.png; maps of another size are resized to the
ground-truth size (bilinear). --per_image saves per-image S-measure, weighted F-measure and MAE as <set>.npz, the
input of tools/bootstrap.py.
"""
import argparse
import os
import sys
from multiprocessing import Pool

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edgecod.evaluate import aggregate, score_one  # noqa: E402


def score_file(args):
    pred_path, gt_path = args
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
    if pred is None:
        raise FileNotFoundError(pred_path)
    if pred.shape != gt.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
    return score_one(pred, gt)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pred', required=True)
    p.add_argument('--gt_root', required=True)
    p.add_argument('--sets', nargs='+', default=['CAMO', 'COD10K', 'NC4K', 'CHAMELEON'])
    p.add_argument('--per_image')
    p.add_argument('--procs', type=int, default=16)
    args = p.parse_args()
    with Pool(args.procs) as pool:
        for s in args.sets:
            names = sorted(os.listdir(os.path.join(args.gt_root, s, 'GT')))
            per = pool.map(score_file, [(os.path.join(args.pred, s, n), os.path.join(args.gt_root, s, 'GT', n))
                                        for n in names], chunksize=16)
            r = aggregate(per)
            print(f'{s:10s} n={r["n"]:5d}  Sm {r["Sm"]:.4f}  Em_mean {r["Em_mean"]:.4f}  Em_max {r["Em_max"]:.4f}  '
                  f'wFm {r["wFm"]:.4f}  MAE {r["MAE"]:.4f}  Fm_mean {r["Fm_mean"]:.4f}')
            if args.per_image:
                os.makedirs(args.per_image, exist_ok=True)
                np.savez(os.path.join(args.per_image, f'{s}.npz'), names=np.array(names),
                         **{k: np.array([x[k] for x in per]) for k in ('Sm', 'wFm', 'MAE')})


if __name__ == '__main__':
    main()
