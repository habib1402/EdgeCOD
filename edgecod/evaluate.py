"""Prediction and scoring with PySODMetrics (Section 4.1).

The probability map is predicted at test_size x test_size, resized to the ground-truth size, stored as an 8-bit PNG
and scored. Every reported metric is a mean over per-image values (the E/F curves are means of per-image curves), so
scoring images in parallel and averaging is exact.
"""
import json
import multiprocessing as mp
import os
import time

import cv2
import numpy as np
import py_sod_metrics as M
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import TestSet, test_collate

MAX_PENDING = 64


def score_one(pred, gt):
    """Per-image S-measure, weighted F-measure, MAE and the E/F-measure curves of one uint8 map."""
    sm, em, wfm, mae, fm = M.Smeasure(), M.Emeasure(), M.WeightedFmeasure(), M.MAE(), M.Fmeasure()
    for m in (sm, em, wfm, mae, fm):
        m.step(pred=pred, gt=gt)
    return {'Sm': sm.sms[0], 'MAE': mae.maes[0], 'wFm': wfm.weighted_fms[0],
            'Em_curve': np.asarray(em.changeable_ems[0], dtype=np.float64), 'Em_adp': em.adaptive_ems[0],
            'Fm_curve': np.asarray(fm.changeable_fms[0], dtype=np.float64), 'Fm_adp': fm.adaptive_fms[0]}


def aggregate(per_image):
    mean = lambda k: np.mean([r[k] for r in per_image], axis=0)  # noqa: E731
    em, fm = mean('Em_curve'), mean('Fm_curve')
    return {'n': len(per_image), 'Sm': float(mean('Sm')),
            'Em_max': float(em.max()), 'Em_mean': float(em.mean()), 'Em_adp': float(mean('Em_adp')),
            'wFm': float(mean('wFm')), 'MAE': float(mean('MAE')),
            'Fm_max': float(fm.max()), 'Fm_mean': float(fm.mean()), 'Fm_adp': float(mean('Fm_adp'))}


@torch.no_grad()
def predict(model, img, hw, device):
    """uint8 probability map of one normalized image tensor (1 x 3 x S x S), resized to hw = (H, W)."""
    logits = model(img.to(device))
    prob = torch.sigmoid(F.interpolate(logits.float(), size=tuple(hw), mode='bilinear', align_corners=False))
    return (prob[0, 0].cpu().numpy() * 255).round().astype(np.uint8)


@torch.no_grad()
def evaluate_pairs(model, pairs, size, device, pool, map_dir=None, workers=4):
    model.eval()
    if map_dir:
        os.makedirs(map_dir, exist_ok=True)
    loader = DataLoader(TestSet(pairs, size), batch_size=1, num_workers=workers, collate_fn=test_collate)
    pending, per_image = [], []
    for img, gt, name, _ in loader:
        pred = predict(model, img, gt.shape, device)
        if map_dir:
            cv2.imwrite(os.path.join(map_dir, name + '.png'), pred)
        pending.append(pool.apply_async(score_one, (pred, gt)))
        if len(pending) >= MAX_PENDING:
            per_image.append(pending.pop(0).get())
    per_image.extend(p.get() for p in pending)
    return aggregate(per_image)


def evaluate_sets(model, sets, size, device, out_dir, tag='last', save_maps=True, log=print, procs=6):
    """Score every {name: pairs} set; writes <out_dir>/results_<tag>.json and, optionally, maps/<tag>/<set>/*.png."""
    results = {}
    with mp.get_context('spawn').Pool(procs) as pool:
        for name, pairs in sets.items():
            t0 = time.time()
            map_dir = os.path.join(out_dir, 'maps', tag, name) if save_maps else None
            r = evaluate_pairs(model, pairs, size, device, pool, map_dir)
            results[name] = r
            log(f'[eval {tag}] {name:10s} n={r["n"]:5d} Sm={r["Sm"]:.4f} Em_max={r["Em_max"]:.4f} '
                f'wFm={r["wFm"]:.4f} MAE={r["MAE"]:.4f} ({time.time() - t0:.0f}s)')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f'results_{tag}.json'), 'w') as f:
        json.dump(results, f, indent=2)
    return results
