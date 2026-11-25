"""Paired bootstrap over test images (Section 4.1): mean difference B minus A and its 95% confidence interval.

  python tools/bootstrap.py --a per_image/WPFormer --b per_image/edgecod_b_s2026 per_image/edgecod_b_s1 per_image/edgecod_b_s2

Each argument is one or more folders of <set>.npz files written by tools/score_maps.py --per_image. Several folders
(e.g. the three seeds of one model) are averaged per image first, as in the paper. 10,000 resamples with seed 0,
per set and over the pooled images of all sets that both models cover. A difference is significant when its CI
excludes 0. For MAE, a negative difference means B is better.
"""
import argparse
import os

import numpy as np

KEYS = ('Sm', 'wFm', 'MAE')


def load(folders, s):
    files = [os.path.join(f, f'{s}.npz') for f in folders]
    if not all(os.path.exists(f) for f in files):
        return None
    ds = [np.load(f) for f in files]
    assert all((d['names'] == ds[0]['names']).all() for d in ds), f'{s}: image lists differ between folders'
    return ds[0]['names'], np.mean([np.stack([d[k] for k in KEYS], 1) for d in ds], 0)


def bootstrap(diff, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    means = np.stack([diff[rng.integers(0, len(diff), len(diff))].mean(0) for _ in range(n_boot)])
    return diff.mean(0), np.percentile(means, 2.5, 0), np.percentile(means, 97.5, 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--a', nargs='+', required=True, help='reference model (one folder per seed)')
    p.add_argument('--b', nargs='+', required=True, help='compared model (one folder per seed)')
    p.add_argument('--sets', nargs='+', default=['CAMO', 'COD10K', 'NC4K', 'CHAMELEON'])
    args = p.parse_args()
    pairs = {}
    for s in args.sets:
        a, b = load(args.a, s), load(args.b, s)
        if a is None or b is None:
            continue
        assert (a[0] == b[0]).all(), f'{s}: image lists differ between the two models'
        pairs[s] = (a[1], b[1])
    pooled = tuple(np.concatenate(x) for x in zip(*pairs.values()))
    print('B minus A, mean [95% CI]')
    for s, (a, b) in list(pairs.items()) + [('pooled ' + '+'.join(pairs), pooled)]:
        m, lo, hi = bootstrap(b - a)
        print(f'{s:32s} n={len(a):5d}  ' + '  '.join(f'{k} {m[i]:+.4f} [{lo[i]:+.4f}, {hi[i]:+.4f}]'
                                                    for i, k in enumerate(KEYS)))


if __name__ == '__main__':
    main()
