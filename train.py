"""Train an EdgeCOD variant, then evaluate the last epoch once on the test sets.

  python train.py --config configs/edgecod_t.yaml --run edgecod_t_s2026
  python train.py --config configs/edgecod_s.yaml --run edgecod_s_s1 --set seed=1
  # design study on the VAL split (no test sets):
  python train.py --config configs/edgecod_t.yaml --run val_t --set data.holdout=0.1 "data.test=[]"
  # resume an interrupted run
  python train.py --config configs/edgecod_t.yaml --run edgecod_t_s2026 --resume

The paper's models are trained for 80 epochs and the checkpoint after the last epoch is evaluated
(no early stopping and no checkpoint selection on any test set).
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from edgecod.config import load_config
from edgecod.data import TrainSet, list_pairs, split_holdout, test_pairs
from edgecod.evaluate import evaluate_pairs, evaluate_sets
from edgecod.losses import total_loss
from edgecod.model import EdgeCOD, count_params


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', action='append', required=True, help='repeatable; merged in order')
    p.add_argument('--set', nargs='*', default=[], help='dotted overrides, e.g. train.epochs=100 seed=1')
    p.add_argument('--run', required=True, help='run name (folder under out_root)')
    p.add_argument('--resume', action='store_true', help='continue from <run>/last.pth')
    return p.parse_args()


class Logger:
    def __init__(self, path):
        self.f = open(path, 'a', buffering=1)

    def __call__(self, msg):
        line = f'{time.strftime("%Y-%m-%d %H:%M:%S")} {msg}'
        print(line, flush=True)
        self.f.write(line + '\n')


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def lr_lambda(tc, total_iters):
    """Linear warm-up, then cosine decay to min_lr_ratio x the base learning rate (or constant)."""
    warm, floor = tc.get('warmup_iters', 0), tc.get('min_lr_ratio', 0.0)

    def f(it):
        if it < warm:
            return (it + 1) / warm
        if tc['schedule'] == 'constant':
            return 1.0
        t = (it - warm) / max(1, total_iters - warm)
        if tc['schedule'] == 'cosine':
            return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * t))
        raise ValueError(tc['schedule'])
    return f


def build_optimizer(model, tc):
    backbone = [p for p in model.backbone.parameters() if p.requires_grad]
    ids = {id(p) for p in backbone}
    rest = [p for p in model.parameters() if p.requires_grad and id(p) not in ids]
    groups = [{'params': backbone, 'lr': tc['backbone_lr']}, {'params': rest, 'lr': tc['lr']}]
    if tc['optimizer'] == 'adamw':
        return torch.optim.AdamW(groups, weight_decay=tc['weight_decay'])
    if tc['optimizer'] == 'sgd':
        return torch.optim.SGD(groups, momentum=tc.get('momentum', 0.9), weight_decay=tc['weight_decay'])
    raise ValueError(tc['optimizer'])


def main():
    args = parse_args()
    cfg = load_config(args.config, args.set)
    run_dir = os.path.join(cfg['out_root'], args.run)
    last_path = os.path.join(run_dir, 'last.pth')
    if os.path.exists(last_path) and not args.resume:
        raise SystemExit(f'{run_dir} already has a checkpoint; use --resume or a new --run name')
    os.makedirs(run_dir, exist_ok=True)
    log = Logger(os.path.join(run_dir, 'train.log'))
    with open(os.path.join(run_dir, 'config.yaml'), 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    seed_all(cfg['seed'])
    device = torch.device('cuda')
    tc, dc = cfg['train'], cfg['data']

    pairs = list_pairs(os.path.join(dc['root'], dc['train']['image']), os.path.join(dc['root'], dc['train']['gt']))
    train_pairs, val_pairs = split_holdout(pairs, dc.get('holdout', 0.0), dc.get('holdout_seed', 0))
    if val_pairs:
        with open(os.path.join(run_dir, 'holdout.txt'), 'w') as f:
            f.writelines(os.path.basename(p[1]) + '\n' for p in val_pairs)
    log(f'run={args.run} train={len(train_pairs)} VAL={len(val_pairs)} overrides={args.set}')

    model = EdgeCOD.from_config(cfg['model'])
    n = count_params(model)
    log(f'model {cfg["model"]["backbone"]}: {n["params_M"]:.2f}M params '
        f'(encoder {n["backbone_M"]:.2f}M, neck + decoder {n["head_M"]:.2f}M)')
    model.to(device)

    loader = DataLoader(TrainSet(train_pairs, max(dc['scales']), tuple(dc['crop']), dc['hflip']),
                        batch_size=tc['batch_size'], shuffle=True, num_workers=dc['workers'],
                        pin_memory=True, drop_last=True, persistent_workers=True)
    total_iters = tc['epochs'] * len(loader)
    opt = build_optimizer(model, tc)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(tc, total_iters))
    amp = {'bf16': torch.bfloat16, 'fp16': torch.float16}.get(tc.get('amp'))

    start_epoch = 0
    if args.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location='cpu')
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        sched.load_state_dict(ck['sched'])
        start_epoch = ck['epoch']
        log(f'resumed from epoch {start_epoch}')

    val_every, val_pool = cfg['eval'].get('val_every', 0), None
    curve_path = os.path.join(run_dir, 'val_curve.json')
    curve = json.load(open(curve_path)) if args.resume and os.path.exists(curve_path) else []
    for epoch in range(start_epoch, tc['epochs']):
        model.train()
        t0, running = time.time(), 0.0
        for step, (img, mask) in enumerate(loader):
            img, mask = img.to(device, non_blocking=True), mask.to(device, non_blocking=True)
            size = random.choice(dc['scales'])
            if size != img.shape[-1]:
                img = F.interpolate(img, size=(size, size), mode='bilinear', align_corners=False)
                mask = F.interpolate(mask, size=(size, size), mode='bilinear', align_corners=False)
            with torch.autocast('cuda', dtype=amp, enabled=amp is not None):
                logits = model(img)
            loss = total_loss(logits, mask, cfg['loss'])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            running += loss.item()
            if (step + 1) % tc['log_every'] == 0:
                log(f'ep {epoch + 1}/{tc["epochs"]} it {step + 1}/{len(loader)} loss {running / (step + 1):.4f} '
                    f'lr {opt.param_groups[1]["lr"]:.2e} {(step + 1) / (time.time() - t0):.2f} it/s '
                    f'mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G')
        log(f'epoch {epoch + 1} done: loss {running / (step + 1):.4f} time {time.time() - t0:.0f}s')
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(),
                    'epoch': epoch + 1}, last_path)
        if val_pairs and val_every and ((epoch + 1) % val_every == 0 or epoch + 1 == tc['epochs']):
            if val_pool is None:
                val_pool = mp.get_context('spawn').Pool(4)
            r = evaluate_pairs(model, val_pairs, dc['test_size'], device, val_pool)
            curve.append({'epoch': epoch + 1, **r})
            with open(curve_path, 'w') as f:
                json.dump(curve, f, indent=2)
            log(f'[VAL ep {epoch + 1}] Sm={r["Sm"]:.4f} Em_max={r["Em_max"]:.4f} wFm={r["wFm"]:.4f} '
                f'MAE={r["MAE"]:.4f}')
    if val_pool is not None:
        val_pool.close()

    sets = {name: test_pairs(dc['root'], name) for name in dc['test']}
    if val_pairs:
        sets['VAL'] = val_pairs
    evaluate_sets(model, sets, dc['test_size'], device, run_dir, 'last', cfg['eval']['save_maps'], log)


if __name__ == '__main__':
    main()
