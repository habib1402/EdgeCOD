"""YAML configs with `_base_` inheritance and dotted command-line overrides (e.g. train.epochs=100)."""
import copy
import os

import yaml


def merge(dst, src):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            merge(dst[k], v)
        else:
            dst[k] = copy.deepcopy(v)
    return dst


def _load(path):
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop('_base_', None)
    if base:
        cfg = merge(_load(os.path.join(os.path.dirname(path), base)), cfg)
    return cfg


def load_config(paths, overrides=()):
    """Merge config files in order, then apply `a.b.c=value` overrides (values parsed as YAML)."""
    cfg = {}
    for p in [paths] if isinstance(paths, str) else paths:
        merge(cfg, _load(p))
    for item in overrides:
        key, val = item.split('=', 1)
        node = cfg
        *parents, leaf = key.split('.')
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(val)
    return cfg
