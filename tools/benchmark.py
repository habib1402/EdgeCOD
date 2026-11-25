"""Parameters, GMACs and inference latency (Section 4.5, Table 8).

  python tools/benchmark.py --config configs/edgecod_t.yaml configs/edgecod_s.yaml configs/edgecod_b.yaml
  python tools/benchmark.py --config configs/edgecod_s.yaml --size 352 --cpu_threads 1 4 --no_gpu

Protocol of the paper:
  - GMACs: torch.utils.flop_counter (FLOPs / 2) for one 416 x 416 image.
  - GPU: batch 1, FP16 and FP32; 50 warm-up + 500 timed passes, each timed with CUDA events; median of 3 repeats.
    "graph" replays the FP16 pass as a CUDA graph (removes Python and kernel-launch overhead).
    Throughput at batch 16 (FP16) and peak allocated memory of one FP16 batch-1 pass (weights included).
  - CPU: FP32, torch.inference_mode, 5 warm-up + 20 timed passes, median, for each thread count.
Timing does not depend on the weights, so the models are built with random weights.
"""
import argparse
import os
import statistics
import sys
import time

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edgecod.config import load_config  # noqa: E402
from edgecod.model import EdgeCOD, count_params  # noqa: E402


@torch.no_grad()
def gmacs(model, size):
    x = torch.randn(1, 3, size, size)
    with FlopCounterMode(display=False) as fc:
        model.eval()(x)
    return fc.get_total_flops() / 2e9


def cache_pos_encodings(model):
    """EdgeNeXt's positional encoding builds a mask on the CPU at every call, which CUDA-graph capture forbids.
    Its output depends only on the input shape and dtype, so it is cached per (shape, dtype); outputs are unchanged."""
    for mod in model.modules():
        if type(mod).__name__ == 'PositionalEncodingFourier':
            fwd, cache = mod.forward, {}

            def cached(shape, fwd=fwd, cache=cache, mod=mod):
                key = (tuple(shape), next(mod.parameters()).dtype)
                if key not in cache:
                    cache[key] = fwd(shape)
                return cache[key]
            mod.forward = cached


@torch.no_grad()
def gpu_latency(model, x, warmup=50, iters=500, graph=False):
    for _ in range(warmup):
        model(x)
    torch.cuda.synchronize()
    run = lambda: model(x)  # noqa: E731
    if graph:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            model(x)
        run = g.replay
        for _ in range(warmup):
            run()
        torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        run()
        e.record()
        e.synchronize()
        times.append(s.elapsed_time(e))
    return float(np.mean(times))


@torch.no_grad()
def throughput(model, x, warmup=20, iters=100):
    for _ in range(warmup):
        model(x)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        model(x)
    e.record()
    e.synchronize()
    return x.shape[0] * iters / (s.elapsed_time(e) / 1000)


@torch.no_grad()
def peak_mem(model, x):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model(x)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 2**20


def cpu_latency(model, size, threads, warmup=5, iters=20):
    torch.set_num_threads(threads)
    x = torch.randn(1, 3, size, size)
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', nargs='+', required=True)
    p.add_argument('--size', type=int, default=416)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--cpu_threads', type=int, nargs='*', default=[1, 4])
    p.add_argument('--no_gpu', action='store_true')
    args = p.parse_args()
    gpu = torch.cuda.is_available() and not args.no_gpu
    if gpu:
        torch.backends.cudnn.benchmark = True
        print(f'GPU: {torch.cuda.get_device_name()}')
    for path in args.config:
        cfg = load_config(path)
        model = EdgeCOD.from_config(cfg['model'], pretrained=False).eval()
        cache_pos_encodings(model)
        line = (f'{os.path.basename(path):16s} @{args.size}  params {count_params(model)["params_M"]:.2f}M  '
                f'GMACs {gmacs(model, args.size):.2f}')
        for t in args.cpu_threads:
            line += f'  CPU {t}t {cpu_latency(model, args.size, t):.1f} ms'
        if gpu:
            m = model.cuda()
            x32 = torch.randn(1, 3, args.size, args.size, device='cuda')
            fp32 = statistics.median(gpu_latency(m, x32) for _ in range(args.repeats))
            m = m.half()
            x16 = x32.half()
            fp16 = statistics.median(gpu_latency(m, x16) for _ in range(args.repeats))
            graph = statistics.median(gpu_latency(m, x16, graph=True) for _ in range(args.repeats))
            xb = torch.randn(args.batch, 3, args.size, args.size, device='cuda', dtype=torch.half)
            tput = statistics.median(throughput(m, xb) for _ in range(args.repeats))
            line += (f'  | FP16 {fp16:.2f} ms ({1000 / fp16:.0f} FPS)  graph {graph:.2f} ms  FP32 {fp32:.2f} ms  '
                     f'bs{args.batch} {tput:.0f} img/s  mem {peak_mem(m, x16):.0f} MB')
        print(line, flush=True)
        del model
        if gpu:
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
