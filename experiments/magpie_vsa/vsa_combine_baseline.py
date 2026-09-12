#!/usr/bin/env python3
"""Blind baseline probe for FastVideo VSA coarse+sparse combine.

This file intentionally contains only the pre-optimization combine. It is used
as a ground-truth rediscovery target for AMD-AGI/Magpie. Do not add the known
optimized implementation here before the agent has written its hypothesis.

The H3-like case matches the shape used in the FastVideo VSA investigation:
B=1, H=56, S=15488, D=128, block_elements=64, bf16.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

MIB = 2**20

SHAPES = {
    "small": dict(batch=1, heads=8, seq=4096, dim=128, block_elements=64),
    "h3": dict(batch=1, heads=56, seq=15488, dim=128, block_elements=64),
}


def baseline_combine(
    out_c: torch.Tensor,
    out_s: torch.Tensor,
    weight: torch.Tensor | None,
    block_elements: int,
) -> torch.Tensor:
    """Pre-optimization FastVideo BHSD combine."""
    batch, heads, n_blocks, dim = out_c.shape
    seq = n_blocks * block_elements
    out_c = (
        out_c.unsqueeze(3)
        .repeat(1, 1, 1, block_elements, 1)
        .view(batch, heads, seq, dim)
    )
    if weight is not None:
        return out_c * weight + out_s
    return out_c + out_s


def _make_tensors(shape: dict[str, int], gated: bool):
    batch = shape["batch"]
    heads = shape["heads"]
    seq = shape["seq"]
    dim = shape["dim"]
    be = shape["block_elements"]
    if seq % be:
        raise ValueError(f"seq={seq} must be divisible by block_elements={be}")
    n_blocks = seq // be

    torch.manual_seed(0)
    out_c = torch.randn(
        batch, heads, n_blocks, dim, device="cuda", dtype=torch.bfloat16
    )
    out_s = torch.randn(
        batch, heads, seq, dim, device="cuda", dtype=torch.bfloat16
    )
    weight = (
        torch.rand(batch, heads, seq, dim, device="cuda", dtype=torch.bfloat16)
        if gated
        else None
    )
    return out_c, out_s, weight


def _reference(
    out_c: torch.Tensor,
    out_s: torch.Tensor,
    weight: torch.Tensor | None,
    block_elements: int,
) -> torch.Tensor:
    coarse = (
        out_c.float()
        .unsqueeze(3)
        .expand(-1, -1, -1, block_elements, -1)
        .reshape(out_s.shape)
    )
    if weight is None:
        ref = coarse + out_s.float()
    else:
        ref = coarse * weight.float() + out_s.float()
    return ref


def run(shape_name: str, gated: bool, warmup: int, iterations: int) -> dict:
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA/ROCm device is required.")

    shape = SHAPES[shape_name]
    out_c, out_s, weight = _make_tensors(shape, gated)
    be = shape["block_elements"]

    got = baseline_combine(out_c, out_s, weight, be)
    ref = _reference(out_c, out_s, weight, be)
    err = (got.float() - ref).abs()
    max_abs = float(err.max().item())
    mean_abs = float(err.mean().item())
    if not torch.isfinite(got).all():
        raise AssertionError("baseline output contains non-finite values")
    if max_abs > 0.05:
        raise AssertionError(f"baseline reference error too large: {max_abs}")

    # Correctness temporaries must not contaminate the measured baseline.
    del got, ref, err
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    for _ in range(warmup):
        baseline_combine(out_c, out_s, weight, be)
    torch.cuda.synchronize()

    samples_ms = []
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base_alloc = torch.cuda.memory_allocated()

    for _ in range(iterations):
        start = time.perf_counter()
        baseline_combine(out_c, out_s, weight, be)
        torch.cuda.synchronize()
        samples_ms.append((time.perf_counter() - start) * 1e3)

    peak_extra_mib = (
        torch.cuda.max_memory_allocated() - base_alloc
    ) / MIB

    full_tensor_mib = out_s.numel() * out_s.element_size() / MIB
    result = {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "shape": shape_name,
        "gated": gated,
        "warmup": warmup,
        "iterations": iterations,
        "latency_ms_median": statistics.median(samples_ms),
        "latency_ms_min": min(samples_ms),
        "peak_extra_mib": peak_extra_mib,
        "one_full_tensor_mib": full_tensor_mib,
        "correctness_max_abs": max_abs,
        "correctness_mean_abs": mean_abs,
    }
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--shape", choices=SHAPES, default="small")
    p.add_argument("--gated", action="store_true")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iterations", type=int, default=5)
    args = p.parse_args()
    run(args.shape, args.gated, args.warmup, args.iterations)


if __name__ == "__main__":
    main()
