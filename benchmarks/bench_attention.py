"""Attention alone, one layer of w2v-BERT 2.0 shape (16 heads of 64, L=64, R=8): milliseconds per call for the
forward and for forward + backward, and peak memory, per backend. CUDA, bf16.

    python benchmarks/bench_attention.py --batch 16 --frames 250 500 1500
"""

import argparse
import json
import math

import torch

from w2vbert_fast.attention import relative_key_attention, relative_key_reference
from w2vbert_fast.model import _dense_sdpa
from w2vbert_fast.triton_attention import fused_relative_key_attention


def timed(fn, repeat=10):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeat, torch.cuda.max_memory_allocated() / 2**30


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--frames", type=int, nargs="+", default=[250, 500, 1500])
    args = parser.parse_args()
    heads, dim, left, right = 16, 64, 64, 8
    backends = {
        "triton": lambda q, k, v, e: fused_relative_key_attention(q, k, v, e, left, right),
        "flash_banded": lambda q, k, v, e: relative_key_attention(q, k, v, e, left, right),
        "sdpa_dense_bias": lambda q, k, v, e: _dense_sdpa(q, k, v, e, left, right, None),
        "hf_eager": lambda q, k, v, e: relative_key_reference(q, k, v, e, left, right),
        "flash_no_bias_floor": lambda q, k, v, e: torch.nn.functional.scaled_dot_product_attention(q, k, v),
    }
    rows = []
    for frames in args.frames:
        q, k, v = (torch.randn(args.batch, heads, frames, dim, device="cuda", dtype=torch.bfloat16,
                               requires_grad=True) for _ in range(3))
        e = torch.randn(left + right + 1, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        grad = torch.randn_like(q)
        for name, fn in backends.items():
            row = {"frames": frames, "backend": name}
            try:
                with torch.no_grad():
                    row["forward_ms"], row["forward_GiB"] = timed(lambda: fn(q, k, v, e))
                row["train_ms"], row["train_GiB"] = timed(lambda: fn(q, k, v, e).backward(grad))
            except torch.OutOfMemoryError:
                row["error"] = "OOM"
            torch.cuda.empty_cache()
            rows.append(row)
            print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
        if torch.cuda.is_available():
            # the model's layout: one packed [B, T, 3, H, D] projection in, [B, T, H, D] out, one packed gradient
            from w2vbert_fast import triton_attention as ta

            qkv = torch.randn(args.batch, frames, 3, heads, dim, device="cuda", dtype=torch.bfloat16,
                              requires_grad=True)
            g = torch.randn(args.batch, frames, heads, dim, device="cuda", dtype=torch.bfloat16)
            row = {"frames": frames, "backend": "triton_packed"}
            with torch.no_grad():
                row["forward_ms"], row["forward_GiB"] = timed(
                    lambda: ta.fused_relative_key_attention_packed(qkv, e, left, right))
            row["train_ms"], row["train_GiB"] = timed(
                lambda: ta.fused_relative_key_attention_packed(qkv, e, left, right).backward(g))
            row["configs"] = {name: str(getattr(ta, name).best_config) for name in
                              ("_forward", "_backward_kv", "_backward_q") if getattr(ta, name).best_config}
            print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)


if __name__ == "__main__":
    main()
