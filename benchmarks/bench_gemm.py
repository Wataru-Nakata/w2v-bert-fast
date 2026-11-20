"""The GEMM ceiling of w2v-BERT 2.0's shapes on this GPU: bare bf16 F.linear forward and forward + backward
(input and weight gradients) at 24000 rows (B=16 x 1500 frames), TFLOPS against the peak."""

import json

import torch

PEAK = 989.4e12


def timed(fn, repeat=30):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeat / 1e3


rows = 24000
for name, n_in, n_out in (("ffn_up", 1024, 4096), ("ffn_down", 4096, 1024), ("qkv", 1024, 3072),
                          ("out/pw2", 1024, 1024), ("pw1", 1024, 2048)):
    x = torch.randn(rows, n_in, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    linear = torch.nn.Linear(n_in, n_out, device="cuda", dtype=torch.bfloat16)
    g = torch.randn(rows, n_out, device="cuda", dtype=torch.bfloat16)
    flops = 2 * rows * n_in * n_out
    with torch.no_grad():
        forward = timed(lambda: linear(x))
    train = timed(lambda: linear(x).backward(g))
    print(json.dumps({"gemm": name, "forward_TFLOPS": round(flops / forward / 1e12), "forward_frac": round(
        flops / forward / PEAK, 3), "train_TFLOPS": round(3 * flops / train / 1e12),
        "train_frac": round(3 * flops / train / PEAK, 3)}), flush=True)
