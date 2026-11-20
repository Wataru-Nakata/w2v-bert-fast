"""The channels-last causal depthwise convolution (K=31) against cuDNN on [B, C, T]: forward and forward +
backward milliseconds at w2v-BERT 2.0's width (1024), bf16."""

import json

import torch

from w2vbert_fast.triton_conv import causal_depthwise_conv, causal_depthwise_conv_reference


def timed(fn, repeat=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeat


for batch, frames in ((16, 500), (16, 1500)):
    x = torch.randn(batch, frames, 1024, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(1024, 1, 31, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    g = torch.randn_like(x)
    for name, fn in (("triton", causal_depthwise_conv), ("cudnn", causal_depthwise_conv_reference)):
        with torch.no_grad():
            forward = timed(lambda: fn(x, w))
        train = timed(lambda: fn(x, w).backward(g))
        print(json.dumps({"batch": batch, "frames": frames, "impl": name, "forward_ms": round(forward, 3),
                          "train_ms": round(train, 3)}), flush=True)
