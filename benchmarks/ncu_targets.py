"""Launches the kernels worth reading in Nsight Compute, a few times each, at w2v-BERT 2.0's training shape
(B=16, T=1500): the depthwise convolution (forward, input and weight gradients) and the attention (forward and
both backward kernels).

    ncu --profile-from-start off --section SpeedOfLight --section Occupancy --section WarpStateStats \\
        python benchmarks/ncu_targets.py

Autotuning runs first, outside the profiled range (cudaProfilerStart/Stop around one more call of each).
"""

import torch

from w2vbert_fast.triton_attention import fused_relative_key_attention_packed
from w2vbert_fast.triton_conv import causal_depthwise_conv

batch, frames = 16, 1500
x = torch.randn(batch, frames, 1024, device="cuda", dtype=torch.bfloat16, requires_grad=True)
w = torch.randn(1024, 1, 31, device="cuda", dtype=torch.bfloat16, requires_grad=True)
qkv = torch.randn(batch, frames, 3, 16, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
e = torch.randn(73, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
grad = torch.ones(batch, frames, 16, 64, device="cuda", dtype=torch.bfloat16)
for _ in range(3):  # the first calls autotune
    causal_depthwise_conv(x, w).backward(torch.ones_like(x))
    fused_relative_key_attention_packed(qkv, e, 64, 8).backward(grad)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
causal_depthwise_conv(x, w).backward(torch.ones_like(x))
fused_relative_key_attention_packed(qkv, e, 64, 8).backward(grad)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
