"""Causal depthwise 1-D convolution in channels-last layout, in Triton (w2v-BERT 2.0's conformer convolution).

``y[b, t, c] = sum_k w[c, k] * x[b, t - (K - 1) + k, c]`` (zero before the start), i.e. ``nn.Conv1d(C, C, K,
groups=C, bias=False)`` after a left padding of K - 1, with x and y as [B, T, C] (no transposes). The backward
gives the input gradient with the reversed kernel and the weight gradient from per-block partial sums.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor


def _configs():
    # the tap loop is not unrolled (an unrolled loop holds every tap's tile in registers: 255 per thread, one block
    # of eight warps per SM), so tiles can be large and many warps resident
    return [triton.Config({"BLOCK_T": t, "BLOCK_C": c}, num_warps=w, num_stages=s)
            for t, c, w, s in ((32, 128, 4, 3), (64, 128, 4, 3), (64, 128, 8, 3), (32, 256, 4, 3), (128, 64, 4, 3),
                               (64, 256, 8, 3), (16, 256, 4, 3), (64, 64, 4, 4))] + [
        triton.Config({"BLOCK_T": t, "BLOCK_C": c}, num_warps=4, num_stages=3, maxnreg=regs)
        for t, c in ((32, 128), (64, 128)) for regs in (64, 96)]


@triton.autotune(configs=_configs(), key=["C", "K"])
@triton.jit
def _conv_forward(X, W, Y, T, C, sx_b, sx_t, sy_b, sy_t, K: tl.constexpr, BLOCK_T: tl.constexpr,
                  BLOCK_C: tl.constexpr):
    pid_t = tl.program_id(0)
    pid_c = tl.program_id(1)
    b = tl.program_id(2)
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_ok = offs_c < C
    acc = tl.zeros([BLOCK_T, BLOCK_C], tl.float32)
    for k in range(K):
        ts = offs_t - (K - 1) + k
        x = tl.load(X + b * sx_b + ts[:, None] * sx_t + offs_c[None, :],
                    mask=((ts >= 0) & (ts < T))[:, None] & c_ok[None, :], other=0.0)
        w = tl.load(W + offs_c * K + k, mask=c_ok, other=0.0)
        acc += x.to(tl.float32) * w.to(tl.float32)[None, :]
    tl.store(Y + b * sy_b + offs_t[:, None] * sy_t + offs_c[None, :], acc.to(Y.dtype.element_ty),
             mask=(offs_t < T)[:, None] & c_ok[None, :])


@triton.autotune(configs=_configs(), key=["C", "K"])
@triton.jit
def _conv_backward_input(G, W, DX, T, C, sg_b, sg_t, sd_b, sd_t, K: tl.constexpr, BLOCK_T: tl.constexpr,
                         BLOCK_C: tl.constexpr):
    pid_t = tl.program_id(0)
    pid_c = tl.program_id(1)
    b = tl.program_id(2)
    offs_s = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_ok = offs_c < C
    acc = tl.zeros([BLOCK_T, BLOCK_C], tl.float32)
    for k in range(K):
        ts = offs_s + (K - 1) - k
        g = tl.load(G + b * sg_b + ts[:, None] * sg_t + offs_c[None, :], mask=(ts < T)[:, None] & c_ok[None, :],
                    other=0.0)
        w = tl.load(W + offs_c * K + k, mask=c_ok, other=0.0)
        acc += g.to(tl.float32) * w.to(tl.float32)[None, :]
    tl.store(DX + b * sd_b + offs_s[:, None] * sd_t + offs_c[None, :], acc.to(DX.dtype.element_ty),
             mask=(offs_s < T)[:, None] & c_ok[None, :])


def _weight_configs():
    return [triton.Config({"BLOCK_C": c}, num_warps=w) for c, w in ((32, 1), (64, 2), (128, 4), (256, 8), (64, 4))]


@triton.autotune(configs=_weight_configs(), key=["C", "K", "ROWS"])
@triton.jit
def _conv_backward_weight(X, G, DW, T, C, sx_b, sx_t, sg_b, sg_t, ROWS, K: tl.constexpr, K_PAD: tl.constexpr,
                          BLOCK_C: tl.constexpr):
    """DW[b * splits + split, k, c] = sum over the split's ROWS time steps t of G[b, t, c] * X[b, t - (K - 1) + k, c].
    One time step per iteration: the [K_PAD, BLOCK_C] window of x times the gradient row, accumulated in registers
    (no reductions); the partials are summed outside."""
    pid_c = tl.program_id(0)
    b = tl.program_id(1)
    split = tl.program_id(2)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_ok = offs_c < C
    taps = tl.arange(0, K_PAD)
    acc = tl.zeros([K_PAD, BLOCK_C], tl.float32)
    for t in range(split * ROWS, tl.minimum((split + 1) * ROWS, T)):
        g = tl.load(G + b * sg_b + t * sg_t + offs_c, mask=c_ok, other=0.0).to(tl.float32)
        ts = t - (K - 1) + taps
        x = tl.load(X + b * sx_b + ts[:, None] * sx_t + offs_c[None, :],
                    mask=((taps < K) & (ts >= 0))[:, None] & c_ok[None, :], other=0.0).to(tl.float32)
        acc += x * g[None, :]
    tl.store(DW + ((b * tl.num_programs(2) + split) * K + taps[:, None]) * C + offs_c[None, :], acc,
             mask=(taps < K)[:, None] & c_ok[None, :])


def _grid(frames, channels, batch):
    return lambda meta: (triton.cdiv(frames, meta["BLOCK_T"]), triton.cdiv(channels, meta["BLOCK_C"]), batch)


@torch.library.custom_op("w2vbert_fast::causal_depthwise_conv", mutates_args=())
def _conv(x: Tensor, weight: Tensor) -> Tensor:
    batch, frames, channels = x.shape
    kernel = weight.size(-1)
    w = weight.reshape(channels, kernel).contiguous()
    y = torch.empty(batch, frames, channels, device=x.device, dtype=x.dtype)
    _conv_forward[_grid(frames, channels, batch)](x, w, y, frames, channels, x.stride(0), x.stride(1), y.stride(0),
                                                   y.stride(1), K=kernel)
    return y


@_conv.register_fake
def _(x, weight):
    return x.new_empty(x.shape)


@torch.library.custom_op("w2vbert_fast::causal_depthwise_conv_backward", mutates_args=())
def _conv_backward(grad: Tensor, x: Tensor, weight: Tensor) -> tuple[Tensor, Tensor]:
    batch, frames, channels = x.shape
    kernel = weight.size(-1)
    w = weight.reshape(channels, kernel).contiguous()
    grad = grad if grad.stride(-1) == 1 else grad.contiguous()
    dx = torch.empty(batch, frames, channels, device=x.device, dtype=x.dtype)
    _conv_backward_input[_grid(frames, channels, batch)](grad, w, dx, frames, channels, grad.stride(0), grad.stride(1),
                                                          dx.stride(0), dx.stride(1), K=kernel)
    rows = 128
    splits = triton.cdiv(frames, rows)
    partial = torch.empty(batch * splits, kernel, channels, device=x.device, dtype=torch.float32)
    _conv_backward_weight[lambda meta: (triton.cdiv(channels, meta["BLOCK_C"]), batch, splits)](
        x, grad, partial, frames, channels, x.stride(0), x.stride(1), grad.stride(0), grad.stride(1), rows, K=kernel,
        K_PAD=triton.next_power_of_2(kernel))
    dw = torch.empty_like(weight).copy_(partial.sum(0).T.reshape(weight.shape))
    return dx, dw


@_conv_backward.register_fake
def _(grad, x, weight):
    return x.new_empty(x.shape), torch.empty_like(weight)


def _conv_setup(ctx, inputs, output):
    ctx.save_for_backward(*inputs)


def _conv_grad(ctx, grad):
    x, weight = ctx.saved_tensors
    return _conv_backward(grad, x, weight)


_conv.register_autograd(_conv_grad, setup_context=_conv_setup)


def causal_depthwise_conv(x: Tensor, weight: Tensor) -> Tensor:
    """x [B, T, C], weight [C, 1, K] (nn.Conv1d's) -> [B, T, C]; CUDA."""
    if not x.is_cuda:
        raise ValueError("causal_depthwise_conv needs CUDA tensors")
    return _conv(x if x.stride(-1) == 1 else x.contiguous(), weight)


def causal_depthwise_conv_reference(x: Tensor, weight: Tensor) -> Tensor:
    kernel = weight.size(-1)
    padded = torch.nn.functional.pad(x.transpose(1, 2), (kernel - 1, 0))
    return torch.nn.functional.conv1d(padded, weight, groups=x.size(-1)).transpose(1, 2)
