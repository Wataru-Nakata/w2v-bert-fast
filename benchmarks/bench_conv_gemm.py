"""Prototype: the causal depthwise convolution (K=31, channels-last) on tensor cores. Per channel the convolution of a
time tile is a Toeplitz matrix times the tile's input window; CB channels at a time make a batched (3-D) dot.
Compared with the package's direct kernels for speed and against cuDNN for values."""

import json

import torch
import triton
import triton.language as tl

from w2vbert_fast.triton_conv import causal_depthwise_conv, causal_depthwise_conv_reference


def _configs(kernel=31):
    # WP: the window, a power of two holding BT + K - 1 steps
    return [triton.Config({"CB": cb, "BT": bt, "NCOL": nc, "WP": triton.next_power_of_2(bt + kernel - 1)},
                          num_warps=w, num_stages=s)
            for cb, bt, nc, w, s in ((8, 32, 32, 4, 2), (8, 32, 16, 4, 2), (4, 32, 32, 4, 2), (8, 64, 16, 4, 2),
                                     (16, 32, 16, 4, 2), (8, 32, 32, 8, 2), (4, 64, 32, 4, 2))]


@triton.autotune(configs=_configs(), key=["C", "K"])
@triton.jit
def _conv_toeplitz(X, W, Y, T, C, sx_b, sx_t, sy_b, sy_t, K: tl.constexpr, REVERSE: tl.constexpr,
                   CB: tl.constexpr, BT: tl.constexpr, NCOL: tl.constexpr, WP: tl.constexpr):
    """Y[b, t, c] = sum_k W[c, k] X[b, t - (K-1) + k, c] (REVERSE: sum_k W[c, K-1-k] X[b, t + k, c], the input
    gradient). Program: CB channels, one sequence, NCOL time tiles of BT steps; window of WP >= BT + K - 1 steps."""
    pid_c = tl.program_id(0)
    b = tl.program_id(1)
    pid_n = tl.program_id(2)
    c = pid_c * CB + tl.arange(0, CB)
    c_ok = c < C
    i = tl.arange(0, BT)
    s = tl.arange(0, WP)
    n = pid_n * NCOL + tl.arange(0, NCOL)
    # Toeplitz [CB, BT, WP]: A[c, i, s] = w[c, s - i] (window s = i + k), reversed taps for the input gradient
    k = s[None, :] - i[:, None]
    if REVERSE:
        tap = K - 1 - k
        start = 0
    else:
        tap = k
        start = K - 1
    a = tl.load(W + c[:, None, None] * K + tap[None, :, :],
                mask=c_ok[:, None, None] & ((k >= 0) & (k < K))[None, :, :], other=0.0)
    # windows [CB, WP, NCOL]: X[b, t0 + s - (K - 1), c] (forward) or X[b, t0 + s, c] (input gradient)
    t = (n * BT)[None, :] + s[:, None] - start
    x = tl.load(X + b * sx_b + t[None, :, :] * sx_t + c[:, None, None],
                mask=c_ok[:, None, None] & ((t >= 0) & (t < T))[None, :, :], other=0.0)
    y = tl.dot(a.to(x.dtype), x)  # [CB, BT, NCOL]
    t_out = (n * BT)[None, :] + i[:, None]  # [BT, NCOL]
    tl.store(Y + b * sy_b + t_out[None, :, :] * sy_t + c[:, None, None], y.to(Y.dtype.element_ty),
             mask=c_ok[:, None, None] & (t_out < T)[None, :, :])


def conv_toeplitz(x, weight, reverse=False):
    batch, frames, channels = x.shape
    kernel = weight.size(-1)
    w = weight.reshape(channels, kernel).contiguous()
    y = torch.empty_like(x)

    def grid(meta):
        return (triton.cdiv(channels, meta["CB"]), batch, triton.cdiv(triton.cdiv(frames, meta["BT"]), meta["NCOL"]))

    _conv_toeplitz[grid](x, w, y, frames, channels, x.stride(0), x.stride(1), y.stride(0), y.stride(1), K=kernel,
                         REVERSE=reverse)
    return y


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
    return start.elapsed_time(end) / repeat


def main():
    for batch, frames in ((16, 500), (16, 1500)):
        x = torch.randn(batch, frames, 1024, device="cuda", dtype=torch.bfloat16)
        w = (torch.randn(1024, 1, 31, device="cuda") / 31 ** 0.5).to(torch.bfloat16)
        ref = causal_depthwise_conv_reference(x.float(), w.float())
        out = conv_toeplitz(x, w)
        error = float((out.float() - ref).norm() / ref.norm())
        # the input gradient: correlate with the reversed taps, looking ahead
        g = torch.randn_like(x)
        xr = x.float().requires_grad_()
        causal_depthwise_conv_reference(xr, w.float()).backward(g.float())
        dx = conv_toeplitz(g, w, reverse=True)
        error_dx = float((dx.float() - xr.grad).norm() / xr.grad.norm())
        with torch.no_grad():
            direct = timed(lambda: causal_depthwise_conv(x, w))
            toeplitz = timed(lambda: conv_toeplitz(x, w))
            toeplitz_dx = timed(lambda: conv_toeplitz(g, w, reverse=True))
        print(json.dumps({"batch": batch, "frames": frames, "direct_fwd_ms": round(direct, 4),
                          "toeplitz_fwd_ms": round(toeplitz, 4), "toeplitz_dx_ms": round(toeplitz_dx, 4),
                          "rel_error_fwd": f"{error:.2e}", "rel_error_dx": f"{error_dx:.2e}",
                          "config": str(_conv_toeplitz.best_config).split(", num_ctas")[0]}), flush=True)


if __name__ == "__main__":
    main()
