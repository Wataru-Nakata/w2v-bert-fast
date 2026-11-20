"""Fused relative-key attention for w2v-BERT 2.0 in Triton (FlashAttention-2 forward and backward).

The relative-key score of query i and key j is ``scale * q_i . k_j + B[i, clip(j - i, -L, R) + L]`` with the bucket
scores ``B = scale * q E^T`` ([B, H, T, L+R+1], computed with one small matmul; kept in log2 units, as the kernels
use exp2). Inside the kernels a key block that
lies entirely left of the band adds the row constant ``B[:, 0]``, one entirely right of it ``B[:, L+R]``, and only
the few blocks crossing the band ``-L < j - i < R`` read the bucket scores per element. Keys at or beyond a
sequence's length are masked (every query row is computed, as the reference does with a key-padding mask).

Backward: dK and dV in one kernel (key blocks outer), dQ and the bucket gradient dB in another (query blocks
outer). Each band bucket (i, d) receives exactly one score gradient, so dB is written without atomics; the
clipped buckets 0 and L+R get row sums. dq += scale * dB E and dE = scale * sum dB^T q follow outside the kernels.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl
from torch import Tensor


def _configs():
    # register-capped variants (maxnreg) trade spills for more resident warps; the tuner keeps whichever is faster
    tiles = ((64, 64, 32, 3, 4), (128, 64, 32, 3, 4), (128, 64, 32, 3, 8), (64, 128, 32, 3, 4), (128, 128, 32, 3, 8),
             (128, 64, 16, 3, 8), (64, 64, 16, 3, 4))
    configs = [triton.Config({"BLOCK_M": m, "BLOCK_N": n, "BAND": band}, num_stages=s, num_warps=w)
               for m, n, band, s, w in tiles]
    configs += [triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BAND": band}, num_stages=3, num_warps=4, maxnreg=regs)
                for band in (16, 32) for regs in (128, 168)]
    return configs


def _forward_configs():
    return [triton.Config({"BLOCK_M": m, "BLOCK_N": n, "BAND_N": band}, num_stages=s, num_warps=w)
            for m, n, band, s, w in ((128, 128, 32, 3, 4), (128, 128, 16, 3, 4), (128, 64, 32, 3, 4),
                                     (128, 128, 32, 3, 8), (128, 128, 32, 4, 4), (128, 64, 16, 3, 4),
                                     (64, 64, 32, 3, 4), (128, 128, 64, 3, 8))] + [
        triton.Config({"BLOCK_M": m, "BLOCK_N": n, "BAND_N": 32}, num_stages=3, num_warps=4, maxnreg=regs)
        for m, n in ((64, 64), (128, 64)) for regs in (128, 168)]


LOG2E = 1.4426950408889634


@triton.jit
def _dot(a, b, IEEE: tl.constexpr):
    if IEEE:
        return tl.dot(a, b, input_precision="ieee")
    else:
        return tl.dot(a, b)


@triton.jit
def _dot_acc(a, b, acc, IEEE: tl.constexpr):
    """acc + a @ b, accumulated by the matrix unit in place."""
    if IEEE:
        return tl.dot(a, b, acc, input_precision="ieee")
    else:
        return tl.dot(a, b, acc)


@triton.jit
def _forward_block(acc, l_i, m_i, q, k_base, v_base, sk_t, sv_t, bucket_row, offs_m, offs_n, offs_d,
                   start_n, row_ok, c_left, c_right, kv_len, qk_scale, REGION: tl.constexpr, MASK: tl.constexpr,
                   LEFT: tl.constexpr, RIGHT: tl.constexpr, IEEE: tl.constexpr):
    """One key block of the online softmax; REGION 0: far left (bias c_left), 1: far right (c_right), 2: any block
    (bias gathered per element)."""
    cols = start_n + offs_n
    if MASK:
        col_ok = cols < kv_len
        k = tl.load(k_base + cols[:, None] * sk_t + offs_d[None, :], mask=col_ok[:, None], other=0.0)
        v = tl.load(v_base + cols[:, None] * sv_t + offs_d[None, :], mask=col_ok[:, None], other=0.0)
    else:
        k = tl.load(k_base + cols[:, None] * sk_t + offs_d[None, :])
        v = tl.load(v_base + cols[:, None] * sv_t + offs_d[None, :])
    s = _dot(q, tl.trans(k), IEEE)
    if REGION == 2:
        dist = cols[None, :] - offs_m[:, None]
        index = tl.minimum(tl.maximum(dist, -LEFT), RIGHT) + LEFT
        s = s * qk_scale + tl.load(bucket_row[:, None] + index, mask=row_ok[:, None], other=0.0)
        if MASK:
            s = tl.where(col_ok[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        p = tl.math.exp2(s - m_new[:, None])
    else:
        # far regions (full blocks only): the bias is the row constant c, so the row maximum is max(s) * scale + c
        # and each probability one fused multiply-add and exp2
        if REGION == 0:
            c = c_left
        else:
            c = c_right
        m_new = tl.maximum(m_i, tl.max(s, 1) * qk_scale + c)
        p = tl.math.exp2(s * qk_scale + (c - m_new)[:, None])
    alpha = tl.math.exp2(m_i - m_new)
    l_i = l_i * alpha + tl.sum(p, 1)
    acc = _dot_acc(p.to(v.dtype), v, acc * alpha[:, None], IEEE)
    return acc, l_i, m_new


@triton.jit
def _far_block(acc, l_i, m_i, q, k_base, v_base, sk_t, sv_t, offs_n, offs_d, start_n, c, qk_scale,
               IEEE: tl.constexpr):
    """One full key block whose bias is the row constant c (far left or far right): one fused multiply-add per
    score."""
    cols = start_n + offs_n
    k = tl.load(k_base + cols[:, None] * sk_t + offs_d[None, :])
    v = tl.load(v_base + cols[:, None] * sv_t + offs_d[None, :])
    s = _dot(q, tl.trans(k), IEEE)
    m_new = tl.maximum(m_i, tl.max(s, 1) * qk_scale + c)
    p = tl.math.exp2(s * qk_scale + (c - m_new)[:, None])
    alpha = tl.math.exp2(m_i - m_new)
    l_i = l_i * alpha + tl.sum(p, 1)
    acc = _dot_acc(p.to(v.dtype), v, acc * alpha[:, None], IEEE)
    return acc, l_i, m_new


@triton.jit
def _regions(start_m, full, LEFT: tl.constexpr, RIGHT: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """Key-block starts (multiples of BLOCK_N, at most full) splitting [0, full) into far left [0, a), the blocks
    that cross the band [a, b) and far right [b, full), for the query rows start_m .. start_m + BLOCK_M - 1."""
    # a block [n, n + BLOCK_N) is far left when n + BLOCK_N - 1 - start_m <= -LEFT
    last_left = start_m - LEFT - BLOCK_N + 1
    a = tl.where(last_left >= 0, (last_left // BLOCK_N + 1) * BLOCK_N, 0)
    # far right when n - (start_m + BLOCK_M - 1) >= RIGHT
    first_right = start_m + BLOCK_M - 1 + RIGHT
    b = ((first_right + BLOCK_N - 1) // BLOCK_N) * BLOCK_N
    a = tl.minimum(a, full)
    b = tl.minimum(tl.maximum(b, a), full)
    return a, b


@triton.autotune(configs=_forward_configs(), key=["TB", "HEAD_DIM", "IEEE"])
@triton.jit
def _forward(Q, K, V, Bk, Lens, O, LSE,
             sq_b, sq_h, sq_t, sk_b, sk_h, sk_t, sv_b, sv_h, sv_t, sb_b, sb_h, sb_t, so_b, so_h, so_t,
             H, T, TB, scale,
             LEFT: tl.constexpr, RIGHT: tl.constexpr, HEAD_DIM: tl.constexpr, IEEE: tl.constexpr,
             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BAND_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    row_ok = offs_m < T
    kv_len = tl.load(Lens + b)
    qk_scale = scale * 1.4426950408889634
    q = tl.load(Q + b * sq_b + h * sq_h + offs_m[:, None] * sq_t + offs_d[None, :], mask=row_ok[:, None], other=0.0)
    bucket_row = Bk + b * sb_b + h * sb_h + offs_m * sb_t
    c_left = tl.load(bucket_row, mask=row_ok, other=0.0)
    c_right = tl.load(bucket_row + LEFT + RIGHT, mask=row_ok, other=0.0)
    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    k_base = K + b * sk_b + h * sk_h
    v_base = V + b * sv_b + h * sv_h
    full = (kv_len // BLOCK_N) * BLOCK_N
    a, b2 = _regions(start_m, full, LEFT, RIGHT, BLOCK_M, BLOCK_N)
    # far blocks, left [0, a) then right [b2, full), in one loop of full tiles (no gathers): the pipelined bulk
    for i in range(0, a // BLOCK_N + (full - b2) // BLOCK_N):
        start_n = i * BLOCK_N
        right = start_n >= a
        start_n = tl.where(right, start_n + (b2 - a), start_n)
        c = tl.where(right, c_right, c_left)
        acc, l_i, m_i = _far_block(acc, l_i, m_i, q, k_base, v_base, sk_t, sv_t, offs_n, offs_d, start_n, c,
                                   qk_scale, IEEE)
    # the band [a, b2) and the partial tail [full, kv_len) in narrow BAND_N-key slices (bias gathered per score)
    offs_band = tl.arange(0, BAND_N)
    for start_n in range(a, b2, BAND_N):
        acc, l_i, m_i = _forward_block(acc, l_i, m_i, q, k_base, v_base, sk_t, sv_t, bucket_row, offs_m, offs_band,
                                       offs_d, start_n, row_ok, c_left, c_right, kv_len, qk_scale, 2, False,
                                       LEFT, RIGHT, IEEE)
    for start_n in range(full, kv_len, BAND_N):
        acc, l_i, m_i = _forward_block(acc, l_i, m_i, q, k_base, v_base, sk_t, sv_t, bucket_row, offs_m, offs_band,
                                       offs_d, start_n, row_ok, c_left, c_right, kv_len, qk_scale, 2, True,
                                       LEFT, RIGHT, IEEE)
    acc = acc / l_i[:, None]
    tl.store(O + b * so_b + h * so_h + offs_m[:, None] * so_t + offs_d[None, :], acc.to(O.dtype.element_ty),
             mask=row_ok[:, None])
    tl.store(LSE + pid_bh * T + offs_m, (m_i + tl.math.log2(l_i)) * 0.6931471805599453, mask=row_ok)


@triton.jit
def _probs(q, k, cols, offs_m, row_ok, bucket_row, c_left, c_right, lse, qk_scale, REGION: tl.constexpr,
           LEFT: tl.constexpr, RIGHT: tl.constexpr, IEEE: tl.constexpr):
    """P = exp2(log2-unit score - log-sum-exp); in the far regions one fused multiply-add per element."""
    s = _dot(q, tl.trans(k), IEEE)
    if REGION == 0:
        p = tl.math.exp2(s * qk_scale + (c_left - lse)[:, None])
    elif REGION == 1:
        p = tl.math.exp2(s * qk_scale + (c_right - lse)[:, None])
    else:
        dist = cols[None, :] - offs_m[:, None]
        index = tl.minimum(tl.maximum(dist, -LEFT), RIGHT) + LEFT
        bias = tl.load(bucket_row[:, None] + index, mask=row_ok[:, None], other=0.0)
        p = tl.math.exp2(s * qk_scale + bias - lse[:, None])
    return p


@triton.jit
def _kv_block(dk, dv, k, v, cols, col_ok, Q, DO, LSE, Delta, Bk, q_base, do_base, sq_t, sdo_t, sb_t, pid_bh, T,
              start_m, offs_d, qk_scale, REGION: tl.constexpr, MASK: tl.constexpr, LEFT: tl.constexpr,
              RIGHT: tl.constexpr, BLOCK_M: tl.constexpr, IEEE: tl.constexpr):
    offs_m = start_m + tl.arange(0, BLOCK_M)
    row_ok = offs_m < T
    if MASK:
        q = tl.load(q_base + offs_m[:, None] * sq_t + offs_d[None, :], mask=row_ok[:, None], other=0.0)
        do = tl.load(do_base + offs_m[:, None] * sdo_t + offs_d[None, :], mask=row_ok[:, None], other=0.0)
        lse = tl.load(LSE + pid_bh * T + offs_m, mask=row_ok, other=0.0) * 1.4426950408889634
        delta = tl.load(Delta + pid_bh * T + offs_m, mask=row_ok, other=0.0)
    else:
        q = tl.load(q_base + offs_m[:, None] * sq_t + offs_d[None, :])
        do = tl.load(do_base + offs_m[:, None] * sdo_t + offs_d[None, :])
        lse = tl.load(LSE + pid_bh * T + offs_m) * 1.4426950408889634
        delta = tl.load(Delta + pid_bh * T + offs_m)
    bucket_row = Bk + offs_m * sb_t
    c_left = tl.load(bucket_row, mask=row_ok, other=0.0)
    c_right = tl.load(bucket_row + LEFT + RIGHT, mask=row_ok, other=0.0)
    p = _probs(q, k, cols, offs_m, row_ok, bucket_row, c_left, c_right, lse, qk_scale, REGION, LEFT, RIGHT, IEEE)
    # rows past T (dO, Delta and q loaded as zero) contribute nothing to dv or dk; keys past the length must not
    p = tl.where(col_ok[None, :], p, 0.0)
    dv = _dot_acc(tl.trans(p.to(do.dtype)), do, dv, IEEE)
    dp = _dot(do, tl.trans(v), IEEE)
    ds = p * (dp - delta[:, None])
    dk = _dot_acc(tl.trans(ds.to(q.dtype)), q, dk, IEEE)
    return dk, dv


@triton.jit
def _kv_far(dk, dv, k, v, col_ok, LSE, Delta, q_base, do_base, bk_base, sq_t, sdo_t, sb_t, pid_bh, T, start_m, offs_d,
            c_off, qk_scale, BLOCK_M: tl.constexpr, IEEE: tl.constexpr):
    """A full query block for which this key block lies wholly in one clipped region (bias column c_off)."""
    offs_m = start_m + tl.arange(0, BLOCK_M)
    q = tl.load(q_base + offs_m[:, None] * sq_t + offs_d[None, :])
    do = tl.load(do_base + offs_m[:, None] * sdo_t + offs_d[None, :])
    lse = tl.load(LSE + pid_bh * T + offs_m) * 1.4426950408889634
    delta = tl.load(Delta + pid_bh * T + offs_m)
    c = tl.load(bk_base + offs_m * sb_t + c_off)
    s = _dot(q, tl.trans(k), IEEE)
    p = tl.math.exp2(s * qk_scale + (c - lse)[:, None])
    p = tl.where(col_ok[None, :], p, 0.0)
    dv = _dot_acc(tl.trans(p.to(do.dtype)), do, dv, IEEE)
    dp = _dot(do, tl.trans(v), IEEE)
    ds = p * (dp - delta[:, None])
    dk = _dot_acc(tl.trans(ds.to(q.dtype)), q, dk, IEEE)
    return dk, dv


@triton.autotune(configs=_configs(), key=["TB", "HEAD_DIM", "IEEE"])
@triton.jit
def _backward_kv(Q, K, V, Bk, Lens, DO, LSE, Delta, DK, DV,
                 sq_b, sq_h, sq_t, sk_b, sk_h, sk_t, sv_b, sv_h, sv_t, sb_b, sb_h, sb_t, sdo_b, sdo_h, sdo_t,
                 sdk_b, sdk_h, sdk_t, sdv_b, sdv_h, sdv_t,
                 H, T, TB, scale,
                 LEFT: tl.constexpr, RIGHT: tl.constexpr, HEAD_DIM: tl.constexpr, IEEE: tl.constexpr,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BAND: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    kv_len = tl.load(Lens + b)
    qk_scale = scale * 1.4426950408889634
    start_n = pid_n * BLOCK_N
    cols = start_n + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    col_ok = cols < kv_len
    k = tl.load(K + b * sk_b + h * sk_h + cols[:, None] * sk_t + offs_d[None, :], mask=col_ok[:, None], other=0.0)
    v = tl.load(V + b * sv_b + h * sv_h + cols[:, None] * sv_t + offs_d[None, :], mask=col_ok[:, None], other=0.0)
    dk = tl.zeros([BLOCK_N, HEAD_DIM], tl.float32)
    dv = tl.zeros([BLOCK_N, HEAD_DIM], tl.float32)
    q_base = Q + b * sq_b + h * sq_h
    do_base = DO + b * sdo_b + h * sdo_h
    bk_base = Bk + b * sb_b + h * sb_h
    if start_n < kv_len:
        full = (T // BLOCK_M) * BLOCK_M
        # query blocks [m, m + BLOCK_M): far right of this key block when start_n - (m + BLOCK_M - 1) >= RIGHT
        last_right = start_n - RIGHT - BLOCK_M + 1
        qa = tl.minimum(tl.where(last_right >= 0, (last_right // BLOCK_M + 1) * BLOCK_M, 0), full)
        # far left when start_n + BLOCK_N - 1 - m <= -LEFT
        first_left = start_n + BLOCK_N - 1 + LEFT
        qb = tl.minimum(tl.maximum(((first_left + BLOCK_M - 1) // BLOCK_M) * BLOCK_M, qa), full)
        # far query blocks, [0, qa) (these keys far right of them) then [qb, full) (far left), in one loop
        for i in range(0, qa // BLOCK_M + (full - qb) // BLOCK_M):
            start_m = i * BLOCK_M
            right = start_m < qa
            start_m = tl.where(right, start_m, start_m + (qb - qa))
            c_off = tl.where(right, LEFT + RIGHT, 0)
            dk, dv = _kv_far(dk, dv, k, v, col_ok, LSE, Delta, q_base, do_base, bk_base, sq_t, sdo_t, sb_t, pid_bh,
                             T, start_m, offs_d, c_off, qk_scale, BLOCK_M, IEEE)
        # the band [qa, qb) and the partial tail [full, T) in narrow BAND-row slices
        for start_m in range(qa, qb, BAND):
            dk, dv = _kv_block(dk, dv, k, v, cols, col_ok, Q, DO, LSE, Delta, bk_base, q_base, do_base, sq_t, sdo_t,
                               sb_t, pid_bh, T, start_m, offs_d, qk_scale, 2, False, LEFT, RIGHT, BAND, IEEE)
        for start_m in range(full, T, BAND):
            dk, dv = _kv_block(dk, dv, k, v, cols, col_ok, Q, DO, LSE, Delta, bk_base, q_base, do_base, sq_t, sdo_t,
                               sb_t, pid_bh, T, start_m, offs_d, qk_scale, 2, True, LEFT, RIGHT, BAND, IEEE)
    tl.store(DK + b * sdk_b + h * sdk_h + cols[:, None] * sdk_t + offs_d[None, :], (dk * scale).to(DK.dtype.element_ty),
             mask=(cols < T)[:, None])
    tl.store(DV + b * sdv_b + h * sdv_h + cols[:, None] * sdv_t + offs_d[None, :], dv.to(DV.dtype.element_ty),
             mask=(cols < T)[:, None])


@triton.jit
def _q_block(dq, d_left, d_right, q, do, lse, delta, k_base, v_base, sk_t, sv_t, bucket_row, db_row, offs_m,
             offs_n, offs_d, start_n, row_ok, c_left, c_right, kv_len, qk_scale, REGION: tl.constexpr,
             MASK: tl.constexpr, LEFT: tl.constexpr, RIGHT: tl.constexpr, IEEE: tl.constexpr):
    cols = start_n + offs_n
    col_ok = cols < kv_len
    if MASK:
        k = tl.load(k_base + cols[:, None] * sk_t + offs_d[None, :], mask=col_ok[:, None], other=0.0)
        v = tl.load(v_base + cols[:, None] * sv_t + offs_d[None, :], mask=col_ok[:, None], other=0.0)
    else:
        k = tl.load(k_base + cols[:, None] * sk_t + offs_d[None, :])
        v = tl.load(v_base + cols[:, None] * sv_t + offs_d[None, :])
    p = _probs(q, k, cols, offs_m, row_ok, bucket_row, c_left, c_right, lse, qk_scale, REGION, LEFT, RIGHT, IEEE)
    dp = _dot(do, tl.trans(v), IEEE)
    if MASK:
        p = tl.where(col_ok[None, :], p, 0.0)
    # rows past T have zero dO and Delta, hence zero dS (their dq is not stored and their dB entries are masked)
    ds = p * (dp - delta[:, None])
    dq = _dot_acc(ds.to(k.dtype), k, dq, IEEE)
    if REGION == 0:
        d_left += tl.sum(ds, 1)
    elif REGION == 1:
        d_right += tl.sum(ds, 1)
    else:
        dist = cols[None, :] - offs_m[:, None]
        d_left += tl.sum(tl.where(dist <= -LEFT, ds, 0.0), 1)
        d_right += tl.sum(tl.where(dist >= RIGHT, ds, 0.0), 1)
        band = (dist > -LEFT) & (dist < RIGHT) & row_ok[:, None] & col_ok[None, :]
        tl.store(db_row[:, None] + dist + LEFT, ds, mask=band)
    return dq, d_left, d_right


@triton.jit
def _q_far(dq, d_left, d_right, q, do, lse, delta, k_base, v_base, sk_t, sv_t, offs_n, offs_d, start_n, c, right,
           qk_scale, IEEE: tl.constexpr):
    """A full key block wholly in one clipped region of these rows (bias c; right selects which row sum)."""
    cols = start_n + offs_n
    k = tl.load(k_base + cols[:, None] * sk_t + offs_d[None, :])
    v = tl.load(v_base + cols[:, None] * sv_t + offs_d[None, :])
    s = _dot(q, tl.trans(k), IEEE)
    p = tl.math.exp2(s * qk_scale + (c - lse)[:, None])
    dp = _dot(do, tl.trans(v), IEEE)
    ds = p * (dp - delta[:, None])
    dq = _dot_acc(ds.to(k.dtype), k, dq, IEEE)
    row_sum = tl.sum(ds, 1)
    d_left += tl.where(right, 0.0, row_sum)
    d_right += tl.where(right, row_sum, 0.0)
    return dq, d_left, d_right


@triton.autotune(configs=_configs(), key=["TB", "HEAD_DIM", "IEEE"])
@triton.jit
def _backward_q(Q, K, V, Bk, Lens, DO, LSE, Delta, DQ, DB, E,
                sq_b, sq_h, sq_t, sk_b, sk_h, sk_t, sv_b, sv_h, sv_t, sb_b, sb_h, sb_t, sdo_b, sdo_h, sdo_t,
                sdq_b, sdq_h, sdq_t,
                H, T, TB, scale,
                LEFT: tl.constexpr, RIGHT: tl.constexpr, HEAD_DIM: tl.constexpr, IEEE: tl.constexpr,
                NB_CHUNK: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BAND: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    kv_len = tl.load(Lens + b)
    qk_scale = scale * 1.4426950408889634
    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    row_ok = offs_m < T
    q = tl.load(Q + b * sq_b + h * sq_h + offs_m[:, None] * sq_t + offs_d[None, :], mask=row_ok[:, None], other=0.0)
    do = tl.load(DO + b * sdo_b + h * sdo_h + offs_m[:, None] * sdo_t + offs_d[None, :], mask=row_ok[:, None],
                 other=0.0)
    lse = tl.load(LSE + pid_bh * T + offs_m, mask=row_ok, other=0.0) * 1.4426950408889634
    delta = tl.load(Delta + pid_bh * T + offs_m, mask=row_ok, other=0.0)
    bucket_row = Bk + b * sb_b + h * sb_h + offs_m * sb_t
    c_left = tl.load(bucket_row, mask=row_ok, other=0.0)
    c_right = tl.load(bucket_row + LEFT + RIGHT, mask=row_ok, other=0.0)
    db_row = DB + pid_bh * T * (LEFT + RIGHT + 1) + offs_m * (LEFT + RIGHT + 1)
    dq = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    d_left = tl.zeros([BLOCK_M], tl.float32)
    d_right = tl.zeros([BLOCK_M], tl.float32)
    k_base = K + b * sk_b + h * sk_h
    v_base = V + b * sv_b + h * sv_h
    full = (kv_len // BLOCK_N) * BLOCK_N
    a, b2 = _regions(start_m, full, LEFT, RIGHT, BLOCK_M, BLOCK_N)
    # far key blocks, left [0, a) then right [b2, full), in one loop of full tiles
    for i in range(0, a // BLOCK_N + (full - b2) // BLOCK_N):
        start_n = i * BLOCK_N
        right = start_n >= a
        start_n = tl.where(right, start_n + (b2 - a), start_n)
        c = tl.where(right, c_right, c_left)
        dq, d_left, d_right = _q_far(dq, d_left, d_right, q, do, lse, delta, k_base, v_base, sk_t, sv_t, offs_n,
                                     offs_d, start_n, c, right, qk_scale, IEEE)
    # the band [a, b2) and the partial tail [full, kv_len) in narrow BAND-key slices (dB band entries stored there)
    offs_band = tl.arange(0, BAND)
    for start_n in range(a, b2, BAND):
        dq, d_left, d_right = _q_block(dq, d_left, d_right, q, do, lse, delta, k_base, v_base, sk_t, sv_t,
                                       bucket_row, db_row, offs_m, offs_band, offs_d, start_n, row_ok, c_left,
                                       c_right, kv_len, qk_scale, 2, False, LEFT, RIGHT, IEEE)
    for start_n in range(full, kv_len, BAND):
        dq, d_left, d_right = _q_block(dq, d_left, d_right, q, do, lse, delta, k_base, v_base, sk_t, sv_t,
                                       bucket_row, db_row, offs_m, offs_band, offs_d, start_n, row_ok, c_left,
                                       c_right, kv_len, qk_scale, 2, True, LEFT, RIGHT, IEEE)
    # dB of these rows is complete once the clipped columns are stored; the bias term of dq is scale * dB E, read
    # back in NB_CHUNK-bucket slices (dS is rounded to the input dtype for it, as for dS K)
    tl.store(db_row, d_left, mask=row_ok)
    tl.store(db_row + LEFT + RIGHT, d_right, mask=row_ok)
    tl.debug_barrier()
    for c0 in tl.static_range(0, LEFT + RIGHT + 1, NB_CHUNK):
        nb = c0 + tl.arange(0, NB_CHUNK)
        nb_ok = nb <= LEFT + RIGHT
        d_bias = tl.load(db_row[:, None] + nb[None, :], mask=row_ok[:, None] & nb_ok[None, :], other=0.0)
        e = tl.load(E + nb[:, None] * HEAD_DIM + offs_d[None, :], mask=nb_ok[:, None], other=0.0)
        dq = _dot_acc(d_bias.to(q.dtype), e.to(q.dtype), dq, IEEE)
    tl.store(DQ + b * sdq_b + h * sdq_h + offs_m[:, None] * sdq_t + offs_d[None, :], (dq * scale).to(DQ.dtype.element_ty),
             mask=row_ok[:, None])


@triton.jit
def _embedding_grad(Q, DB, DE, sq_b, sq_h, sq_t, H, T, SPLIT, NB: tl.constexpr, NB_PAD: tl.constexpr,
                    HEAD_DIM: tl.constexpr, BLOCK_T: tl.constexpr, IEEE: tl.constexpr):
    """DE[program] = dB^T q over one (batch, head) and a 1/SPLIT share of its rows (partials, summed outside)."""
    pid_bh = tl.program_id(0)
    split = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    nb = tl.arange(0, NB_PAD)
    offs_d = tl.arange(0, HEAD_DIM)
    acc = tl.zeros([NB_PAD, HEAD_DIM], tl.float32)
    chunk = tl.cdiv(tl.cdiv(T, SPLIT), BLOCK_T) * BLOCK_T
    for start in range(split * chunk, tl.minimum((split + 1) * chunk, T), BLOCK_T):
        offs_t = start + tl.arange(0, BLOCK_T)
        t_ok = offs_t < T
        q = tl.load(Q + b * sq_b + h * sq_h + offs_t[:, None] * sq_t + offs_d[None, :], mask=t_ok[:, None], other=0.0)
        d_bias = tl.load(DB + (pid_bh * T + offs_t[:, None]) * NB + nb[None, :], mask=t_ok[:, None] & (nb < NB)[None, :],
                         other=0.0)
        acc = _dot_acc(tl.trans(d_bias.to(q.dtype)), q, acc, IEEE)
    tl.store(DE + (pid_bh * SPLIT + split) * NB * HEAD_DIM + nb[:, None] * HEAD_DIM + offs_d[None, :], acc,
             mask=(nb < NB)[:, None])


@triton.jit
def _buckets(Q, E, Out, sq_b, sq_h, sq_t, H, T, factor, NB: tl.constexpr, NB_PAD: tl.constexpr,
             HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, IEEE: tl.constexpr):
    """Out[b, h, t, :] = factor * q[b, h, t] . E^T in float32, reading q in its own dtype and layout."""
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_n = tl.arange(0, NB_PAD)
    row_ok = offs_m < T
    q = tl.load(Q + b * sq_b + h * sq_h + offs_m[:, None] * sq_t + offs_d[None, :], mask=row_ok[:, None], other=0.0)
    e = tl.load(E + offs_n[:, None] * HEAD_DIM + offs_d[None, :], mask=(offs_n < NB)[:, None], other=0.0)
    out = _dot(q, tl.trans(e.to(q.dtype)), IEEE) * factor
    tl.store(Out + (pid_bh * T + offs_m[:, None]) * NB + offs_n[None, :], out,
             mask=row_ok[:, None] & (offs_n < NB)[None, :])


@triton.jit
def _delta(O, DO, Delta, so_b, so_h, so_t, sdo_b, sdo_h, sdo_t, H, T, HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr):
    """Delta[b, h, t] = rowsum(dO * O) in float32."""
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    row_ok = offs_m < T
    o = tl.load(O + b * so_b + h * so_h + offs_m[:, None] * so_t + offs_d[None, :], mask=row_ok[:, None], other=0.0)
    do = tl.load(DO + b * sdo_b + h * sdo_h + offs_m[:, None] * sdo_t + offs_d[None, :], mask=row_ok[:, None],
                 other=0.0)
    tl.store(Delta + pid_bh * T + offs_m, tl.sum(o.to(tl.float32) * do.to(tl.float32), 1), mask=row_ok)


def _strides(t: Tensor) -> tuple[int, int, int]:
    return t.stride(0), t.stride(1), t.stride(2)


def _nb_pad(left: int, right: int) -> int:
    return max(16, triton.next_power_of_2(left + right + 1))


def _attention_forward(query, key, value, embedding, lengths, left, right, out):
    """Writes the attention of [B, H, T, D] (strided) inputs into out (any [B, H, T, D] strides); returns the
    log-sum-exp [B, H, T] and the log2-unit bucket scores [B, H, T, L+R+1]."""
    batch, heads, frames, dim = query.shape
    scale = 1.0 / math.sqrt(dim)
    ieee = query.dtype == torch.float32
    buckets = torch.empty(batch, heads, frames, left + right + 1, device=query.device, dtype=torch.float32)
    _buckets[(triton.cdiv(frames, 64), batch * heads)](
        query, embedding.contiguous(), buckets, *_strides(query), heads, frames, scale * LOG2E, NB=left + right + 1,
        NB_PAD=_nb_pad(left, right), HEAD_DIM=dim, BLOCK_M=64, IEEE=ieee, num_warps=4)
    lse = torch.empty(batch, heads, frames, device=query.device, dtype=torch.float32)

    def grid(meta):
        return (triton.cdiv(frames, meta["BLOCK_M"]), batch * heads)

    _forward[grid](query, key, value, buckets, lengths, out, lse, *_strides(query), *_strides(key), *_strides(value),
                   *_strides(buckets), *_strides(out), heads, frames, triton.next_power_of_2(frames), scale,
                   LEFT=left, RIGHT=right, HEAD_DIM=dim, IEEE=ieee)
    return lse, buckets


def _attention_backward(grad_out, query, key, value, embedding, lengths, buckets, out, lse, left, right, dq, dk, dv):
    """Writes dq, dk, dv (any [B, H, T, D] strides) and returns dE."""
    batch, heads, frames, dim = query.shape
    scale = 1.0 / math.sqrt(dim)
    ieee = query.dtype == torch.float32
    grad_out = grad_out if grad_out.stride(-1) == 1 else grad_out.contiguous()
    delta = torch.empty(batch, heads, frames, device=query.device, dtype=torch.float32)
    _delta[(triton.cdiv(frames, 64), batch * heads)](out, grad_out, delta, *_strides(out), *_strides(grad_out), heads,
                                                     frames, HEAD_DIM=dim, BLOCK_M=64, num_warps=4)
    nb_pad = _nb_pad(left, right)
    db = torch.zeros(batch, heads, frames, left + right + 1, device=query.device, dtype=torch.float32)
    common = (*_strides(query), *_strides(key), *_strides(value), *_strides(buckets), *_strides(grad_out))
    tail = (heads, frames, triton.next_power_of_2(frames), scale)
    _backward_kv[lambda meta: (triton.cdiv(frames, meta["BLOCK_N"]), batch * heads)](
        query, key, value, buckets, lengths, grad_out, lse, delta, dk, dv, *common, *_strides(dk), *_strides(dv),
        *tail, LEFT=left, RIGHT=right, HEAD_DIM=dim, IEEE=ieee)
    _backward_q[lambda meta: (triton.cdiv(frames, meta["BLOCK_M"]), batch * heads)](
        query, key, value, buckets, lengths, grad_out, lse, delta, dq, db, embedding.contiguous(), *common,
        *_strides(dq), *tail, LEFT=left, RIGHT=right, HEAD_DIM=dim, IEEE=ieee, NB_CHUNK=32)
    split = max(1, min(8, frames // 256))
    de = torch.empty(batch * heads * split, left + right + 1, dim, device=query.device, dtype=torch.float32)
    _embedding_grad[(batch * heads, split)](query, db, de, *_strides(query), heads, frames, split, NB=left + right + 1,
                                            NB_PAD=nb_pad, HEAD_DIM=dim, BLOCK_T=64, IEEE=ieee, num_warps=4)
    return (de.sum(0) * scale).to(embedding.dtype)


# ---- custom ops: q, k, v as separate [B, H, T, D] tensors

@torch.library.custom_op("w2vbert_fast::relative_attention", mutates_args=())
def _relative_attention(query: Tensor, key: Tensor, value: Tensor, embedding: Tensor, lengths: Tensor, left: int,
                        right: int) -> tuple[Tensor, Tensor, Tensor]:
    out = torch.empty_like(query)
    lse, buckets = _attention_forward(query, key, value, embedding, lengths, left, right, out)
    return out, lse, buckets


@_relative_attention.register_fake
def _(query, key, value, embedding, lengths, left, right):
    batch, heads, frames, _ = query.shape
    return (torch.empty_like(query), query.new_empty(batch, heads, frames, dtype=torch.float32),
            query.new_empty(batch, heads, frames, left + right + 1, dtype=torch.float32))


@torch.library.custom_op("w2vbert_fast::relative_attention_backward", mutates_args=())
def _relative_attention_backward(grad_out: Tensor, query: Tensor, key: Tensor, value: Tensor, embedding: Tensor,
                                 lengths: Tensor, buckets: Tensor, out: Tensor, lse: Tensor, left: int,
                                 right: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    dq, dk, dv = torch.empty_like(query), torch.empty_like(key), torch.empty_like(value)
    de = _attention_backward(grad_out, query, key, value, embedding, lengths, buckets, out, lse, left, right, dq, dk,
                             dv)
    return dq, dk, dv, de


@_relative_attention_backward.register_fake
def _(grad_out, query, key, value, embedding, lengths, buckets, out, lse, left, right):
    return (torch.empty_like(query), torch.empty_like(key), torch.empty_like(value), torch.empty_like(embedding))


def _setup(ctx, inputs, output):
    query, key, value, embedding, lengths, left, right = inputs
    out, lse, buckets = output
    ctx.save_for_backward(query, key, value, embedding, lengths, buckets, out, lse)
    ctx.dims = (left, right)


def _backward(ctx, grad_out, grad_lse, grad_buckets):
    query, key, value, embedding, lengths, buckets, out, lse = ctx.saved_tensors
    dq, dk, dv, de = _relative_attention_backward(grad_out, query, key, value, embedding, lengths, buckets, out, lse,
                                                  *ctx.dims)
    return dq, dk, dv, de, None, None, None


_relative_attention.register_autograd(_backward, setup_context=_setup)


# ---- custom ops: the packed projection [B, T, 3, H, D] -> [B, T, H, D] (no copies in either direction)

def _split(qkv: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    return tuple(t.transpose(1, 2) for t in qkv.unbind(2))  # each [B, H, T, D], strided


@torch.library.custom_op("w2vbert_fast::relative_attention_packed", mutates_args=())
def _packed(qkv: Tensor, embedding: Tensor, lengths: Tensor, left: int, right: int) -> tuple[Tensor, Tensor, Tensor]:
    batch, frames, _, heads, dim = qkv.shape
    out = qkv.new_empty(batch, frames, heads, dim)
    lse, buckets = _attention_forward(*_split(qkv), embedding, lengths, left, right, out.transpose(1, 2))
    return out, lse, buckets


@_packed.register_fake
def _(qkv, embedding, lengths, left, right):
    batch, frames, _, heads, dim = qkv.shape
    return (qkv.new_empty(batch, frames, heads, dim), qkv.new_empty(batch, heads, frames, dtype=torch.float32),
            qkv.new_empty(batch, heads, frames, left + right + 1, dtype=torch.float32))


@torch.library.custom_op("w2vbert_fast::relative_attention_packed_backward", mutates_args=())
def _packed_backward(grad_out: Tensor, qkv: Tensor, embedding: Tensor, lengths: Tensor, buckets: Tensor, out: Tensor,
                     lse: Tensor, left: int, right: int) -> tuple[Tensor, Tensor]:
    dqkv = torch.empty_like(qkv)
    dq, dk, dv = _split(dqkv)
    de = _attention_backward(grad_out.transpose(1, 2), *_split(qkv), embedding, lengths, buckets, out.transpose(1, 2),
                             lse, left, right, dq, dk, dv)
    return dqkv, de


@_packed_backward.register_fake
def _(grad_out, qkv, embedding, lengths, buckets, out, lse, left, right):
    return torch.empty_like(qkv), torch.empty_like(embedding)


def _packed_setup(ctx, inputs, output):
    qkv, embedding, lengths, left, right = inputs
    out, lse, buckets = output
    ctx.save_for_backward(qkv, embedding, lengths, buckets, out, lse)
    ctx.dims = (left, right)


def _packed_grad(ctx, grad_out, grad_lse, grad_buckets):
    qkv, embedding, lengths, buckets, out, lse = ctx.saved_tensors
    grad_out = grad_out if grad_out.stride(-1) == 1 else grad_out.contiguous()
    dqkv, de = _packed_backward(grad_out, qkv, embedding, lengths, buckets, out, lse, *ctx.dims)
    return dqkv, de, None, None, None


_packed.register_autograd(_packed_grad, setup_context=_packed_setup)


def _lengths(lengths, batch, frames, device):
    if lengths is None:
        return torch.full((batch,), frames, device=device, dtype=torch.int32)
    return lengths.to(torch.int32).contiguous()


def fused_relative_key_attention(query: Tensor, key: Tensor, value: Tensor, embedding: Tensor, left: int, right: int,
                                 lengths: Tensor | None = None) -> Tensor:
    """Relative-key attention ([B, H, T, D], embedding [L+R+1, D]) in fused Triton kernels, with gradients for
    query, key, value and the embedding. lengths [B] masks keys at or beyond each sequence's length (query rows are
    all computed, as with the reference key-padding mask). CUDA; fp16, bf16 or fp32 (fp32 without TF32)."""
    if not query.is_cuda:
        raise ValueError("fused_relative_key_attention needs CUDA tensors")
    if query.size(-1) not in (16, 32, 64, 128):
        raise ValueError(f"head size {query.size(-1)} is not supported (16, 32, 64, 128)")
    query, key, value = (t if t.stride(-1) == 1 else t.contiguous() for t in (query, key, value))
    lengths = _lengths(lengths, query.size(0), query.size(2), query.device)
    return _relative_attention(query, key, value, embedding, lengths, int(left), int(right))[0]


def fused_relative_key_attention_packed(qkv: Tensor, embedding: Tensor, left: int, right: int,
                                        lengths: Tensor | None = None) -> Tensor:
    """The same from a packed projection qkv [B, T, 3, H, D] (contiguous; e.g. one linear layer's output viewed
    so); returns [B, T, H, D], ready for the output projection. The backward returns one [B, T, 3, H, D] gradient."""
    if not qkv.is_cuda:
        raise ValueError("fused_relative_key_attention_packed needs CUDA tensors")
    if qkv.size(-1) not in (16, 32, 64, 128):
        raise ValueError(f"head size {qkv.size(-1)} is not supported (16, 32, 64, 128)")
    qkv = qkv.contiguous()
    lengths = _lengths(lengths, qkv.size(0), qkv.size(1), qkv.device)
    return _packed(qkv, embedding, lengths, int(left), int(right))[0]
