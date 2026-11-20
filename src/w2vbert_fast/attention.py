"""Exact, fast relative-key self-attention for w2v-BERT 2.0 (Hugging Face ``Wav2Vec2BertSelfAttention`` with
``position_embeddings_type="relative_key"``).

The relative-key score is ``s_ij = q_i . (k_j + E[clip(j - i, -L, R) + L]) / sqrt(d)``, with a learned distance
embedding ``E`` of ``L + R + 1`` rows shared by all heads (w2v-BERT 2.0: ``L = 64``, ``R = 8``). Outside the band
``-L < j - i < R`` the clipped distance is constant, so the bias is a per-query constant there:

* far left  (``j - i <= -L``): keys ``k_j + E[0]``,     a causal attention of ``q[L:]`` over ``k[:T-L]``;
* far right (``j - i >= R``):  keys ``k_j + E[L+R]``,   a causal attention of the time-reversed ``q[:T-R]`` over
  the time-reversed ``k[R:]``;
* band      (``-L < j - i < R``, ``L + R - 1`` keys per query): the gathered bias, computed densely in blocks.

The far regions run on PyTorch's FlashAttention kernel (no bias, causal), the band in float32, and the three
partial softmaxes are merged by their log-sum-exps. The backward reuses the merged output and log-sum-exp in each
region (as ring attention does), so every region's gradient is exact. The result equals the reference up to
floating-point rounding; no [B, H, T, T] tensor is ever built.

Backends of the far regions: ``flash`` (CUDA, fp16/bf16, head size <= 256), ``math`` (anything else: an explicit
float32/float64 implementation used for CPU and for tests).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

# --------------------------------------------------------------------------------------------- reference


def relative_key_reference(query: Tensor, key: Tensor, value: Tensor, embedding: Tensor, left: int, right: int,
                           attention_mask: Tensor | None = None) -> Tensor:
    """The Hugging Face computation (``[B, H, T, D]`` inputs, ``embedding [L+R+1, D]``), for tests and fallbacks.
    attention_mask is additive ``[B, 1, T, T]`` (or broadcastable), as the HF encoder passes it."""
    frames = query.size(-2)
    positions = torch.arange(frames, device=query.device)
    distance = (positions[None, :] - positions[:, None]).clamp(-left, right) + left
    scores = torch.matmul(query, key.transpose(-2, -1))
    scores = scores + torch.einsum("bhld,lrd->bhlr", query, embedding[distance].to(query.dtype))
    scores = scores / math.sqrt(query.size(-1))
    if attention_mask is not None:
        scores = scores + attention_mask
    return torch.matmul(torch.softmax(scores, dim=-1), value)


# ----------------------------------------------------------------------------------- far-region kernels


def _flash_ok(query: Tensor) -> bool:
    return (query.is_cuda and query.dtype in (torch.float16, torch.bfloat16) and query.size(-1) <= 256
            and query.size(-1) % 8 == 0)


def _causal_forward(query: Tensor, key: Tensor, value: Tensor, scale: float) -> tuple[Tensor, Tensor, tuple]:
    """Causal attention (query a attends keys 0..a; equal lengths) -> output, float32 log-sum-exp [B, H, Lq] and
    the state the backward needs."""
    if _flash_ok(query):
        out, lse, cum_q, cum_k, max_q, max_k, seed, offset, _ = torch.ops.aten._scaled_dot_product_flash_attention(
            query, key, value, 0.0, True, False, scale=scale)
        return out, lse, ("flash", cum_q, cum_k, max_q, max_k, seed, offset)
    compute = torch.promote_types(query.dtype, torch.float32)
    scores = torch.matmul(query.to(compute), key.to(compute).transpose(-2, -1)) * scale
    causal = torch.ones(scores.shape[-2:], dtype=torch.bool, device=query.device).tril()
    scores = scores.masked_fill(~causal, float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)
    out = torch.matmul(torch.exp(scores - lse[..., None]), value.to(compute)).to(query.dtype)
    return out, lse, ("math",)


def _causal_backward(grad_out: Tensor, query: Tensor, key: Tensor, value: Tensor, out: Tensor, lse: Tensor,
                     scale: float, state: tuple) -> tuple[Tensor, Tensor, Tensor]:
    """Gradients of a causal attention given the (merged) output and log-sum-exp of its rows."""
    if state[0] == "flash":
        _, cum_q, cum_k, max_q, max_k, seed, offset = state
        return torch.ops.aten._scaled_dot_product_flash_attention_backward(
            grad_out.contiguous(), query, key, value, out.contiguous(), lse.contiguous(), cum_q, cum_k, max_q, max_k,
            0.0, True, seed, offset, scale=scale)
    compute = torch.promote_types(query.dtype, torch.float32)
    q, k, v, d_out = (t.to(compute) for t in (query, key, value, grad_out))
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    causal = torch.ones(scores.shape[-2:], dtype=torch.bool, device=query.device).tril()
    probs = torch.exp(scores - lse.to(compute)[..., None]).masked_fill(~causal, 0.0)
    grad_value = torch.matmul(probs.transpose(-2, -1), d_out)
    delta = (d_out * out.to(compute)).sum(-1, keepdim=True)
    grad_scores = probs * (torch.matmul(d_out, v.transpose(-2, -1)) - delta)
    grad_query = torch.matmul(grad_scores, k) * scale
    grad_key = torch.matmul(grad_scores.transpose(-2, -1), q) * scale
    return grad_query.to(query.dtype), grad_key.to(key.dtype), grad_value.to(value.dtype)


# ----------------------------------------------------------------------------------------------- band

BLOCK = 64  # query rows per band block


def _band_layout(frames: int, left: int, right: int):
    width = left + right - 1  # band keys per query: distances -(L-1) .. R-1
    blocks = -(-frames // BLOCK)
    window = BLOCK + width - 1  # keys one query block can reach
    return width, blocks, window


def _windows(tensor: Tensor, frames: int, left: int, right: int) -> Tensor:
    """[B, H, T, D] -> [B, H, blocks, window, D]: block n holds keys n*BLOCK - (L-1) .. n*BLOCK + BLOCK + R - 2
    (zero outside 0..T-1)."""
    width, blocks, window = _band_layout(frames, left, right)
    padded = F.pad(tensor, (0, 0, left - 1, blocks * BLOCK - frames + right - 1))
    return padded.unfold(2, window, BLOCK).transpose(-1, -2)


def _band_index(device, left: int, right: int) -> Tensor:
    width, _, _ = _band_layout(BLOCK, left, right)
    return torch.arange(BLOCK, device=device)[:, None] + torch.arange(width, device=device)[None, :]  # [BLOCK, W]


def _band_valid(frames: int, left: int, right: int, device) -> Tensor:
    """[blocks, BLOCK, W]: query and key inside 0..T-1."""
    width, blocks, _ = _band_layout(frames, left, right)
    query = torch.arange(blocks * BLOCK, device=device).view(blocks, BLOCK, 1)
    key = query + torch.arange(width, device=device) - (left - 1)
    return (query < frames) & (key >= 0) & (key < frames)


def _compute_dtype(tensor: Tensor) -> torch.dtype:
    return torch.promote_types(tensor.dtype, torch.float32)


def _band_scores(query: Tensor, key: Tensor, embedding: Tensor, left: int, right: int, scale: float) -> Tensor:
    """Band scores [B, H, blocks, BLOCK, W] in float32 (float64 for float64 inputs), -inf outside the sequence."""
    batch, heads, frames, dim = query.shape
    width, blocks, _ = _band_layout(frames, left, right)
    compute = _compute_dtype(query)
    q = F.pad(query.to(compute), (0, 0, 0, blocks * BLOCK - frames)).view(batch, heads, blocks, BLOCK, dim)
    keys = _windows(key.to(compute), frames, left, right)  # [B,H,n,window,D]
    full = torch.matmul(q, keys.transpose(-1, -2))  # [B,H,n,BLOCK,window]
    index = _band_index(query.device, left, right).expand(batch, heads, blocks, BLOCK, width)
    scores = full.gather(-1, index) + torch.matmul(q, embedding[1:left + right].to(compute).T)
    valid = _band_valid(frames, left, right, query.device)
    return (scores * scale).masked_fill(~valid, float("-inf")), q, keys, index, valid


def _scatter_band(band: Tensor, index: Tensor, window: int) -> Tensor:
    return band.new_zeros(*band.shape[:-1], window).scatter(-1, index, band)


def _overlap_add(windows: Tensor, frames: int, left: int, right: int) -> Tensor:
    """Inverse of _windows for gradients: [B, H, blocks, window, D] summed back to [B, H, T, D]."""
    batch, heads, blocks, window, dim = windows.shape
    start = torch.arange(blocks, device=windows.device)[:, None] * BLOCK + torch.arange(window, device=windows.device)
    padded = windows.new_zeros(batch, heads, blocks * BLOCK + window - BLOCK, dim)
    padded.index_add_(2, start.reshape(-1), windows.reshape(batch, heads, blocks * window, dim))
    return padded[:, :, left - 1:left - 1 + frames]


# ------------------------------------------------------------------------------------- the autograd op


class _BandedRelativeAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, embedding, left, right):
        batch, heads, frames, dim = query.shape
        scale = 1.0 / math.sqrt(dim)
        width, blocks, window = _band_layout(frames, left, right)
        ctx.embedding_dtype = embedding.dtype
        embedding = embedding.to(query.dtype)
        neg = torch.full((batch, heads, frames), float("-inf"), device=query.device, dtype=_compute_dtype(query))
        parts_out, parts_lse = [], []
        far = []
        if frames > left:  # far left: queries L.., keys ..T-L-1
            q_l = query[:, :, left:].contiguous()
            k_l = (key[:, :, :frames - left] + embedding[0]).contiguous()
            v_l = value[:, :, :frames - left].contiguous()
            out, lse, state = _causal_forward(q_l, k_l, v_l, scale)
            parts_out.append(F.pad(out, (0, 0, left, 0)))
            parts_lse.append(torch.cat((neg[:, :, :left], lse.to(neg.dtype)), -1))
            far.append(("left", q_l, k_l, v_l, state))
        if frames > right:  # far right, time-reversed: queries ..T-R-1, keys R..
            q_r = query[:, :, :frames - right].flip(2).contiguous()
            k_r = (key[:, :, right:] + embedding[left + right]).flip(2).contiguous()
            v_r = value[:, :, right:].flip(2).contiguous()
            out, lse, state = _causal_forward(q_r, k_r, v_r, scale)
            parts_out.append(F.pad(out.flip(2), (0, 0, 0, right)))
            parts_lse.append(torch.cat((lse.flip(-1).to(neg.dtype), neg[:, :, :right]), -1))
            far.append(("right", q_r, k_r, v_r, state))
        scores, _, _, index, _ = _band_scores(query, key, embedding, left, right, scale)
        band_lse = torch.logsumexp(scores, dim=-1)  # [B,H,n,BLOCK]
        probs = _scatter_band(torch.exp(scores - band_lse[..., None]).nan_to_num(0.0), index, window)
        band_out = torch.matmul(probs, _windows(value.to(probs.dtype), frames, left, right))
        parts_out.append(band_out.reshape(batch, heads, blocks * BLOCK, dim)[:, :, :frames])
        parts_lse.append(band_lse.reshape(batch, heads, blocks * BLOCK)[:, :, :frames])
        lse = torch.stack(parts_lse)  # [parts,B,H,T]
        total = torch.logsumexp(lse, dim=0)  # every query has its own key in the band, so finite
        weights = torch.exp(lse - total)
        output = sum(w[..., None] * part.to(weights.dtype) for w, part in zip(weights, parts_out)).to(query.dtype)
        ctx.save_for_backward(query, key, value, embedding, output, total)
        ctx.far = far
        ctx.dims = (left, right, scale)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        query, key, value, embedding, output, total = ctx.saved_tensors
        left, right, scale = ctx.dims
        batch, heads, frames, dim = query.shape
        width, blocks, window = _band_layout(frames, left, right)
        compute = _compute_dtype(query)
        grad_query = torch.zeros_like(query, dtype=compute)
        grad_key = torch.zeros_like(key, dtype=compute)
        grad_value = torch.zeros_like(value, dtype=compute)
        grad_embedding = torch.zeros_like(embedding, dtype=compute)
        for name, q, k, v, state in ctx.far:
            if name == "left":
                rows = slice(left, frames)
                d_out, out, lse = grad_output[:, :, rows], output[:, :, rows], total[:, :, rows]
                dq, dk, dv = _causal_backward(d_out, q, k, v, out, lse, scale, state)
                grad_query[:, :, rows] += dq.to(compute)
                grad_key[:, :, :frames - left] += dk.to(compute)
                grad_value[:, :, :frames - left] += dv.to(compute)
                grad_embedding[0] += dk.to(compute).sum((0, 1, 2))
            else:
                rows = slice(0, frames - right)
                d_out = grad_output[:, :, rows].flip(2)
                out, lse = output[:, :, rows].flip(2), total[:, :, rows].flip(-1)
                dq, dk, dv = _causal_backward(d_out, q, k, v, out, lse, scale, state)
                grad_query[:, :, rows] += dq.to(compute).flip(2)
                grad_key[:, :, right:] += dk.to(compute).flip(2)
                grad_value[:, :, right:] += dv.to(compute).flip(2)
                grad_embedding[left + right] += dk.to(compute).sum((0, 1, 2))
        # band, with the merged log-sum-exp: P = exp(S - LSE), dS = P (dP - D), D = rowsum(dO * O)
        scores, q_blocks, key_windows, index, valid = _band_scores(query, key, embedding, left, right, scale)
        pad = blocks * BLOCK - frames
        lse_blocks = F.pad(total, (0, pad)).view(batch, heads, blocks, BLOCK)
        probs = torch.exp(scores - lse_blocks[..., None]).masked_fill(~valid, 0.0)
        d_out = F.pad(grad_output.to(compute), (0, 0, 0, pad)).view(batch, heads, blocks, BLOCK, dim)
        delta = F.pad((grad_output.to(compute) * output.to(compute)).sum(-1), (0, pad)).view(
            batch, heads, blocks, BLOCK)
        value_windows = _windows(value.to(compute), frames, left, right)
        d_probs = torch.matmul(d_out, value_windows.transpose(-1, -2)).gather(
            -1, index.expand(batch, heads, blocks, BLOCK, width))
        d_scores = probs * (d_probs - delta[..., None]) * scale  # [B,H,n,BLOCK,W], scale folded in
        d_full = _scatter_band(d_scores, index, window)
        grad_query += (torch.matmul(d_full, key_windows) + torch.matmul(d_scores, embedding[1:left + right].to(compute))
                       ).reshape(batch, heads, blocks * BLOCK, dim)[:, :, :frames]
        grad_key += _overlap_add(torch.matmul(d_full.transpose(-1, -2), q_blocks), frames, left, right)
        grad_value += _overlap_add(torch.matmul(_scatter_band(probs, index, window).transpose(-1, -2), d_out),
                                   frames, left, right)
        grad_embedding[1:left + right] += torch.einsum("bhntw,bhntd->wd", d_scores, q_blocks)
        return (grad_query.to(query.dtype), grad_key.to(key.dtype), grad_value.to(value.dtype),
                grad_embedding.to(ctx.embedding_dtype), None, None)


def relative_key_attention(query: Tensor, key: Tensor, value: Tensor, embedding: Tensor, left: int,
                           right: int) -> Tensor:
    """Exact relative-key attention of unpadded sequences ([B, H, T, D]; embedding [L+R+1, D]), with gradients for
    query, key, value and the embedding."""
    if left < 1 or right < 1:
        raise ValueError(f"left and right must be positive, not {left}, {right}")
    if embedding.size(0) != left + right + 1:
        raise ValueError(f"embedding has {embedding.size(0)} rows, expected {left + right + 1}")
    out = _BandedRelativeAttention.apply(query, key, value, embedding, left, right)
    return out
