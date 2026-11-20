"""Speed-ups for a Hugging Face ``Wav2Vec2BertModel`` (w2v-BERT 2.0) that keep its weights, state-dict keys and
outputs.

``optimize(model)`` swaps the class of every relative-key self-attention module for ``FastSelfAttention`` (same
parameters, same keys; ``isinstance`` still holds), optionally keeps only the first ``num_layers`` encoder layers,
and optionally compiles the feed-forward and convolution modules. ``load()`` does the same on a fresh model.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor

from .attention import relative_key_attention, relative_key_reference

Backend = Literal["auto", "triton", "flash", "sdpa", "eager"]


def _key_lengths(attention_mask: Tensor | None, frames: int) -> Tensor | None:
    """Per-sequence key lengths [B] (int32, on the mask's device) of the Hugging Face encoder's additive key-padding
    mask: [B, 1, T, T] expanded from [B, 1, 1, T], 0 for valid keys and the dtype's minimum for padded ones. None
    without a mask. Raises ValueError for any other shape or layout (the caller then falls back), decided from
    metadata alone. Computed afresh for every call and without a host synchronization (nothing to cache across
    batches, nothing to break a compiled graph); that the padding is a suffix of 0 / minimum values is asserted on
    the device (torch._assert_async), since the kernels mask keys at or past each length."""
    if attention_mask is None:
        return None
    if (attention_mask.dim() != 4 or attention_mask.size(1) != 1 or attention_mask.size(-1) != frames
            or attention_mask.size(-2) != frames or attention_mask.stride(-2) != 0
            or not attention_mask.dtype.is_floating_point):
        raise ValueError("not an expanded additive key-padding mask")
    row = attention_mask[:, 0, 0, :]
    valid = row == 0
    lengths = valid.sum(-1, dtype=torch.int32)
    suffix = valid == (torch.arange(frames, device=row.device)[None, :] < lengths[:, None])
    padded = valid | (row <= torch.finfo(row.dtype).min / 2)
    torch._assert_async(torch.logical_and(suffix, padded).all(),
                        "w2vbert_fast: the attention mask must pad keys at the end of each sequence")
    return lengths


class FastSelfAttention(torch.nn.Module):
    """``Wav2Vec2BertSelfAttention`` with an exact, faster relative-key attention (set by class swap; see optimize).

    backend ``triton``: the fused Triton kernels (``triton_attention``; CUDA, head size 16/32/64/128), with the
    key-padding mask applied inside the kernel, so every position matches the reference. ``flash``: the banded
    FlashAttention decomposition (``attention.relative_key_attention``); padded batches run per sequence length and
    padded query rows return zeros (the HF encoder's convolution module zeroes them, so no valid position changes).
    ``sdpa``: one scaled-dot-product call with a dense relative bias. ``eager``: the original module. ``auto``:
    triton where it applies, else flash. The original code is used whenever ``output_attentions`` is requested, the mask
    is not a key-padding mask, or attention dropout is active."""

    fast_backend: Backend = "auto"

    def _eager(self, *args, **kwargs):
        from transformers.models.wav2vec2_bert.modeling_wav2vec2_bert import Wav2Vec2BertSelfAttention

        return Wav2Vec2BertSelfAttention.forward(self, *args, **kwargs)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor | None = None,
                relative_position_embeddings: Tensor | None = None, output_attentions: bool = False):
        if (self.fast_backend == "eager" or output_attentions or self.position_embeddings_type != "relative_key"
                or (self.training and self.dropout.p > 0)):
            return self._eager(hidden_states, attention_mask, relative_position_embeddings, output_attentions)
        batch, frames, _ = hidden_states.shape

        projections = (self.linear_q, self.linear_k, self.linear_v)
        if all(type(m) is torch.nn.Linear for m in projections):
            # q, k and v in one matmul (the weights stay in their own modules; views of one [B, T, 3, H, D] output)
            weight = torch.cat([m.weight for m in projections])
            bias = None if any(m.bias is None for m in projections) else torch.cat([m.bias for m in projections])
            qkv = torch.nn.functional.linear(hidden_states, weight, bias)
        else:
            # wrapped projections (e.g. PEFT LoRA layers, whose .weight is only the base weight): call each module
            qkv = torch.stack([m(hidden_states) for m in projections], dim=2)
        qkv = qkv.view(batch, frames, 3, self.num_heads, self.head_size)
        query, key, value = (t.transpose(1, 2) for t in qkv.unbind(2))
        left, right = self.left_max_position_embeddings, self.right_max_position_embeddings
        embedding = self.distance_embedding.weight
        try:
            lengths = _key_lengths(attention_mask, frames)
        except ValueError:
            lengths, backend = None, "sdpa"
        else:
            backend = self.fast_backend
        if backend in ("auto", "triton") and query.is_cuda and self.head_size in (16, 32, 64, 128):
            from .triton_attention import fused_relative_key_attention_packed

            output = fused_relative_key_attention_packed(qkv, embedding, left, right, lengths)  # [B, T, H, D]
            return self.linear_out(output.reshape(batch, frames, self.num_heads * self.head_size)), None
        elif backend == "triton":
            raise ValueError("the triton backend needs CUDA tensors and a head size of 16, 32, 64 or 128")
        elif backend == "sdpa":
            output = _dense_sdpa(query, key, value, embedding, left, right, attention_mask)
        elif lengths is None or bool((lengths == frames).all()):
            output = relative_key_attention(query, key, value, embedding, left, right)
        else:
            output = query.new_zeros(query.shape)
            for length in sorted(set(lengths.tolist())):
                index = (lengths == length).nonzero().squeeze(1)
                output[index, :, :length] = relative_key_attention(
                    query[index, :, :length], key[index, :, :length], value[index, :, :length], embedding, left,
                    right)
        output = output.transpose(1, 2).reshape(batch, frames, self.num_heads * self.head_size)
        return self.linear_out(output), None


class FastConvolutionModule(torch.nn.Module):
    """``Wav2Vec2BertConvolutionModule`` computed in [B, T, C] layout (set by class swap; see optimize): the 1x1
    convolutions as matmuls, the gated linear unit on the last dimension, and the causal depthwise convolution in a
    channels-last Triton kernel on CUDA. Same parameters and keys; the same values as the original."""

    def forward(self, hidden_states: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        if not (type(self.pointwise_conv1) is type(self.pointwise_conv2) is type(self.depthwise_conv)
                is torch.nn.Conv1d):
            # wrapped convolutions (e.g. PEFT LoRA layers): the original forward calls the modules themselves
            from transformers.models.wav2vec2_bert.modeling_wav2vec2_bert import Wav2Vec2BertConvolutionModule

            return Wav2Vec2BertConvolutionModule.forward(self, hidden_states, attention_mask)
        hidden = self.layer_norm(hidden_states)
        if attention_mask is not None:
            hidden = hidden.masked_fill(~attention_mask.bool().unsqueeze(-1), 0.0)
        hidden = torch.nn.functional.linear(hidden, self.pointwise_conv1.weight.squeeze(-1))
        value, gate = hidden.chunk(2, dim=-1)
        hidden = value * torch.sigmoid(gate)
        if hidden.is_cuda:
            from .triton_conv import causal_depthwise_conv

            hidden = causal_depthwise_conv(hidden, self.depthwise_conv.weight)
        else:
            from .triton_conv import causal_depthwise_conv_reference

            hidden = causal_depthwise_conv_reference(hidden, self.depthwise_conv.weight)
        hidden = self.activation(self.depthwise_layer_norm(hidden))
        hidden = torch.nn.functional.linear(hidden, self.pointwise_conv2.weight.squeeze(-1))
        return self.dropout(hidden)


def _dense_sdpa(query, key, value, embedding, left, right, attention_mask):
    """The relative bias gathered from [B, H, T, L+R+1] bucket scores (no [T, T, D] tensor), then one SDPA call."""
    frames = query.size(-2)
    positions = torch.arange(frames, device=query.device)
    distance = ((positions[None, :] - positions[:, None]).clamp(-left, right) + left)
    buckets = torch.matmul(query, embedding.to(query.dtype).T)
    bias = buckets.gather(-1, distance.expand(*buckets.shape[:2], frames, frames)) / math.sqrt(query.size(-1))
    if attention_mask is not None:
        bias = bias + attention_mask.to(bias.dtype)
    return torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=bias)


RECOMPILE_LIMIT = 64


def _raise_recompile_limit() -> None:
    """Every encoder layer of every compiled model runs the same forward code, so dynamo's per-code recompile limit
    (8 by default) is shared by all of them: train and eval modes, new sequence lengths and a second model in the same
    process each take entries, and past the limit layers silently run uncompiled. Raise it to RECOMPILE_LIMIT."""
    import torch._dynamo

    config = torch._dynamo.config
    for name in ("recompile_limit", "cache_size_limit"):
        if hasattr(config, name) and getattr(config, name) < RECOMPILE_LIMIT:
            setattr(config, name, RECOMPILE_LIMIT)


_CLASSES: dict = {}


def _fast_class(base: type, fast: type = None) -> type:
    """One subclass per original class: the fast forward first, then the original's attributes and methods."""
    fast = FastSelfAttention if fast is None else fast
    if (base, fast) not in _CLASSES:
        _CLASSES[(base, fast)] = type(f"Fast{base.__name__}", (fast, base), {})
    return _CLASSES[(base, fast)]


def optimize(model: torch.nn.Module, *, backend: Backend = "auto", num_layers: int | None = None,
             compile: bool = False, convolution: bool = True) -> torch.nn.Module:
    """In place: fast relative-key attention and (convolution=True) the channels-last convolution module in every
    encoder layer, optional truncation to the first num_layers layers (outputs of those layers are unchanged),
    optional torch.compile of each whole encoder layer (the Triton kernels are custom ops inside the graph).
    Returns the model."""
    from transformers.models.wav2vec2_bert.modeling_wav2vec2_bert import (Wav2Vec2BertConvolutionModule,
                                                                          Wav2Vec2BertSelfAttention)

    encoder = model.encoder if hasattr(model, "encoder") else model
    if num_layers is not None:
        if not 1 <= num_layers <= len(encoder.layers):
            raise ValueError(f"num_layers must be in 1..{len(encoder.layers)}, not {num_layers}")
        encoder.layers = encoder.layers[:num_layers]
        if hasattr(model, "config"):
            model.config.num_hidden_layers = num_layers
    swapped = 0
    for module in model.modules():
        if isinstance(module, Wav2Vec2BertSelfAttention) and getattr(module, "position_embeddings_type", None) == \
                "relative_key":
            if not isinstance(module, FastSelfAttention):
                module.__class__ = _fast_class(type(module))
            module.fast_backend = backend
            swapped += 1
    if convolution and backend != "eager":
        for module in model.modules():
            if isinstance(module, Wav2Vec2BertConvolutionModule) and not isinstance(module, FastConvolutionModule):
                module.__class__ = _fast_class(type(module), FastConvolutionModule)
    if compile:
        _raise_recompile_limit()
        for layer in encoder.layers:
            layer.compile()
    model._w2vbert_fast = {"backend": backend, "attention_modules": swapped, "num_layers": len(encoder.layers),
                           "compile": compile, "convolution": convolution}
    return model


def load(name: str = "facebook/w2v-bert-2.0", *, num_layers: int | None = None, backend: Backend = "auto",
         compile: bool = False, device: torch.device | str | None = None, dtype: torch.dtype | None = None,
         **kwargs) -> torch.nn.Module:
    """``Wav2Vec2BertModel.from_pretrained(name, **kwargs)`` with optimize() applied."""
    from transformers import Wav2Vec2BertModel

    model = Wav2Vec2BertModel.from_pretrained(name, **kwargs)
    optimize(model, backend=backend, num_layers=num_layers, compile=False)
    if device is not None or dtype is not None:
        model = model.to(device=device, dtype=dtype)
    if compile:
        optimize(model, backend=backend, compile=True)
    return model


__all__ = ["FastConvolutionModule", "FastSelfAttention", "optimize", "load", "relative_key_reference"]
