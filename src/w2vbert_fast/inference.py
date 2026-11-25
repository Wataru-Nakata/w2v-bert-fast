"""Inference with CUDA graphs: the optimized model (``optimize(..., compile=True)``, bf16) captured once per input
shape bucket and replayed, so each call costs one graph launch instead of hundreds of kernel launches.

Input shapes are rounded up to buckets so that a handful of graphs serves every length: the number of frames to a
multiple of ``frame_step`` (padded frames are masked as the feature extractor's padding is, so valid positions are
unchanged) and the batch size to the next of ``batch_sizes`` (the extra rows are dropped). All graphs share one
memory pool; they replay one at a time on the current stream.
"""

from __future__ import annotations

import collections
import math

import torch
from torch import Tensor

from .model import optimize


class InferenceEncoder:
    """w2v-BERT 2.0 for inference, replayed from CUDA graphs.

    ``encoder(input_features, attention_mask)`` takes [B, T, 160] SeamlessM4T features (``extract_features`` gives
    them, with the mask) and a [B, T] mask (1 for valid frames, the padding at the end) or None, and returns the
    hidden states of ``output_layers``: one [B, T, 1024] tensor, or a dict {layer: tensor} when several are asked
    for (0 is the encoder input, i the output of layer i). ``encoder.from_waveforms(waves)`` does the feature
    extraction too.

    model: a ``Wav2Vec2BertModel`` (used in place) or a name for ``from_pretrained``. num_layers: keep only the first
    layers. frame_step: frames are rounded up to a multiple of it (1 frame = 20 ms; 25 = 0.5 s). batch_sizes: the batch
    buckets; larger batches run in chunks of the largest. max_graphs: graphs kept (least recently used dropped).
    copy: return fresh tensors (default); with copy=False the results are views of the graph's static outputs, valid
    until the next call. cuda_graphs=False runs the same bucketed computation without graphs (for comparison)."""

    def __init__(self, model="facebook/w2v-bert-2.0", *, num_layers: int | None = None, output_layers=None,
                 dtype: torch.dtype = torch.bfloat16, device: torch.device | str = "cuda", frame_step: int = 25,
                 batch_sizes=(1, 2, 4, 8, 16), max_graphs: int = 32, copy: bool = True, cuda_graphs: bool = True):
        if isinstance(model, str):
            from transformers import Wav2Vec2BertModel

            model = Wav2Vec2BertModel.from_pretrained(model)
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("InferenceEncoder needs a CUDA device")
        model = model.eval()
        model.config.layerdrop = 0.0
        optimize(model, num_layers=num_layers)
        self.model = optimize(model.to(self.device, dtype), compile=True)
        self.num_layers = len(self.model.encoder.layers)
        self.output_layers = sorted(set([self.num_layers] if output_layers is None else output_layers))
        if any(not 0 <= i <= self.num_layers for i in self.output_layers):
            raise ValueError(f"output_layers must be in 0..{self.num_layers}")
        self.dtype = dtype
        self.frame_step = frame_step
        self.batch_sizes = sorted(set(batch_sizes))
        self.max_graphs = max_graphs
        self.copy = copy
        self.cuda_graphs = cuda_graphs
        self.width = self.model.config.feature_projection_input_dim
        self._graphs: collections.OrderedDict = collections.OrderedDict()
        self._pool = None

    def _forward(self, features: Tensor, mask: Tensor) -> tuple[Tensor, ...]:
        everything = self.output_layers != [self.num_layers]
        out = self.model(features, attention_mask=mask, output_hidden_states=everything)
        if not everything:
            return (out.last_hidden_state,)
        return tuple(out.hidden_states[i] for i in self.output_layers)

    def _bucket(self, batch: int, frames: int) -> tuple[int, int]:
        rows = next((size for size in self.batch_sizes if size >= batch), self.batch_sizes[-1])
        return rows, self.frame_step * math.ceil(frames / self.frame_step)

    def _entry(self, rows: int, frames: int):
        key = (rows, frames)
        if key in self._graphs:
            self._graphs.move_to_end(key)
            return self._graphs[key]
        features = torch.zeros(rows, frames, self.width, device=self.device, dtype=self.dtype)
        mask = torch.ones(rows, frames, device=self.device, dtype=torch.long)
        current = torch.cuda.current_stream(self.device)
        side = torch.cuda.Stream(self.device)
        side.wait_stream(current)
        with torch.cuda.stream(side), torch.no_grad():
            for _ in range(2):  # compilation and kernel autotuning happen here, not under capture
                self._forward(features, mask)
        current.wait_stream(side)
        torch.cuda.synchronize(self.device)
        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(graph, pool=self._pool):
            outputs = self._forward(features, mask)
        self._graphs[key] = (graph, features, mask, outputs)
        while len(self._graphs) > self.max_graphs:
            self._graphs.popitem(last=False)
        return self._graphs[key]

    def _run_chunk(self, features: Tensor, mask: Tensor | None) -> tuple[Tensor, ...]:
        batch, frames, _ = features.shape
        rows, padded = self._bucket(batch, frames)
        if self.cuda_graphs:
            graph, static_features, static_mask, outputs = self._entry(rows, padded)
        else:
            static_features = torch.zeros(rows, padded, self.width, device=self.device, dtype=self.dtype)
            static_mask = torch.ones(rows, padded, device=self.device, dtype=torch.long)
        static_features[:batch, :frames].copy_(features)
        static_features[:batch, frames:].zero_()
        static_features[batch:].zero_()
        static_mask[batch:].fill_(1)  # unused rows: whole sequences, so nothing divides by an empty key set
        static_mask[:batch, frames:].zero_()
        if mask is None:
            static_mask[:batch, :frames].fill_(1)
        else:
            static_mask[:batch, :frames].copy_(mask)
        if self.cuda_graphs:
            graph.replay()
        else:
            with torch.no_grad():
                outputs = self._forward(static_features, static_mask)
        return tuple(out[:batch, :frames] for out in outputs)

    @torch.no_grad()
    def __call__(self, input_features: Tensor, attention_mask: Tensor | None = None):
        features = input_features.to(self.device)
        mask = None if attention_mask is None else attention_mask.to(self.device)
        largest = self.batch_sizes[-1]
        chunks = []
        for start in range(0, features.size(0), largest):
            part = self._run_chunk(features[start:start + largest],
                                   None if mask is None else mask[start:start + largest])
            chunks.append(tuple(t.clone() for t in part) if self.copy or features.size(0) > largest else part)
        outputs = [torch.cat(parts) if len(chunks) > 1 else parts[0] for parts in zip(*chunks)]
        if len(outputs) == 1:
            return outputs[0]
        return dict(zip(self.output_layers, outputs))

    def from_waveforms(self, waves, lengths: Tensor | None = None, sampling_rate: int = 16000):
        """The same from 16 kHz waveforms (a list of 1-D tensors, or [B, N] with lengths): features computed on the
        GPU (``extract_features``) first."""
        from .features import extract_features

        inputs = extract_features(waves, lengths, sampling_rate=sampling_rate, device=self.device)
        return self(inputs["input_features"], inputs["attention_mask"])
