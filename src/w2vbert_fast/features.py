"""Batched, device-side SeamlessM4T features for w2v-BERT 2.0 (``SeamlessM4TFeatureExtractor`` of Hugging Face
transformers, the ``facebook/w2v-bert-2.0`` preprocessor), computed with torch on any device.

Per waveform (16 kHz, float in [-1, 1]), as the reference does:
* scaled by 2**15 (Kaldi's int16 convention);
* 25 ms frames every 10 ms without centering (``1 + (n - 400) // 160`` frames);
* DC removal, pre-emphasis 0.97 (the first sample scaled by 0.03), Povey window, 512-point power spectrum;
* 80 Kaldi mel bins (20 Hz to 8 kHz), ``log(max(mel, 1.1920929e-07))``;
* per-bin normalization over the waveform's own frames, unbiased variance + 1e-7;
* padded to the longest (padding value 1.0, as the preprocessor config sets) and to an even frame count, then pairs
  of frames stacked to 160 features; the attention mask keeps every second frame.

The mel bank and the window come from transformers' own audio utilities, so the constants are the reference's.
"""

from __future__ import annotations

import torch
from torch import Tensor

SAMPLE_RATE = 16000
FRAME, HOP, FFT = 400, 160, 512
STRIDE = 2
PADDING_VALUE = 1.0
_CONSTANTS: dict = {}


def _constants(device, dtype) -> tuple[Tensor, Tensor]:
    key = (str(device), dtype)
    if key not in _CONSTANTS:
        from transformers.audio_utils import mel_filter_bank, window_function

        mel = mel_filter_bank(num_frequency_bins=FFT // 2 + 1, num_mel_filters=80, min_frequency=20,
                              max_frequency=SAMPLE_RATE // 2, sampling_rate=SAMPLE_RATE, norm=None,
                              mel_scale="kaldi", triangularize_in_mel_space=True)
        window = window_function(FRAME, "povey", periodic=False)
        _CONSTANTS[key] = (torch.as_tensor(window, dtype=dtype, device=device),
                           torch.as_tensor(mel, dtype=dtype, device=device))
    return _CONSTANTS[key]


def log_mel(waves: Tensor) -> Tensor:
    """[B, N] 16 kHz waveforms (zero padded) -> [B, frames, 80] log mel energies of every full frame."""
    window, mel = _constants(waves.device, waves.dtype)
    frames = (waves * 32768.0).unfold(-1, FRAME, HOP)  # [B, F, 400]
    frames = frames - frames.mean(-1, keepdim=True)
    frames = torch.cat((frames[..., :1] * (1 - 0.97), frames[..., 1:] - 0.97 * frames[..., :-1]), dim=-1)
    power = torch.fft.rfft(frames * window, n=FFT).abs().pow(2)
    return torch.matmul(power, mel).clamp_min(1.1920929e-07).log()


def frame_count(samples: Tensor) -> Tensor:
    return torch.where(samples >= FRAME, 1 + (samples - FRAME) // HOP, torch.zeros_like(samples))


def extract_features(waves: Tensor | list[Tensor], lengths: Tensor | None = None, sampling_rate: int = SAMPLE_RATE,
                     device: torch.device | str | None = None, dtype: torch.dtype = torch.float32
                     ) -> dict[str, Tensor]:
    """w2v-BERT 2.0 inputs from waveforms: a list of 1-D tensors, or a [B, N] tensor with optional lengths [B].

    Returns ``input_features`` [B, T, 160] and ``attention_mask`` [B, T] (long), as the reference extractor's
    ``__call__(..., return_tensors="pt")``. Inputs at another sampling rate are resampled with torchaudio first.
    Each waveform must hold at least one 25 ms frame (400 samples at 16 kHz)."""
    if isinstance(waves, (list, tuple)):
        lengths = torch.tensor([w.numel() for w in waves])
        waves = torch.nn.utils.rnn.pad_sequence([w.reshape(-1) for w in waves], batch_first=True)
    if waves.ndim != 2:
        raise ValueError("waves must be [B, N] or a list of 1-D tensors")
    if device is not None:
        waves = waves.to(device)
    waves = waves.to(dtype)
    lengths = (torch.full((waves.size(0),), waves.size(1)) if lengths is None else lengths).to(waves.device).long()
    if sampling_rate != SAMPLE_RATE:
        import torchaudio

        waves = torchaudio.functional.resample(waves, sampling_rate, SAMPLE_RATE)
        lengths = (lengths * SAMPLE_RATE + sampling_rate - 1) // sampling_rate
        lengths = lengths.clamp_max(waves.size(1))
    frames = frame_count(lengths)
    if bool((frames < 2).any()):
        raise ValueError("every waveform needs at least two 25 ms frames (560 samples at 16 kHz)")
    features = log_mel(waves)  # [B, F, 80]
    valid = torch.arange(features.size(1), device=waves.device)[None, :] < frames[:, None]
    count = frames[:, None, None].to(features.dtype)
    masked = features.masked_fill(~valid[..., None], 0.0)
    mean = masked.sum(1, keepdim=True) / count
    var = ((features - mean).masked_fill(~valid[..., None], 0.0).pow(2).sum(1, keepdim=True)) / (count - 1)
    features = ((features - mean) / torch.sqrt(var + 1e-7)).masked_fill(~valid[..., None], PADDING_VALUE)
    longest = int(frames.max())
    even = longest + longest % STRIDE
    features = features[:, :longest]
    valid = valid[:, :longest]
    if even > longest:
        features = torch.nn.functional.pad(features, (0, 0, 0, even - longest), value=PADDING_VALUE)
        valid = torch.nn.functional.pad(valid, (0, even - longest), value=False)
    batch = features.size(0)
    return {"input_features": features.reshape(batch, even // STRIDE, 80 * STRIDE),
            "attention_mask": valid[:, 1::STRIDE].long()}
