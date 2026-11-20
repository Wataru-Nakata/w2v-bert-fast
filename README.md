# w2v-bert-fast

Faster training and inference for [w2v-BERT 2.0](https://huggingface.co/facebook/w2v-bert-2.0)
(Hugging Face `Wav2Vec2BertModel`), with the same weights, the same state-dict keys and the same outputs.

```bash
pip install git+https://github.com/Wataru-Nakata/w2v-bert-fast
```

```python
import torch
import w2vbert_fast

# the pretrained model with the fast modules, each encoder layer compiled
model = w2vbert_fast.load("facebook/w2v-bert-2.0", device="cuda", dtype=torch.bfloat16, compile=True)

# SeamlessM4T features computed on the GPU (the values of the Hugging Face feature extractor)
inputs = w2vbert_fast.extract_features(waveforms, device="cuda")  # a list of 1-D 16 kHz tensors
with torch.no_grad():
    hidden = model(inputs["input_features"].to(torch.bfloat16),
                   attention_mask=inputs["attention_mask"]).last_hidden_state
```

A model you already have is sped up in place (its state-dict keys do not change), here keeping its first 8 layers:

```python
from transformers import Wav2Vec2BertModel

model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0").to("cuda", torch.bfloat16)
w2vbert_fast.optimize(model, num_layers=8, compile=True)
```

## What changes

`optimize()` swaps the classes of the encoder's self-attention and convolution modules for subclasses with a
faster `forward`. Parameters, buffers and their names stay where they were, so `state_dict()`, `load_state_dict()`,
optimizers, `isinstance` checks and wrappers keep working.

* **Relative-key attention in fused Triton kernels.** w2v-BERT 2.0 adds a learned relative-position term
  `q_i · E[clip(j - i, -64, 8)]` to every attention score, which rules out FlashAttention and makes the Hugging Face
  code build a `[T, T, 64]` position tensor. The kernels here compute the exact same attention in FlashAttention
  style: outside a 73-frame band the term is a per-row constant, so most key blocks need nothing beyond one extra
  multiply-add, and only the band reads the per-distance scores. The backward returns the gradients of the
  queries, keys, values and the distance embedding. Padding masks become per-sequence key lengths inside the kernel.
* **One projection for q, k and v,** whose output the attention kernel reads in place, and whose gradient it
  writes in place.
* **The causal depthwise convolution (kernel 31) in channels-last Triton kernels,** without the two transposes,
  and the 1x1 convolutions as matrix products.
* **`compile=True`** compiles each encoder layer with `torch.compile`; the Triton kernels are registered custom
  operators, so the layers compile without graph breaks.
* **`num_layers=n`** keeps only the first `n` layers, for models that read an intermediate layer.
* **`extract_features`** computes the SeamlessM4T log-mel features (the `facebook/w2v-bert-2.0` preprocessor) in
  batch on any device.

Padded batches (the feature extractor's `attention_mask`) are handled inside the attention kernel as per-sequence
key lengths, computed on the GPU without synchronizing; the padding must come at the end of each sequence, as the
Hugging Face feature extractor produces it (checked on the device).

The original code is used where the fast path does not apply: `output_attentions=True`, attention dropout above
zero in training, other position-embedding types, CPU tensors (attention falls back to an exact banded
decomposition), and projection or convolution layers wrapped by adapters such as PEFT LoRA (the wrapped modules are
called as they are, so the adapters keep working).

## Exactness

The tests compare against the Hugging Face implementation:

* the whole model, float64 on CPU: outputs and every parameter gradient, with and without padding, with LoRA
  adapters, truncated, and the fallbacks;
* the Triton attention against a float64 reference: fp32 with IEEE dot products to a relative error of `1e-3` (at
  every region boundary), and bf16 to `2e-2` at 300 and 1500 frames, with and without key lengths;
* the convolution and the feature extractor against PyTorch and the Hugging Face extractor;
* the custom operators with `torch.library.opcheck`.

```bash
python -m unittest discover -s tests
```

## Speed

All 24 layers of `facebook/w2v-bert-2.0`, batch 16, bf16, one H200. Milliseconds per step and model FLOPs
utilization (MFU) against the H200's dense bf16 peak of 989.4 TFLOPS. Training is a forward and backward pass with
every parameter trainable and the pretrained configuration's dropout (layer drop and SpecAugment off). 500 and 1500
encoder frames are 10 s and 30 s of audio.

| | 10 s, inference | 10 s, training | 30 s, inference | 30 s, training |
|---|---|---|---|---|
| Hugging Face | 62.2 ms (15.7%) | 190 ms (15.4%) | 320 ms (9.9%) | 1035 ms (9.2%) |
| `optimize(model)` | 31.7 ms (30.8%) | 102 ms (28.8%) | 101 ms (31.3%) | 322 ms (29.5%) |
| `optimize(model, compile=True)` | **28.3 ms (34.5%)** | **93.1 ms (31.5%)** | **90.7 ms (34.9%)** | **298 ms (31.9%)** |

That is 2.2x (10 s) to 3.5x (30 s) faster inference and 2.0x to 3.5x faster training than the Hugging Face model,
and 42 GiB instead of 76 GiB of memory for a 30 s training step. (Compiling the Hugging Face layers with
`torch.compile` alone speeds up its inference, to 43.8 ms and 198 ms, but made its training slower in our runs.) The relative-key attention itself runs at
0.69 ms forward and 2.9 ms forward + backward for 16 x 16 heads x 1500 frames, against 0.55 ms and 2.5 ms for
PyTorch's FlashAttention *without* the relative term.

MFU counts the attention as dense (`4 T d` FLOPs per token and layer). cuBLAS itself reaches 52-65% of the peak on
this model's matrix shapes (24000 x 1024 x 4096 and the like; `benchmarks/bench_gemm.py`) and FlashAttention
without the relative term 18-27% (backward and forward), so the matrix products and the attention floor alone put the ceiling near
42% for inference and 36% for training on this GPU. An Nsight Systems timeline of back-to-back steps shows the GPU
busy 98.5% of the time (`benchmarks/steady_state.py`): what remains is kernel time.

```bash
# one process per variant: compiled variants share dynamo's per-code recompile budget
python benchmarks/bench_model.py --batch 16 --frames 500 1500 --variants triton+compile
python benchmarks/bench_attention.py --batch 16 --frames 500 1500
```

`compile=True` raises dynamo's recompile limit to 64 (`torch._dynamo.config.recompile_limit`): every compiled layer
runs the same `forward`, so modes, sequence lengths and other compiled models in the process share the default
budget of 8, past which layers silently run uncompiled.

## Requirements

PyTorch 2.4 or later, transformers 4.40 or later, and for the fast GPU path Triton (bundled with PyTorch on Linux)
and an NVIDIA GPU. Head sizes 16, 32, 64 and 128 are supported (w2v-BERT 2.0 uses 64).

## License

Apache-2.0
