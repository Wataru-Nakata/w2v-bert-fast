"""The README's examples, run as written (random 16 kHz noise for the waveforms), checked against the Hugging Face
model and feature extractor."""

import torch
from transformers import AutoFeatureExtractor, Wav2Vec2BertModel

import w2vbert_fast

torch.manual_seed(0)
waveforms = [torch.randn(16000 * 7) * 0.1, torch.randn(16000 * 4) * 0.1]

model = w2vbert_fast.load("facebook/w2v-bert-2.0", device="cuda", dtype=torch.bfloat16, compile=True)
inputs = w2vbert_fast.extract_features(waveforms, device="cuda")
with torch.no_grad():
    hidden = model(inputs["input_features"].to(torch.bfloat16),
                   attention_mask=inputs["attention_mask"]).last_hidden_state

extractor = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
reference_inputs = extractor([w.numpy() for w in waveforms], sampling_rate=16000, return_tensors="pt", padding=True)
reference = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0").to("cuda", torch.bfloat16).eval()
with torch.no_grad():
    expected = reference(reference_inputs["input_features"].to("cuda", torch.bfloat16),
                         attention_mask=reference_inputs["attention_mask"].cuda()).last_hidden_state
valid = inputs["attention_mask"][..., None]
error = float(((hidden.float() - expected.float()) * valid).norm() / (expected.float() * valid).norm())
print(f"example 1: hidden {tuple(hidden.shape)}, relative difference to Hugging Face bf16: {error:.3g}")
assert hidden.shape == expected.shape and error < 0.05, error

short = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0").to("cuda", torch.bfloat16)
w2vbert_fast.optimize(short, num_layers=8, compile=True)
out = short(inputs["input_features"].to(torch.bfloat16), attention_mask=inputs["attention_mask"]).last_hidden_state
print(f"example 2: {len(short.encoder.layers)} layers, hidden {tuple(out.shape)}")

encoder = w2vbert_fast.InferenceEncoder("facebook/w2v-bert-2.0")
graphed = encoder.from_waveforms(waveforms)
error = float(((graphed.float() - expected.float()) * valid).norm() / (expected.float() * valid).norm())
print(f"example 3: InferenceEncoder {tuple(graphed.shape)}, relative difference to Hugging Face bf16: {error:.3g}")
assert graphed.shape == expected.shape and error < 0.05, error

encoder = w2vbert_fast.InferenceEncoder("facebook/w2v-bert-2.0", num_layers=20, output_layers=[8, 20])
layers = encoder.from_waveforms(waveforms)
print(f"example 4: layers {sorted(layers)}, shapes {[tuple(v.shape) for v in layers.values()]}")
assert sorted(layers) == [8, 20]
