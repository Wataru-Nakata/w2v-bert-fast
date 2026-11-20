"""Where the time goes: torch.profiler over a forward (inference) or a forward + backward (training) of w2v-BERT
2.0 in bf16 on CUDA; CUDA kernel time summed by kernel name, and by encoder sub-module (ffn, attention, conv
module, layer norms) through record_function scopes.

    python benchmarks/profile_model.py --batch 16 --frames 500 --variant triton --mode train
"""

import argparse
import collections

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from w2vbert_fast import optimize


def scoped(module, name):
    forward = module.forward

    def wrapped(*args, **kwargs):
        with record_function(f"scope::{name}"):
            return forward(*args, **kwargs)

    module.forward = wrapped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--frames", type=int, default=500)
    parser.add_argument("--variant", default="triton")
    parser.add_argument("--mode", choices=["infer", "train"], default="train")
    args = parser.parse_args()
    from transformers import Wav2Vec2BertModel

    model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0")
    model.config.layerdrop = 0.0
    model.config.mask_time_prob = 0.0
    if args.variant != "eager":
        optimize(model, backend=args.variant.split("+")[0], compile="compile" in args.variant)
    model = model.to("cuda", torch.bfloat16)
    if "compile" not in args.variant:  # the scope wrappers would make dynamo recompile every layer
        for layer in model.encoder.layers:
            for name in ("ffn1", "ffn1_layer_norm", "self_attn_layer_norm", "self_attn", "conv_module", "ffn2",
                         "ffn2_layer_norm", "final_layer_norm"):
                scoped(getattr(layer, name), name)
    features = torch.randn(args.batch, args.frames, 160, device="cuda", dtype=torch.bfloat16)
    grad = torch.randn(args.batch, args.frames, 1024, device="cuda", dtype=torch.bfloat16)

    def step():
        if args.mode == "infer":
            model.eval()
            with torch.no_grad():
                model(features)
        else:
            model.train()
            model.zero_grad(set_to_none=True)
            model(features).last_hidden_state.backward(grad)

    for _ in range(3):
        step()
    torch.cuda.synchronize()
    # report every host-device synchronization inside a step (each one drains the launch queue)
    torch.cuda.set_sync_debug_mode("warn")
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        step()
    torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()
    syncs = collections.Counter(f"{w.filename}:{w.lineno}" for w in caught if "synchroniz" in str(w.message).lower())
    print(f"== synchronizing calls in one step: {sum(syncs.values())}")
    for message, count in syncs.most_common(5):
        print(f"{count:6d}x  {message}")
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
        step()
        torch.cuda.synchronize()
    events = prof.key_averages()
    total = sum(e.device_time_total for e in events if e.device_type.name == "CUDA") or 1
    kernels = sorted((e for e in events if e.device_type.name == "CUDA"), key=lambda e: -e.self_device_time_total)
    print(f"== {args.variant} {args.mode} B={args.batch} T={args.frames}: CUDA kernel time {total / 1e3:.1f} ms")
    for e in kernels[:30]:
        print(f"{e.self_device_time_total / 1e3:9.2f} ms {100 * e.self_device_time_total / total:5.1f}%  "
              f"{e.count:6d}x  {e.key[:110]}")
    # idle device time between consecutive kernels (launch or host gaps), and around which kernels it falls
    spans = sorted((e.time_range.start, e.time_range.end, e.name) for e in prof.events()
                   if e.device_type.name == "CUDA" and e.time_range.end > e.time_range.start)
    gaps = collections.Counter()
    idle = 0.0
    for (s0, e0, n0), (s1, e1, n1) in zip(spans, spans[1:]):
        gap = s1 - max(e0, s0)
        if gap > 2:  # microseconds
            idle += gap
            gaps[f"{n0[:50]} -> {n1[:50]}"] += gap
    wall = spans[-1][1] - spans[0][0] if spans else 0
    print(f"== device span {wall / 1e3:.1f} ms, idle gaps {idle / 1e3:.1f} ms; largest gap sources:")
    for name, value in gaps.most_common(12):
        print(f"{value / 1e3:9.2f} ms  {name}")
    scopes = collections.Counter()
    for e in events:
        if e.key.startswith("scope::"):
            scopes[e.key[7:]] += e.device_time_total
    print("== by sub-module (forward scopes; device time incl. children)")
    for name, value in scopes.most_common():
        print(f"{value / 1e3:9.2f} ms  {name}")


if __name__ == "__main__":
    main()
