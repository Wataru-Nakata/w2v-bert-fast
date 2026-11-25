"""InferenceEncoder (CUDA graphs) against the optimized model run eagerly (per-layer torch.compile) and the Hugging
Face model: milliseconds per call (results copied out), MFU, and the distance to the Hugging Face fp32 output.

    python benchmarks/bench_inference.py --layers 24 --shapes 1x150 1x500 1x1500 1x6000 16x500 16x1500
"""

import argparse
import copy
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from bench_model import model_flops, peak, timed  # noqa: E402

from w2vbert_fast import InferenceEncoder, optimize  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=24)
    parser.add_argument("--shapes", nargs="+", default=["1x500", "16x1500"])
    parser.add_argument("--frame-step", type=int, default=25)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--no-hf", action="store_true", help="skip the Hugging Face eager baseline")
    args = parser.parse_args()
    from transformers import Wav2Vec2BertModel

    base = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0").eval()
    base.encoder.layers = base.encoder.layers[:args.layers]
    base.config.num_hidden_layers = args.layers
    base.config.layerdrop = 0.0
    encoder = InferenceEncoder(copy.deepcopy(base), frame_step=args.frame_step)
    compiled = optimize(copy.deepcopy(base), compile=True).to("cuda", torch.bfloat16).eval()
    reference = copy.deepcopy(base).cuda()
    hf = copy.deepcopy(base).to("cuda", torch.bfloat16)
    for shape in args.shapes:
        batch, frames = (int(v) for v in shape.split("x"))
        torch.manual_seed(0)
        features = torch.randn(batch, frames, 160, device="cuda")
        mask = torch.ones(batch, frames, device="cuda", dtype=torch.long)
        for row in range(1, batch):
            mask[row, frames - row * frames // (2 * batch):] = 0
        valid = mask[..., None].float()
        with torch.no_grad():
            exact = reference(features, attention_mask=mask).last_hidden_state * valid
        flops = model_flops(base.config, batch, frames)

        def error(out):
            return round(float((out.float() * valid - exact).norm() / exact.norm()), 4)

        runs = {"InferenceEncoder": lambda: encoder(features, mask),
                "optimize+compile": lambda: compiled(features.to(torch.bfloat16), attention_mask=mask).last_hidden_state}
        if not args.no_hf:
            runs["huggingface"] = lambda: hf(features.to(torch.bfloat16), attention_mask=mask).last_hidden_state
        for name, fn in runs.items():
            with torch.no_grad():
                out = fn()
                seconds = timed(fn, args.repeat)[0]
            print(json.dumps({"layers": args.layers, "batch": batch, "frames": frames, "impl": name,
                              "ms": round(seconds * 1e3, 3), "mfu": round(flops / seconds / peak(), 4),
                              "err_vs_fp32": error(out)}), flush=True)


if __name__ == "__main__":
    main()
