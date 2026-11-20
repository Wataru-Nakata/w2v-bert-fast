"""Whole-model speed and MFU of w2v-BERT 2.0 (facebook/w2v-bert-2.0, 24 layers) on CUDA in bf16.

MFU = model FLOPs / time / peak. Model FLOPs are counted analytically (matmul and attention FLOPs only, 2 per
multiply-add; a training step = 3 x forward): per token and layer, two feed-forward blocks (2 x 2 x 2 x d x 4d),
the q/k/v/out projections (4 x 2 x d^2), attention (2 x 2 x T x d), the convolution module's pointwise layers
(2 x d x 2d + 2 x d x d) and its depthwise convolution (2 x d x 31), plus the 160 -> d feature projection.

    python benchmarks/bench_model.py --batch 16 --frames 1500 --variants eager triton triton+compile
"""

import argparse
import copy
import json

import torch

from w2vbert_fast import optimize

PEAK = {"H200": 989.4e12, "H100": 989.4e12, "A100": 312e12}


def model_flops(config, batch: int, frames: int) -> float:
    d, f, layers = config.hidden_size, config.intermediate_size, config.num_hidden_layers
    per_token = (2 * 2 * 2 * d * f + 4 * 2 * d * d + 2 * 2 * frames * d + 2 * d * 2 * d + 2 * d * d
                 + 2 * d * config.conv_depthwise_kernel_size) * layers + 2 * config.feature_projection_input_dim * d
    return per_token * batch * frames


def peak() -> float:
    name = torch.cuda.get_device_name()
    return next((value for key, value in PEAK.items() if key in name), 989.4e12)


def timed(fn, repeat):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeat / 1e3, torch.cuda.max_memory_allocated() / 2**30


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--frames", type=int, nargs="+", default=[1500])
    parser.add_argument("--variants", nargs="+", default=["eager", "sdpa", "flash", "triton", "triton+compile"])
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--pad", action="store_true", help="a padded batch: lengths spread over 50-100%% of frames")
    args = parser.parse_args()
    from transformers import Wav2Vec2BertModel

    base = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0")
    base.config.layerdrop = 0.0
    base.config.mask_time_prob = 0.0
    for frames in args.frames:
        flops = model_flops(base.config, args.batch, frames)
        mask = None
        if args.pad:
            lengths = torch.linspace(frames // 2, frames, args.batch, device="cuda").long()
            mask = (torch.arange(frames, device="cuda")[None, :] < lengths[:, None]).long()
        features = torch.randn(args.batch, frames, base.config.feature_projection_input_dim, device="cuda",
                               dtype=torch.bfloat16)
        for variant in args.variants:
            model = copy.deepcopy(base)
            parts = variant.split("+")
            if parts[0] != "eager":
                optimize(model, backend=parts[0], compile="compile" in parts)
            model = model.to("cuda", torch.bfloat16)
            if parts[0] == "eager" and "compile" in parts:  # the unmodified model, each layer torch.compile'd
                for layer in model.encoder.layers:
                    layer.compile()
            # whole-model compilation: +model (default mode), +graphs (CUDA graphs), +maxat (max-autotune GEMMs)
            mode = {"model": "default", "graphs": "reduce-overhead", "maxat": "max-autotune-no-cudagraphs"}
            for part in parts[1:]:
                if part in mode:
                    model = torch.compile(model, mode=mode[part])
            row = {"frames": frames, "batch": args.batch, "variant": variant + ("+padded" if args.pad else "")}
            try:
                model.eval()
                def infer():
                    if "graphs" in parts:
                        torch.compiler.cudagraph_mark_step_begin()
                    return model(features, attention_mask=mask).last_hidden_state

                with torch.no_grad():
                    seconds, memory = timed(infer, args.repeat)
                row.update(infer_ms=seconds * 1e3, infer_mfu=flops / seconds / peak(), infer_GiB=memory)
                if not args.skip_train:
                    model.train()
                    grad = torch.randn(args.batch, frames, base.config.hidden_size, device="cuda", dtype=torch.bfloat16)

                    def step():
                        model.zero_grad(set_to_none=True)
                        model(features, attention_mask=mask).last_hidden_state.backward(grad)
                        if "graphs" in parts:
                            torch.compiler.cudagraph_mark_step_begin()

                    seconds, memory = timed(step, args.repeat)
                    row.update(train_ms=seconds * 1e3, train_mfu=3 * flops / seconds / peak(), train_GiB=memory)
            except torch.OutOfMemoryError:
                row["error"] = "OOM"
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
