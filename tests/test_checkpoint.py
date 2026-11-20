"""The real facebook/w2v-bert-2.0 checkpoint on a GPU, against the Hugging Face model: every layer's hidden states
and parameter gradients in fp32 (tight), bf16 with compile=True (no further from fp32 than Hugging Face's own bf16),
consecutive batches with different padding, LoRA-wrapped projections, and two compiled models in one process.
Skipped without CUDA or without the checkpoint in the local Hugging Face cache."""

import copy
import logging
import unittest

import torch

from w2vbert_fast import optimize


def relative(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm())


def masks(frames):
    """Two padded batches of the same shape (different lengths) and an unpadded one."""
    out = []
    for lengths in ((frames, frames * 3 // 4, frames // 2, 50), (frames // 3, frames, 77, frames - 1)):
        mask = torch.zeros(4, frames, dtype=torch.long, device="cuda")
        for row, length in enumerate(lengths):
            mask[row, :length] = 1
        out.append(mask)
    out.append(None)
    return out


@unittest.skipUnless(torch.cuda.is_available(), "CUDA")
class CheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import Wav2Vec2BertModel

        try:
            base = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0", local_files_only=True)
        except OSError:
            raise unittest.SkipTest("facebook/w2v-bert-2.0 is not in the local cache")
        base.config.layerdrop = 0.0
        base.config.mask_time_prob = 0.0
        cls.base = base.eval()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        generator = torch.Generator(device="cuda").manual_seed(0)
        cls.frames = 600
        cls.features = torch.randn(4, cls.frames, 160, device="cuda", generator=generator)

    def hidden_states(self, model, dtype, mask):
        with torch.no_grad():
            states = model(self.features.to(dtype), attention_mask=mask, output_hidden_states=True).hidden_states
        valid = 1.0 if mask is None else mask[..., None].to(dtype)
        return [h * valid for h in states]

    def test_fp32_layers_and_gradients_match(self):
        reference = copy.deepcopy(self.base).cuda()
        fast = optimize(copy.deepcopy(self.base)).cuda()
        for mask in masks(self.frames):
            a = self.hidden_states(reference, torch.float32, mask)
            b = self.hidden_states(fast, torch.float32, mask)
            for layer, (x, y) in enumerate(zip(a, b)):
                self.assertLess(relative(y, x), 1e-4, f"layer {layer} padded={mask is not None}")
        # gradients of the whole model for a loss on the valid positions of a padded batch
        mask = masks(self.frames)[0]
        weight = torch.randn(4, self.frames, 1024, device="cuda") * mask[..., None]
        grads = []
        for model in (reference, fast):
            model.zero_grad(set_to_none=True)
            (model(self.features, attention_mask=mask).last_hidden_state * weight).sum().backward()
            grads.append({n: p.grad for n, p in model.named_parameters() if p.grad is not None})
        self.assertEqual(set(grads[0]), set(grads[1]))
        names = sorted(grads[0])
        everything = [torch.cat([g[n].flatten() for n in names]) for g in grads]
        self.assertLess(relative(everything[1], everything[0]), 1e-4)
        # per parameter, against a floor of the largest gradient's scale (some gradients are near zero)
        scale = max(float(grads[0][n].norm()) for n in names)
        for name in names:
            error = float((grads[1][name] - grads[0][name]).norm()) / (float(grads[0][name].norm()) + 1e-3 * scale)
            self.assertLess(error, 1e-3, name)

    def test_bf16_compiled_is_as_close_to_fp32_as_hugging_face_bf16(self):
        reference = copy.deepcopy(self.base).cuda()
        hf_bf16 = copy.deepcopy(self.base).to("cuda", torch.bfloat16)
        fast = optimize(copy.deepcopy(self.base).to("cuda", torch.bfloat16), compile=True)
        for mask in masks(self.frames):
            exact = self.hidden_states(reference, torch.float32, mask)
            ours = self.hidden_states(fast, torch.bfloat16, mask)
            theirs = self.hidden_states(hf_bf16, torch.bfloat16, mask)
            for layer, (x, y, z) in enumerate(zip(exact, ours, theirs)):
                self.assertLess(relative(y, x), 1.5 * relative(z, x) + 1e-3,
                                f"layer {layer} padded={mask is not None}: ours {relative(y, x):.3g}, "
                                f"Hugging Face bf16 {relative(z, x):.3g}")

    def test_lora_wrapped_projections(self):
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            self.skipTest("peft is not installed")
        config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, bias="none",
                            target_modules=["linear_q", "linear_k", "linear_v", "intermediate_dense"])
        reference = get_peft_model(copy.deepcopy(self.base), config).cuda().eval()
        torch.manual_seed(0)
        for name, parameter in reference.named_parameters():
            if "lora_B" in name:
                torch.nn.init.normal_(parameter, std=0.02)
        fast = copy.deepcopy(reference)
        optimize(fast.base_model.model)
        mask = masks(self.frames)[1]
        a = self.hidden_states(reference, torch.float32, mask)
        b = self.hidden_states(fast, torch.float32, mask)
        for layer, (x, y) in enumerate(zip(a, b)):
            self.assertLess(relative(y, x), 1e-4, f"layer {layer}")

    def test_two_compiled_models_stay_compiled(self):
        records = []

        class Catch(logging.Handler):
            def emit(self, record):
                records.append(record.getMessage())

        handler = Catch(level=logging.WARNING)
        logging.getLogger("torch._dynamo").addHandler(handler)
        try:
            short = optimize(copy.deepcopy(self.base).to("cuda", torch.bfloat16), num_layers=8, compile=True)
            full = optimize(copy.deepcopy(self.base).to("cuda", torch.bfloat16), compile=True)
            grad = torch.ones(4, self.frames, 1024, device="cuda", dtype=torch.bfloat16)
            for frames in (self.frames, self.frames // 2):
                features = self.features[:, :frames].to(torch.bfloat16)
                for model in (short, full):
                    model.eval()
                    with torch.no_grad():
                        model(features)
                    model.train()
                    model(features).last_hidden_state.backward(grad[:, :frames])
        finally:
            logging.getLogger("torch._dynamo").removeHandler(handler)
        limits = [m for m in records if "recompile_limit" in m or "cache_size_limit" in m]
        self.assertEqual(limits, [])


if __name__ == "__main__":
    unittest.main()
