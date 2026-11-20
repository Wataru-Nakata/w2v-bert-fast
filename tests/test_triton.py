"""The fused Triton kernels against the Hugging Face computation on CUDA: values and gradients (query, key, value,
distance embedding), with and without per-sequence key lengths, in fp32 (IEEE dots; tight) and bf16."""

import unittest

import torch

from w2vbert_fast.attention import relative_key_reference


def make(batch, heads, frames, dim, left, right, dtype, seed=0):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    q, k, v = (torch.randn(batch, heads, frames, dim, device="cuda", generator=generator).to(dtype).requires_grad_()
               for _ in range(3))
    e = torch.randn(left + right + 1, dim, device="cuda", generator=generator).to(dtype).requires_grad_()
    w = torch.randn(batch, heads, frames, dim, device="cuda", generator=generator)
    return q, k, v, e, w


def run(fn, q, k, v, e, w, mask=None):
    out = fn(q, k, v, e)
    weight = w if mask is None else w * mask[:, None, :, None]
    (out.to(weight.dtype) * weight).sum().backward()
    result = [out.detach().double()] + [t.grad.double().clone() for t in (q, k, v, e)]
    for t in (q, k, v, e):
        t.grad = None
    return result


_ALL_CONFIGS: dict = {}


def setUpModule():
    """Tune over two tile shapes only (the smallest and a large one), so that both code paths are checked without
    compiling and timing every configuration for every sequence-length bucket; test_every_tuning_configuration
    checks the values of all of them once."""
    if not torch.cuda.is_available():
        return
    from w2vbert_fast import triton_attention

    for kernel in (triton_attention._forward, triton_attention._backward_kv, triton_attention._backward_q):
        configs = kernel.configs
        _ALL_CONFIGS[id(kernel)] = list(configs)
        small = min(configs, key=lambda c: c.kwargs["BLOCK_M"] * c.kwargs["BLOCK_N"])
        large = max(configs, key=lambda c: c.kwargs["BLOCK_M"] * c.kwargs["BLOCK_N"])
        kernel.configs = [small, large]


@unittest.skipUnless(torch.cuda.is_available(), "CUDA")
class TritonTests(unittest.TestCase):
    def compare(self, frames, dtype, tolerance, lengths=None, left=64, right=8, batch=2, heads=2, dim=64):
        from w2vbert_fast.triton_attention import fused_relative_key_attention

        q, k, v, e, w = make(batch, heads, frames, dim, left, right, dtype, seed=frames)
        additive = None
        if lengths is not None:
            lengths = torch.tensor(lengths, device="cuda")
            keys = torch.arange(frames, device="cuda")[None, :] < lengths[:, None]
            additive = torch.zeros(batch, 1, frames, frames, device="cuda").masked_fill(~keys[:, None, None, :],
                                                                                          float("-inf"))
        fast = run(lambda *a: fused_relative_key_attention(*a, left, right, lengths), q, k, v, e, w)
        q64, k64, v64, e64 = (t.detach().double().requires_grad_() for t in (q, k, v, e))
        ref = run(lambda *a: relative_key_reference(*a, left, right, None if additive is None else additive.double()),
                  q64, k64, v64, e64, w.double())
        for name, a, b in zip(("out", "dq", "dk", "dv", "dE"), fast, ref):
            # relative to the reference's norm, floored so that a gradient that is ~0 (e.g. dq at T=1) is not
            # compared by its rounding noise alone
            error = float((a - b).norm() / (b.norm() + 1e-3 * b.numel() ** 0.5))
            self.assertLess(error, tolerance, f"{name} T={frames} {dtype} lengths={lengths}: {error:.3g}")

    def test_fp32_matches_tightly_across_region_boundaries(self):
        for frames in (1, 7, 8, 9, 63, 64, 65, 72, 73, 100, 129, 300):
            self.compare(frames, torch.float32, 1e-3)

    def test_bf16_matches_long_sequences(self):
        for frames in (300, 1500):
            self.compare(frames, torch.bfloat16, 2e-2)

    def test_key_lengths_mask_padded_keys(self):
        self.compare(150, torch.float32, 1e-3, lengths=[150, 37])
        self.compare(1000, torch.bfloat16, 2e-2, lengths=[1000, 517])

    def test_every_tuning_configuration(self):
        # the tuner may pick any configuration (register caps, narrow band slices): check each one's values
        from w2vbert_fast import triton_attention

        kernels = (triton_attention._forward, triton_attention._backward_kv, triton_attention._backward_q)
        saved = [kernel.configs for kernel in kernels]
        try:
            for kernel, configs, name in zip(kernels, saved, ("forward", "backward_kv", "backward_q")):
                for config in _ALL_CONFIGS[id(kernel)]:
                    kernel.configs = [config]
                    kernel.cache.clear()
                    with self.subTest(kernel=name, config=str(config)):
                        self.compare(300, torch.bfloat16, 2e-2, lengths=[300, 211])
                kernel.configs = configs
                kernel.cache.clear()
        finally:
            for kernel, configs in zip(kernels, saved):
                kernel.configs = configs
                kernel.cache.clear()

    def test_small_clipping(self):
        for frames in (1, 5, 33, 130):
            self.compare(frames, torch.float32, 1e-3, left=3, right=2, dim=32)

    def test_custom_op_registration(self):
        from w2vbert_fast.triton_attention import _relative_attention

        q, k, v, e, _ = make(2, 2, 70, 64, 64, 8, torch.bfloat16)
        q = q.detach().transpose(1, 2).contiguous().transpose(1, 2).requires_grad_()  # a [B,T,H,D]-strided view
        lengths = torch.tensor([70, 40], device="cuda", dtype=torch.int32)
        torch.library.opcheck(_relative_attention, (q, k, v, e, lengths, 64, 8))

    def test_packed_projection_matches_the_reference(self):
        from w2vbert_fast.triton_attention import fused_relative_key_attention_packed

        for frames, lengths in ((150, None), (300, [300, 211])):
            torch.manual_seed(frames)
            qkv = torch.randn(2, frames, 3, 4, 64, device="cuda").to(torch.bfloat16).requires_grad_()
            e = torch.randn(73, 64, device="cuda").to(torch.bfloat16).requires_grad_()
            w = torch.randn(2, frames, 4, 64, device="cuda")
            lens = None if lengths is None else torch.tensor(lengths, device="cuda")
            out = fused_relative_key_attention_packed(qkv, e, 64, 8, lens)
            (out.float() * w).sum().backward()
            fast = (out.detach().double(), qkv.grad.double(), e.grad.double())
            q64 = qkv.detach().double().requires_grad_()
            e64 = e.detach().double().requires_grad_()
            additive = None
            if lens is not None:
                keys = torch.arange(frames, device="cuda")[None, :] < lens[:, None]
                additive = torch.zeros(2, 1, frames, frames, device="cuda", dtype=torch.float64).masked_fill(
                    ~keys[:, None, None, :], float("-inf"))
            q, k, v = (t.transpose(1, 2) for t in q64.unbind(2))
            ref = relative_key_reference(q, k, v, e64, 64, 8, additive).transpose(1, 2)
            (ref * w.double()).sum().backward()
            for name, a, b in zip(("out", "dqkv", "dE"), fast, (ref.detach(), q64.grad, e64.grad)):
                error = float((a - b).norm() / (b.norm() + 1e-3 * b.numel() ** 0.5))
                self.assertLess(error, 2e-2, f"{name} T={frames} lengths={lengths}: {error:.3g}")


if __name__ == "__main__":
    unittest.main()
