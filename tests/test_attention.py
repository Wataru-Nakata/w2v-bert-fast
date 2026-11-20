"""relative_key_attention against the Hugging Face computation: values and gradients (query, key, value and the
distance embedding) in float64 on CPU across the region boundaries, and in bf16 on CUDA through FlashAttention."""

import itertools
import unittest

import torch

from w2vbert_fast.attention import relative_key_attention, relative_key_reference


def case(batch, heads, frames, dim, left, right, dtype=torch.float64, device="cpu", seed=0):
    generator = torch.Generator().manual_seed(seed)
    make = lambda *shape: torch.randn(*shape, generator=generator, dtype=torch.float64).to(device, dtype)  # noqa: E731
    q, k, v = (make(batch, heads, frames, dim).requires_grad_() for _ in range(3))
    embedding = make(left + right + 1, dim).requires_grad_()
    weight = make(batch, heads, frames, dim)
    return q, k, v, embedding, weight


def grads(fn, q, k, v, embedding, weight):
    out = fn(q, k, v, embedding)
    (out * weight).sum().backward()
    result = [out.detach()] + [t.grad.clone() for t in (q, k, v, embedding)]
    for t in (q, k, v, embedding):
        t.grad = None
    return result


class Float64Tests(unittest.TestCase):
    def check(self, frames, left, right, batch=2, heads=2, dim=8):
        q, k, v, embedding, weight = case(batch, heads, frames, dim, left, right, seed=frames)
        fast = grads(lambda *a: relative_key_attention(*a, left, right), q, k, v, embedding, weight)
        reference = grads(lambda *a: relative_key_reference(*a, left, right), q, k, v, embedding, weight)
        for name, a, b in zip(("out", "dq", "dk", "dv", "dE"), fast, reference):
            torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-10, msg=f"{name} T={frames} L={left} R={right}")

    def test_w2v_bert_clipping_across_region_boundaries(self):
        for frames in (1, 2, 7, 8, 9, 63, 64, 65, 71, 72, 73, 128, 130, 200):
            self.check(frames, 64, 8)

    def test_small_clipping_many_blocks(self):
        for frames, (left, right) in itertools.product((1, 3, 4, 5, 9, 63, 64, 65, 129), ((1, 1), (2, 1), (3, 2),
                                                                                         (1, 4))):
            self.check(frames, left, right, batch=1, heads=1, dim=4)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA")
class FlashTests(unittest.TestCase):
    def test_bf16_flash_matches_the_float32_reference(self):
        for frames in (65, 300, 1500):
            q, k, v, embedding, weight = case(2, 4, frames, 64, 64, 8, dtype=torch.bfloat16, device="cuda")
            fast = grads(lambda *a: relative_key_attention(*a, 64, 8), q, k, v, embedding, weight)
            q32, k32, v32, e32 = (t.detach().float().requires_grad_() for t in (q, k, v, embedding))
            reference = grads(lambda *a: relative_key_reference(*a, 64, 8), q32, k32, v32, e32, weight.float())
            for name, a, b in zip(("out", "dq", "dk", "dv", "dE"), fast, reference):
                error = (a.float() - b).norm() / b.norm()
                self.assertLess(float(error), 2e-2, f"{name} T={frames}: relative error {float(error):.3g}")


if __name__ == "__main__":
    unittest.main()
