"""The channels-last causal depthwise convolution against nn.Conv1d (left padding K-1): values and gradients."""

import unittest

import torch

from w2vbert_fast.triton_conv import causal_depthwise_conv_reference


@unittest.skipUnless(torch.cuda.is_available(), "CUDA")
class ConvTests(unittest.TestCase):
    def test_matches_conv1d_values_and_gradients(self):
        from w2vbert_fast.triton_conv import causal_depthwise_conv

        for batch, frames, channels, kernel in ((2, 1, 8, 31), (3, 37, 100, 31), (2, 500, 1024, 31), (1, 64, 64, 5)):
            for dtype, tolerance in ((torch.float32, 1e-5), (torch.bfloat16, 1e-2)):
                torch.manual_seed(frames)
                x = torch.randn(batch, frames, channels, device="cuda").to(dtype).requires_grad_()
                w = (0.2 * torch.randn(channels, 1, kernel, device="cuda")).to(dtype).requires_grad_()
                g = torch.randn(batch, frames, channels, device="cuda")
                out = causal_depthwise_conv(x, w)
                (out.float() * g).sum().backward()
                fast = (out.detach().float(), x.grad.float(), w.grad.float())
                x32, w32 = (t.detach().float().requires_grad_() for t in (x, w))
                ref_out = causal_depthwise_conv_reference(x32, w32)
                (ref_out * g).sum().backward()
                for name, a, b in zip(("out", "dx", "dw"), fast, (ref_out.detach(), x32.grad, w32.grad)):
                    error = float((a - b).norm() / b.norm().clamp_min(1e-12))
                    self.assertLess(error, tolerance, f"{name} {batch}x{frames}x{channels} K={kernel} {dtype}")

    def test_custom_op_registration(self):
        from w2vbert_fast.triton_conv import _conv

        x = torch.randn(2, 50, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        w = torch.randn(64, 1, 31, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        torch.library.opcheck(_conv, (x, w))


if __name__ == "__main__":
    unittest.main()
