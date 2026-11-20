"""optimize() on a small random Wav2Vec2BertModel: same outputs and gradients as the original (unpadded and
padded batches, valid positions), same state-dict keys, truncation, and the fallbacks."""

import copy
import unittest

import torch
from transformers import Wav2Vec2BertConfig, Wav2Vec2BertModel

from w2vbert_fast import optimize


def tiny(left=6, right=3, layers=3):
    config = Wav2Vec2BertConfig(hidden_size=32, num_hidden_layers=layers, num_attention_heads=4,
                                intermediate_size=64, feature_projection_input_dim=16, conv_depthwise_kernel_size=5,
                                position_embeddings_type="relative_key", left_max_position_embeddings=left,
                                right_max_position_embeddings=right, layerdrop=0.0, hidden_dropout=0.0,
                                attention_dropout=0.0, activation_dropout=0.0, feat_proj_dropout=0.0,
                                conformer_conv_dropout=0.0, mask_time_prob=0.0, mask_feature_prob=0.0)
    torch.manual_seed(0)
    return Wav2Vec2BertModel(config).double()


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.original = tiny()
        self.fast = optimize(copy.deepcopy(self.original))
        torch.manual_seed(1)
        self.features = torch.randn(3, 40, 16, dtype=torch.float64)
        self.mask = torch.ones(3, 40, dtype=torch.long)
        self.mask[1, 31:] = 0
        self.mask[2, 9:] = 0

    def test_state_dict_keys_and_values_are_unchanged(self):
        a, b = self.original.state_dict(), self.fast.state_dict()
        self.assertEqual(list(a), list(b))
        for key in a:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)

    def test_outputs_match_without_padding(self):
        for model in (self.original, self.fast):
            model.eval()
        with torch.no_grad():
            a = self.original(self.features, output_hidden_states=True).hidden_states
            b = self.fast(self.features, output_hidden_states=True).hidden_states
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y, rtol=1e-9, atol=1e-10)

    def test_outputs_and_gradients_match_on_valid_positions_of_a_padded_batch(self):
        for model in (self.original, self.fast):
            model.train()
        weight = torch.randn(3, 40, 32, dtype=torch.float64) * self.mask[..., None]
        results = []
        for model in (self.original, self.fast):
            model.zero_grad()
            out = model(self.features, attention_mask=self.mask).last_hidden_state
            (out * weight).sum().backward()
            results.append((out.detach() * self.mask[..., None],
                            {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}))
        torch.testing.assert_close(results[0][0], results[1][0], rtol=1e-9, atol=1e-10)
        self.assertEqual(set(results[0][1]), set(results[1][1]))
        for name in results[0][1]:
            torch.testing.assert_close(results[0][1][name], results[1][1][name], rtol=1e-8, atol=1e-9, msg=name)

    def test_consecutive_batches_with_different_padding(self):
        # same shapes, different lengths, one after the other (a mask buffer may be reused at the same address)
        for model in (self.original, self.fast):
            model.eval()
        for cut in ((40, 31, 9), (12, 40, 25), (40, 40, 40), (7, 7, 39)):
            mask = torch.zeros(3, 40, dtype=torch.long)
            for row, length in enumerate(cut):
                mask[row, :length] = 1
            with torch.no_grad():
                a = self.original(self.features, attention_mask=mask).last_hidden_state * mask[..., None]
                b = self.fast(self.features, attention_mask=mask).last_hidden_state * mask[..., None]
            torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-10, msg=str(cut))

    def test_truncation_keeps_the_first_layers(self):
        self.original.eval()
        short = optimize(copy.deepcopy(self.original), num_layers=2).eval()
        with torch.no_grad():
            full = self.original(self.features, output_hidden_states=True).hidden_states
            out = short(self.features).last_hidden_state
        self.assertEqual(len(short.encoder.layers), 2)
        torch.testing.assert_close(out, full[2], rtol=1e-9, atol=1e-10)

    def test_output_attentions_and_the_sdpa_backend_fall_back_correctly(self):
        self.original.eval()
        self.fast.eval()
        with torch.no_grad():
            a = self.original(self.features, attention_mask=self.mask, output_attentions=True)
            b = self.fast(self.features, attention_mask=self.mask, output_attentions=True)
            self.assertEqual(len(b.attentions), 3)
            torch.testing.assert_close(a.last_hidden_state, b.last_hidden_state, rtol=1e-9, atol=1e-10)
            sdpa = optimize(copy.deepcopy(self.original), backend="sdpa").eval()
            c = sdpa(self.features, attention_mask=self.mask).last_hidden_state
            torch.testing.assert_close(a.last_hidden_state * self.mask[..., None], c * self.mask[..., None],
                                       rtol=1e-9, atol=1e-10)

    def test_lora_wrapped_projections_keep_their_adapters(self):
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            self.skipTest("peft is not installed")
        config = LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, bias="none",
                            target_modules=["linear_q", "linear_k", "linear_v", "pointwise_conv1"])
        original = get_peft_model(copy.deepcopy(self.original), config)
        torch.manual_seed(2)
        for name, parameter in original.named_parameters():
            if "lora_B" in name:
                torch.nn.init.normal_(parameter, std=0.1)  # PEFT starts B at zero; make the adapters matter
        fast = copy.deepcopy(original)
        optimize(fast.base_model.model)
        outputs = []
        for model in (original, fast):
            model.train()
            hidden = model(self.features, attention_mask=self.mask).last_hidden_state
            weight = self.mask[..., None].to(hidden.dtype)
            (hidden * weight).pow(2).sum().backward()
            grads = {n: p.grad for n, p in model.named_parameters() if p.requires_grad}
            outputs.append((hidden * weight, grads))
        torch.testing.assert_close(outputs[0][0], outputs[1][0], rtol=1e-9, atol=1e-10)
        self.assertEqual(sorted(outputs[0][1]), sorted(outputs[1][1]))
        for name in outputs[0][1]:
            torch.testing.assert_close(outputs[0][1][name], outputs[1][1][name], rtol=1e-8, atol=1e-9, msg=name)


if __name__ == "__main__":
    unittest.main()
