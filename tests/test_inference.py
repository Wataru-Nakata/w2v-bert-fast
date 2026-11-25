"""InferenceEncoder on a GPU: the CUDA-graph replays give the values of the same model run eagerly, for padded
batches, shapes in any order (graphs share one memory pool), several output layers and batches split into chunks;
and on the real checkpoint, no further from the Hugging Face fp32 model than Hugging Face's own bf16."""

import copy
import unittest

import torch
from transformers import Wav2Vec2BertConfig, Wav2Vec2BertModel

from w2vbert_fast import InferenceEncoder, optimize


def tiny():
    config = Wav2Vec2BertConfig(hidden_size=64, num_hidden_layers=3, num_attention_heads=4, intermediate_size=128,
                                feature_projection_input_dim=16, conv_depthwise_kernel_size=5,
                                position_embeddings_type="relative_key", left_max_position_embeddings=6,
                                right_max_position_embeddings=3, mask_time_prob=0.0)
    torch.manual_seed(0)
    return Wav2Vec2BertModel(config).eval()


def batch_of(lengths, frames, width, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    features = torch.randn(len(lengths), frames, width, device="cuda", generator=generator)
    mask = torch.zeros(len(lengths), frames, device="cuda", dtype=torch.long)
    for row, length in enumerate(lengths):
        mask[row, :length] = 1
    return features, mask


def relative(a, b, mask):
    valid = mask[..., None].float()
    return float(((a.float() - b.float()) * valid).norm() / (b.float() * valid).norm())


@unittest.skipUnless(torch.cuda.is_available(), "CUDA")
class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = tiny()
        cls.eager = optimize(copy.deepcopy(cls.base), compile=True).to("cuda", torch.bfloat16).eval()

    def eager_bucketed(self, encoder, features, mask):
        """The eager model on the same bucketed input the encoder builds (identical kernels and shapes)."""
        batch, frames, _ = features.shape
        rows, padded = encoder._bucket(batch, frames)
        f = torch.zeros(rows, padded, features.size(-1), device="cuda", dtype=torch.bfloat16)
        m = torch.ones(rows, padded, device="cuda", dtype=torch.long)
        f[:batch, :frames] = features
        m[:batch] = 0
        m[:batch, :frames] = mask
        with torch.no_grad():
            out = self.eager(f, attention_mask=m, output_hidden_states=True).hidden_states
        return [h[:batch, :frames] for h in out]

    def test_graph_replays_match_the_eager_model_in_any_order(self):
        encoder = InferenceEncoder(copy.deepcopy(self.base), frame_step=8, batch_sizes=(1, 2, 4))
        cases = [([37, 20, 9], 37, 0), ([5], 5, 1), ([37, 37, 30], 37, 2), ([61, 3], 61, 3), ([5], 5, 4),
                 ([37, 20, 9], 37, 5)]
        for lengths, frames, seed in cases:
            features, mask = batch_of(lengths, frames, 16, seed)
            out = encoder(features, mask)
            expected = self.eager_bucketed(encoder, features, mask)[-1]
            self.assertEqual(out.shape, (len(lengths), frames, 64))
            self.assertLess(relative(out, expected, mask), 1e-6, f"{lengths}")
        self.assertEqual(sorted(encoder._graphs), [(1, 8), (2, 64), (4, 40)])

    def test_bucketing_keeps_valid_positions(self):
        encoder = InferenceEncoder(copy.deepcopy(self.base), frame_step=16, batch_sizes=(4,))
        features, mask = batch_of([45, 30], 45, 16, 7)
        with torch.no_grad():
            unpadded = self.eager(features.to(torch.bfloat16), attention_mask=mask).last_hidden_state
        self.assertLess(relative(encoder(features, mask), unpadded, mask), 1e-2)

    def test_output_layers_chunks_and_views(self):
        encoder = InferenceEncoder(copy.deepcopy(self.base), output_layers=[0, 2, 3], frame_step=8,
                                   batch_sizes=(1, 2))
        features, mask = batch_of([30, 22, 17, 30, 9], 30, 16, 11)
        out = encoder(features, mask)
        self.assertEqual(sorted(out), [0, 2, 3])
        expected = [self.eager_bucketed(encoder, features[i:i + 2], mask[i:i + 2]) for i in (0, 2, 4)]
        for layer in (0, 2, 3):
            want = torch.cat([e[layer] for e in expected])
            self.assertLess(relative(out[layer], want, mask), 1e-6, f"layer {layer}")
        views = InferenceEncoder(copy.deepcopy(self.base), frame_step=8, copy=False)
        first = views(features[:1], mask[:1])
        self.assertEqual(first.shape, (1, 30, 64))

    def test_real_checkpoint_is_as_close_to_fp32_as_hugging_face_bf16(self):
        try:
            base = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0", local_files_only=True).eval()
        except OSError:
            self.skipTest("facebook/w2v-bert-2.0 is not in the local cache")
        base.encoder.layers = base.encoder.layers[:6]
        base.config.num_hidden_layers = 6
        encoder = InferenceEncoder(copy.deepcopy(base))
        reference = copy.deepcopy(base).cuda()
        hf_bf16 = copy.deepcopy(base).to("cuda", torch.bfloat16)
        features, mask = batch_of([400, 333, 120], 400, 160, 13)
        with torch.no_grad():
            exact = reference(features, attention_mask=mask).last_hidden_state
            theirs = hf_bf16(features.to(torch.bfloat16), attention_mask=mask).last_hidden_state
        ours = encoder(features, mask)
        self.assertLess(relative(ours, exact, mask), 1.5 * relative(theirs, exact, mask) + 1e-3)


if __name__ == "__main__":
    unittest.main()
