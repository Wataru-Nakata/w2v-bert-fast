"""extract_features against transformers' SeamlessM4TFeatureExtractor with the w2v-BERT 2.0 preprocessor settings."""

import unittest

import numpy as np
import torch
from transformers import SeamlessM4TFeatureExtractor

from w2vbert_fast import extract_features


class FeatureTests(unittest.TestCase):
    def test_matches_the_reference_extractor(self):
        reference = SeamlessM4TFeatureExtractor(feature_size=80, num_mel_bins=80, sampling_rate=16000,
                                                padding_value=1, stride=2)
        generator = np.random.default_rng(0)
        waves = [(0.1 * generator.standard_normal(n)).astype(np.float32) for n in (16000, 12345, 5000, 1000)]
        expected = reference(waves, sampling_rate=16000, return_tensors="pt")
        got = extract_features([torch.from_numpy(w) for w in waves], dtype=torch.float64)
        self.assertEqual(tuple(got["input_features"].shape), tuple(expected["input_features"].shape))
        torch.testing.assert_close(got["attention_mask"], expected["attention_mask"].long(), rtol=0, atol=0)
        valid = expected["attention_mask"].bool()
        torch.testing.assert_close(got["input_features"].float()[valid], expected["input_features"][valid],
                                   rtol=1e-3, atol=2e-3)
        torch.testing.assert_close(got["input_features"].float()[~valid], expected["input_features"][~valid])


if __name__ == "__main__":
    unittest.main()
