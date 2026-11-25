"""Fast, exact w2v-BERT 2.0 (Hugging Face ``Wav2Vec2BertModel``) for training and inference."""

from .attention import relative_key_attention, relative_key_reference
from .features import extract_features
from .inference import InferenceEncoder
from .model import FastSelfAttention, load, optimize

__all__ = ["FastSelfAttention", "InferenceEncoder", "extract_features", "load", "optimize", "relative_key_attention",
           "relative_key_reference"]
__version__ = "0.1.0"
