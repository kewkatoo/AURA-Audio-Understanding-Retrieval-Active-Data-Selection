"""
Frozen Wav2Vec2 encoder.

Requirements satisfied here (per Phase 1 spec):
- frozen parameters (requires_grad_(False), eval mode)
- no gradients (torch.no_grad() around every forward pass)
- deterministic inference where practical (eval mode disables dropout;
  no other stochastic ops are used)
- mean pooling over the temporal dimension
- L2-normalized final embedding (configurable)
- configurable device, with automatic CPU fallback
- configurable batch size
- output dimensionality recorded automatically from the loaded model
  config, never hard-coded
"""
from __future__ import annotations

import numpy as np
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

from aura.encoders.base import AudioEncoder, EncoderInfo


def _resolve_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        # Explicit CPU fallback per spec -- never raise if CUDA was
        # requested but unavailable.
        return "cpu"
    # "auto"
    return "cuda" if torch.cuda.is_available() else "cpu"


class Wav2Vec2Encoder(AudioEncoder):
    def __init__(
        self,
        checkpoint: str = "facebook/wav2vec2-base",
        pooling: str = "mean",
        normalize: bool = True,
        device: str = "auto",
        batch_size: int = 8,
    ):
        if pooling != "mean":
            raise NotImplementedError(
                f"Only 'mean' pooling is implemented in Phase 1, got {pooling!r}."
            )
        self.checkpoint = checkpoint
        self.pooling = pooling
        self.normalize = normalize
        self.batch_size = batch_size
        self.device = _resolve_device(device)

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(checkpoint)
        self.model = Wav2Vec2Model.from_pretrained(checkpoint)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.to(self.device)

        embedding_dim = self.model.config.hidden_size  # recorded automatically

        self._info = EncoderInfo(
            name="wav2vec2",
            checkpoint=checkpoint,
            embedding_dim=int(embedding_dim),
            pooling=pooling,
            l2_normalized=normalize,
            device=self.device,
        )

    @property
    def info(self) -> EncoderInfo:
        return self._info

    @torch.no_grad()
    def encode_batch(
        self, waveforms: list[np.ndarray], sample_rate: int
    ) -> np.ndarray:
        if len(waveforms) == 0:
            return np.zeros((0, self._info.embedding_dim), dtype=np.float32)

        expected_sr = self.feature_extractor.sampling_rate
        if sample_rate != expected_sr:
            raise ValueError(
                f"Wav2Vec2Encoder expects {expected_sr}Hz input, got {sample_rate}Hz. "
                "Resample before calling encode_batch (aura.ingest.audio handles this)."
            )

        all_embeddings = []
        for start in range(0, len(waveforms), self.batch_size):
            chunk = waveforms[start:start + self.batch_size]
            inputs = self.feature_extractor(
                [w.astype(np.float32) for w in chunk],
                sampling_rate=expected_sr,
                return_tensors="pt",
                padding=True,
            )
            input_values = inputs["input_values"].to(self.device)
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            outputs = self.model(
                input_values=input_values, attention_mask=attention_mask
            )
            hidden_states = outputs.last_hidden_state  # (batch, time, hidden)

            if attention_mask is not None:
                # Mean pool only over valid (non-padded) timesteps. The
                # feature extractor's attention_mask is at the input-sample
                # rate; Wav2Vec2's conv stack downsamples time, so we
                # derive an output-length mask via the model's own helper
                # rather than assuming a fixed stride.
                output_lengths = self.model._get_feat_extract_output_lengths(
                    attention_mask.sum(-1)
                ).to(torch.long)
                batch_size, max_len, _ = hidden_states.shape
                time_idx = torch.arange(max_len, device=hidden_states.device).unsqueeze(0)
                valid_mask = (time_idx < output_lengths.unsqueeze(1)).unsqueeze(-1)
                summed = (hidden_states * valid_mask).sum(dim=1)
                counts = valid_mask.sum(dim=1).clamp(min=1)
                pooled = summed / counts
            else:
                pooled = hidden_states.mean(dim=1)

            if self.normalize:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)

            all_embeddings.append(pooled.cpu().numpy().astype(np.float32))

        return np.concatenate(all_embeddings, axis=0)
