# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Modules below used for the audio encoder component in: models/nano_nemotron_vl.py
"""

from collections.abc import Iterable
from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn
from transformers import BatchFeature
from transformers import ParakeetEncoder as HFParakeetEncoder
from transformers import ParakeetFeatureExtractor, PretrainedConfig

from vllm.model_executor.layers.activation import ReLUSquaredActivation
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.transformers_utils.configs.parakeet import ExtractorConfig, ParakeetConfig


class GPUMelSpectrogram(nn.Module):
    """GPU-accelerated mel spectrogram matching HF ParakeetFeatureExtractor.

    Replaces CPU-bound numpy/librosa mel extraction with torch.stft on CUDA.
    Produces numerically equivalent output to the HF feature extractor:
    preemphasis -> STFT -> power spectrum -> mel filterbank -> log -> normalize.
    """

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 400,
        n_mels: int = 80,
        sample_rate: int = 16000,
        preemphasis: float = 0.97,
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.preemphasis = preemphasis

        self.register_buffer("window", torch.hann_window(win_length))
        mel_fb = self._make_mel_filterbank(n_fft, n_mels, sample_rate)
        self.register_buffer("mel_fb", mel_fb)

    @staticmethod
    def _make_mel_filterbank(
        n_fft: int,
        n_mels: int,
        sample_rate: int,
        fmin: float = 0.0,
        fmax: float | None = None,
    ) -> torch.Tensor:
        """Build Slaney-normalized mel filterbank (matches librosa default)."""
        if fmax is None:
            fmax = sample_rate / 2.0
        n_bins = n_fft // 2 + 1

        mel_min = 2595.0 * np.log10(1.0 + fmin / 700.0)
        mel_max = 2595.0 * np.log10(1.0 + fmax / 700.0)
        mels = np.linspace(mel_min, mel_max, n_mels + 2)
        freqs = 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

        fft_freqs = np.linspace(0, sample_rate / 2.0, n_bins)
        fb = np.zeros((n_mels, n_bins), dtype=np.float32)
        for i in range(n_mels):
            low, center, high = freqs[i], freqs[i + 1], freqs[i + 2]
            up = (fft_freqs - low) / max(center - low, 1e-10)
            down = (high - fft_freqs) / max(high - center, 1e-10)
            fb[i] = np.maximum(0.0, np.minimum(up, down))
            # Slaney normalization: scale by 2 / bandwidth
            enorm = 2.0 / max(high - low, 1e-10)
            fb[i] *= enorm
        return torch.from_numpy(fb)

    def forward(
        self, waveforms: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute normalized log-mel spectrogram on GPU.

        Args:
            waveforms: (batch, max_samples) float tensor
            lengths: (batch,) long tensor of valid sample counts
        Returns:
            features: (batch, max_frames, n_mels) normalized log-mel
            attention_mask: (batch, max_frames) boolean mask
        """
        device = waveforms.device
        batch_size, max_samples = waveforms.shape

        # Preemphasis: y[n] = x[n] - coeff * x[n-1]
        preemph = torch.empty_like(waveforms)
        preemph[:, 0] = waveforms[:, 0]
        preemph[:, 1:] = waveforms[:, 1:] - self.preemphasis * waveforms[:, :-1]

        # Mask padding to zero (matches HF behavior)
        sample_idx = torch.arange(max_samples, device=device).unsqueeze(0)
        sample_mask = sample_idx < lengths.unsqueeze(1)
        preemph = preemph * sample_mask

        # Ensure buffers are on the correct device
        window = self.window.to(device=device)
        mel_fb = self.mel_fb.to(device=device)

        # STFT (center=True is torch default, matching HF)
        stft_out = torch.stft(
            preemph,
            self.n_fft,
            self.hop_length,
            self.win_length,
            window,
            return_complex=True,
        )  # (batch, n_fft//2+1, frames)
        power = stft_out.real.pow(2) + stft_out.imag.pow(2)

        # Mel filterbank
        mel_fb = mel_fb.to(dtype=power.dtype)
        mel = torch.matmul(mel_fb, power)  # (batch, n_mels, frames)
        mel = mel.transpose(1, 2)  # (batch, frames, n_mels)

        # Log scale (epsilon = 2^-24 matches HF)
        mel = torch.log(mel + 2**-24)

        # Compute valid frame counts
        # HF formula: floor((audio_len - n_fft//2) / hop_length)
        num_frames = mel.shape[1]
        feat_lengths = (lengths - self.n_fft // 2) // self.hop_length
        feat_lengths = feat_lengths.clamp(min=1, max=num_frames)

        frame_idx = torch.arange(num_frames, device=device).unsqueeze(0)
        feat_mask = frame_idx < feat_lengths.unsqueeze(1)

        # Per-clip mean-variance normalization
        feat_mask_f = feat_mask.unsqueeze(-1).float()
        n_valid = feat_lengths.float().unsqueeze(-1).clamp(min=1)  # (batch, 1)

        mean = (mel * feat_mask_f).sum(dim=1) / n_valid  # (batch, n_mels)
        diff = (mel - mean.unsqueeze(1)) * feat_mask_f
        var = diff.pow(2).sum(dim=1) / n_valid
        std = (var + 1e-5).sqrt()

        mel = (mel - mean.unsqueeze(1)) / std.unsqueeze(1)
        mel = mel * feat_mask_f

        return mel, feat_mask


class ParakeetProjection(nn.Module):
    def __init__(self, config: ParakeetConfig) -> None:
        super().__init__()
        sound_hidden_size = config.hidden_size
        proj_hidden_size = config.projection_hidden_size
        llm_hidden_size = config.llm_hidden_size
        bias = config.projection_bias

        self.norm = RMSNorm(sound_hidden_size, eps=config.projection_eps)
        self.linear1 = nn.Linear(sound_hidden_size, proj_hidden_size, bias=bias)
        self.activation = ReLUSquaredActivation()
        self.linear2 = nn.Linear(proj_hidden_size, llm_hidden_size, bias=bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm(hidden_states)
        hidden_states = self.linear1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.linear2(hidden_states)
        return hidden_states


class ProjectedParakeet(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        *,
        dtype: torch.dtype,
        llm_hidden_size: int,
        max_model_len: int,
    ) -> None:
        super().__init__()
        self.config = ParakeetConfig.from_hf_config(
            config, llm_hidden_size=llm_hidden_size, max_model_len=max_model_len
        )
        self.mel_transform = GPUMelSpectrogram(
            n_fft=512,
            hop_length=getattr(config, "hop_length", 160),
            win_length=getattr(config, "win_length", 400),
            n_mels=getattr(config, "num_mel_bins", 80),
            sample_rate=getattr(config, "sampling_rate", 16000),
            preemphasis=getattr(config, "preemphasis", 0.97),
        )
        self.encoder = HFParakeetEncoder(self.config)
        self.encoder = self.encoder.to(dtype)
        self.projection = ParakeetProjection(self.config)
        self.projection = self.projection.to(dtype)

    def forward(
        self, input_features: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        outputs = self.encoder(
            input_features=input_features, attention_mask=attention_mask
        )
        outputs = outputs.last_hidden_state
        outputs = outputs.to(dtype=torch.bfloat16)
        outputs = self.projection(outputs)
        return outputs

    def forward_from_waveforms(
        self, waveforms: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute mel on GPU then run encoder + projection.

        Args:
            waveforms: (batch, max_samples) raw audio on GPU
            lengths: (batch,) valid sample counts
        Returns:
            embeddings: (batch, out_frames, llm_hidden_size)
            attention_mask: (batch, mel_frames) for valid frame tracking
        """
        mel_features, attention_mask = self.mel_transform(waveforms, lengths)
        mel_features = mel_features.to(dtype=next(self.encoder.parameters()).dtype)
        outputs = self.forward(mel_features, attention_mask)
        return outputs, attention_mask

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded_params: set[str] = set()
        params_dict = dict(self.named_parameters())
        buffers_dict = dict(self.named_buffers())

        if isinstance(weights, dict):
            weights_list = list(weights.items())
        else:
            weights_list = list(weights)

        for name, weight in weights_list:
            if name.startswith("sound_encoder.encoder.feature_extractor."):
                # Feature extractor buffers are handled outside the encoder.
                continue
            if name.startswith("sound_encoder."):
                target_name = name[len("sound_encoder.") :]
            elif name.startswith("sound_projection."):
                target_name = f"projection.{name[len('sound_projection.') :]}"
            else:
                continue

            target = params_dict.get(target_name)
            if target is None:
                target = buffers_dict.get(target_name)
            if target is None:
                # Skip unknown weights (e.g. mel_transform buffers not in ckpt)
                continue
            weight_loader = getattr(target, "weight_loader", default_weight_loader)
            with torch.no_grad():
                weight_loader(target, weight)
            loaded_params.add(target_name)

        return loaded_params


class ParakeetExtractor(ParakeetFeatureExtractor):
    def __init__(self, config: PretrainedConfig) -> None:
        self.config = ExtractorConfig.from_hf_config(config)
        super().__init__(**asdict(self.config))
        self._clip_target_samples = int(
            round(self.config.clip_duration_s * self.sampling_rate)
        )
        self._tail_min_samples = int(
            round(self.config.clip_min_duration_s * self.sampling_rate)
        )

    def _clip_sizes(self, audio_len: int) -> list[int]:
        audio_len = max(audio_len, self._tail_min_samples)
        num_full_clips, remainder = divmod(audio_len, self._clip_target_samples)
        clip_sizes = [self._clip_target_samples] * num_full_clips
        if remainder > 0:
            clip_sizes.append(max(remainder, self._tail_min_samples))
        return clip_sizes

    def audio_token_count(self, audio_len: int) -> int:
        total_tokens = 0
        for clip_size in self._clip_sizes(audio_len):
            num_frames = clip_size // self.hop_length
            n_tokens = HFParakeetEncoder._get_subsampling_output_length(
                self, torch.tensor([num_frames], dtype=torch.float)
            )
            total_tokens += int(n_tokens.item())
        return max(1, total_tokens)

    def split_audio_into_clips(self, audio: np.ndarray) -> list[np.ndarray]:
        assert audio.ndim == 1
        audio_len = int(audio.shape[0])
        clip_sizes = self._clip_sizes(audio_len)
        target_len = sum(clip_sizes)
        if audio_len < target_len:
            audio = np.pad(audio, (0, target_len - audio_len))

        clips = list[np.ndarray]()
        offset = 0
        for clip_size in clip_sizes:
            clips.append(audio[offset : offset + clip_size])
            offset += clip_size
        return clips

    def __call__(self, raw_speech: list[np.ndarray], *args, **kwargs):
        audio_clips = list[np.ndarray]()
        audio_num_clips = list[int]()
        for audio in raw_speech:
            clips = self.split_audio_into_clips(audio)
            audio_clips.extend(clips)
            audio_num_clips.append(len(clips))

        # Return raw waveforms instead of CPU mel spectrograms.
        # Mel extraction is deferred to GPU via GPUMelSpectrogram.
        clip_lengths = [len(c) for c in audio_clips]
        max_len = max(clip_lengths) if clip_lengths else 0
        padded = np.zeros((len(audio_clips), max_len), dtype=np.float32)
        for i, clip in enumerate(audio_clips):
            padded[i, : len(clip)] = clip

        return BatchFeature(
            {
                "input_audio_waveforms": torch.from_numpy(padded),
                "audio_clip_lengths": torch.tensor(clip_lengths, dtype=torch.long),
                "audio_num_clips": audio_num_clips,
            }
        )

    def audio_length(self, audio_tokens: int) -> int:
        return int(audio_tokens * self.config.subsampling_factor * self.hop_length)
