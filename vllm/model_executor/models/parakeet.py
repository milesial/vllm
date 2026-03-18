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
import torch.nn.functional as F
from transformers import ParakeetEncoder as HFParakeetEncoder
from transformers import (ParakeetFeatureExtractor, PretrainedConfig,
                          feature_extraction_utils)

from vllm.model_executor.layers.activation import ReLUSquaredActivation
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.transformers_utils.configs.parakeet import ExtractorConfig, ParakeetConfig


class ParakeetProjection(nn.Module):
    def __init__(self, config: ParakeetConfig) -> None:
        super().__init__()
        sound_hidden_size = config.hidden_size
        proj_hidden_size = config.projection_hidden_size
        llm_hidden_size = config.llm_hidden_size
        bias = config.projection_bias

        self.norm = nn.LayerNorm(sound_hidden_size, eps=config.projection_eps)
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
                raise ValueError(f"Unknown weight: {name}")
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
        self._window_cache: dict[str, torch.Tensor] = {}
        self._mel_filter_cache: dict[str, torch.Tensor] = {}

    def _normalize_audio_length(self, audio_len: int) -> int:
        # Match mcore's compute_params() logic for clip/minduration handling.
        target_len = max(audio_len, self._tail_min_samples)
        tail_remainder = target_len % self._clip_target_samples
        if 0 < tail_remainder < self._tail_min_samples:
            padding = self._tail_min_samples - tail_remainder
            target_len += padding
        assert isinstance(target_len, int)
        return target_len

    def audio_token_count(self, audio_len: int) -> int:
        audio_len = self._normalize_audio_length(audio_len)
        num_frames = audio_len // self.hop_length
        n_tokens = HFParakeetEncoder._get_subsampling_output_length(
            self, torch.tensor([num_frames], dtype=torch.float)
        )
        return max(1, n_tokens.item())

    def __call__(self, raw_speech: list[np.ndarray], *args, **kwargs):
        device = kwargs.get("device", "cpu")
        if device is None:
            device = "cpu"
        device_obj = torch.device(device)
        if device_obj.type == "cuda":
            return self._torch_gpu_call(raw_speech, *args, **kwargs)

        padded = []
        for p in raw_speech:
            assert p.ndim == 1
            audio_len = int(p.shape[0])
            target_len = self._normalize_audio_length(audio_len)
            p = np.pad(p, (0, target_len - audio_len))
            padded.append(p)
        return super().__call__(padded, *args, **kwargs)

    def _torch_extract_fbank_features(
        self,
        waveform: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        cache_key = str(device)
        window = self._window_cache.get(cache_key)
        if window is None:
            window = torch.hann_window(
                self.win_length,
                periodic=False,
                device=device,
            )
            self._window_cache[cache_key] = window

        mel_filters = self._mel_filter_cache.get(cache_key)
        if mel_filters is None:
            mel_filters = self.mel_filters.to(device)
            self._mel_filter_cache[cache_key] = mel_filters

        stft = torch.stft(
            waveform,
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
            pad_mode="constant",
        )
        power_spec = stft.real.square() + stft.imag.square()
        mel_spec = torch.matmul(mel_filters, power_spec)
        mel_spec = torch.log(mel_spec.clamp_min_(2**-24))
        return mel_spec.transpose(1, 2).contiguous()

    def _torch_gpu_call(
        self,
        raw_speech: list[np.ndarray],
        truncation: bool = False,
        pad_to_multiple_of: int | None = None,
        return_tensors: str | None = None,
        return_attention_mask: bool | None = None,
        padding: str | None = "longest",
        max_length: int | None = None,
        sampling_rate: int | None = None,
        do_normalize: bool | None = None,
        device: str | None = "cuda",
        return_token_timestamps: bool | None = None,
        **kwargs,
    ):
        del truncation, pad_to_multiple_of, return_attention_mask
        del padding, max_length, do_normalize, return_token_timestamps, kwargs

        if sampling_rate is not None and sampling_rate != self.sampling_rate:
            raise ValueError(
                f"The model corresponding to this feature extractor: "
                f"{self.__class__.__name__} was trained using a sampling "
                f"rate of {self.sampling_rate}. Please make sure that the "
                f"provided `raw_speech` input was sampled with "
                f"{self.sampling_rate} and not {sampling_rate}."
            )

        device_obj = torch.device(device or "cuda")
        clip_tensors: list[torch.Tensor] = []
        target_lengths: list[int] = []
        for speech in raw_speech:
            assert speech.ndim == 1
            audio_len = int(speech.shape[0])
            target_len = self._normalize_audio_length(audio_len)
            clip_tensors.append(
                torch.as_tensor(speech, dtype=torch.float32, device=device_obj))
            target_lengths.append(target_len)

        audio_lengths = torch.tensor(
            target_lengths,
            dtype=torch.long,
            device=device_obj,
        )
        max_audio_len = int(audio_lengths.max().item()) if target_lengths else 0
        input_features = torch.zeros(
            (len(clip_tensors), max_audio_len),
            dtype=torch.float32,
            device=device_obj,
        )
        for idx, clip in enumerate(clip_tensors):
            input_features[idx, :clip.numel()] = clip

        if self.preemphasis is not None:
            time_mask = (
                torch.arange(max_audio_len, device=device_obj).unsqueeze(0)
                < audio_lengths.unsqueeze(1)
            )
            input_features = torch.cat(
                [
                    input_features[:, :1],
                    input_features[:, 1:]
                    - self.preemphasis * input_features[:, :-1],
                ],
                dim=1,
            )
            input_features = input_features.masked_fill(~time_mask, 0.0)

        input_features = self._torch_extract_fbank_features(
            input_features, device_obj)
        feature_lengths = torch.floor_divide(
            audio_lengths + self.n_fft // 2 * 2 - self.n_fft,
            self.hop_length,
        )
        attention_mask = (
            torch.arange(input_features.shape[1], device=device_obj)[None, :]
            < feature_lengths[:, None]
        )
        mask = attention_mask.unsqueeze(-1)
        input_features_masked = input_features * mask
        mean = input_features_masked.sum(dim=1) / feature_lengths.unsqueeze(-1)
        mean = mean.unsqueeze(1)
        variance = (
            ((input_features_masked - mean) ** 2 * mask).sum(dim=1)
            / (feature_lengths - 1).unsqueeze(-1)
        )
        std = torch.sqrt(variance).unsqueeze(1)
        input_features = (input_features - mean) / (std + 1e-5)
        input_features *= mask

        return feature_extraction_utils.BatchFeature(
            data={
                "input_features": input_features,
                "attention_mask": attention_mask,
            },
            tensor_type=return_tensors,
        )

    def audio_length(self, audio_tokens: int) -> int:
        return int(audio_tokens * self.config.subsampling_factor * self.hop_length)
