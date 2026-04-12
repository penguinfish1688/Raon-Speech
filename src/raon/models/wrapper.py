# Copyright 2026 The RAON Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Offline inference helpers for raon text and audio generation."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypedDict, cast

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import trange
from transformers import (
    LogitsProcessorList,
    StaticCache,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeAudioEncoderConfig

from ..modules import EmbeddingAdaptorOutput
from ..modules.audio_encoder import AuTStreamingState, AuTWrapper
from ..modules.audio_tokenizer import (
    MimiConv1dPaddingCache,
    StaticMimiConv1dPaddingCache,
)
from ..modules.concurrent_audio_decoder import ConcurrentAudioDecoder
from ..modules.voxtral_encoder import VoxtralRealtimeEncoderConfig
from ..modules.voxtral_wrapper import VoxtralStreamingState, VoxtralWrapper
from ..utils.delay import undelay_audio_codes
from ..utils.misc import cast_to_module_dtype
from ..utils.special_tokens import (
    AUDIO_END,
    AUDIO_INPUT_PLACEHOLDER,
    AUDIO_OUTPUT_BC,
    AUDIO_OUTPUT_END_PAD,
    AUDIO_OUTPUT_PAD,
    AUDIO_OUTPUT_PLACEHOLDER,
    AUDIO_START,
    IM_END,
    IM_START,
    LOSS_IGNORE_INDEX,
    PAD,
)
from ..utils.state_machine import DuplexMachineState, DuplexPhase, DuplexStateConfig, DuplexStateManager

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..types import AudioDecoderOutput, AudioEncoderOutput, AudioTokenizerOutput
    from .raon import RaonModel


# ---------------------------------------------------------------------------
# Type definitions (from inference_utils.py)
# ---------------------------------------------------------------------------

# Set of audio special token IDs used to distinguish text tokens from audio tokens
_AUDIO_SPECIAL_TOKEN_IDS = {
    AUDIO_OUTPUT_PLACEHOLDER.id,
    AUDIO_INPUT_PLACEHOLDER.id,
    AUDIO_OUTPUT_PAD.id,
    AUDIO_OUTPUT_END_PAD.id,
}


class RaonLosses(TypedDict):
    """Training loss container: combined loss, text loss, and per-codebook audio loss.

    loss: Scalar combined loss. Dtype: float.
    text_loss: Scalar text LM loss. Dtype: float.
    audio_loss: Per-codebook audio loss. Shape: [num_code_groups]. Dtype: float.
    """

    loss: torch.Tensor
    text_loss: torch.Tensor
    audio_loss: torch.Tensor


AudioInputEncoderCache = tuple[StaticCache, StaticMimiConv1dPaddingCache | None] | AuTStreamingState | VoxtralStreamingState


@dataclass
class RaonDecodingState:
    """Mutable state for full-duplex streaming decoding.

    Tracks sequences, attention masks, audio codes, KV cache, encoder cache,
    decoder stream ID, sampling config, and acoustic-delay state.
    """

    sequences: torch.Tensor
    attention_mask: torch.Tensor
    audio_codes: torch.Tensor
    audio_codes_mask: torch.Tensor
    past_key_values: Any  # KV cache from the text model (DynamicCache or similar)
    audio_input_encoder_cache: AudioInputEncoderCache
    audio_decoder_stream_id: int
    do_sample: bool
    logits_processor: LogitsProcessorList
    num_code_groups: int = 8
    # For acoustic delay processing: stores semantic codes from previous frame
    semantic_buffer: torch.Tensor | None = None
    # Penalty to subtract from eos logit (higher = longer responses)
    eos_penalty: float = 0.0
    # Penalty to subtract from SIL logit (higher = less silence).
    sil_penalty: float = 0.0
    # Penalty to subtract from BC logit in SIL phase (positive = suppress, negative = boost).
    bc_penalty: float = 0.0
    # Mealy state machine for logit masking
    machine_state: DuplexMachineState | None = None
    # number of frames remaining where SIL must be forced (listen-first warmup)
    forced_sil_remaining: int = 0

    def _reset(self) -> None:
        """Reset the decoding state for reuse."""
        device = self.sequences.device
        self.sequences = torch.zeros(1, 0, dtype=torch.long, device=device)
        self.attention_mask = torch.zeros(1, 0, dtype=torch.long, device=device)
        self.audio_codes = torch.zeros(1, 0, self.num_code_groups, dtype=torch.long, device=device)
        self.audio_codes_mask = torch.zeros(1, 0, dtype=torch.bool, device=device)
        self.machine_state = None
        self.forced_sil_remaining = 0

        if self.semantic_buffer is not None:
            self.semantic_buffer = None

        if not isinstance(self.audio_input_encoder_cache, (AuTStreamingState, VoxtralStreamingState)):
            if self.audio_input_encoder_cache[0] is not None:
                self.audio_input_encoder_cache[0].reset()

            if self.audio_input_encoder_cache[1] is not None:
                self.audio_input_encoder_cache[1].reset()


# ---------------------------------------------------------------------------
# Output type definitions (from inference.py)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Sampling helpers (from inference_utils.py)
# ---------------------------------------------------------------------------


@torch.inference_mode()
def apply_repetition_aware_sampling(
    sampled_ids: torch.Tensor,
    logits: torch.Tensor,
    audio_codes: torch.Tensor,
    window_size: int,
    repetition_threshold: float,
    skip_frames: int = 0,
) -> torch.Tensor:
    """Resample overly repetitive first-code predictions."""
    result_ids = sampled_ids.clone()
    first_group_codes = audio_codes[:, skip_frames:, 0]

    for batch_index in range(sampled_ids.shape[0]):
        sampled_token = sampled_ids[batch_index, 0].item()
        codes_seq = first_group_codes[batch_index]
        window = codes_seq[max(0, codes_seq.shape[0] - window_size) :]
        if window.numel() == 0:
            continue

        repetition_ratio = (window == sampled_token).sum().item() / window.numel()
        if repetition_ratio > repetition_threshold:
            probs = F.softmax(logits[batch_index], dim=-1, dtype=torch.float32)
            probs = probs.clamp_min(torch.finfo(probs.dtype).tiny)
            result_ids[batch_index, 0] = torch.multinomial(probs, num_samples=1)[0]

    return result_ids


def make_audio_code_sampler(
    sequences: torch.Tensor,
    logits_processor: LogitsProcessorList,
    audio_codes: torch.Tensor,
    ras_enabled: bool,
    ras_window_size: int,
    ras_repetition_threshold: float,
    ras_skip_frames: int = 0,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build the sampler used for the first generated audio code."""
    def sample_audio_code(logits: torch.Tensor) -> torch.Tensor:
        processed_logits = (
            logits_processor(input_ids=sequences, scores=logits)  # type: ignore[arg-type]
            if len(logits_processor) > 0
            else logits
        )
        probs = F.softmax(processed_logits, dim=-1, dtype=torch.float32)
        probs = probs.clamp_min(torch.finfo(probs.dtype).tiny)
        sampled_ids = torch.multinomial(probs, num_samples=1)
        if ras_enabled and audio_codes.shape[1] > 0:
            sampled_ids = apply_repetition_aware_sampling(
                sampled_ids=sampled_ids,
                logits=logits,
                audio_codes=audio_codes,
                window_size=ras_window_size,
                repetition_threshold=ras_repetition_threshold,
                skip_frames=ras_skip_frames,
            )
        return sampled_ids

    return sample_audio_code


# ---------------------------------------------------------------------------
# Inference model (from inference.py)
# ---------------------------------------------------------------------------


class GenerateOutput(TypedDict):
    sequences: torch.Tensor
    audio_codes: torch.Tensor | None
    audio_codes_mask: torch.Tensor | None
    audio: torch.Tensor | None
    audio_lengths: torch.Tensor | None


class RaonInferenceModel(ABC):
    """Abstract base class for offline raon inference."""

    vocab_size: int
    codebook_size: int
    num_code_groups: int
    sampling_rate: int
    frame_rate: float
    tokenizer: Any | None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        assert hasattr(self, "vocab_size"), "Model must have vocab_size attribute."
        assert hasattr(self, "codebook_size"), "Model must have codebook_size attribute."
        assert hasattr(self, "num_code_groups"), "Model must have num_code_groups attribute."
        assert hasattr(self, "sampling_rate"), "Model must have sampling_rate attribute."
        assert hasattr(self, "frame_rate"), "Model must have frame_rate attribute."
        self.concurrent_audio_decoder: ConcurrentAudioDecoder | None = None

    @abstractmethod
    def inference_forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor,
        audio_input: torch.Tensor | None = None,
        audio_output: torch.Tensor | None = None,
        audio_input_lengths: torch.Tensor | None = None,
        audio_output_lengths: torch.Tensor | None = None,
        audio_output_codes: torch.Tensor | None = None,
        audio_output_codes_mask: torch.Tensor | None = None,
        audio_input_embeds: torch.Tensor | None = None,
        audio_input_embeds_mask: torch.Tensor | None = None,
        speaker_embeds: torch.Tensor | None = None,
        use_cache: bool | None = False,
        past_key_values: Any = None,
        cache_position: torch.Tensor | None = None,
        return_hidden_layer_means: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

    @abstractmethod
    def tokenize_audio(
        self,
        audio: torch.Tensor | None = None,
        audio_lengths: torch.Tensor | None = None,
        sampling_rate: int | None = None,
        num_code_groups: int = 8,
        return_mimi_features: bool = False,
        encoder_past_key_values: StaticCache | None = None,
        conv_padding_cache: StaticMimiConv1dPaddingCache | None = None,
        use_streaming: bool | None = None,
    ) -> AudioTokenizerOutput: ...

    @abstractmethod
    def get_audio_input_embeds(
        self,
        audio: torch.Tensor | None = None,
        audio_lengths: torch.Tensor | None = None,
        sampling_rate: int | None = None,
        num_code_groups: int = 8,
        encoder_past_key_values: StaticCache | None = None,
        conv_padding_cache: MimiConv1dPaddingCache | None = None,
        use_streaming: bool | None = None,
    ) -> AudioEncoderOutput: ...

    @abstractmethod
    def get_proj_code(self) -> nn.Linear:
        """Return the audio code projection layer."""
        ...

    @abstractmethod
    def decode_audio(
        self,
        audio_codes: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        use_streaming: bool | None = None,
    ) -> AudioDecoderOutput: ...

    @abstractmethod
    def generate_audio_codes(
        self,
        talker_last_hidden_state: torch.Tensor,
        first_code_sampler: Callable[[torch.Tensor], torch.Tensor] | None = None,
        allow_audio_end: bool = True,
    ) -> torch.Tensor: ...

    @abstractmethod
    def get_model(self) -> RaonModel: ...

    @abstractmethod
    def init_past_key_values(
        self,
        batch_size: int,
        max_sequence_length: int,
        prev_cache: Any | None = None,
    ) -> Any: ...

    @abstractmethod
    def free_past_key_values(self, past_key_values: Any) -> None: ...

    def start_concurrent_audio_decoder(self, timeout: float = 5.0) -> None:
        if self.concurrent_audio_decoder is None:
            self.concurrent_audio_decoder = ConcurrentAudioDecoder(self)

        if not self.concurrent_audio_decoder.is_running:
            self.concurrent_audio_decoder.start(timeout=timeout)

    def stop_concurrent_audio_decoder(self, timeout: float | None = 5.0) -> None:
        """Stop the background audio decoder worker and release resources."""
        if self.concurrent_audio_decoder is not None:
            self.concurrent_audio_decoder.stop(timeout=timeout)
            self.concurrent_audio_decoder = None

    def create_audio_decoder_stream(self) -> int:
        assert self.concurrent_audio_decoder is not None and self.concurrent_audio_decoder.is_running, (
            "Concurrent audio decoder must be running."
        )
        return self.concurrent_audio_decoder.create_stream()

    def _destroy_audio_decoder_stream(self, stream_id: int) -> None:
        assert self.concurrent_audio_decoder is not None and self.concurrent_audio_decoder.is_running, (
            "Concurrent audio decoder must be running."
        )
        self.concurrent_audio_decoder.destroy_stream(stream_id)

    def get_silence_codes(self, device: torch.device) -> torch.Tensor:
        """Return cached silence audio codes for keeping the decoder warm during SIL frames."""
        if not hasattr(self, "_silence_codes") or self._silence_codes is None:
            samples_per_frame = int(self.sampling_rate / self.frame_rate)
            silence_audio = torch.zeros(1, 1, samples_per_frame, device=device)
            silence_lengths = torch.tensor([samples_per_frame], device=device)
            with torch.no_grad():
                result = self.tokenize_audio(
                    audio=silence_audio,
                    audio_lengths=silence_lengths,
                    num_code_groups=self.num_code_groups,
                )
            self._silence_codes = result.audio_codes[0, 0].to(device)
        return self._silence_codes.to(device)

    def push_audio_codes(self, audio_codes: torch.Tensor, stream_id: int) -> None:
        assert self.concurrent_audio_decoder is not None and self.concurrent_audio_decoder.is_running, (
            "Concurrent audio decoder must be running."
        )
        assert audio_codes.ndim == 1 and audio_codes.shape[0] == self.num_code_groups, (
            f"Expected 1D audio codes with shape `[{self.num_code_groups}]` but got `{audio_codes.shape}`."
        )
        self.concurrent_audio_decoder.push_audio_codes(stream_id, audio_codes[None, None])

    def pull_audio(self, stream_id: int) -> torch.Tensor:
        assert self.concurrent_audio_decoder is not None and self.concurrent_audio_decoder.is_running, (
            "Concurrent audio decoder must be running."
        )
        results = self.concurrent_audio_decoder.drain_to(max_pending=1, stream_id=stream_id)
        assert len(results) == 1, f"Expected exactly one audio result but got `{len(results)}`."
        _, decoded_audio = results[0]
        return decoded_audio

    def _drain_audio_decoding_queue(self, stream_id: int) -> list[tuple[int, torch.Tensor]]:
        assert self.concurrent_audio_decoder is not None and self.concurrent_audio_decoder.is_running, (
            "Concurrent audio decoder must be running."
        )
        return self.concurrent_audio_decoder.drain_to(max_pending=0, stream_id=stream_id)

    def free_duplex_decoding_state(self, state: RaonDecodingState) -> None:
        self._drain_audio_decoding_queue(state.audio_decoder_stream_id)
        self._destroy_audio_decoder_stream(state.audio_decoder_stream_id)
        self.free_past_key_values(state.past_key_values)
        state._reset()

    def init_audio_encoder_cache(
        self,
        prev_cache: AudioInputEncoderCache | None = None,
    ) -> AudioInputEncoderCache:
        """Initialize or reset the audio input encoder cache for streaming.

        For Mimi encoders, returns a (StaticCache, conv_padding_cache) tuple.
        For causal AuT encoders, returns an AuTStreamingState.

        Args:
            prev_cache: Previous cache to reset and reuse. If None, creates fresh.

        Returns:
            Initialized encoder cache suitable for streaming audio input.
        """
        model = self.get_model()
        audio_encoder_config = model.config.audio_encoder_config

        if isinstance(audio_encoder_config, VoxtralRealtimeEncoderConfig):
            assert isinstance(model.audio_encoder, VoxtralWrapper)
            return model.audio_encoder.init_streaming_state()
        elif isinstance(audio_encoder_config, Qwen3OmniMoeAudioEncoderConfig):
            assert model.aut_is_causal, (
                "Duplex streaming requires a causal audio encoder. "
                "Set `aut_is_causal=True` in the model config to enable "
                "causal AuT streaming for duplex decoding."
            )
            assert isinstance(model.audio_encoder, AuTWrapper)
            return model.audio_encoder.init_streaming_state()
        else:
            if prev_cache is not None:
                assert isinstance(prev_cache, tuple), "Expected Mimi cache tuple."
                prev_cache[0].reset()
                if prev_cache[1] is not None:
                    prev_cache[1].reset()
                return prev_cache

            assert audio_encoder_config.sliding_window is not None
            past_key_values = StaticCache(
                audio_encoder_config,
                max_cache_len=audio_encoder_config.sliding_window,
            )
            return past_key_values, None

    def _streaming_tokenize_audio_with_cache(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor | None,
        audio_encoder_cache: tuple[StaticCache, StaticMimiConv1dPaddingCache],
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[StaticCache, StaticMimiConv1dPaddingCache]]:
        outputs = self.tokenize_audio(
            audio=audio,
            audio_lengths=audio_lengths,
            encoder_past_key_values=audio_encoder_cache[0],
            conv_padding_cache=audio_encoder_cache[1],
            use_streaming=True,
            num_code_groups=self.num_code_groups,
        )
        assert outputs.audio_codes is not None, "Expected `audio_codes` to be not None."
        assert outputs.audio_codes_mask is not None, "Expected `audio_codes_mask` to be not None."
        assert outputs.encoder_cache is not None, "Expected `encoder_cache` to be not None."
        assert isinstance(
            outputs.encoder_cache[0],
            StaticCache,
        ), f"Expected `encoder_cache[0]` to be `StaticCache` but got `{type(outputs.encoder_cache[0]).__name__}`."
        assert isinstance(outputs.encoder_cache[1], StaticMimiConv1dPaddingCache), (
            f"Expected `encoder_cache[1]` to be `StaticMimiConv1dPaddingCache` "
            f"but got `{type(outputs.encoder_cache[1]).__name__}`."
        )
        return (
            outputs.audio_codes,
            outputs.audio_codes_mask,
            (outputs.encoder_cache[0], outputs.encoder_cache[1]),
        )

    def streaming_tokenize_audio(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor | None,
        audio_encoder_cache: tuple[StaticCache, StaticMimiConv1dPaddingCache | None],
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[StaticCache, StaticMimiConv1dPaddingCache]]:
        """Tokenize audio in streaming mode with encoder cache for incremental encoding.

        Args:
            audio: Raw audio frame. Shape: [batch_size, num_samples]. Dtype: float.
            audio_lengths: Per-sample lengths. Shape: [batch_size]. Dtype: long.
            audio_encoder_cache: Current encoder cache (StaticCache, conv padding cache).

        Returns:
            Tuple of (audio_codes, audio_codes_mask, updated_encoder_cache).
            audio_codes: Shape [batch_size, num_frames, num_code_groups]. Dtype: long.
            audio_codes_mask: Shape [batch_size, num_frames]. Dtype: bool.
        """
        if audio_encoder_cache[1] is None or not audio_encoder_cache[1].is_initialized:
            outputs = self.tokenize_audio(
                audio=audio,
                audio_lengths=audio_lengths,
                encoder_past_key_values=audio_encoder_cache[0],
                conv_padding_cache=None,
                use_streaming=True,
                num_code_groups=self.num_code_groups,
            )
            assert outputs.audio_codes is not None, "`audio_codes` must not be None."
            assert outputs.audio_codes_mask is not None, "`audio_codes_mask` must not be None."
            assert outputs.encoder_cache is not None, "`encoder_cache` must not be None."
            assert isinstance(
                outputs.encoder_cache[0],
                StaticCache,
            ), f"Expected `encoder_cache[0]` to be `StaticCache` but got `{type(outputs.encoder_cache[0]).__name__}`."
            assert isinstance(
                dynamic_padding_cache := outputs.encoder_cache[1],
                MimiConv1dPaddingCache,
            ), (
                f"Expected `encoder_cache[1]` to be `MimiConv1dPaddingCache` "
                f"but got `{type(outputs.encoder_cache[1]).__name__}`."
            )

            if audio_encoder_cache[1] is None:
                static_conv_padding_cache = StaticMimiConv1dPaddingCache(
                    per_layer_padding=[int(padding) for padding in dynamic_padding_cache.per_layer_padding],
                    padding_cache=dynamic_padding_cache.padding_cache,  # type: ignore
                )
            else:
                audio_encoder_cache[1].initialize(dynamic_padding_cache.padding_cache)  # type: ignore
                static_conv_padding_cache = audio_encoder_cache[1]

            return (
                outputs.audio_codes,
                outputs.audio_codes_mask,
                (outputs.encoder_cache[0], static_conv_padding_cache),
            )
        else:
            return self._streaming_tokenize_audio_with_cache(
                audio=audio.flatten().view(audio.shape),
                audio_lengths=audio_lengths,
                audio_encoder_cache=audio_encoder_cache,  # type: ignore
            )

    def _streaming_get_audio_input_embeds_with_cache(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor | None,
        audio_encoder_cache: tuple[StaticCache, StaticMimiConv1dPaddingCache],
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[StaticCache, StaticMimiConv1dPaddingCache]]:
        outputs = self.get_audio_input_embeds(
            audio=audio,
            audio_lengths=audio_lengths,
            encoder_past_key_values=audio_encoder_cache[0],
            conv_padding_cache=audio_encoder_cache[1],
            use_streaming=True,
        )
        assert outputs.audio_embeds is not None, "Expected `audio_embeds` to be not None."
        assert outputs.audio_embeds_mask is not None, "Expected `audio_embeds_mask` to be not None."
        assert outputs.encoder_cache is not None, "Expected `encoder_cache` to be not None."
        assert isinstance(
            outputs.encoder_cache[0],
            StaticCache,
        ), f"Expected `encoder_cache[0]` to be `StaticCache` but got `{type(outputs.encoder_cache[0]).__name__}`."
        assert isinstance(outputs.encoder_cache[1], StaticMimiConv1dPaddingCache), (
            f"Expected `encoder_cache[1]` to be `StaticMimiConv1dPaddingCache` "
            f"but got `{type(outputs.encoder_cache[1]).__name__}`."
        )
        return (
            outputs.audio_embeds,
            outputs.audio_embeds_mask,
            (outputs.encoder_cache[0], outputs.encoder_cache[1]),
        )

    def _streaming_get_audio_input_embeds(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor | None,
        audio_encoder_cache: tuple[StaticCache, StaticMimiConv1dPaddingCache | None],
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[StaticCache, StaticMimiConv1dPaddingCache]]:
        if audio_encoder_cache[1] is None or not audio_encoder_cache[1].is_initialized:
            outputs = self.get_audio_input_embeds(
                audio=audio,
                audio_lengths=audio_lengths,
                encoder_past_key_values=audio_encoder_cache[0],
                conv_padding_cache=None,
                use_streaming=True,
            )
            assert outputs.audio_embeds is not None, "Expected `audio_embeds` to be not None."
            assert outputs.audio_embeds_mask is not None, "Expected `audio_embeds_mask` to be not None."
            assert outputs.encoder_cache is not None, "Expected `encoder_cache` to be not None."
            assert isinstance(
                outputs.encoder_cache[0],
                StaticCache,
            ), f"Expected `encoder_cache[0]` to be `StaticCache` but got `{type(outputs.encoder_cache[0]).__name__}`."
            assert isinstance(
                dynamic_padding_cache := outputs.encoder_cache[1],
                MimiConv1dPaddingCache,
            ), (
                f"Expected `encoder_cache[1]` to be `MimiConv1dPaddingCache` "
                f"but got `{type(outputs.encoder_cache[1]).__name__}`."
            )

            if audio_encoder_cache[1] is None:
                static_conv_padding_cache = StaticMimiConv1dPaddingCache(
                    per_layer_padding=[int(padding) for padding in dynamic_padding_cache.per_layer_padding],
                    padding_cache=dynamic_padding_cache.padding_cache,  # type: ignore
                )
            else:
                audio_encoder_cache[1].initialize(dynamic_padding_cache.padding_cache)  # type: ignore
                static_conv_padding_cache = audio_encoder_cache[1]

            return (
                outputs.audio_embeds,
                outputs.audio_embeds_mask,
                (outputs.encoder_cache[0], static_conv_padding_cache),
            )
        else:
            return self._streaming_get_audio_input_embeds_with_cache(
                audio=audio.flatten().view(audio.shape),
                audio_lengths=audio_lengths,
                audio_encoder_cache=audio_encoder_cache,  # type: ignore
            )

    def _streaming_get_audio_input_embeds_aut(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor | None,
        streaming_state: AuTStreamingState,
    ) -> tuple[torch.Tensor, torch.Tensor, AuTStreamingState]:
        """Get audio input embeddings using AuT causal streaming encoder.

        Runs the AuT wrapper in streaming mode (one chunk at a time) and passes
        the output through the input adaptor.

        Args:
            audio: Raw audio frame. Shape: [1, num_samples]. Dtype: float.
            audio_lengths: Per-sample lengths. Shape: [1]. Dtype: long.
            streaming_state: Current AuT streaming state.

        Returns:
            Tuple of (audio_embeds, audio_embeds_mask, updated_streaming_state).
            audio_embeds: Shape [1, num_frames, hidden_size]. Dtype: float.
            audio_embeds_mask: Shape [1, num_frames]. Dtype: bool.
        """
        model = self.get_model()
        assert isinstance(model.audio_encoder, AuTWrapper)

        # Reshape audio to [batch_size, num_channels, num_samples] for the wrapper.
        if audio.ndim == 2:
            audio_3d = audio[:, None, :]
        else:
            audio_3d = audio

        audio_3d = cast_to_module_dtype(audio_3d, model.audio_encoder)

        encoder_outputs = model.audio_encoder(
            audio_3d,
            streaming_state=streaming_state,
        )

        assert encoder_outputs.embeds is not None
        audio_embeds = encoder_outputs.embeds  # [1, new_frames, output_dim]
        updated_state = encoder_outputs.streaming_state
        assert isinstance(updated_state, AuTStreamingState)

        # Build mask: all frames are valid.
        audio_embeds_mask = torch.ones(
            audio_embeds.shape[:2],
            dtype=torch.bool,
            device=audio_embeds.device,
        )

        # Pass through input adaptor.
        assert model.input_adaptor is not None, "input_adaptor is unavailable when supports_audio_input is False."
        adaptor_outputs = model.input_adaptor(audio_embeds, mask=audio_embeds_mask)
        assert isinstance(adaptor_outputs, EmbeddingAdaptorOutput)
        assert (audio_embeds := adaptor_outputs.outputs_embeds) is not None
        assert (audio_embeds_mask := adaptor_outputs.mask) is not None  # type: ignore

        return audio_embeds, audio_embeds_mask, updated_state

    def _streaming_get_audio_input_embeds_voxtral(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor | None,
        streaming_state: VoxtralStreamingState,
    ) -> tuple[torch.Tensor, torch.Tensor, VoxtralStreamingState]:
        """Get audio input embeddings using Voxtral causal streaming encoder.

        Runs the Voxtral wrapper in streaming mode (one chunk at a time) and
        passes the output through the input adaptor.

        Args:
            audio: Raw audio frame. Shape: [1, num_samples]. Dtype: float.
            audio_lengths: Per-sample lengths. Shape: [1]. Dtype: long.
            streaming_state: Current Voxtral streaming state.

        Returns:
            Tuple of (audio_embeds, audio_embeds_mask, updated_streaming_state).
            audio_embeds: Shape [1, num_frames, hidden_size]. Dtype: float.
            audio_embeds_mask: Shape [1, num_frames]. Dtype: bool.
        """
        model = self.get_model()
        assert isinstance(model.audio_encoder, VoxtralWrapper)

        if audio.ndim == 2:
            audio_3d = audio[:, None, :]
        else:
            audio_3d = audio

        audio_3d = cast_to_module_dtype(audio_3d, model.audio_encoder)

        encoder_outputs = model.audio_encoder(
            audio_3d,
            streaming_state=streaming_state,
        )

        assert encoder_outputs.embeds is not None
        audio_embeds = encoder_outputs.embeds
        updated_state = encoder_outputs.streaming_state
        assert isinstance(updated_state, VoxtralStreamingState)

        audio_embeds_mask = torch.ones(
            audio_embeds.shape[:2],
            dtype=torch.bool,
            device=audio_embeds.device,
        )

        assert model.input_adaptor is not None, "input_adaptor is unavailable when supports_audio_input is False."
        adaptor_outputs = model.input_adaptor(audio_embeds, mask=audio_embeds_mask)
        assert isinstance(adaptor_outputs, EmbeddingAdaptorOutput)
        assert (audio_embeds := adaptor_outputs.outputs_embeds) is not None
        assert (audio_embeds_mask := adaptor_outputs.mask) is not None  # type: ignore

        return audio_embeds, audio_embeds_mask, updated_state

    def compile_audio_modules(self, duplex: bool = True, max_sequence_length: int = 8192) -> RaonDecodingState | None:
        """torch.compile audio code predictor and optionally duplex streaming path; run warmup.

        Args:
            duplex: If True, compile duplex streaming path and run warmup; else only code predictor.
            max_sequence_length: Max sequence length for KV cache during warmup.

        Returns:
            RaonDecodingState after warmup if duplex=True; None otherwise.
        """
        model = self.get_model()
        code_predictor = model.code_predictor
        assert code_predictor is not None, "compile_audio_modules requires a model with audio output support."

        code_predictor.generate_greedy = torch.compile(
            code_predictor.generate_greedy,
            fullgraph=True,
            dynamic=False,
            backend="inductor",
            mode="max-autotune",
        )

        if isinstance(model.audio_encoder, VoxtralWrapper):
            model.audio_encoder.compile_encoder()

        if duplex:
            self._streaming_get_audio_input_embeds_with_cache = torch.compile(
                self._streaming_get_audio_input_embeds_with_cache,
                fullgraph=False,
                dynamic=None,
                backend="inductor",
                mode="default",
            )

            with torch.inference_mode():
                device = self.get_model().device
                dtype = self.get_model().dtype
                samples_per_frame = int(self.sampling_rate / self.frame_rate)

                safe_vocab = min(self.vocab_size, 1000)
                state = self.init_duplex_decoding_state(
                    sequences=torch.randint(0, safe_vocab, (1, 1), device=device),
                    attention_mask=torch.ones(1, 1, dtype=torch.long, device=device),
                    do_sample=False,
                    max_sequence_length=max_sequence_length,
                )

                is_uta_warmup = getattr(self, "sequence_mode", "tua") == "uta"
                text_warmup_pos = -2 if is_uta_warmup else -3
                for step in trange(8, desc="Warmup 1/3", mininterval=0):
                    state, _ = self.duplex_decoding_step(
                        state=state,
                        audio_input=torch.randn(1, samples_per_frame, device=device, dtype=dtype),
                    )
                    if state.sequences.shape[1] >= 3:
                        if step % 2 == 0:
                            state.sequences[0, text_warmup_pos] = 0
                        else:
                            state.sequences[0, text_warmup_pos] = AUDIO_OUTPUT_PLACEHOLDER.id

                self._drain_audio_decoding_queue(state.audio_decoder_stream_id)

                state = self.init_duplex_decoding_state(
                    sequences=torch.randint(0, safe_vocab, (1, 20), device=device),
                    attention_mask=torch.ones(1, 20, dtype=torch.long, device=device),
                    do_sample=False,
                    max_sequence_length=max_sequence_length,
                    prev_state=state,
                )
                for step in trange(8, desc="Warmup 2/3", mininterval=0):
                    state, _ = self.duplex_decoding_step(
                        state=state,
                        audio_input=torch.randn(1, samples_per_frame, device=device, dtype=dtype),
                    )
                    if state.sequences.shape[1] >= 3:
                        if step % 2 == 0:
                            state.sequences[0, text_warmup_pos] = 0
                        else:
                            state.sequences[0, text_warmup_pos] = AUDIO_OUTPUT_PLACEHOLDER.id

                self._drain_audio_decoding_queue(state.audio_decoder_stream_id)

                audio_input_encoder_cache: AudioInputEncoderCache = state.audio_input_encoder_cache
                for _ in trange(256, desc="Warmup 3/3", mininterval=0):
                    if isinstance(audio_input_encoder_cache, AuTStreamingState):
                        _, _, audio_input_encoder_cache = self._streaming_get_audio_input_embeds_aut(
                            audio=torch.randn(1, samples_per_frame, device=device, dtype=dtype),
                            audio_lengths=torch.tensor([samples_per_frame], device=device),
                            streaming_state=audio_input_encoder_cache,
                        )
                    elif isinstance(audio_input_encoder_cache, VoxtralStreamingState):
                        _, _, audio_input_encoder_cache = self._streaming_get_audio_input_embeds_voxtral(
                            audio=torch.randn(1, samples_per_frame, device=device, dtype=dtype),
                            audio_lengths=torch.tensor([samples_per_frame], device=device),
                            streaming_state=audio_input_encoder_cache,
                        )
                    else:
                        _, _, audio_input_encoder_cache = self._streaming_get_audio_input_embeds(
                            audio=torch.randn(1, samples_per_frame, device=device, dtype=dtype),
                            audio_lengths=torch.tensor([samples_per_frame], device=device),
                            audio_encoder_cache=audio_input_encoder_cache,
                        )

                self.free_duplex_decoding_state(state)
                return state

        return None

    def _compute_speaker_embeds(
        self,
        speaker_audio: torch.Tensor,
        speaker_audio_lengths: torch.Tensor | None,
    ) -> torch.Tensor:
        from ..modules import PretrainedSpeakerEncoder

        model = self.get_model()
        assert model.speaker_encoder is not None, "_compute_speaker_embeds requires a model with a speaker encoder."

        if speaker_audio_lengths is None:
            speaker_audio_lengths = torch.full(
                (speaker_audio.shape[0],),
                speaker_audio.shape[1],
                device=speaker_audio.device,
                dtype=torch.long,
            )

        if isinstance(model.speaker_encoder, PretrainedSpeakerEncoder):
            return model.speaker_encoder(speaker_audio, speaker_audio_lengths)

        tokenizer_output = self.tokenize_audio(
            audio=speaker_audio,
            audio_lengths=speaker_audio_lengths,
            return_mimi_features=True,
        )
        return model.speaker_encoder(
            tokenizer_output.mimi_features,
            mask=tokenizer_output.audio_codes_mask,
        )

    def _pad_audio_input(self, audio_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        sampling_rate = self.sampling_rate
        frame_rate: float = self.frame_rate
        assert (samples_per_frame := int(sampling_rate / frame_rate)) == sampling_rate / frame_rate, (
            f"Expected `sampling_rate / frame_rate` to be an integer but got `{sampling_rate / frame_rate}`."
        )

        if audio_input.shape[1] < samples_per_frame:
            audio_input = F.pad(audio_input, (0, samples_per_frame - audio_input.shape[1]))
            logger.warning(
                f"Duplex decoding uses {samples_per_frame} samples per frame, "
                f"but {audio_input.shape[1]} samples were input. "
                "The input audio has been padded accordingly."
            )
        elif audio_input.shape[1] > samples_per_frame:
            audio_input = audio_input[:, :samples_per_frame]
            logger.warning(
                f"Duplex decoding uses {samples_per_frame} samples per frame, "
                f"but {audio_input.shape[1]} samples were input. "
                "The input audio has been truncated accordingly."
            )

        audio_input_lengths = torch.tensor([audio_input.shape[1]], device=audio_input.device)
        return audio_input, audio_input_lengths

    @torch.inference_mode()
    def _update_duplex_sequences_and_generate_audio_codes(
        self,
        new_logits: torch.Tensor,
        new_last_hidden_state: torch.Tensor,
        sequences: torch.Tensor,
        attention_mask: torch.Tensor,
        audio_codes: torch.Tensor,
        audio_codes_mask: torch.Tensor,
        do_sample: bool,
        logits_processor: LogitsProcessorList,
        eos_penalty: float = 0.0,
        sil_penalty: float = 0.0,
        bc_penalty: float = 0.0,
        machine_state: DuplexMachineState | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, DuplexMachineState | None]:
        if new_logits.shape[0] != 1:
            raise NotImplementedError(f"Only batch size 1 is supported but got `{new_logits.shape[0]}`.")

        # Text prediction is always read from the token before [A].
        user_logits = new_logits[:, -2:-1, : self.vocab_size]
        # Apply pad penalty to reduce pad probability (encourage longer responses)
        if eos_penalty > 0:
            user_logits = user_logits.clone()
            user_logits[:, :, self.duplex_pad_token_id] -= eos_penalty
        if sil_penalty > 0 and getattr(self, "use_sil_token", False):
            user_logits = user_logits.clone()
            user_logits[:, :, self.duplex_sil_token_id] -= sil_penalty
        # Apply BC penalty to adjust backchannel probability in SIL phase.
        # Positive values suppress BC, negative values boost BC.
        if (
            bc_penalty != 0
            and getattr(self, "use_backchannel_token", False)
            and machine_state is not None
            and machine_state.phase == DuplexPhase.SIL
        ):
            user_logits = user_logits.clone()
            user_logits[:, :, self.duplex_bc_token_id] -= bc_penalty

        # Apply state-machine logit masking to enforce valid transitions.
        if machine_state is not None and self.use_duplex_end_pad:
            user_logits = self._state_manager.apply_logit_mask(user_logits, machine_state, self.vocab_size)

        if do_sample:
            user_logits = logits_processor(input_ids=sequences, scores=user_logits[:, -1])  # type: ignore
            user_probs = F.softmax(user_logits, dim=-1, dtype=torch.float32)
            user_probs = user_probs.clamp_min(torch.finfo(user_probs.dtype).tiny)
            text_or_eos_id = torch.multinomial(user_probs, num_samples=1)
        else:
            text_or_eos_id = user_logits[:, -1:].argmax(dim=-1)

        predicted_token_id = int(text_or_eos_id.item())

        # Determine current speech phase for conditional audio code generation.
        is_in_speech = machine_state is not None and machine_state.phase == DuplexPhase.SPEECH

        # Only generate audio codes when currently in speech (pre-transition).
        # For onset frames (SIL→SPEECH), fresh codes are generated after transition.
        new_audio_codes = None
        if is_in_speech:
            first_code_sampler = None
            if do_sample:
                first_code_sampler = make_audio_code_sampler(
                    sequences=sequences,
                    logits_processor=logits_processor,
                    audio_codes=audio_codes,
                    ras_enabled=False,
                    ras_window_size=40,
                    ras_repetition_threshold=0.1,
                )
            new_audio_codes = self.generate_audio_codes(
                talker_last_hidden_state=new_last_hidden_state[:, -1:],
                first_code_sampler=first_code_sampler,
                allow_audio_end=False,
            )
            new_audio_codes = new_audio_codes.clone()

        new_machine_state: DuplexMachineState | None = None
        # Use state machine for frame construction
        new_machine_state, frame_tokens, emitted_audio = self._state_manager.transition(
            machine_state, predicted_token_id, sequences.device
        )
        input_ids = torch.tensor([frame_tokens], device=sequences.device)

        # Conditionally append audio codes based on whether audio was emitted.
        if emitted_audio:
            if new_audio_codes is None:
                # Onset frame (SIL→SPEECH): generate fresh codes now.
                first_code_sampler = None
                if do_sample:
                    first_code_sampler = make_audio_code_sampler(
                        sequences=sequences,
                        logits_processor=logits_processor,
                        audio_codes=audio_codes,
                        ras_enabled=False,
                        ras_window_size=40,
                        ras_repetition_threshold=0.1,
                    )
                new_audio_codes = self.generate_audio_codes(
                    talker_last_hidden_state=new_last_hidden_state[:, -1:],
                    first_code_sampler=first_code_sampler,
                    allow_audio_end=False,
                )
                new_audio_codes = new_audio_codes.clone()
            audio_end_predicted_mask = new_audio_codes[:, 0] == self.codebook_size
            if audio_end_predicted_mask.any():
                # Clamp audio-end sentinel back into Mimi codebook range.
                new_audio_codes[audio_end_predicted_mask, 0] = 0
            audio_codes = torch.cat((audio_codes, new_audio_codes[None]), dim=1)
            audio_codes_mask = torch.cat(
                (
                    audio_codes_mask,
                    torch.tensor([[True]], device=audio_codes.device, dtype=torch.bool),
                ),
                dim=1,
            )
        # else: SIL frame — no audio codes generated or appended (matches Reference)

        sequences = torch.cat((sequences, input_ids), dim=1)
        attention_mask = F.pad(attention_mask, (0, input_ids.shape[1]), value=1)
        return input_ids, sequences, attention_mask, audio_codes, audio_codes_mask, new_machine_state

    @torch.inference_mode()
    def duplex_decoding_step(
        self,
        state: RaonDecodingState,
        audio_input: torch.Tensor,
        hidden_collector: list[dict[str, torch.Tensor]] | None = None,
    ) -> tuple[RaonDecodingState, torch.Tensor]:
        """Run one duplex decoding step: encode user audio, predict tokens/codes, push codes, pull waveform.

        Args:
            state: Current duplex decoding state.
            audio_input: One frame of user audio. Shape: [1, num_samples_per_frame]. Dtype: float.

        Returns:
            Tuple of (updated_state, decoded_audio).
            decoded_audio: Decoded waveform for this frame. Shape: [1, num_samples_per_frame]. Dtype: float.
        """
        if state.sequences.shape[0] != 1:
            raise NotImplementedError(f"Only batch size 1 is supported but got `{state.sequences.shape[0]}`.")

        last_token = int(state.sequences[0, -1].item())
        valid_last_tokens = {AUDIO_OUTPUT_PLACEHOLDER.id, AUDIO_START.id}
        if last_token not in valid_last_tokens:
            raise ValueError(f"Last token must be one of `{sorted(valid_last_tokens)}` but got `{last_token}`.")

        prev_audio_codes_length = state.audio_codes.shape[1]

        audio_input, audio_input_lengths = self._pad_audio_input(audio_input=audio_input)

        # Dispatch to AuT, Voxtral, or Mimi streaming encoder based on cache type.
        audio_input_encoder_cache: AudioInputEncoderCache
        if isinstance(state.audio_input_encoder_cache, AuTStreamingState):
            audio_input_embeds, audio_input_embeds_mask, audio_input_encoder_cache = (
                self._streaming_get_audio_input_embeds_aut(
                    audio=audio_input,
                    audio_lengths=audio_input_lengths,
                    streaming_state=state.audio_input_encoder_cache,
                )
            )
        elif isinstance(state.audio_input_encoder_cache, VoxtralStreamingState):
            audio_input_embeds, audio_input_embeds_mask, audio_input_encoder_cache = (
                self._streaming_get_audio_input_embeds_voxtral(
                    audio=audio_input,
                    audio_lengths=audio_input_lengths,
                    streaming_state=state.audio_input_encoder_cache,
                )
            )
        else:
            audio_input_embeds, audio_input_embeds_mask, audio_input_encoder_cache = self._streaming_get_audio_input_embeds(
                audio=audio_input,
                audio_lengths=audio_input_lengths,
                audio_encoder_cache=state.audio_input_encoder_cache,
            )

        # Determine num_input_tokens from state machine
        num_input_tokens = state.machine_state.num_input_tokens
        has_text_input = num_input_tokens == 3

        step_audio_codes = state.audio_codes[:, -1:] if state.audio_codes.shape[1] > 0 else None
        step_audio_codes_mask = state.audio_codes_mask[:, -1:] if state.audio_codes_mask.shape[1] > 0 else None

        full_position_ids = state.attention_mask.cumsum(dim=1) - 1
        seq_len = state.attention_mask.shape[1]
        cache_position = torch.arange(seq_len - num_input_tokens, seq_len, device=state.sequences.device)
        step_input_ids = state.sequences[:, -num_input_tokens:]
        inference_outputs = self.inference_forward(
            return_hidden_layer_means=hidden_collector is not None,
            input_ids=state.sequences[:, -num_input_tokens:],
            attention_mask=None,
            position_ids=full_position_ids[:, -num_input_tokens:],
            audio_output_codes=step_audio_codes,
            audio_output_codes_mask=step_audio_codes_mask,
            audio_input_embeds=audio_input_embeds,
            audio_input_embeds_mask=audio_input_embeds_mask,
            speaker_embeds=None,
            use_cache=True,
            past_key_values=state.past_key_values,
            cache_position=cache_position,
        )
        if hidden_collector is not None:
            talker_last_hidden_state, text_logits, hidden_layer_means = cast(
                tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                inference_outputs,
            )
            hidden_collector.append(
                {
                    "input_token_ids": step_input_ids[0].detach().cpu().long(),
                    "text_hidden_layers": hidden_layer_means[0].detach().cpu().float(),
                }
            )
        else:
            talker_last_hidden_state, text_logits = cast(
                tuple[torch.Tensor, torch.Tensor],
                inference_outputs,
            )

        # Force SIL for the remaining listen-first warmup frames.
        if state.forced_sil_remaining > 0 and getattr(self, "use_sil_token", False):
            forced_logits = torch.full_like(text_logits, fill_value=-1e9)
            forced_logits[:, -2, self.duplex_sil_token_id] = 0.0
            text_logits = forced_logits

        # Standard mode (with optional EPAD support)
        new_machine_state: DuplexMachineState | None = state.machine_state
        _, sequences, attention_mask, audio_codes, audio_codes_mask, new_machine_state = (
            self._update_duplex_sequences_and_generate_audio_codes(
                new_logits=text_logits,
                new_last_hidden_state=talker_last_hidden_state,
                sequences=state.sequences,
                attention_mask=state.attention_mask,
                audio_codes=state.audio_codes,
                audio_codes_mask=state.audio_codes_mask,
                do_sample=state.do_sample,
                logits_processor=state.logits_processor,
                eos_penalty=state.eos_penalty,
                sil_penalty=state.sil_penalty,
                bc_penalty=state.bc_penalty,
                machine_state=state.machine_state,
            )
        )

        # Detect if current frame is SIL-no-audio using state machine
        is_current_sil_no_audio = not new_machine_state.emitted_audio

        if is_current_sil_no_audio:
            # Clear semantic buffer so post-SIL onset starts fresh (matches Reference).
            new_semantic_buffer = None
            # Push silence codes to keep decoder conv state warm (matches Reference).
            silence_codes = self.get_silence_codes(state.sequences.device)
            self.push_audio_codes(audio_codes=silence_codes, stream_id=state.audio_decoder_stream_id)
            decoded_audio = self.pull_audio(state.audio_decoder_stream_id)
            if decoded_audio.device != state.sequences.device:
                decoded_audio = decoded_audio.to(state.sequences.device)
        else:
            # Normal speech frame
            valid_trailing_tokens = {AUDIO_OUTPUT_PLACEHOLDER.id, AUDIO_START.id}
            assert int(sequences[0, -1].item()) in valid_trailing_tokens, (
                f"Last token must be one of `{sorted(valid_trailing_tokens)}` but got `{sequences[0, -1]}`."
            )
            # Audio codes grow by 1 only when emitted_audio=True (speech frames).
            expected_codes = prev_audio_codes_length + 1
            assert audio_codes.shape[1] == expected_codes, (
                f"Expected `{expected_codes}` audio codes but got `{audio_codes.shape[1]}`."
            )

            # Handle acoustic delay if configured
            new_semantic_buffer = state.semantic_buffer
            if self.max_delay > 0:
                current_codes = audio_codes[0, -1]
                semantic_code = current_codes[0:1]
                acoustic_codes = current_codes[1:]

                if state.semantic_buffer is None:
                    new_semantic_buffer = semantic_code
                    output_codes = torch.zeros_like(current_codes)
                    output_codes[0] = semantic_code[0]
                else:
                    output_codes = torch.cat([state.semantic_buffer, acoustic_codes], dim=0)
                    new_semantic_buffer = semantic_code

                self.push_audio_codes(audio_codes=output_codes, stream_id=state.audio_decoder_stream_id)
            else:
                self.push_audio_codes(audio_codes=audio_codes[0, -1], stream_id=state.audio_decoder_stream_id)

            decoded_audio = self.pull_audio(state.audio_decoder_stream_id)

        updated_state = RaonDecodingState(
            sequences=sequences,
            attention_mask=attention_mask,
            audio_codes=audio_codes,
            audio_codes_mask=audio_codes_mask,
            past_key_values=state.past_key_values,
            audio_input_encoder_cache=audio_input_encoder_cache,
            audio_decoder_stream_id=state.audio_decoder_stream_id,
            do_sample=state.do_sample,
            logits_processor=state.logits_processor,
            num_code_groups=state.num_code_groups,
            semantic_buffer=new_semantic_buffer,
            eos_penalty=state.eos_penalty,
            sil_penalty=state.sil_penalty,
            bc_penalty=state.bc_penalty,
            machine_state=new_machine_state,
            forced_sil_remaining=max(0, state.forced_sil_remaining - 1),
        )

        return updated_state, decoded_audio

    def init_duplex_decoding_state(
        self,
        sequences: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_k: int = 20,
        top_p: float = 0.8,
        max_sequence_length: int = 8192,
        prev_state: RaonDecodingState | None = None,
        eos_penalty: float = 0.0,
        sil_penalty: float = 0.0,
        bc_penalty: float = 0.0,
        speaker_embeds: torch.Tensor | None = None,
        speak_first: bool = False,
    ) -> RaonDecodingState:
        """Initialize duplex decoding state and run the first frame to obtain [U][A] prompt.

        Args:
            sequences: Initial text tokens (system prompt). Shape: [1, seq_length]. Dtype: long.
            attention_mask: Mask for valid positions. Shape: [1, seq_length]. Dtype: long.
            do_sample: Whether to sample (vs. greedy).
            temperature: Sampling temperature.
            top_k: Top-k filtering.
            top_p: Top-p filtering.
            max_sequence_length: Max sequence length for KV cache.
            prev_state: Previous state to reuse caches (e.g. from warmup).
            eos_penalty: Penalty to subtract from pad/eos logit to encourage longer output.
            sil_penalty: Penalty to subtract from SIL logit.

        Returns:
            RaonDecodingState ready for duplex_decoding_step.
        """
        self.start_concurrent_audio_decoder()

        # Lazy init state manager for logit masking
        if not hasattr(self, "_state_manager"):
            self._state_manager = DuplexStateManager(
                DuplexStateConfig(
                    use_duplex_end_pad=self.use_duplex_end_pad,
                    use_sil_token=getattr(self, "use_sil_token", False),
                    no_audio_in_sil=getattr(self, "no_audio_in_sil", False),
                    sequence_mode=getattr(self, "sequence_mode", "tua"),
                    duplex_pad_token_id=self.duplex_pad_token_id,
                    duplex_end_pad_token_id=self.duplex_end_pad_token_id,
                    duplex_sil_token_id=getattr(self, "duplex_sil_token_id", -1),
                    use_backchannel_token=getattr(self, "use_backchannel_token", False),
                    duplex_bc_token_id=getattr(self, "duplex_bc_token_id", AUDIO_OUTPUT_BC.id),
                )
            )

        if isinstance(self.get_model().config.audio_encoder_config, Qwen3OmniMoeAudioEncoderConfig):
            aut_is_causal = getattr(self.get_model(), "aut_is_causal", False)
            if not aut_is_causal:
                raise NotImplementedError(
                    "Duplex streaming decoding requires a causal audio encoder. "
                    "Set `aut_is_causal=True` in the model config to enable "
                    "causal AuT streaming for duplex decoding."
                )

        if sequences.shape[0] != 1:
            raise NotImplementedError(f"Only batch size 1 is supported but got `{sequences.shape[0]}`.")

        if self.max_delay > 1:
            raise NotImplementedError(
                f"Duplex decoding only supports acoustic_delay of 0 or 1, "
                f"got max_delay={self.max_delay}. semantic_buffer assumes single-step delay."
            )
        if self.max_delay > 0 and self.delays[0] != 0:
            raise ValueError(
                f"Semantic codebook (index 0) must have delay=0 for duplex decoding, "
                f"got delays[0]={self.delays[0]}. The semantic_buffer logic assumes "
                f"delays=[0, N, N, ..., N]."
            )

        # Auto-insert speaker token if speaker_embeds provided but not in sequences
        if speaker_embeds is not None and self.speaker_token_id is not None:
            if not (sequences == self.speaker_token_id).any():
                speaker_token = torch.full(
                    (sequences.shape[0], 1),
                    fill_value=self.speaker_token_id,
                    dtype=sequences.dtype,
                    device=sequences.device,
                )
                sequences = torch.cat((sequences, speaker_token), dim=1)
                if attention_mask is not None:
                    attention_mask = F.pad(attention_mask, (0, 1), value=1)

        if attention_mask is None:
            attention_mask = torch.ones_like(sequences)

        _audio_ids = torch.tensor(list(_AUDIO_SPECIAL_TOKEN_IDS), device=sequences.device)
        assert not torch.isin(sequences, _audio_ids).any() and (attention_mask == 1).all(), (
            "All `sequences` must be text tokens and all `attention_mask` values must be 1. "
            f"`{sequences=}`, `{attention_mask=}`."
        )

        logits_processor = LogitsProcessorList()
        if do_sample and temperature and temperature != 1.0:
            logits_processor.append(TemperatureLogitsWarper(temperature=temperature))
        if do_sample and top_k and top_k > 0:
            logits_processor.append(TopKLogitsWarper(top_k=top_k))
        if do_sample and top_p and top_p < 1.0:
            logits_processor.append(TopPLogitsWarper(top_p=top_p))

        if prev_state is not None:
            past_key_values = self.init_past_key_values(
                batch_size=1,
                max_sequence_length=max_sequence_length,
                prev_cache=prev_state.past_key_values,
            )
            audio_input_encoder_cache = self.init_audio_encoder_cache(prev_cache=prev_state.audio_input_encoder_cache)
            self._drain_audio_decoding_queue(stream_id=prev_state.audio_decoder_stream_id)
            self._destroy_audio_decoder_stream(prev_state.audio_decoder_stream_id)
        else:
            past_key_values = self.init_past_key_values(batch_size=1, max_sequence_length=max_sequence_length)
            audio_input_encoder_cache = self.init_audio_encoder_cache()

        audio_decoder_stream_id = self.create_audio_decoder_stream()

        audio_codes = torch.zeros(1, 0, self.num_code_groups, dtype=torch.long, device=sequences.device)
        audio_codes_mask = torch.zeros(1, 0, dtype=torch.bool, device=sequences.device)

        input_ids = torch.cat(
            [
                sequences,
                torch.tensor(
                    [[IM_START.id, AUDIO_START.id]],
                    device=sequences.device,
                ),
            ],
            dim=1,
        )
        position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        cache_position = torch.arange(input_ids.shape[1], device=input_ids.device)

        talker_last_hidden_state, text_logits = cast(
            tuple[torch.Tensor, torch.Tensor],
            self.inference_forward(
            input_ids=input_ids,
            attention_mask=None,
            position_ids=position_ids,
            speaker_embeds=speaker_embeds,
            use_cache=True,
            past_key_values=past_key_values,
            cache_position=cache_position,
            ),
        )

        initial_machine_state = self._state_manager.initial_state(speak_first=speak_first)

        # Force the first [U] prediction explicitly.
        forced_initial_prediction_id = self._state_manager.initial_forced_prediction_id(speak_first)
        if forced_initial_prediction_id is not None:
            forced_logits = torch.full_like(text_logits, fill_value=-1e9)
            forced_logits[:, -2, forced_initial_prediction_id] = 0.0
            text_logits = forced_logits

        _, sequences, attention_mask, audio_codes, audio_codes_mask, initial_machine_state = (
            self._update_duplex_sequences_and_generate_audio_codes(
                new_logits=text_logits,
                new_last_hidden_state=talker_last_hidden_state,
                sequences=input_ids,
                attention_mask=torch.ones_like(input_ids),
                audio_codes=audio_codes,
                audio_codes_mask=audio_codes_mask,
                do_sample=do_sample,
                logits_processor=logits_processor,
                machine_state=initial_machine_state,
            )
        )

        # Check if audio was emitted (listen-first SIL may not emit audio).
        emitted_audio = initial_machine_state.emitted_audio if initial_machine_state is not None else True

        initial_semantic_buffer = None
        if not emitted_audio:
            # Listen-first: no audio emitted on first frame. Push silence to keep decoder warm.
            silence_codes = self.get_silence_codes(sequences.device)
            self.push_audio_codes(audio_codes=silence_codes, stream_id=audio_decoder_stream_id)
        elif self.max_delay > 0:
            # When acoustic delay is active, the first frame is a placeholder:
            # only the semantic code (CB0) is valid; acoustic codes (CB1-7) are zeros
            # because no previous frame exists to provide delayed acoustic predictions.
            # The duplex_step handler (semantic_buffer logic) will produce properly
            # aligned frames from the second step onward. This initial placeholder frame
            # is expected and acceptable — the audio decoder handles it gracefully.
            first_codes = audio_codes[0, -1]
            semantic_code = first_codes[0:1]
            output_codes = torch.zeros_like(first_codes)
            output_codes[0] = semantic_code[0]
            initial_semantic_buffer = semantic_code
            self.push_audio_codes(audio_codes=output_codes, stream_id=audio_decoder_stream_id)
        else:
            self.push_audio_codes(audio_codes=audio_codes[0, -1], stream_id=audio_decoder_stream_id)

        # Listen-first warmup: force one additional SIL frame after init.
        forced_sil_remaining = 1 if (not speak_first and getattr(self, "use_sil_token", False)) else 0

        state = RaonDecodingState(
            sequences=sequences,
            attention_mask=attention_mask,
            audio_codes=audio_codes,
            audio_codes_mask=audio_codes_mask,
            past_key_values=past_key_values,
            audio_input_encoder_cache=audio_input_encoder_cache,
            audio_decoder_stream_id=audio_decoder_stream_id,
            do_sample=do_sample,
            logits_processor=logits_processor,
            num_code_groups=self.num_code_groups,
            semantic_buffer=initial_semantic_buffer,
            eos_penalty=eos_penalty,
            sil_penalty=float(sil_penalty),
            bc_penalty=float(bc_penalty),
            machine_state=initial_machine_state,
            forced_sil_remaining=forced_sil_remaining,
        )
        return state

    def _extract_gt_tokens_per_frame(
        self,
        gt_input_ids: torch.Tensor,
        system_prefix_len: int,
    ) -> list[list[int]]:
        """
        Extract GT tokens for each frame from the training input_ids sequence.

        The GT sequence structure after system prefix:
        - [im_start] [audio_start] then per-frame tokens
        - Each frame ends with [A] token
        - Silence frame: [U] [A] (2 tokens)
        - EPAD frame: EPAD [U] [A] (3 tokens)
        - Text frame: text [U] [A] (3 tokens)

        Returns:
            List of token lists, one per frame. Each list contains the tokens
            to add for that frame (e.g., [text, U, A] or [U, A]).
        """
        input_ids = gt_input_ids[0].tolist() if gt_input_ids.dim() > 1 else gt_input_ids.tolist()

        frame_tokens: list[list[int]] = []
        # Skip preamble tokens after system prefix.
        # Preamble may vary ([IM_START], [IM_START, AUDIO_START], [IM_START, EPAD, AUDIO_START]).
        preamble_token_ids = {IM_START.id, AUDIO_START.id}
        position = system_prefix_len
        while position < len(input_ids) and input_ids[position] in preamble_token_ids:
            position += 1
        sequence_mode = self._state_manager._config.effective_sequence_mode

        while position < len(input_ids):
            token_id = input_ids[position]

            if token_id == AUDIO_INPUT_PLACEHOLDER.id:
                if sequence_mode == "uta" and position + 2 < len(input_ids):
                    frame_end_token = input_ids[position + 2]
                    if frame_end_token in (AUDIO_OUTPUT_PLACEHOLDER.id, AUDIO_START.id):
                        frame_tokens.append(input_ids[position : position + 3])
                        position += 3
                        continue
                if position + 1 < len(input_ids) and input_ids[position + 1] == AUDIO_OUTPUT_PLACEHOLDER.id:
                    frame_tokens.append(input_ids[position : position + 2])
                    position += 2
                    continue
            elif sequence_mode == "tua" and position + 2 < len(input_ids):
                if input_ids[position + 1] == AUDIO_INPUT_PLACEHOLDER.id and input_ids[position + 2] in (
                    AUDIO_OUTPUT_PLACEHOLDER.id,
                    AUDIO_START.id,
                ):
                    frame_tokens.append(input_ids[position : position + 3])
                    position += 3
                    continue

            position += 1

        return frame_tokens

    def _sample_from_logits(
        self,
        sequences: torch.Tensor,
        logits: torch.Tensor,
        force_audio_output: bool,
        force_text_output: bool,
        do_sample: bool,
        logits_processor: LogitsProcessorList,
    ) -> torch.Tensor:
        if force_audio_output:
            return torch.full((logits.shape[0], 1), AUDIO_OUTPUT_PAD.id, dtype=torch.long, device=logits.device)

        next_token_logits = logits[:, -1].clone()
        if force_text_output:
            next_token_logits[..., AUDIO_OUTPUT_PAD.id] = torch.finfo(next_token_logits.dtype).min

        if do_sample:
            processed_logits = logits_processor(input_ids=sequences, scores=next_token_logits)  # type: ignore[arg-type]
            probs = F.softmax(processed_logits, dim=-1, dtype=torch.float32)
            probs = probs.clamp_min(torch.finfo(probs.dtype).tiny)
            return torch.multinomial(probs, num_samples=1)
        return next_token_logits.argmax(dim=-1, keepdim=True)

    def _update_sequences_and_generate_audio_codes(
        self,
        new_logits: torch.Tensor,
        new_last_hidden_state: torch.Tensor,
        sequences: torch.Tensor,
        attention_mask: torch.Tensor,
        audio_codes: torch.Tensor,
        audio_codes_mask: torch.Tensor,
        is_complete: torch.Tensor,
        pad_token_id: int,
        force_audio_output: bool,
        force_text_output: bool,
        do_sample: bool,
        logits_processor: LogitsProcessorList,
        is_generating_audio: torch.Tensor | None = None,
        ras_enabled: bool = False,
        ras_window_size: int = 50,
        ras_repetition_threshold: float = 0.5,
        suppress_audio_eos: bool = False,
        ras_skip_frames: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Sample next token and generate audio codes for one autoregressive step.

        Args:
            new_logits: Logits from the latest forward pass. Shape: [batch_size, 1, vocab_size].
            new_last_hidden_state: Talker hidden state. Shape: [batch_size, 1, hidden_dim].
            sequences: Current token sequence. Shape: [batch_size, seq_length].
            attention_mask: Attention mask. Shape: [batch_size, seq_length].
            audio_codes: Audio code history. Shape: [batch_size, num_frames, num_code_groups].
            audio_codes_mask: Audio code mask. Shape: [batch_size, num_frames].
            is_complete: Per-sample completion flag. Shape: [batch_size].
            pad_token_id: Token ID used for padding completed sequences.
            force_audio_output: If True, always generate audio codes.
            force_text_output: If True, suppress audio-start tokens.
            do_sample: Whether to sample (vs. greedy).
            logits_processor: Logits processing pipeline.
            is_generating_audio: Per-sample audio generation state. Shape: [batch_size] or None.
            ras_enabled: Enable repetition-aware sampling for audio codes.
            ras_window_size: Window size for repetition detection.
            ras_repetition_threshold: Threshold for repetition penalty.

        Returns:
            Tuple of (sequences, attention_mask, audio_codes, audio_codes_mask, is_complete, is_generating_audio).
        """
        next_is_generating_audio = is_generating_audio.clone() if is_generating_audio is not None else None

        if is_generating_audio is None:
            new_ids = self._sample_from_logits(
                sequences=sequences,
                logits=new_logits,
                force_audio_output=force_audio_output,
                force_text_output=force_text_output,
                do_sample=do_sample,
                logits_processor=logits_processor,
            )
        else:
            new_ids = torch.full((new_logits.shape[0], 1), pad_token_id, dtype=torch.long, device=new_logits.device)
            text_mode_mask = ~is_generating_audio
            if text_mode_mask.any():
                new_ids[text_mode_mask] = self._sample_from_logits(
                    sequences=sequences[text_mode_mask],
                    logits=new_logits[text_mode_mask],
                    force_audio_output=False,
                    force_text_output=True,
                    do_sample=do_sample,
                    logits_processor=logits_processor,
                )
            if is_generating_audio.any():
                new_ids[is_generating_audio] = AUDIO_OUTPUT_PAD.id

        new_ids[is_complete, -1] = pad_token_id
        is_complete |= new_ids[:, -1] == IM_END.id

        if next_is_generating_audio is not None:
            assert is_generating_audio is not None
            start_audio_mask = (~is_complete) & (~is_generating_audio) & (new_ids[:, -1] == AUDIO_START.id)
            next_is_generating_audio[start_audio_mask] = True
            is_audio_output = (~is_complete) & is_generating_audio
        else:
            is_audio_output = (~is_complete) & (new_ids[:, -1] == AUDIO_OUTPUT_PAD.id)
            new_ids[is_audio_output, -1] = AUDIO_OUTPUT_PLACEHOLDER.id

        final_audio_output_mask = is_audio_output.clone()
        sequences_with_new_ids = torch.cat((sequences, new_ids), dim=1)
        attention_mask = F.pad(attention_mask, (0, 1), value=1)
        audio_codes_mask = torch.cat((audio_codes_mask, final_audio_output_mask[:, None]), dim=1)
        audio_codes = F.pad(audio_codes, (0, 0, 0, 1))

        if is_audio_output.any():
            first_code_sampler = None
            if do_sample:
                first_code_sampler = make_audio_code_sampler(
                    sequences=sequences_with_new_ids[is_audio_output],
                    logits_processor=logits_processor,
                    audio_codes=audio_codes[is_audio_output, :-1],
                    ras_enabled=ras_enabled,
                    ras_window_size=ras_window_size,
                    ras_repetition_threshold=ras_repetition_threshold,
                    ras_skip_frames=ras_skip_frames,
                )

            generated_audio_codes = self.generate_audio_codes(
                talker_last_hidden_state=new_last_hidden_state[is_audio_output, -1:],
                first_code_sampler=first_code_sampler,
            )
            generated_audio_end_mask = generated_audio_codes[:, 0] == self.codebook_size
            if suppress_audio_eos:
                generated_audio_end_mask[:] = False
            new_ids[is_audio_output, -1] = AUDIO_OUTPUT_PLACEHOLDER.id

            if generated_audio_end_mask.any():
                local_non_end_mask = ~generated_audio_end_mask
                global_non_end_mask = is_audio_output.clone()
                global_non_end_mask[is_audio_output] = local_non_end_mask
                final_audio_output_mask = global_non_end_mask
                audio_end_global_mask = is_audio_output.clone()
                audio_end_global_mask[is_audio_output] = generated_audio_end_mask
                new_ids[audio_end_global_mask, -1] = AUDIO_END.id
                if next_is_generating_audio is not None:
                    next_is_generating_audio[audio_end_global_mask] = False
                if local_non_end_mask.any():
                    audio_codes[global_non_end_mask, -1] = generated_audio_codes[local_non_end_mask]
            else:
                audio_codes[is_audio_output, -1] = generated_audio_codes

        audio_codes_mask[:, -1] = final_audio_output_mask
        sequences = torch.cat((sequences, new_ids), dim=1)
        if next_is_generating_audio is not None:
            next_is_generating_audio[is_complete] = False

        return sequences, attention_mask, audio_codes, audio_codes_mask, is_complete, next_is_generating_audio

    def _generation_prefill(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        audio_input: torch.Tensor | None,
        audio_output: torch.Tensor | None,
        audio_input_lengths: torch.Tensor | None,
        audio_output_lengths: torch.Tensor | None,
        audio_output_codes: torch.Tensor | None,
        pad_token_id: int,
        force_audio_output: bool,
        force_text_output: bool,
        disable_eos_on_first_output: bool,
        do_sample: bool,
        logits_processor: LogitsProcessorList,
        max_sequence_length: int,
        audio_input_embeds: torch.Tensor | None = None,
        audio_input_embeds_mask: torch.Tensor | None = None,
        ras_enabled: bool = False,
        ras_window_size: int = 50,
        ras_repetition_threshold: float = 0.5,
        speaker_embeds: torch.Tensor | None = None,
        suppress_audio_eos: bool = False,
        ras_skip_frames: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, Any]:
        """Run the prefill phase: encode inputs and generate the first output token and audio codes.

        Args:
            input_ids: Input token IDs. Shape: [batch_size, seq_length].
            attention_mask: Attention mask. Shape: [batch_size, seq_length] or None.
            audio_input: Raw input audio. Shape: [batch_size, num_samples] or None.
            audio_output: Raw output audio for teacher-forced codes. Shape: [batch_size, num_samples] or None.
            audio_input_lengths: Per-sample input audio lengths. Shape: [batch_size] or None.
            audio_output_lengths: Per-sample output audio lengths. Shape: [batch_size] or None.
            audio_output_codes: Pre-computed output audio codes or None.
            pad_token_id: Token ID used for padding completed sequences.
            force_audio_output: If True, always generate audio codes.
            force_text_output: If True, suppress audio-start tokens.
            disable_eos_on_first_output: If True, prevent EOS on the first generated token.
            do_sample: Whether to sample (vs. greedy).
            logits_processor: Logits processing pipeline.
            max_sequence_length: Maximum sequence length for KV cache allocation.
            audio_input_embeds: Pre-computed audio input embeddings or None.
            audio_input_embeds_mask: Mask for pre-computed audio input embeddings or None.
            ras_enabled: Enable repetition-aware sampling.
            ras_window_size: Window size for repetition detection.
            ras_repetition_threshold: Threshold for repetition penalty.
            speaker_embeds: Speaker conditioning embeddings or None.
            suppress_audio_eos: If True, prevent audio EOS on the first generated audio frame.
            ras_skip_frames: Number of leading frames to ignore in RAS history.

        Returns:
            Tuple of (sequences, attention_mask, audio_codes, audio_codes_mask,
            is_complete, is_generating_audio, past_key_values).
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        position_ids = attention_mask.cumsum(dim=1) - 1
        past_key_values = self.init_past_key_values(batch_size=input_ids.shape[0], max_sequence_length=max_sequence_length)
        cache_position = torch.arange(input_ids.shape[1], device=input_ids.device)
        # Auto-create audio_output_codes_mask when codes are provided without mask.
        audio_output_codes_mask = None
        if audio_output_codes is not None:
            audio_output_codes_mask = torch.ones(
                audio_output_codes.shape[0],
                audio_output_codes.shape[1],
                dtype=torch.bool,
                device=audio_output_codes.device,
            )
        talker_last_hidden_state, text_logits = cast(
            tuple[torch.Tensor, torch.Tensor],
            self.inference_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            audio_input=audio_input,
            audio_output=audio_output,
            audio_input_lengths=audio_input_lengths,
            audio_output_lengths=audio_output_lengths,
            audio_output_codes=audio_output_codes,
            audio_output_codes_mask=audio_output_codes_mask,
            audio_input_embeds=audio_input_embeds,
            audio_input_embeds_mask=audio_input_embeds_mask,
            speaker_embeds=speaker_embeds,
            use_cache=True,
            past_key_values=past_key_values,
            cache_position=cache_position,
            ),
        )

        if disable_eos_on_first_output:
            text_logits[..., IM_END.id] = torch.finfo(text_logits.dtype).min

        batch_size = input_ids.shape[0]
        sequences = input_ids
        audio_codes = torch.zeros(batch_size, 0, self.num_code_groups, dtype=torch.long, device=input_ids.device)
        audio_codes_mask = torch.zeros(batch_size, 0, dtype=torch.bool, device=input_ids.device)
        is_complete = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)

        if force_audio_output:
            is_generating_audio = torch.ones(batch_size, dtype=torch.bool, device=input_ids.device)
        else:
            is_generating_audio = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)

        sequences, attention_mask, audio_codes, audio_codes_mask, is_complete, is_generating_audio = (
            self._update_sequences_and_generate_audio_codes(
                new_logits=text_logits,
                new_last_hidden_state=talker_last_hidden_state,
                sequences=sequences,
                attention_mask=attention_mask,
                audio_codes=audio_codes,
                audio_codes_mask=audio_codes_mask,
                is_complete=is_complete,
                pad_token_id=pad_token_id,
                force_audio_output=force_audio_output,
                force_text_output=force_text_output,
                do_sample=do_sample,
                logits_processor=logits_processor,
                is_generating_audio=is_generating_audio,
                ras_enabled=ras_enabled,
                ras_window_size=ras_window_size,
                ras_repetition_threshold=ras_repetition_threshold,
                suppress_audio_eos=suppress_audio_eos,
                ras_skip_frames=ras_skip_frames,
            )
        )
        return sequences, attention_mask, audio_codes, audio_codes_mask, is_complete, is_generating_audio, past_key_values

    def _decoding_step(
        self,
        sequences: torch.Tensor,
        attention_mask: torch.Tensor,
        audio_codes: torch.Tensor,
        audio_codes_mask: torch.Tensor,
        is_complete: torch.Tensor,
        past_key_values: Any,
        pad_token_id: int,
        force_audio_output: bool,
        force_text_output: bool,
        is_generating_audio: torch.Tensor | None,
        do_sample: bool,
        logits_processor: LogitsProcessorList,
        ras_enabled: bool = False,
        ras_window_size: int = 50,
        ras_repetition_threshold: float = 0.5,
        suppress_audio_eos: bool = False,
        ras_skip_frames: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Run one autoregressive decoding step using cached KV states.

        Args:
            sequences: Current token sequence. Shape: [batch_size, seq_length].
            attention_mask: Attention mask. Shape: [batch_size, seq_length].
            audio_codes: Audio code history. Shape: [batch_size, num_frames, num_code_groups].
            audio_codes_mask: Audio code mask. Shape: [batch_size, num_frames].
            is_complete: Per-sample completion flag. Shape: [batch_size].
            past_key_values: Cached key-value states from previous steps.
            pad_token_id: Token ID used for padding completed sequences.
            force_audio_output: If True, always generate audio codes.
            force_text_output: If True, suppress audio-start tokens.
            is_generating_audio: Per-sample audio generation state or None.
            do_sample: Whether to sample (vs. greedy).
            logits_processor: Logits processing pipeline.
            ras_enabled: Enable repetition-aware sampling.
            ras_window_size: Window size for repetition detection.
            ras_repetition_threshold: Threshold for repetition penalty.

        Returns:
            Tuple of (sequences, attention_mask, audio_codes, audio_codes_mask,
            is_complete, is_generating_audio).
        """
        supports_audio_output = bool(getattr(self, "supports_audio_output", True))
        cache_position = attention_mask.sum(dim=1, keepdim=False) - 1
        talker_last_hidden_state, text_logits = cast(
            tuple[torch.Tensor, torch.Tensor],
            self.inference_forward(
            input_ids=sequences[:, -1:],
            position_ids=cache_position.unsqueeze(1),
            attention_mask=attention_mask,
            audio_output_codes=audio_codes[:, -1:] if supports_audio_output else None,
            audio_output_codes_mask=audio_codes_mask[:, -1:] if supports_audio_output else None,
            past_key_values=past_key_values,
            use_cache=True,
            cache_position=cache_position,
            ),
        )
        return self._update_sequences_and_generate_audio_codes(
            new_logits=text_logits,
            new_last_hidden_state=talker_last_hidden_state,
            sequences=sequences,
            attention_mask=attention_mask,
            audio_codes=audio_codes,
            audio_codes_mask=audio_codes_mask,
            is_complete=is_complete,
            pad_token_id=pad_token_id,
            force_audio_output=force_audio_output,
            force_text_output=force_text_output,
            do_sample=do_sample,
            logits_processor=logits_processor,
            is_generating_audio=is_generating_audio,
            ras_enabled=ras_enabled,
            ras_window_size=ras_window_size,
            ras_repetition_threshold=ras_repetition_threshold,
            suppress_audio_eos=suppress_audio_eos,
            ras_skip_frames=ras_skip_frames,
        )

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        audio_input: torch.Tensor | None = None,
        audio_output: torch.Tensor | None = None,
        audio_input_lengths: torch.Tensor | None = None,
        audio_output_lengths: torch.Tensor | None = None,
        audio_output_codes: torch.Tensor | None = None,
        audio_input_embeds: torch.Tensor | None = None,
        audio_input_embeds_mask: torch.Tensor | None = None,
        max_new_tokens: int = 128,
        pad_token_id: int = PAD.id,
        num_code_groups: int | None = None,
        force_audio_output: bool = False,
        force_text_output: bool = False,
        disable_eos_on_first_output: bool = True,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_k: int = 20,
        top_p: float = 0.8,
        disable_tqdm: bool = False,
        ras_enabled: bool = False,
        ras_window_size: int = 50,
        ras_repetition_threshold: float = 0.5,
        speaker_embeds: torch.Tensor | None = None,
        speaker_audio: torch.Tensor | None = None,
        speaker_audio_lengths: torch.Tensor | None = None,
        continuation_silence_frames: int = 0,
    ) -> GenerateOutput:
        """Run offline (non-streaming) inference: generate text and/or audio from input tokens.

        Args:
            input_ids: Input token IDs. Shape: [batch_size, seq_length].
            attention_mask: Attention mask. Shape: [batch_size, seq_length] or None.
            audio_input: Raw input audio. Shape: [batch_size, num_samples] or None.
            audio_output: Raw output audio for teacher-forced codes or None.
            audio_input_lengths: Per-sample input audio lengths or None.
            audio_output_lengths: Per-sample output audio lengths or None.
            audio_output_codes: Pre-computed output audio codes or None.
            audio_input_embeds: Pre-computed audio input embeddings or None.
            audio_input_embeds_mask: Mask for pre-computed audio input embeddings or None.
            max_new_tokens: Maximum number of new tokens to generate.
            pad_token_id: Token ID used for padding completed sequences.
            num_code_groups: Number of audio codebook groups to use.
            force_audio_output: If True, always generate audio codes.
            force_text_output: If True, suppress audio-start tokens.
            disable_eos_on_first_output: If True, prevent EOS on the first generated token.
            do_sample: Whether to sample (vs. greedy).
            temperature: Sampling temperature.
            top_k: Top-k filtering.
            top_p: Top-p filtering.
            disable_tqdm: If True, suppress progress bar.
            ras_enabled: Enable repetition-aware sampling for audio codes.
            ras_window_size: Window size for repetition detection.
            ras_repetition_threshold: Threshold for repetition penalty.
            speaker_embeds: Pre-computed speaker conditioning embeddings or None.
            speaker_audio: Raw speaker reference audio for on-the-fly embedding computation or None.
            speaker_audio_lengths: Per-sample speaker audio lengths or None.
            continuation_silence_frames: Number of initial generated audio frames to
                replace with Mimi-encoded silence during TTS continuation warmup.

        Returns:
            GenerateOutput with generated sequences, audio codes, and masks.
        """
        if speaker_audio is not None and speaker_embeds is None:
            speaker_embeds = self._compute_speaker_embeds(speaker_audio, speaker_audio_lengths)

        if num_code_groups is None:
            num_code_groups = self.num_code_groups
        assert num_code_groups <= self.num_code_groups, (
            f"Expected num_code_groups <= {self.num_code_groups}, got {num_code_groups}."
        )

        if not bool(getattr(self, "supports_audio_output", True)):
            force_audio_output = False
            force_text_output = True

        logits_processor = LogitsProcessorList()
        if do_sample and temperature and temperature != 1.0:
            logits_processor.append(TemperatureLogitsWarper(temperature=temperature))
        if do_sample and top_k and top_k > 0:
            logits_processor.append(TopKLogitsWarper(top_k=top_k))
        if do_sample and top_p and top_p < 1.0:
            logits_processor.append(TopPLogitsWarper(top_p=top_p))

        sequences, attention_mask, audio_codes, audio_codes_mask, is_complete, is_generating_audio, past_key_values = (
            self._generation_prefill(
                input_ids=input_ids,
                attention_mask=attention_mask,
                audio_input=audio_input,
                audio_output=audio_output,
                audio_input_lengths=audio_input_lengths,
                audio_output_lengths=audio_output_lengths,
                audio_output_codes=audio_output_codes,
                audio_input_embeds=audio_input_embeds,
                audio_input_embeds_mask=audio_input_embeds_mask,
                speaker_embeds=speaker_embeds,
                pad_token_id=pad_token_id,
                force_audio_output=force_audio_output,
                force_text_output=force_text_output,
                disable_eos_on_first_output=disable_eos_on_first_output,
                do_sample=do_sample,
                logits_processor=logits_processor,
                max_sequence_length=8 * (1 + (input_ids.shape[1] + max_new_tokens) // 8),
                ras_enabled=ras_enabled,
                ras_window_size=ras_window_size,
                ras_repetition_threshold=ras_repetition_threshold,
                suppress_audio_eos=continuation_silence_frames > 0,
                ras_skip_frames=continuation_silence_frames,
            )
        )

        silence_codes = None
        generated_audio_frame_count = 0
        if continuation_silence_frames > 0:
            silence_codes = self.get_silence_codes(input_ids.device)

        if silence_codes is not None and audio_codes_mask.any():
            generated_audio_frame_count = 1
            if generated_audio_frame_count <= continuation_silence_frames:
                audio_codes[:, -1] = silence_codes

        for _ in trange(max_new_tokens - 1, disable=disable_tqdm):
            if is_complete.all():
                break
            in_silence_window = silence_codes is not None and generated_audio_frame_count < continuation_silence_frames
            sequences, attention_mask, audio_codes, audio_codes_mask, is_complete, is_generating_audio = self._decoding_step(
                sequences=sequences,
                attention_mask=attention_mask,
                audio_codes=audio_codes,
                audio_codes_mask=audio_codes_mask,
                is_complete=is_complete,
                past_key_values=past_key_values,
                pad_token_id=pad_token_id,
                force_audio_output=force_audio_output,
                force_text_output=force_text_output,
                is_generating_audio=is_generating_audio,
                do_sample=do_sample,
                logits_processor=logits_processor,
                ras_enabled=ras_enabled,
                ras_window_size=ras_window_size,
                ras_repetition_threshold=ras_repetition_threshold,
                suppress_audio_eos=in_silence_window,
                ras_skip_frames=continuation_silence_frames,
            )

            if silence_codes is not None and audio_codes_mask[:, -1].any():
                generated_audio_frame_count += 1
                if generated_audio_frame_count <= continuation_silence_frames:
                    audio_codes[audio_codes_mask[:, -1], -1] = silence_codes

        audio = None
        audio_lengths = None
        if not force_text_output and audio_codes_mask.any():
            contiguous_audio_sequences = pad_sequence(
                [seq[mask] for seq, mask in zip(audio_codes, audio_codes_mask, strict=True)],
                batch_first=True,
                padding_value=0,
            )
            # Realign delayed codes before decoding
            if self.max_delay > 0:
                contiguous_audio_sequences = undelay_audio_codes(self.delays, contiguous_audio_sequences, padding_value=0)
            # Build padding_mask so Mimi decoder trims extra samples from causal ConvTranspose1d.
            audio_lengths = (audio_codes_mask.float().sum(dim=1) * self.sampling_rate / self.frame_rate).floor().long()
            max_audio_len = int(audio_lengths.max().item())
            padding_mask = torch.arange(max_audio_len, device=audio_lengths.device).unsqueeze(0) < audio_lengths.unsqueeze(1)
            audio = self.decode_audio(
                audio_codes=contiguous_audio_sequences,
                padding_mask=padding_mask.long(),
                use_streaming=None,
            ).audio

            if continuation_silence_frames > 0:
                trim_samples = int(continuation_silence_frames * self.sampling_rate / self.frame_rate)
                audio = audio[:, trim_samples:]
                audio_lengths = (audio_lengths - trim_samples).clamp(min=0)

        self.free_past_key_values(past_key_values)
        return {
            "sequences": sequences,
            "audio_codes": audio_codes,
            "audio_codes_mask": audio_codes_mask,
            "audio": audio,
            "audio_lengths": audio_lengths,
        }
