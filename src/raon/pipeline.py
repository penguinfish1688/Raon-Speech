# coding=utf-8
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

"""High-level pipeline API for RAON inference."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import yaml

logger = logging.getLogger(__name__)

from raon.utils.audio_io import load_audio as _load_audio_shared
from raon.utils.misc import DTYPE_MAP

from raon.models.raon import RaonModel
from raon.utils.processor import (
    RaonProcessor,
    get_default_stt_prompt,
    get_default_tts_prompt,
)
from raon.utils.special_tokens import (
    AUDIO_OUTPUT_PLACEHOLDER,
    AUDIO_START,
    SPEAKER_EMBEDDING_PLACEHOLDER,
)

_DEFAULT_INFERENCE_CONFIG = Path(__file__).parent.parent.parent / "config" / "infer.yaml"
_DEFAULT_DUPLEX_CONFIG = Path(__file__).parent.parent.parent / "config" / "duplex_infer.yaml"


class RaonPipeline:
    """High-level inference API for RAON speech LLM.

    Loads the model and processor once, and exposes task-specific methods for
    STT, TTS, TextQA, and Speech-Chat.

    Example::

        pipe = RaonPipeline("/path/to/model")
        text = pipe.stt("audio.wav")
        waveform, sr = pipe.tts("Hello, world!")
        RaonPipeline.save_audio((waveform, sr), "out.wav")
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        attn_implementation: str = "sdpa",
        config: str | None = None,
        duplex_config: str | None = None,
    ) -> None:
        """Load model, processor, and default task parameters from infer.yaml.

        Args:
            model_path: Path to the pretrained RAON model directory.
            device: Device to run inference on (e.g. ``"cuda"``, ``"cpu"``).
            dtype: Torch dtype string — one of ``"bfloat16"``, ``"float16"``, ``"float32"``.
            attn_implementation: Attention backend string (``"sdpa"``, ``"eager"``, or ``"fa"``).
            config: Optional path to ``infer.yaml``-style task defaults.
            duplex_config: Optional path to ``duplex_infer.yaml``-style defaults.
        """
        torch_dtype = DTYPE_MAP[dtype]
        self.device = device
        self.dtype = torch_dtype

        self.model: RaonModel = RaonModel.from_pretrained(model_path, dtype=torch_dtype).to(device).eval()
        if attn_implementation == "fa":
            attn_implementation = "flash_attention_2"
        if attn_implementation not in {"sdpa", "eager", "flash_attention_2"}:
            raise ValueError(
                f"Invalid attn_implementation: {attn_implementation}. "
                "Use one of: sdpa, eager, fa."
            )
        self.model._set_attention_implementation(attn_implementation)
        logger.info("Pipeline attention implementation: %s", attn_implementation)
        self.processor: RaonProcessor = RaonProcessor.from_pretrained(model_path)

        inference_config_path = Path(config) if config is not None else _DEFAULT_INFERENCE_CONFIG
        with open(inference_config_path, encoding="utf-8") as f:
            self.task_params: dict[str, dict] = yaml.safe_load(f)

        duplex_config_path = Path(duplex_config) if duplex_config is not None else _DEFAULT_DUPLEX_CONFIG
        if duplex_config_path.exists():
            with open(duplex_config_path, encoding="utf-8") as f:
                raw = yaml.safe_load(f)
            self.duplex_params: dict = raw.get("duplex", {}) if raw else {}
        else:
            self.duplex_params = {}

    # ------------------------------------------------------------------
    # Core method
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict],
        *,
        force_audio_output: bool = False,
        force_text_output: bool = True,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        speaker_audio: str | None = None,
        max_audio_chunk_length: int | None = 192000,
        **gen_kwargs,
    ) -> str | tuple[torch.Tensor, int]:
        """Process messages through the model and return the result.

        Args:
            messages: HF-style message list, e.g.
                ``[{"role": "user", "content": "Hello"}]`` or multimodal
                ``[{"role": "user", "content": [{"type": "audio", "audio": "path"}, {"type": "text", "text": "..."}]}]``.
            force_audio_output: If ``True``, generate audio output (for TTS).
            force_text_output: If ``True``, generate text output.
            max_new_tokens: Maximum number of new tokens to generate.
                Defaults to 512 for audio output, 1024 for text output.
            temperature: Sampling temperature.
                Defaults to 1.2 for audio output, 0.7 for text output.
            speaker_audio: Path to speaker reference audio for voice conditioning.
            max_audio_chunk_length: If set, split audio_input into chunks of at
                most this many samples (matching training-time chunking).
            **gen_kwargs: Additional keyword arguments forwarded to ``model.generate()``.

        Returns:
            ``str`` for text output, or ``(waveform_tensor, sampling_rate)`` for audio output.
        """
        has_audio_input = any(
            isinstance(m.get("content"), list) and any(p.get("type") == "audio" for p in m["content"]) for m in messages
        )
        if force_audio_output:
            default_key = "default_audio"
        elif has_audio_input:
            default_key = "default_text_with_audio_input"
        else:
            default_key = "default_text"
        defaults = self.task_params.get(default_key, {})
        if max_new_tokens is None:
            max_new_tokens = defaults.get("max_new_tokens", 1024)
        if temperature is None:
            temperature = defaults.get("temperature", 0.7)
        if force_audio_output:
            force_text_output = False
            # Prepend speaker embedding placeholder to the first user message
            # so the model conditions on the speaker reference audio.
            if speaker_audio is not None:
                speaker_token = str(SPEAKER_EMBEDDING_PLACEHOLDER)
                for msg in messages:
                    if msg.get("role") == "user":
                        content = msg.get("content", "")
                        if isinstance(content, str) and speaker_token not in content:
                            msg["content"] = speaker_token + content
                        elif isinstance(content, list):
                            has_speaker = any(
                                isinstance(p, dict) and p.get("type") == "text" and speaker_token in p.get("text", "")
                                for p in content
                            )
                            if not has_speaker:
                                content.insert(0, {"type": "text", "text": speaker_token})
                        break

        inputs = self.processor(
            messages,
            add_generation_prompt=True,
            force_audio_output=force_audio_output,
            device=self.device,
            dtype=self.dtype,
            max_audio_chunk_length=max_audio_chunk_length,
        )

        input_length = int(inputs["attention_mask"].sum().item())

        # Load speaker audio if a path was given
        speaker_audio_tensor: torch.Tensor | None = None
        if speaker_audio is not None:
            speaker_audio_tensor = self._load_speaker_audio(speaker_audio)

        output = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            audio_input=inputs.get("audio_input"),
            audio_input_lengths=inputs.get("audio_input_lengths"),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            force_audio_output=force_audio_output,
            force_text_output=force_text_output,
            speaker_audio=speaker_audio_tensor,
            **gen_kwargs,
        )

        if force_audio_output:
            audio = output["audio"]
            audio_lengths = output["audio_lengths"]
            waveform = audio[0]
            length = int(audio_lengths[0].item())
            return self._trim_last_frame(waveform[:length]), self.processor.sampling_rate

        sequences = output["sequences"]
        generated_ids = sequences[0, input_length:]
        return self.processor.tokenizer.decode(generated_ids, skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Task-specific convenience methods
    # ------------------------------------------------------------------

    def stt(self, audio: str, prompt: str | None = None) -> str:
        """STT: audio → text.

        Args:
            audio: Path to the input audio file.
            prompt: Optional transcription instruction. Defaults to the
                standard STT prompt.

        Returns:
            Transcribed text string.
        """
        effective_prompt = prompt if prompt is not None else get_default_stt_prompt()
        messages: list[dict] = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio},
                    {"type": "text", "text": effective_prompt},
                ],
            }
        ]
        params = self.task_params.get("stt", {})
        return self.chat(  # type: ignore[return-value]
            messages,
            force_audio_output=params.get("force_audio_output", False),
            force_text_output=params.get("force_text_output", True),
            max_new_tokens=params.get("max_new_tokens", 512),
            temperature=params.get("temperature", 0.2),
            max_audio_chunk_length=params.get("max_audio_chunk_length"),
        )

    def tts(
        self,
        text: str,
        speaker_audio: str | None = None,
    ) -> tuple[torch.Tensor, int]:
        """TTS: text → audio.

        Args:
            text: The text to synthesize.
            speaker_audio: Optional path to a speaker reference audio file for
                voice conditioning.

        Returns:
            ``(waveform, sampling_rate)`` tuple.
        """
        speaker_token = str(SPEAKER_EMBEDDING_PLACEHOLDER) if speaker_audio is not None else ""
        prompt = get_default_tts_prompt()
        messages: list[dict] = [
            {
                "role": "user",
                "content": f"{speaker_token}{prompt}:\n{text}",
            }
        ]
        params = self.task_params.get("tts", {})
        waveform, sampling_rate = self.chat(  # type: ignore[misc]
            messages,
            force_audio_output=params.get("force_audio_output", True),
            force_text_output=params.get("force_text_output", False),
            max_new_tokens=params.get("max_new_tokens", 512),
            temperature=params.get("temperature", 1.2),
            speaker_audio=speaker_audio,
            ras_enabled=params.get("ras_enabled", False),
            ras_window_size=params.get("ras_window_size", 50),
            ras_repetition_threshold=params.get("ras_repetition_threshold", 0.5),
        )
        return waveform, sampling_rate

    def tts_continuation(
        self,
        target_text: str,
        ref_audio: str,
        ref_text: str | None = None,
        speaker_audio: str | None = None,
    ) -> tuple[torch.Tensor, int]:
        """TTS continuation: prefill reference audio as generated output, then continue for target text.

        Constructs the sequence as if the model already generated the reference audio,
        then continues generating audio for the target text. This produces speech that
        naturally continues from the reference, preserving speaker characteristics.

        Args:
            target_text: Text to generate speech for.
            ref_audio: Path to the reference audio file.
            ref_text: Transcription of the reference audio. If ``None``, the
                reference audio is automatically transcribed via :meth:`stt`.
            speaker_audio: Optional path to a separate speaker reference audio
                for voice conditioning. If ``None``, ``ref_audio`` is used.

        Returns:
            ``(waveform, sampling_rate)`` tuple.
        """
        if ref_text is None:
            ref_text = self.stt(ref_audio)
        if speaker_audio is None:
            speaker_audio = ref_audio

        ref_audio_tensor = self._load_speaker_audio(ref_audio).to(self.dtype)
        ref_audio_lengths = torch.tensor([ref_audio_tensor.shape[1]], device=self.device)

        # Tokenize reference audio into codes (with chunking to match training).
        model_config = self.model.config if hasattr(self.model, "config") else self.model.get_model().config
        max_output_chunk = getattr(model_config, "max_audio_output_seq_length", 192000)
        ref_samples = ref_audio_tensor.shape[-1]
        with torch.no_grad():
            num_code_groups = self.model.num_code_groups
            if ref_samples <= max_output_chunk:
                ref_codes = self.model.tokenize_audio(
                    audio=ref_audio_tensor,
                    audio_lengths=ref_audio_lengths,
                    num_code_groups=num_code_groups,
                ).audio_codes
            else:
                code_chunks = []
                offset = 0
                while offset < ref_samples:
                    end = min(offset + max_output_chunk, ref_samples)
                    chunk = ref_audio_tensor[:, offset:end]
                    chunk_len = torch.tensor([end - offset], device=self.device)
                    chunk_codes = self.model.tokenize_audio(
                        audio=chunk,
                        audio_lengths=chunk_len,
                        num_code_groups=num_code_groups,
                    ).audio_codes
                    code_chunks.append(chunk_codes)
                    offset = end
                ref_codes = torch.cat(code_chunks, dim=1)
        num_ref_frames = ref_codes.shape[1]

        speaker_token = str(SPEAKER_EMBEDDING_PLACEHOLDER) if speaker_audio is not None else ""
        combined_text = f"{ref_text} {target_text}"
        prompt = get_default_tts_prompt()
        messages = [{"role": "user", "content": f"{speaker_token}{prompt}:\n{combined_text}"}]

        inputs = self.processor(
            messages,
            add_generation_prompt=True,
            device=self.device,
            dtype=self.dtype,
        )

        # Manually append <audio_start> + <audio_output_placeholder> * num_ref_frames.
        audio_prefix = torch.full(
            (1, 1 + num_ref_frames),
            AUDIO_OUTPUT_PLACEHOLDER.id,
            dtype=torch.long,
            device=self.device,
        )
        audio_prefix[0, 0] = AUDIO_START.id
        input_ids = torch.cat([inputs["input_ids"], audio_prefix], dim=1)
        attention_mask = torch.cat([inputs["attention_mask"], torch.ones_like(audio_prefix)], dim=1)

        speaker_audio_tensor = self._load_speaker_audio(speaker_audio)

        params = self.task_params.get("tts_continuation", self.task_params.get("tts", {}))
        with torch.no_grad():
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                audio_output=ref_audio_tensor,
                audio_output_lengths=ref_audio_lengths,
                max_new_tokens=params.get("max_new_tokens", 512),
                temperature=params.get("temperature", 1.2),
                top_k=params.get("top_k", 20),
                top_p=params.get("top_p", 0.8),
                do_sample=True,
                force_audio_output=True,
                speaker_audio=speaker_audio_tensor,
                ras_enabled=params.get("ras_enabled", False),
                ras_window_size=params.get("ras_window_size", 50),
                ras_repetition_threshold=params.get("ras_repetition_threshold", 0.5),
                continuation_silence_frames=params.get("continuation_silence_frames", 0),
            )

        audio = output["audio"]
        audio_lengths = output["audio_lengths"]
        waveform = audio[0]
        length = int(audio_lengths[0].item())
        return waveform[:length], self.processor.sampling_rate

    def speech_chat(self, audio: str) -> str:
        """Speech-Chat: audio → text.

        Args:
            audio: Path to the audio file containing the spoken question.

        Returns:
            Text answer string.
        """
        messages: list[dict] = [
            {
                "role": "user",
                "content": [{"type": "audio", "audio": audio}],
            }
        ]
        params = self.task_params.get("speech-chat", {})
        return self.chat(  # type: ignore[return-value]
            messages,
            force_audio_output=params.get("force_audio_output", False),
            force_text_output=params.get("force_text_output", True),
            max_new_tokens=params.get("max_new_tokens", 1024),
            temperature=params.get("temperature", 0.7),
            max_audio_chunk_length=params.get("max_audio_chunk_length"),
        )

    def textqa(self, text: str, audio: str | None = None) -> str:
        """TextQA: text + optional audio → text.

        Args:
            text: The input text prompt or question.
            audio: Optional path to an audio file providing context.

        Returns:
            Generated text string.
        """
        if audio is not None:
            content: str | list[dict] = [
                {"type": "audio", "audio": audio},
                {"type": "text", "text": text},
            ]
        else:
            content = text
        messages: list[dict] = [{"role": "user", "content": content}]
        params = self.task_params.get("textqa", {})
        return self.chat(  # type: ignore[return-value]
            messages,
            force_audio_output=params.get("force_audio_output", False),
            force_text_output=params.get("force_text_output", True),
            max_new_tokens=params.get("max_new_tokens", 1024),
            temperature=params.get("temperature", 0.7),
            max_audio_chunk_length=params.get("max_audio_chunk_length"),
        )

    # ------------------------------------------------------------------
    # Duplex
    # ------------------------------------------------------------------

    def load_audio(
        self,
        path: str,
        channel: int | None = None,
    ) -> torch.Tensor:
        """Load an audio file, resample to the model's sampling rate.

        For stereo duplex audio: channel 0 = assistant, channel 1 = user.

        Args:
            path: Path to the audio file.
            channel: Channel index to extract from stereo audio. ``None`` = mono mix.

        Returns:
            Audio tensor of shape ``[1, num_samples]`` on ``self.device``.
        """
        audio, _ = _load_audio_shared(
            path,
            self.processor.sampling_rate,
            mono=channel is None,
            channel=channel,
            device=self.device,
            dtype=self.dtype,
        )
        return audio

    def duplex(
        self,
        audio_input: torch.Tensor,
        output_dir: str,
        *,
        save_hidden: bool = False,
        steering_layer: int | None = None,
        steering_vectors: list[torch.Tensor | None] | None = None,
        system_prompt: str | None = None,
        speak_first: bool | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        sil_penalty: float | None = None,
        bc_penalty: float | None = None,
        speaker_embeds: torch.Tensor | None = None,
        speaker_audio: str | None = None,
        eos_penalty: float | None = None,
    ) -> dict:
        """Run full-duplex inference.

        Parameters default to values from ``config/duplex_infer.yaml``.

        Args:
            audio_input: User audio tensor of shape ``[1, num_samples]``.
            output_dir: Directory to save output files.
            system_prompt: System prompt text.
            save_hidden: If True, save per-step hidden payload to output_hidden.pt.
            steering_layer: Optional thinker layer index to inject steering vectors.
            steering_vectors: Optional per-step vectors; each step can be None (no injection) or a 1D vector.
            speak_first: If True, the model speaks first.
            temperature: Sampling temperature.
            top_p: Top-p sampling threshold.
            top_k: Top-k filtering.
            sil_penalty: Penalty subtracted from SIL token logit.
            speaker_embeds: Optional precomputed speaker embeddings.
            speaker_audio: Optional speaker reference audio path. Used only
                when ``speaker_embeds`` is not provided.
            eos_penalty: EOS penalty value.

        Returns:
            Summary dict with durations and sample counts.
        """
        from raon.duplex_generate import run_duplex_inference

        cfg = self.duplex_params
        if system_prompt is None:
            system_prompt = cfg.get("system_prompt", "You are engaging in real-time conversation.")
        if speak_first is None:
            speak_first = cfg.get("speak_first", False)
        if temperature is None:
            temperature = cfg.get("temperature", 0.9)
        if top_p is None:
            top_p = cfg.get("top_p", 0.95)
        if top_k is None:
            top_k = cfg.get("top_k", 66)
        if sil_penalty is None:
            sil_penalty = cfg.get("sil_penalty", 0.0)
        if bc_penalty is None:
            bc_penalty = cfg.get("bc_penalty", 0.0)
        if eos_penalty is None:
            eos_penalty = cfg.get("eos_penalty", 0.0)
        if speaker_embeds is None and speaker_audio is not None:
            speaker_audio_tensor = self._load_speaker_audio(speaker_audio)
            speaker_embeds = self.model._compute_speaker_embeds(speaker_audio_tensor, None)

        return run_duplex_inference(
            model=self.model,
            processor=self.processor,
            audio_input=audio_input,
            output_dir=Path(output_dir),
            system_prompt=system_prompt,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            sil_penalty=sil_penalty,
            bc_penalty=bc_penalty,
            speaker_embeds=speaker_embeds,
            device=self.device,
            dtype=self.dtype,
            eos_penalty=eos_penalty,
            speak_first=speak_first,
            save_hidden=save_hidden,
            steering_layer=steering_layer,
            steering_vectors=steering_vectors,
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def save_audio(result: tuple[torch.Tensor, int], path: str) -> None:
        """Save an audio result to a WAV file.

        Args:
            result: ``(waveform, sampling_rate)`` tuple as returned by
                :meth:`tts` or :meth:`chat`.
            path: Destination file path (e.g. ``"output.wav"``).
        """
        from raon.utils.audio_io import save_audio as _save

        waveform, sampling_rate = result
        _save(waveform, sampling_rate, path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _trim_last_frame(self, waveform: torch.Tensor) -> torch.Tensor:
        """Drop one codec frame from generated audio to avoid trailing artifacts."""
        samples_per_frame = int(self.model.sampling_rate / self.model.frame_rate)
        if waveform.shape[-1] <= samples_per_frame:
            logger.warning(
                "Generated audio too short (%d samples <= %d samples_per_frame); returning empty waveform.",
                waveform.shape[-1],
                samples_per_frame,
            )
            return waveform[:0]
        return waveform[:-samples_per_frame]

    def _load_speaker_audio(self, path: str) -> torch.Tensor:
        """Load and preprocess a speaker reference audio file.

        Resamples to the model's sampling rate and converts to mono.

        Args:
            path: File path to the speaker audio.

        Returns:
            Float tensor of shape ``[1, num_samples]`` on ``self.device``.
        """
        audio, _ = _load_audio_shared(
            path,
            self.processor.sampling_rate,
            mono=True,
            device=self.device,
            dtype=self.dtype,
        )
        return audio
