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

"""Full-duplex inference script for RAON.

Usage:
    python -m raon.duplex_generate \
        --model_path /path/to/model \
        --audio_input /path/to/input.wav \
        --output_dir /path/to/output \
        --device cuda \
        --dtype bfloat16

    python -m raon.duplex_generate \
        --model_path /path/to/model \
        --data_dir /path/to/data-dir \
        --output_dir /path/to/output \
        --device cuda \
        --dtype bfloat16
"""

from __future__ import annotations

import argparse
import json
import logging
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from tqdm.auto import trange

from raon.utils.audio_io import load_audio as _load_audio_shared

from raon.models.raon import RaonDuplexModel, RaonModel  # noqa: F401 — registers raon_duplex with AutoModel
from raon.utils.processor import RaonProcessor
from raon.utils.special_tokens import (
    AUDIO_INPUT_PLACEHOLDER,
    AUDIO_OUTPUT_END_PAD,
    AUDIO_OUTPUT_PAD,
    AUDIO_OUTPUT_PLACEHOLDER,
    AUDIO_START,
    IM_END,
    IM_START,
)

logger = logging.getLogger(__name__)


def _load_yaml_config(config_path: str | Path) -> dict:
    """Load duplex inference config from YAML, returning the 'duplex' section."""
    import yaml

    with open(config_path) as f:
        raw = yaml.safe_load(f)
    return raw.get("duplex", {}) if raw else {}


def _resolve_metadata_audio_path(path_value: str, metadata_jsonl_path: Path | None) -> str:
    """Resolve metadata-provided audio path to an existing filesystem path when possible."""
    raw = Path(path_value).expanduser()
    candidates: list[Path]
    if raw.is_absolute():
        candidates = [raw]
    else:
        candidates = [raw]
        if metadata_jsonl_path is not None:
            candidates.append((metadata_jsonl_path.parent / raw))
            candidates.append((metadata_jsonl_path.parent.parent / raw))

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])


def _extract_speaker_audio_from_metadata(metadata: dict) -> str | None:
    """Extract speaker reference audio path from metadata JSON (preferred key: ``speaker_audio``)."""
    speaker_audio = metadata.get("speaker_audio")
    if isinstance(speaker_audio, str) and speaker_audio.strip():
        return speaker_audio.strip()

    speaker_ref_audios = metadata.get("speaker_ref_audios")
    if isinstance(speaker_ref_audios, list) and speaker_ref_audios and isinstance(speaker_ref_audios[0], str):
        first = speaker_ref_audios[0].strip()
        if first:
            return first
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAON full-duplex inference on an input audio file.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pretrained model directory.")
    parser.add_argument(
        "--audio_input",
        type=str,
        default=None,
        help="Path to the input audio file (.wav). Use this for single-sample inference.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Directory scanned recursively for *.jsonl. Each non-empty line is treated as one sample.",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output audio.")
    parser.add_argument("--device", type=str, default="cuda", help="Device (default: cuda).")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Torch dtype (default: bfloat16).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to duplex_infer.yaml. Defaults to config/duplex_infer.yaml relative to repo root.",
    )
    parser.add_argument(
        "--temperature", type=float, default=None, help="Sampling temperature (default: from config or 0.9)."
    )
    parser.add_argument("--top_k", type=int, default=None, help="Top-k filtering (default: from config or 66).")
    parser.add_argument("--top_p", type=float, default=None, help="Top-p filtering (default: from config or 0.95).")
    parser.add_argument(
        "--sil_penalty",
        type=float,
        default=None,
        help="Penalty subtracted from SIL token logit (default: from config or 0.0).",
    )
    parser.add_argument(
        "--bc_penalty",
        type=float,
        default=None,
        help="Penalty subtracted from BC token logit (default: from config or 0.0).",
    )
    parser.add_argument(
        "--eos_penalty",
        type=float,
        default=None,
        help="Penalty subtracted from EOS/pad token logit to encourage longer output (default: from config or 0.0).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed for reproducible duplex decoding.")
    parser.add_argument(
        "--speak_first",
        action="store_true",
        default=None,
        help="Force model to speak first. Default: listen first.",
    )
    parser.add_argument(
        "--persona",
        type=str,
        default=None,
        help="Persona key (from catalog) or raw persona description string.",
    )
    parser.add_argument(
        "--context",
        type=str,
        default=None,
        help="Additional context sentence appended to system prompt when persona is set.",
    )
    parser.add_argument(
        "--system_prompt_style",
        type=str,
        default=None,
        choices=["base", "persona", "persona_context", "custom"],
        help="System prompt style. Default: base.",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=None,
        help="Custom system prompt text. Only used when --system_prompt_style=custom.",
    )
    parser.add_argument(
        "--speaker_audio",
        type=str,
        default=None,
        help="Speaker reference audio for voice conditioning (e.g., data/duplex/eval/audio/spk_ref.wav).",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="eager",
        choices=["fa", "sdpa", "eager"],
        help="Attention implementation (default: eager). Use `fa` for FlashAttention.",
    )
    args = parser.parse_args()
    if bool(args.audio_input) == bool(args.data_dir):
        parser.error("Provide exactly one of --audio_input or --data_dir.")
    if args.attn_implementation == "fa":
        args.attn_implementation = "flash_attention_2"

    # Auto-detect metadata JSONL alongside audio input.
    # e.g. /path/to/duplex_00.wav → /path/to/duplex_00.jsonl or /path/to/../duplex_00.jsonl
    _meta = None
    _meta_jsonl_path: Path | None = None
    if args.audio_input:
        audio_path = Path(args.audio_input)
        jsonl_candidates = [
            audio_path.with_suffix(".jsonl"),  # same dir as audio
            audio_path.parent.parent / f"{audio_path.stem}.jsonl",  # parent dir (e.g. audio/ → ../)
        ]
        for jsonl_path in jsonl_candidates:
            if jsonl_path.exists():
                with open(jsonl_path) as _f:
                    _meta = json.loads(_f.readline())
                _meta_jsonl_path = jsonl_path
                logger.info("Loaded metadata from %s", jsonl_path)
                break

    if _meta is not None:
        # CLI args take precedence over metadata values.
        if args.speak_first is None:
            args.speak_first = _meta.get("speak_first", False)
        if args.persona is None:
            args.persona = _meta.get("persona")
        if args.context is None:
            args.context = _meta.get("context")
        if args.system_prompt is None:
            args.system_prompt = _meta.get("system_prompt")
        if args.speaker_audio is None:
            metadata_speaker_audio = _extract_speaker_audio_from_metadata(_meta)
            if metadata_speaker_audio is not None:
                args.speaker_audio = _resolve_metadata_audio_path(
                    metadata_speaker_audio,
                    _meta_jsonl_path,
                )
                logger.info("Using speaker reference audio from metadata: %s", args.speaker_audio)
        args._meta_name = _meta.get("name")
    else:
        args._meta_name = None

    # Resolve config path: explicit --config > config/duplex_infer.yaml relative to repo root.
    if args.config is None:
        repo_root = Path(__file__).resolve().parents[2]
        default_config = repo_root / "config" / "duplex_infer.yaml"
        if default_config.exists():
            args.config = str(default_config)

    # Load sampling parameters from yaml config. CLI args take precedence.
    cfg = _load_yaml_config(args.config) if args.config else {}
    if args.temperature is None:
        args.temperature = float(cfg.get("temperature", 0.9))
    if args.top_p is None:
        args.top_p = float(cfg.get("top_p", 0.95))
    if args.top_k is None:
        args.top_k = int(cfg.get("top_k", 66))
    args.do_sample = bool(cfg.get("do_sample", True))
    if args.eos_penalty is None:
        args.eos_penalty = float(cfg.get("eos_penalty", 0.0))
    if args.sil_penalty is None:
        args.sil_penalty = float(cfg.get("sil_penalty", 0.0))
    if args.bc_penalty is None:
        args.bc_penalty = float(cfg.get("bc_penalty", 0.0))
    if args.seed is None and cfg.get("seed") is not None:
        args.seed = int(cfg["seed"])

    # CLI args take precedence over yaml config.
    if args.speak_first is None:
        args.speak_first = bool(cfg.get("speak_first", False))
    if args.system_prompt_style is None:
        args.system_prompt_style = cfg.get("system_prompt_style", "base")
    if args.persona is None:
        args.persona = cfg.get("persona") or None
    if args.context is None:
        args.context = cfg.get("context") or None

    # Build system prompt based on style.
    from raon.utils.duplex_prompt_catalog import build_system_prompt

    if args.system_prompt is not None:
        # Already set (explicit --system_prompt or from metadata).
        system_prompt_text = args.system_prompt
    elif args.system_prompt_style == "custom" and args.system_prompt:
        system_prompt_text = args.system_prompt
    else:
        system_prompt_text = build_system_prompt(
            persona=args.persona,
            context=args.context,
            name=getattr(args, "_meta_name", None),
            deterministic=True,
        )
    args.system_prompt = system_prompt_text

    return args


from raon.utils.misc import resolve_dtype


def load_audio(path: str, target_sr: int, device: str, dtype: torch.dtype, channel: int | None = None) -> torch.Tensor:
    """Load an audio file, resample to *target_sr*, and return as ``[1, num_samples]``.

    For stereo duplex audio: channel 0 = assistant, channel 1 = user.
    Use ``channel=1`` for user input, ``channel=0`` for speaker embedding source.

    Args:
        path: Path to the audio file.
        target_sr: Target sampling rate in Hz.
        device: Torch device string (e.g. ``"cuda"``).
        dtype: Torch dtype for the output tensor.
        channel: Channel index to extract from stereo audio. None = mono mix.

    Returns:
        Audio tensor of shape ``[1, num_samples]`` on *device* with *dtype*.
    """
    audio, _ = _load_audio_shared(
        path,
        target_sr,
        mono=channel is None,
        channel=channel,
        device=device,
        dtype=dtype,
    )
    return audio


from raon.utils.audio_io import save_audio


def _resolve_dataset_path(path_value: str | None, jsonl_path: Path) -> str | None:
    if not path_value:
        return None
    raw = Path(path_value).expanduser()
    if raw.is_absolute():
        return str(raw)
    candidates = [raw, jsonl_path.parent / raw, jsonl_path.parent.parent / raw]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return None


def save_stereo_audio(
    user_np: np.ndarray,
    assistant_np: np.ndarray,
    sampling_rate: int,
    output_path: Path,
) -> None:
    """Write stereo audio (L=user, R=assistant) to a ``.wav`` file.

    Pads the shorter channel with zeros so both have equal length.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    max_len = max(len(user_np), len(assistant_np))
    user_padded = np.pad(user_np, (0, max_len - len(user_np)))
    assistant_padded = np.pad(assistant_np, (0, max_len - len(assistant_np)))
    stereo = np.stack([user_padded, assistant_padded], axis=-1)  # [samples, 2]
    sf.write(str(output_path), stereo, sampling_rate)


def save_summary(summary: dict, output_path: Path) -> None:
    """Write inference summary as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def run_duplex_inference(
    model,
    processor: RaonProcessor,
    audio_input: torch.Tensor,
    output_dir: Path,
    system_prompt: str = "",
    do_sample: bool = True,
    temperature: float = 0.9,
    top_p: float = 0.8,
    top_k: int = 20,
    speaker_embeds: torch.Tensor | None = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    eos_penalty: float = 0.0,
    sil_penalty: float = 0.0,
    bc_penalty: float = 0.0,
    speak_first: bool = False,
    save_hidden: bool = False,
    steering_layer: int | None = None,
    steering_vectors: list[torch.Tensor | None] | None = None,
) -> dict:
    """Run full-duplex inference on a single audio input and save results.

    Performs frame-by-frame duplex decoding, saves assistant audio, stereo mix,
    decoded text, and a JSON summary to *output_dir*.

    Args:
        model: RAON duplex model (must support ``init_duplex_decoding_state``).
        processor: Processor with tokenizer and audio config.
        audio_input: User audio tensor of shape ``[1, num_samples]``.
        output_dir: Directory to save output files.
        system_prompt: Optional system prompt text.
        do_sample: Enable sampling.
        temperature: Sampling temperature.
        top_p: Top-p sampling threshold.
        top_k: Top-k filtering.
        speaker_embeds: Optional precomputed speaker embeddings.
        device: Torch device string.
        dtype: Torch dtype.

    Returns:
        Summary dict with durations and sample counts.
    """
    tokenizer = processor.tokenizer
    sr = processor.sampling_rate

    # Build system prompt tokens
    system_messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
    if system_messages:
        inputs = processor(system_messages, add_generation_prompt=False, device=device, dtype=dtype)
        system_tokens = inputs["input_ids"]
    else:
        system_tokens = torch.zeros((1, 0), dtype=torch.long, device=device)

    logger.info("Running duplex generation ...")    
    samples_per_frame = int(sr / processor.frame_rate)
    audio_input_length = audio_input.shape[-1]
    expected_decode_steps = ((audio_input_length - samples_per_frame) // samples_per_frame) + 1
    if steering_layer is not None and steering_vectors is not None:
        _active_steps = sum(1 for v in steering_vectors if v is not None)
        logger.info(
            "Steering active: layer=%d, vectors=%d, active_steps=%d",
            int(steering_layer),
            len(steering_vectors),
            _active_steps,
        )

    if audio_input_length < samples_per_frame:
        raise ValueError(
            f"Audio input too short ({audio_input_length} samples) for duplex decoding "
            f"(minimum {samples_per_frame} samples = 1 frame at {sr}Hz / {processor.frame_rate}fps)."
        )

    with torch.inference_mode():
        state = model.init_duplex_decoding_state(
            sequences=system_tokens,
            attention_mask=torch.ones_like(system_tokens),
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            speaker_embeds=speaker_embeds,
            eos_penalty=eos_penalty,
            sil_penalty=sil_penalty,
            bc_penalty=bc_penalty,
            speak_first=speak_first,
        )

        audio_output_frames: list[torch.Tensor] = []
        hidden_steps: list[dict[str, torch.Tensor]] = []
        # [DEBUG-LOG] Frame-level logging for text-audio delay analysis
        _prev_seq_len = int(state.sequences.shape[1])
        _frame_log_path = output_dir / "frame_log.txt"
        _frame_log_path.parent.mkdir(parents=True, exist_ok=True)
        _frame_log = open(str(_frame_log_path), "w", buffering=1)
        _text_vocab_size = int(getattr(model, "text_vocab_size", 0) or 0)
        _frame_idx = 0
        try:
            for i in trange(
                0,
                audio_input_length - samples_per_frame + 1,
                samples_per_frame,
                mininterval=0,
                desc="Duplex Generation",
            ):
                audio_input_frame = audio_input[:, i : i + samples_per_frame]
                step_steering_vector = None
                if steering_vectors is not None and _frame_idx < len(steering_vectors):
                    step_steering_vector = steering_vectors[_frame_idx]
                _steer_tag = "-"
                if steering_layer is not None and step_steering_vector is not None:
                    _steer_norm = float(step_steering_vector.float().norm().item())
                    _steer_tag = f"L{int(steering_layer)}|norm={_steer_norm:.4f}"
                state, audio_output_frame = model.duplex_decoding_step(
                    state=state,
                    audio_input=audio_input_frame,
                    hidden_collector=hidden_steps if save_hidden else None,
                    steering_layer=steering_layer,
                    steering_vector=step_steering_vector,
                )
                audio_output_frames.append(audio_output_frame)

                # [DEBUG-LOG] Extract text delta and audio RMS for this frame
                _cur_seq_len = int(state.sequences.shape[1])
                _new_tokens = state.sequences[0, _prev_seq_len:_cur_seq_len].tolist()
                _text_tokens = [
                    t
                    for t in _new_tokens
                    if t < _text_vocab_size
                    and t
                    not in {
                        AUDIO_INPUT_PLACEHOLDER.id,
                        AUDIO_OUTPUT_PLACEHOLDER.id,
                        AUDIO_OUTPUT_PAD.id,
                        AUDIO_OUTPUT_END_PAD.id,
                        AUDIO_START.id,
                        IM_START.id,
                        IM_END.id,
                    }
                ]
                _text_str = tokenizer.decode(_text_tokens, skip_special_tokens=False) if _text_tokens else ""
                _out_rms = float(audio_output_frame.float().pow(2).mean().sqrt())
                _in_rms = float(audio_input_frame.float().pow(2).mean().sqrt())
                _phase = state.machine_state.phase.name if state.machine_state is not None else "?"
                _frame_log.write(
                    f"[{_phase}] f={_frame_idx} text={repr(_text_str) if _text_str else '-'} "
                    f"out_rms={_out_rms:.4f} in_rms={_in_rms:.4f} ntok={_cur_seq_len - _prev_seq_len} steer={_steer_tag}\n"
                )
                _prev_seq_len = _cur_seq_len
                _frame_idx += 1
        finally:
            _frame_log.close()
            logger.info("Saved frame log -> %s", _frame_log_path)
            model.free_duplex_decoding_state(state)

    audio_output = torch.cat(audio_output_frames, dim=1)  # [1, num_output_samples]

    # -- Save outputs --
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. assistant.wav — model-generated assistant audio (mono)
    assistant_np = audio_output[0].float().cpu().numpy()
    assistant_path = output_dir / "assistant.wav"
    save_audio(assistant_np, sr, assistant_path)
    logger.info(
        "Saved assistant audio: %d samples (%.2f sec) -> %s",
        len(assistant_np),
        len(assistant_np) / sr,
        assistant_path,
    )

    # 2. user_assistant.wav — stereo: L=user, R=assistant
    user_np = audio_input[0].float().cpu().numpy()
    stereo_path = output_dir / "user_assistant.wav"
    save_stereo_audio(user_np, assistant_np, sr, stereo_path)
    max_len = max(len(user_np), len(assistant_np))
    logger.info(
        "Saved stereo audio: %d samples (%.2f sec) -> %s",
        max_len,
        max_len / sr,
        stereo_path,
    )

    # 3. output.json — summary
    summary = {
        "assistant_duration_sec": len(assistant_np) / sr,
        "user_duration_sec": len(user_np) / sr,
        "assistant_samples": len(assistant_np),
        "user_samples": len(user_np),
        "sampling_rate": sr,
    }
    json_path = output_dir / "output.json"
    save_summary(summary, json_path)
    logger.info("Saved summary -> %s", json_path)

    if save_hidden:
        if len(hidden_steps) != expected_decode_steps:
            raise RuntimeError(
                "Hidden step count mismatch: "
                f"expected {expected_decode_steps} (from input audio frames), "
                f"got {len(hidden_steps)}."
            )

    if save_hidden and hidden_steps:
        frame_rate = float(processor.frame_rate)
        text_hidden_layers = torch.stack([step["text_hidden_layers"] for step in hidden_steps], dim=0)
        talker_hidden_states = torch.stack([step["talker_hidden_state"] for step in hidden_steps], dim=0)

        max_width = max(int(step["input_token_ids"].shape[0]) for step in hidden_steps)
        input_token_ids = torch.full((len(hidden_steps), max_width), -1, dtype=torch.long)
        for idx, step in enumerate(hidden_steps):
            ids = step["input_token_ids"].long()
            input_token_ids[idx, : ids.shape[0]] = ids

        # Per-step output tokens: [text_token, audio_token] where missing values are -1.
        if all("output_token_ids" in step for step in hidden_steps):
            output_token_ids = torch.stack(
                [step["output_token_ids"].long() for step in hidden_steps],
                dim=0,
            )
        else:
            # Fallback for legacy collector shape.
            fallback_text = [
                int(step["input_token_ids"][-2].item()) if step["input_token_ids"].numel() >= 2 else int(step["input_token_ids"][-1].item())
                for step in hidden_steps
            ]
            output_token_ids = torch.stack(
                [
                    torch.tensor([text_id, -1], dtype=torch.long)
                    for text_id in fallback_text
                ],
                dim=0,
            )

        # Keep token_ids/token_names as text-stream summary for compatibility.
        token_ids = output_token_ids[:, 0].clone()
        token_ids_list = [int(x.item()) for x in token_ids]

        token_names: list[str] = []
        tokenizer = processor.tokenizer
        for token_id in token_ids_list:
            try:
                piece = tokenizer.convert_ids_to_tokens(int(token_id))
                token_names.append(str(piece) if piece is not None else f"<id:{token_id}>")
            except Exception:  # noqa: BLE001
                token_names.append(f"<id:{token_id}>")

        hidden_payload = {
            "schema_version": 6,
            "input_wav": None,
            "output_wav": str(assistant_path),
            "output_text": str(json_path),
            "frame_rate": frame_rate,
            "token_ids": token_ids,
            "token_names": token_names,
            "input_token_ids": input_token_ids,
            "input_token_width": int(input_token_ids.shape[1]),
            "output_token_ids": output_token_ids,
            "talker_hidden_states": talker_hidden_states,
            # Preserve every layer: [T, L, D]
            "hidden_states": text_hidden_layers,
        }
        hidden_path = output_dir / "output_hidden.pt"
        torch.save(hidden_payload, hidden_path)
        logger.info("Saved hidden payload -> %s", hidden_path)

    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    if args.data_dir:
        data_dir = Path(args.data_dir)
        output_root = Path(args.output_dir)
        jsonl_files = sorted(data_dir.rglob("*.jsonl"))
        if not jsonl_files:
            raise SystemExit(f"No JSONL files found under: {data_dir}")

        processed = 0
        for jsonl_path in jsonl_files:
            with open(jsonl_path, encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    audio_path = _resolve_dataset_path(record.get("audio_path"), jsonl_path)
                    if not audio_path:
                        raise SystemExit(f"Missing audio_path in {jsonl_path}:{idx + 1}")

                    sample_out = output_root / f"{jsonl_path.stem}_{idx:05d}"
                    cmd = [
                        sys.executable,
                        "-m",
                        "raon.duplex_generate",
                        "--model_path",
                        args.model_path,
                        "--audio_input",
                        audio_path,
                        "--output_dir",
                        str(sample_out),
                        "--device",
                        args.device,
                        "--dtype",
                        args.dtype,
                    ]
                    if args.config:
                        cmd += ["--config", args.config]

                    attn_arg = "fa" if args.attn_implementation == "flash_attention_2" else args.attn_implementation
                    cmd += ["--attn_implementation", attn_arg]

                    if args.temperature is not None:
                        cmd += ["--temperature", str(args.temperature)]
                    if args.top_k is not None:
                        cmd += ["--top_k", str(args.top_k)]
                    if args.top_p is not None:
                        cmd += ["--top_p", str(args.top_p)]
                    if args.sil_penalty is not None:
                        cmd += ["--sil_penalty", str(args.sil_penalty)]
                    if args.bc_penalty is not None:
                        cmd += ["--bc_penalty", str(args.bc_penalty)]
                    if args.eos_penalty is not None:
                        cmd += ["--eos_penalty", str(args.eos_penalty)]

                    speaker_audio = args.speaker_audio or _resolve_dataset_path(
                        record.get("speaker_audio") or ((record.get("speaker_ref_audios") or [None])[0]),
                        jsonl_path,
                    )
                    if speaker_audio:
                        cmd += ["--speaker_audio", speaker_audio]

                    if isinstance(record.get("speak_first"), bool) and record["speak_first"]:
                        cmd.append("--speak_first")
                    if isinstance(record.get("persona"), str) and record["persona"].strip():
                        cmd += ["--persona", record["persona"].strip()]
                    if isinstance(record.get("context"), str) and record["context"].strip():
                        cmd += ["--context", record["context"].strip()]
                    if isinstance(record.get("system_prompt"), str) and record["system_prompt"].strip():
                        cmd += ["--system_prompt", record["system_prompt"].strip()]
                    if args.seed is not None:
                        cmd += ["--seed", str(args.seed)]

                    logger.info("[%s:%05d] %s", jsonl_path.name, idx, " ".join(shlex.quote(part) for part in cmd))
                    subprocess.run(cmd, check=True)
                    processed += 1

        logger.info("Processed %d duplex sample(s).", processed)
        return

    torch_dtype = resolve_dtype(args.dtype)
    output_dir = Path(args.output_dir)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        logger.info("Using RNG seed %d", args.seed)

    logger.info("Loading model from %s ...", args.model_path)
    from transformers import AutoModel

    model = AutoModel.from_pretrained(args.model_path, torch_dtype=torch_dtype, trust_remote_code=False)
    model._set_attention_implementation(args.attn_implementation)
    logger.info("Attention implementation: %s", args.attn_implementation)
    model = model.to(args.device).eval()

    logger.info("Loading processor from %s ...", args.model_path)
    processor = RaonProcessor.from_pretrained(args.model_path)

    # Load input audio: mono → pass through, stereo → average channels (matches duplex-model inference)
    logger.info("Loading audio from %s ...", args.audio_input)
    is_stereo = sf.info(args.audio_input).channels == 2
    if is_stereo:
        logger.info("Stereo detected: averaging channels to mono")
    audio_input = load_audio(args.audio_input, processor.sampling_rate, args.device, torch_dtype)
    logger.info(
        "Audio loaded: %d samples (%.2f sec)",
        audio_input.shape[1],
        audio_input.shape[1] / processor.sampling_rate,
    )

    # Compute speaker embeddings if --speaker_audio is provided.
    # Speaker conditioning injects an ECAPA-TDNN embedding into the decoding init state
    # via the speaker placeholder token. The embedding is baked into the KV cache at init
    # and conditions all subsequent frames. Note: effectiveness depends on the checkpoint —
    # early-stage checkpoints (e.g. v1-iter10k) may not exhibit strong voice adaptation.
    speaker_embeds = None
    if args.speaker_audio and hasattr(model, "speaker_encoder") and model.speaker_encoder is not None:
        logger.info("Computing speaker embeddings from %s ...", args.speaker_audio)
        speaker_audio = load_audio(args.speaker_audio, processor.sampling_rate, args.device, torch_dtype)
        speaker_embeds = model._compute_speaker_embeds(speaker_audio, None)
        logger.info("Speaker embeddings computed: shape %s", speaker_embeds.shape)

    run_duplex_inference(
        model=model,
        processor=processor,
        audio_input=audio_input,
        output_dir=output_dir,
        system_prompt=args.system_prompt,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        speaker_embeds=speaker_embeds,
        device=args.device,
        dtype=torch_dtype,
        eos_penalty=args.eos_penalty,
        sil_penalty=args.sil_penalty,
        bc_penalty=args.bc_penalty,
        speak_first=args.speak_first,
    )

    logger.info("Done. Output saved to %s", output_dir)


if __name__ == "__main__":
    main()
