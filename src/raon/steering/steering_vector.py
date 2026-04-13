"""Build Personaplex-compatible steering_vector.json files for RAON.

This script computes a mean steering direction from a mode-class dataset:
    direction = mu_speak - mu_listen

Token windows for labeling (per sample):
- listening: question_start + 1s < t < question_end - 1s
- speaking:  question_end + 1s < t < question_end + 11s

Where t is token time in seconds derived from frame_rate.

Then it writes steering vectors into root_dir/*/steering_vector.json using the
same nested format used by Personaplex:
    {
      "layer_0": {"0": null, ..., "k": [..D..], ...},
      "layer_1": { ... }
    }

Injection index is aligned to input_timing.json["interrupt_start"].
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import wave
from pathlib import Path
from typing import Any

import torch


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=str(path.parent), delete=False) as tf:
        json.dump(payload, tf, indent=2, ensure_ascii=False)
        tf.flush()
        os.fsync(tf.fileno())
        tmp_name = tf.name
    os.replace(tmp_name, path)


def _load_existing_steering_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict in {path}, got {type(data)}")
    return data


def _load_hidden(path: Path) -> tuple[torch.Tensor, float]:
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict payload at {path}, got {type(payload).__name__}")
    if "hidden_states" not in payload:
        raise KeyError(f"Missing 'hidden_states' in {path}")

    hidden = payload["hidden_states"]
    if not isinstance(hidden, torch.Tensor):
        hidden = torch.as_tensor(hidden)
    if hidden.ndim == 2:
        hidden = hidden.unsqueeze(1)
    if hidden.ndim != 3:
        raise ValueError(f"Expected hidden_states [T,L,D] or [T,D], got {tuple(hidden.shape)} in {path}")

    frame_rate = float(payload.get("frame_rate", 12.5))
    assert frame_rate > 0.0, f"Invalid frame_rate {frame_rate} in {path}"

    return hidden.float(), frame_rate


def _load_timing(path: Path) -> tuple[float, float, float]:
    with path.open("r", encoding="utf-8") as f:
        timing = json.load(f)
    if not isinstance(timing, dict):
        raise ValueError(f"Expected dict in {path}, got {type(timing)}")
    for key in ("question_start", "question_end", "interrupt_start"):
        if key not in timing:
            raise KeyError(f"Missing '{key}' in {path}")

    question_start = float(timing["question_start"])
    question_end = float(timing["question_end"])
    interrupt_start = float(timing["interrupt_start"])
    assert question_end > question_start, f"Expected question_end > question_start in {path}"

    return question_start, question_end, interrupt_start


def _load_question_timing(path: Path) -> tuple[float, float]:
    with path.open("r", encoding="utf-8") as f:
        timing = json.load(f)
    if not isinstance(timing, dict):
        raise ValueError(f"Expected dict in {path}, got {type(timing)}")
    for key in ("question_start", "question_end"):
        if key not in timing:
            raise KeyError(f"Missing '{key}' in {path}")

    question_start = float(timing["question_start"])
    question_end = float(timing["question_end"])
    assert question_end > question_start, f"Expected question_end > question_start in {path}"
    return question_start, question_end


def _wav_duration_seconds(wav_path: Path) -> float:
    try:
        with wave.open(str(wav_path), "rb") as wf:
            nframes = wf.getnframes()
            framerate = wf.getframerate()
        if framerate <= 0:
            raise ValueError(f"Invalid sample rate in WAV: {wav_path}")
        return float(nframes) / float(framerate)
    except wave.Error:
        import soundfile as sf

        info = sf.info(str(wav_path))
        if info.samplerate <= 0:
            raise ValueError(f"Invalid sample rate in WAV: {wav_path}")
        return float(info.frames) / float(info.samplerate)


def _build_mode_masks(num_steps: int, frame_rate: float, question_start: float, question_end: float) -> tuple[torch.Tensor, torch.Tensor]:
    idx = torch.arange(num_steps, dtype=torch.float32)
    times = idx / float(frame_rate)

    listen_mask = (times > (question_start + 1.0)) & (times < (question_end - 1.0))
    speak_mask = (times > (question_end + 1.0)) & (times < (question_end + 11.0))

    return listen_mask, speak_mask


def _discover_mode_samples(mode_class_dataset: Path) -> list[Path]:
    sample_dirs = [p for p in mode_class_dataset.iterdir() if p.is_dir()]
    sample_dirs.sort(key=lambda p: (0, int(p.name)) if p.name.isdigit() else (1, p.name))

    valid: list[Path] = []
    for sd in sample_dirs:
        if (sd / "output_hidden.pt").is_file() and (sd / "input_timing.json").is_file():
            valid.append(sd)
    return valid


def _compute_mean_direction(mode_class_dataset: Path, alpha: float) -> tuple[torch.Tensor, float]:
    sample_dirs = _discover_mode_samples(mode_class_dataset)
    if not sample_dirs:
        raise FileNotFoundError(
            f"No valid samples found in {mode_class_dataset}. Need */output_hidden.pt and */input_timing.json"
        )

    listen_sum: torch.Tensor | None = None
    speak_sum: torch.Tensor | None = None
    listen_count: int = 0
    speak_count: int = 0
    dataset_frame_rate: float | None = None

    for sd in sample_dirs:
        hidden, frame_rate = _load_hidden(sd / "output_hidden.pt")  # [T,L,D]
        if dataset_frame_rate is None:
            dataset_frame_rate = float(frame_rate)
        else:
            assert abs(float(frame_rate) - float(dataset_frame_rate)) < 1e-6, (
                f"Inconsistent frame_rate in mode_class_dataset: {frame_rate} vs {dataset_frame_rate}"
            )
        q_start, q_end = _load_question_timing(sd / "input_timing.json")

        num_steps = int(hidden.shape[0])
        listen_mask, speak_mask = _build_mode_masks(num_steps, frame_rate, q_start, q_end)
        sample_listen_count = int(listen_mask.sum())
        sample_speak_count = int(speak_mask.sum())
        print(
            f"[steering_vector][mode_class] sample={sd.name} "
            f"listening_tokens={sample_listen_count} speaking_tokens={sample_speak_count} "
            f"total_steps={num_steps}"
        )

        if sample_listen_count > 0:
            listen_chunk = hidden[listen_mask].sum(dim=0).cpu()  # [L,D]
            if listen_sum is None:
                listen_sum = torch.zeros_like(listen_chunk)
            assert listen_sum.shape == listen_chunk.shape, "listen shape mismatch"
            listen_sum += listen_chunk
            listen_count += sample_listen_count

        if sample_speak_count > 0:
            speak_chunk = hidden[speak_mask].sum(dim=0).cpu()  # [L,D]
            if speak_sum is None:
                speak_sum = torch.zeros_like(speak_chunk)
            assert speak_sum.shape == speak_chunk.shape, "speak shape mismatch"
            speak_sum += speak_chunk
            speak_count += sample_speak_count

    if listen_sum is None or speak_sum is None:
        raise RuntimeError("Failed to collect both listening and speaking hidden chunks.")
    assert listen_count > 0, "No listening tokens collected from mode_class_dataset"
    assert speak_count > 0, "No speaking tokens collected from mode_class_dataset"
    print(
        f"[steering_vector][mode_class] aggregate listening_tokens={listen_count} "
        f"speaking_tokens={speak_count}"
    )

    mu_listen = listen_sum / float(listen_count)  # [L,D]
    mu_speak = speak_sum / float(speak_count)  # [L,D]
    direction = (mu_speak - mu_listen) * float(alpha)
    assert dataset_frame_rate is not None
    return direction, float(dataset_frame_rate)


def _resolve_total_tokens(entry_dir: Path, frame_rate: float) -> int:
    hidden_path = entry_dir / "output_hidden.pt"
    if hidden_path.is_file():
        hidden, hidden_rate = _load_hidden(hidden_path)
        assert abs(hidden_rate - frame_rate) < 1e-6, (
            f"Frame-rate mismatch in {entry_dir}: hidden={hidden_rate}, expected={frame_rate}"
        )
        return int(hidden.shape[0])

    wav_path = entry_dir / "input.wav"
    if wav_path.is_file():
        dur = _wav_duration_seconds(wav_path)
        total_tokens = int(math.ceil(dur * frame_rate))
        assert total_tokens > 0, f"Computed non-positive tokens for {wav_path}"
        return total_tokens

    raise FileNotFoundError(
        f"Cannot resolve total tokens for {entry_dir}. Need output_hidden.pt or input.wav"
    )


def _build_layer_payload(vec: torch.Tensor, total_tokens: int, start_idx: int, decay_span: int) -> dict[str, list[float] | None]:
    payload: dict[str, list[float] | None] = {str(i): None for i in range(total_tokens)}
    payload[str(start_idx)] = vec.tolist()

    for k in range(1, int(decay_span) + 1):
        tid = start_idx + k
        if tid >= total_tokens:
            break
        decay = 1.0 - (float(k) / float(decay_span)) if decay_span > 0 else 0.0
        if decay <= 0.0:
            payload[str(tid)] = None
            continue
        payload[str(tid)] = (vec * float(decay)).tolist()

    return payload


def write_steering_vectors(
    root_dir: str,
    mode_class_dataset: str,
    alpha: float,
    decay_span: int = 0,
) -> None:
    root = Path(root_dir)
    mode_root = Path(mode_class_dataset)
    if not root.is_dir():
        raise FileNotFoundError(f"root_dir not found: {root}")
    if not mode_root.is_dir():
        raise FileNotFoundError(f"mode_class_dataset not found: {mode_root}")
    if decay_span < 0:
        raise ValueError(f"decay_span must be >= 0, got {decay_span}")

    direction, frame_rate = _compute_mean_direction(mode_root, alpha=alpha)  # [L,D], scalar
    n_layers = int(direction.shape[0])

    target_dirs = [p for p in root.iterdir() if p.is_dir() and (p / "input_timing.json").is_file()]
    target_dirs.sort(key=lambda p: (0, int(p.name)) if p.name.isdigit() else (1, p.name))
    if not target_dirs:
        raise FileNotFoundError(f"No target entries under {root} with input_timing.json")

    updated = 0
    for entry_dir in target_dirs:
        q_start, q_end, interrupt_start = _load_timing(entry_dir / "input_timing.json")
        # Alignment sanity: interruption should happen after question starts.
        assert interrupt_start >= q_start, (
            f"interrupt_start ({interrupt_start}) must be >= question_start ({q_start}) in {entry_dir}"
        )

        total_tokens = _resolve_total_tokens(entry_dir, frame_rate=frame_rate)
        start_idx = int(interrupt_start * frame_rate)
        start_idx = max(0, min(start_idx, total_tokens - 1))

        start_time_from_idx = start_idx / frame_rate
        # Quantization sanity: idx->time should be within one frame.
        assert abs(start_time_from_idx - interrupt_start) <= (1.0 / frame_rate), (
            f"interrupt_start alignment too large in {entry_dir}: "
            f"interrupt_start={interrupt_start:.6f}, start_idx={start_idx}, idx_time={start_time_from_idx:.6f}"
        )

        steering_path = entry_dir / "steering_vector.json"
        existing = _load_existing_steering_payload(steering_path)

        for layer_idx in range(n_layers):
            key = f"layer_{layer_idx}"
            layer_vec = direction[layer_idx].detach().cpu().float()
            existing[key] = _build_layer_payload(
                vec=layer_vec,
                total_tokens=total_tokens,
                start_idx=start_idx,
                decay_span=decay_span,
            )

        _atomic_write_json(steering_path, existing)
        updated += 1

    print(
        f"[steering_vector] Done. Updated {updated} entries at {root}. "
        f"layers={n_layers}, alpha={alpha}, decay_span={decay_span}, frame_rate={frame_rate}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Personaplex-compatible steering_vector.json for RAON.")
    ap.add_argument("--root-dir", type=str, required=True, help="Target dataset root containing */input_timing.json")
    ap.add_argument(
        "--mode-class-dataset",
        type=str,
        required=True,
        help="Mode-class dataset root containing */output_hidden.pt and */input_timing.json",
    )
    ap.add_argument("--alpha", type=float, required=True, help="Scale factor for steering direction (mu_speak - mu_listen)")
    ap.add_argument("--decay-span", type=int, default=0, help="Optional linear decay length in tokens after interrupt_start")
    args = ap.parse_args()

    write_steering_vectors(
        root_dir=args.root_dir,
        mode_class_dataset=args.mode_class_dataset,
        alpha=float(args.alpha),
        decay_span=int(args.decay_span),
    )


if __name__ == "__main__":
    main()
