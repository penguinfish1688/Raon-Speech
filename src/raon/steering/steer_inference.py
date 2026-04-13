from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from tqdm import tqdm

import torchaudio
import torch
from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from raon.utils.duplex_data import CHANNEL_DUPLEX_TO_SYSTEM_MESSAGE, get_duplex_system_message_key

logger = logging.getLogger(__name__)


def _is_local_path(model_path: str) -> bool:
    return model_path.startswith(("/", "./", "../", "~"))


def _validate_local_model_path(model_path: str) -> Path:
    model_dir = Path(model_path).expanduser().resolve()
    config_path = model_dir / "config.json"
    if not model_dir.exists():
        raise FileNotFoundError(f"Local model path does not exist: {model_dir}")
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Local model path is not a directory: {model_dir}")
    if not config_path.exists():
        raise FileNotFoundError(
            f"Invalid local model directory (missing config.json): {config_path}. "
            "Pass a HF repo id (e.g. KRAFTON/Raon-SpeechChat-9B) or a valid local checkpoint folder."
        )
    return model_dir


def _load_pipeline_class(model_path: str):
    """Load RaonPipeline class from HF remote code, mirroring duplex_example.py."""
    if _is_local_path(model_path):
        _validate_local_model_path(model_path)
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    revision = getattr(cfg, "_commit_hash", None)
    return get_class_from_dynamic_module(
        "modeling_raon.RaonPipeline",
        model_path,
        revision=revision,
    )


def _create_pipeline(
    model_path: str,
    *,
    device: str,
    dtype: str,
    attn_implementation: str,
):
    """Create pipeline preferring local raon import so local code edits are effective."""
    if _is_local_path(model_path):
        # Local path should be a proper HF-style checkpoint dir; fail fast with clear message.
        _validate_local_model_path(model_path)

    try:
        from raon.pipeline import RaonPipeline

        logger.info("Loaded RaonPipeline from local raon package for %s", model_path)
        return RaonPipeline(
            model_path=model_path,
            device=device,
            dtype=dtype,
            attn_implementation=attn_implementation,
        )
    except Exception as exc:  # noqa: BLE001
        if _is_local_path(model_path):
            raise RuntimeError(
                f"Failed to load local model from '{model_path}' via local raon pipeline: {exc}"
            ) from exc
        logger.warning("Local raon pipeline load failed (%s). Falling back to HF dynamic module.", exc)
        pipeline_cls = _load_pipeline_class(model_path)
        logger.info("Loaded RaonPipeline via HF dynamic module for %s", model_path)
        return pipeline_cls(model_path, device=device, dtype=dtype, attn_implementation=attn_implementation)


def _resolve_audio_channel(audio_path: Path, channel_mode: str = "auto-user") -> int | None:
    """Resolve which channel to use for duplex input.

    - auto-user: channel 1 for stereo, mono otherwise
    - mono: mix channels
    - left/right: force channel 0/1
    """
    if channel_mode == "mono":
        return None
    if channel_mode == "left":
        return 0
    if channel_mode == "right":
        return 1

    # Older torchaudio builds may not expose torchaudio.info.
    if hasattr(torchaudio, "info"):
        info = torchaudio.info(str(audio_path))
        return 1 if info.num_channels >= 2 else None

    waveform, _sample_rate = torchaudio.load(str(audio_path))
    return 1 if waveform.shape[0] >= 2 else None


def _resolve_system_prompt(prompt_or_key: str) -> str:
    """Resolve canonical duplex prompt keys to full text."""
    if prompt_or_key in CHANNEL_DUPLEX_TO_SYSTEM_MESSAGE:
        return CHANNEL_DUPLEX_TO_SYSTEM_MESSAGE[prompt_or_key]

    parts = [part.strip() for part in prompt_or_key.split(":") if part.strip()]
    if len(parts) in {2, 3}:
        if len(parts) == 2:
            language = "eng"
            channel, speak_mode = parts
        else:
            language, channel, speak_mode = parts

        if language == "eng" and channel in {"full_duplex", "duplex_instruct"} and speak_mode in {"speak-first", "listen-first"}:
            key = get_duplex_system_message_key(
                language=language,
                channel=channel,
                speak_first=(speak_mode == "speak-first"),
            )
            return CHANNEL_DUPLEX_TO_SYSTEM_MESSAGE.get(key, prompt_or_key)

    return prompt_or_key


def _resolve_speaker_audio_path(speaker_audio: str | None) -> str | None:
    """Use explicit speaker audio if provided; otherwise prefer demo default if present."""
    if speaker_audio:
        p = Path(speaker_audio).expanduser()
        return str(p)

    repo_root = Path(__file__).resolve().parents[3]
    default_ref = repo_root / "data" / "duplex" / "eval" / "audio" / "spk_ref.wav"
    if default_ref.exists():
        return str(default_ref)
    return None


def _load_step_steering_vectors(sample_dir: Path, layer: int) -> list[torch.Tensor | None]:
    steering_path = sample_dir / "steering_vector.json"
    if not steering_path.exists():
        raise FileNotFoundError(f"Missing steering file: {steering_path}")

    with steering_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict in {steering_path}, got {type(payload)}")

    layer_key = f"layer_{int(layer)}"
    if layer_key not in payload:
        available = sorted(k for k in payload.keys() if isinstance(k, str) and k.startswith("layer_"))
        raise KeyError(
            f"Missing key '{layer_key}' in {steering_path}. Available layer keys: {available}"
        )

    layer_payload = payload[layer_key]
    if not isinstance(layer_payload, dict):
        raise ValueError(f"Expected dict at {layer_key} in {steering_path}")

    parsed: dict[int, torch.Tensor | None] = {}
    max_idx = -1
    for key, value in layer_payload.items():
        idx = int(key)
        if idx < 0:
            continue
        max_idx = max(max_idx, idx)
        if value is None:
            parsed[idx] = None
        else:
            vec = torch.as_tensor(value, dtype=torch.float32).reshape(-1)
            parsed[idx] = vec

    if max_idx < 0:
        return []

    out: list[torch.Tensor | None] = [None] * (max_idx + 1)
    for idx, vec in parsed.items():
        out[idx] = vec
    non_null = sum(1 for v in out if v is not None)
    first_dim = next((int(v.numel()) for v in out if v is not None), -1)
    first_non_none_idx = next((i for i, v in enumerate(out) if v is not None), -1)
    max_l2 = 0.0
    for v in out:
        if v is None:
            continue
        l2 = float(v.float().norm().item())
        if l2 > max_l2:
            max_l2 = l2

    frame_rate = 12.5
    hidden_path = sample_dir / "output_hidden.pt"
    if hidden_path.exists():
        try:
            hidden_payload = torch.load(str(hidden_path), map_location="cpu", weights_only=False)
            if isinstance(hidden_payload, dict) and "frame_rate" in hidden_payload:
                frame_rate = float(hidden_payload["frame_rate"])
        except Exception:  # noqa: BLE001
            pass

    first_time_sec = (first_non_none_idx / frame_rate) if first_non_none_idx >= 0 else -1.0
    logger.info(
        (
            "Loaded steering vectors for %s layer=%d: total_steps=%d, active_steps=%d, "
            "vector_dim=%d, max_l2=%.6f, first_active_idx=%d, first_active_time=%.6fs"
        ),
        sample_dir,
        int(layer),
        len(out),
        non_null,
        first_dim,
        max_l2,
        first_non_none_idx,
        first_time_sec,
    )
    return out


def inference_batch(
    root_dir: str | Path,
    steer: int | None = None,
    *,
    model_path: str = "KRAFTON/Raon-SpeechChat-9B",
    device: str = "cuda",
    dtype: str = "bfloat16",
    attn_implementation: str = "sdpa",
    system_prompt: str | None = None,
    temperature: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    eos_penalty: float | None = None,
    sil_penalty: float | None = None,
    bc_penalty: float | None = None,
    channel_mode: str = "mono",
    speak_first: bool | None = None,
    speaker_audio: str | None = None,
    save_hidden: bool = False,
) -> dict[str, int]:
    """Run duplex inference for every ``root/*/input.wav``.

    Each sample directory gets:
    - ``assistant.wav`` and ``user_assistant.wav`` (from ``pipe.duplex``)
    - ``output.wav`` (copy of ``assistant.wav`` for personaplex-style parity)

    Args:
        root_dir: Dataset root containing sample folders.
        steer: Optional steering target layer index. If set, apply per-step vectors from steering_vector.json.
        model_path: RAON duplex model path or HF repo id.
        device: Inference device.
        dtype: Torch dtype string.
        attn_implementation: Attention backend (sdpa/eager/fa).
        speaker_audio: Optional speaker reference audio.
        save_hidden: Save per-step hidden payload to output_hidden.pt under each sample directory.
    """
    if steer is not None:
        logger.info("Steering enabled at layer=%s", steer)

    root = Path(root_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Invalid root_dir: {root}")

    if root.is_file() and root.name == "input.wav":
        input_paths = [root]
    elif root.is_dir():
        sample_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
        input_paths = [sample_dir / "input.wav" for sample_dir in sample_dirs if (sample_dir / "input.wav").exists()]
    else:
        raise FileNotFoundError(f"Expected a dataset directory or an input.wav file, got: {root}")

    if not input_paths:
        raise FileNotFoundError(f"No input files found under {root} with pattern */input.wav")

    pipe = _create_pipeline(
        model_path=model_path,
        device=device,
        dtype=dtype,
        attn_implementation=attn_implementation,
    )
    resolved_prompt = _resolve_system_prompt(system_prompt) if system_prompt else None
    resolved_speaker_audio = _resolve_speaker_audio_path(speaker_audio)

    ok = 0
    failed = 0
    for input_path in tqdm(input_paths):
        sample_dir = input_path.parent
        try:
            # channel = _resolve_audio_channel(input_path, channel_mode=channel_mode)
            audio_input = pipe.load_audio(str(input_path)) # channel=channel)
            duplex_kwargs = {
                "audio_input": audio_input,
                "output_dir": str(sample_dir),
                "speaker_audio": resolved_speaker_audio,
                "save_hidden": save_hidden,
            }
            if steer is not None:
                duplex_kwargs["steering_layer"] = int(steer)
                duplex_kwargs["steering_vectors"] = _load_step_steering_vectors(sample_dir, int(steer))
            if speak_first is not None:
                duplex_kwargs["speak_first"] = speak_first
            if resolved_prompt is not None:
                duplex_kwargs["system_prompt"] = resolved_prompt
            if temperature is not None:
                duplex_kwargs["temperature"] = temperature
            if top_k is not None:
                duplex_kwargs["top_k"] = top_k
            if top_p is not None:
                duplex_kwargs["top_p"] = top_p
            if eos_penalty is not None:
                duplex_kwargs["eos_penalty"] = eos_penalty
            if sil_penalty is not None:
                duplex_kwargs["sil_penalty"] = sil_penalty
            if bc_penalty is not None:
                duplex_kwargs["bc_penalty"] = bc_penalty

            pipe.duplex(
                **duplex_kwargs,
            )

            assistant_wav = sample_dir / "assistant.wav"
            output_wav = sample_dir / "output.wav"
            if not assistant_wav.exists():
                raise FileNotFoundError(f"Expected assistant output not found: {assistant_wav}")
            shutil.copyfile(assistant_wav, output_wav)
            ok += 1
            logger.info("[%d/%d] done: %s", ok + failed, len(input_paths), sample_dir)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            logger.exception("Failed on %s: %s", sample_dir, exc)

    summary = {"total": len(input_paths), "ok": ok, "failed": failed}
    logger.info("Batch finished: %s", summary)
    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run RAON duplex inference for root/*/input.wav dataset.")
    ap.add_argument("root_dir", type=str, help="Dataset root directory.")
    ap.add_argument("--steer", type=int, default=None, metavar="LAYER", help="Steering layer index (0-based thinker layer).")
    ap.add_argument("--model-path", type=str, default="KRAFTON/Raon-SpeechChat-9B", help="Model path or HF repo id.")
    ap.add_argument("--device", type=str, default="cuda", help="Inference device.")
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"], help="Torch dtype.")
    ap.add_argument("--attn-implementation", type=str, default="sdpa", choices=["sdpa", "eager", "fa"], help="Attention backend.")
    ap.add_argument("--system-prompt", type=str, default=None, help="System prompt text or prompt key. Default uses duplex config.")
    ap.add_argument("--temperature", type=float, default=None, help="Sampling temperature. Default uses duplex config.")
    ap.add_argument("--top-k", type=int, default=None, help="Top-k filtering. Default uses duplex config.")
    ap.add_argument("--top-p", type=float, default=None, help="Top-p sampling. Default uses duplex config.")
    ap.add_argument("--eos-penalty", type=float, default=None, help="EOS penalty. Default uses duplex config.")
    ap.add_argument("--sil-penalty", type=float, default=None, help="SIL penalty. Default uses duplex config.")
    ap.add_argument("--bc-penalty", type=float, default=None, help="Backchannel penalty. Default uses duplex config.")
    ap.add_argument(
        "--channel-mode",
        type=str,
        default="mono",
        choices=["auto-user", "mono", "left", "right"],
        help="Input channel selection for stereo wavs. Notebook parity uses mono.",
    )
    speak_group = ap.add_mutually_exclusive_group()
    speak_group.add_argument("--speak-first", action="store_true", help="Force speak-first mode.")
    speak_group.add_argument("--listen-first", action="store_true", help="Force listen-first mode.")
    ap.add_argument("--speaker-audio", type=str, default=None, help="Optional speaker reference wav path.")
    ap.add_argument("--save-hidden", action="store_true", help="Save output_hidden.pt for each sample directory.")
    return ap


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _build_arg_parser().parse_args()
    speak_first = None
    if args.speak_first:
        speak_first = True
    elif args.listen_first:
        speak_first = False

    inference_batch(
        root_dir=args.root_dir,
        steer=args.steer,
        model_path=args.model_path,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        system_prompt=args.system_prompt,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        eos_penalty=args.eos_penalty,
        sil_penalty=args.sil_penalty,
        bc_penalty=args.bc_penalty,
        channel_mode=args.channel_mode,
        speak_first=speak_first,
        speaker_audio=args.speaker_audio,
        save_hidden=args.save_hidden,
    )



if __name__ == "__main__":
    main()

