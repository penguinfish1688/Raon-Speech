from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import torchaudio

from raon.pipeline import RaonPipeline

logger = logging.getLogger(__name__)


def _resolve_audio_channel(audio_path: Path) -> int | None:
    """Use user channel for stereo duplex audio (channel 1), otherwise mono."""
    info = torchaudio.info(str(audio_path))
    return 1 if info.num_channels >= 2 else None


def inference_batch(
    root_dir: str | Path,
    steer: bool = False,
    *,
    model_path: str = "KRAFTON/Raon-SpeechChat-9B",
    device: str = "cuda",
    dtype: str = "bfloat16",
    attn_implementation: str = "sdpa",
    speaker_audio: str | None = None,
) -> dict[str, int]:
    """Run listen-first duplex inference for every ``root/*/input.wav``.

    Each sample directory gets:
    - ``assistant.wav`` and ``user_assistant.wav`` (from ``pipe.duplex``)
    - ``output.wav`` (copy of ``assistant.wav`` for personaplex-style parity)

    Args:
        root_dir: Dataset root containing sample folders.
        steer: Reserved flag for future steering logic.
        model_path: RAON duplex model path or HF repo id.
        device: Inference device.
        dtype: Torch dtype string.
        attn_implementation: Attention backend (sdpa/eager/fa).
        speaker_audio: Optional speaker reference audio.
    """
    if steer:
        logger.warning("steer=True is not implemented yet; running unsteered inference.")

    root = Path(root_dir).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Invalid root_dir: {root}")

    input_paths = sorted(root.glob("*/input.wav"))
    if not input_paths:
        raise FileNotFoundError(f"No input files found under {root} with pattern */input.wav")

    pipe = RaonPipeline(
        model_path=model_path,
        device=device,
        dtype=dtype,
        attn_implementation=attn_implementation,
    )

    ok = 0
    failed = 0
    for input_path in input_paths:
        sample_dir = input_path.parent
        try:
            channel = _resolve_audio_channel(input_path)
            audio_input = pipe.load_audio(str(input_path), channel=channel)
            pipe.duplex(
                audio_input=audio_input,
                output_dir=str(sample_dir),
                speak_first=False,  # listen-first mode
                speaker_audio=speaker_audio,
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
    ap.add_argument("--steer", action="store_true", help="Reserved; not implemented yet.")
    ap.add_argument("--model-path", type=str, default="KRAFTON/Raon-SpeechChat-9B", help="Model path or HF repo id.")
    ap.add_argument("--device", type=str, default="cuda", help="Inference device.")
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"], help="Torch dtype.")
    ap.add_argument("--attn-implementation", type=str, default="sdpa", choices=["sdpa", "eager", "fa"], help="Attention backend.")
    ap.add_argument("--speaker-audio", type=str, default=None, help="Optional speaker reference wav path.")
    return ap


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _build_arg_parser().parse_args()
    inference_batch(
        root_dir=args.root_dir,
        steer=args.steer,
        model_path=args.model_path,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        speaker_audio=args.speaker_audio,
    )


if __name__ == "__main__":
    main()

