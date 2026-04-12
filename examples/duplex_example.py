from __future__ import annotations

from multiprocessing import freeze_support
import re
from pathlib import Path

import torch
from IPython.display import Audio, display
from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

# ── Configuration ──────────────────────────────────────────────────────────
MODEL_ID = "KRAFTON/Raon-SpeechChat-9B"
DEVICE = "cuda"
DTYPE = "bfloat16"
AUDIO_INPUT = "../data/duplex/eval/audio/duplex_00.wav"
AUDIO_INPUT_PERSONA = "../data/duplex/eval/audio/duplex_01.wav"
SPEAKER_REF_AUDIO = "../data/duplex/eval/audio/spk_ref.wav"
OUTPUT_ROOT = Path("output/duplex_notebook")

def load_hub_classes(model_id: str):
    print(f"Resolving Hub module from {model_id}...")
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    revision = getattr(cfg, "_commit_hash", None)
    raon_pipeline = get_class_from_dynamic_module(
        "modeling_raon.RaonPipeline",
        model_id,
        revision=revision,
    )
    build_prompt = get_class_from_dynamic_module(
        "modeling_raon.build_system_prompt",
        model_id,
        revision=revision,
    )
    return raon_pipeline, build_prompt

def run_and_display(pipe, user_audio, label: str, audio_input_path: str | None = None, **kwargs) -> dict:
    """Run duplex inference via pipeline and display results inline."""
    out_dir = OUTPUT_ROOT / label.lower().replace(" ", "_")
    kwargs.setdefault("speaker_audio", SPEAKER_REF_AUDIO)

    print(f"\n{'=' * 60}")
    print(f"  {label}")
    for k, v in kwargs.items():
        print(f"  {k}: {v!r}")
    print(f"{'=' * 60}")

    audio = pipe.load_audio(audio_input_path) if audio_input_path else user_audio
    summary = pipe.duplex(audio_input=audio, output_dir=str(out_dir), **kwargs)

    frame_log = (out_dir / "frame_log.txt").read_text()
    tokens = re.findall(r"text='([^']+)'", frame_log)
    decoded = "".join(t for t in tokens if t != "-")

    print(f"\nDecoded text: {decoded}")
    print(f"Duration: {summary['assistant_duration_sec']:.1f}s")
    display(Audio(str(out_dir / "assistant.wav")))

    return summary


def main() -> None:
    freeze_support()

    raon_pipeline, _build_system_prompt = load_hub_classes(MODEL_ID)

    print("Creating RaonPipeline...")
    pipe = raon_pipeline(MODEL_ID, device=DEVICE, dtype=DTYPE)
    print("Pipeline ready.")

    user_audio = pipe.load_audio(AUDIO_INPUT)
    display(Audio(user_audio[0].float().cpu().numpy(), rate=pipe.processor.sampling_rate))

    _ = run_and_display(pipe, user_audio, "Listen First", audio_input_path=AUDIO_INPUT)


if __name__ == "__main__":
    main()