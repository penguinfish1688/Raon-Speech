"""Generate RAON mode-class labels from per-step log-likelihood thresholds.

This mirrors Personaplex mode labeling semantics, but uses RAON step-level payloads:
- hidden states are saved per step in output_hidden.pt
- each step may carry multiple token IDs in input_token_ids/output_token_ids

Output schema per sample directory (root/*/input.json):
{
  "input": <text>,
  "modes": {
    "listening": [[start_step, end_step], ...],
    "speaking": [[start_step, end_step], ...]
  }
}
"""

from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import torch


def _load_hidden_payload(path: Path) -> dict[str, Any]:
    data = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict payload at {path}, got {type(data).__name__}")
    return data


def _extract_hidden_layers(payload: dict[str, Any]) -> torch.Tensor:
    if "hidden_states" not in payload:
        raise KeyError("Payload missing 'hidden_states'.")
    hidden = payload["hidden_states"]
    if not isinstance(hidden, torch.Tensor):
        hidden = torch.as_tensor(hidden)
    if hidden.ndim == 2:
        hidden = hidden.unsqueeze(1)
    if hidden.ndim != 3:
        raise ValueError(f"Expected hidden_states [T,L,D] or [T,D], got {tuple(hidden.shape)}")
    return hidden.float()


def _extract_ids(payload: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    if "input_token_ids" not in payload or "output_token_ids" not in payload:
        raise KeyError("Payload requires input_token_ids and output_token_ids for LL labeling.")
    input_ids = payload["input_token_ids"]
    output_ids = payload["output_token_ids"]
    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.as_tensor(input_ids)
    if not isinstance(output_ids, torch.Tensor):
        output_ids = torch.as_tensor(output_ids)
    if input_ids.ndim != 2 or output_ids.ndim != 2:
        raise ValueError(
            f"Expected input/output token ids [T,K], got {tuple(input_ids.shape)} and {tuple(output_ids.shape)}"
        )
    return input_ids.long(), output_ids.long()


def _extract_talker_hidden(payload: dict[str, Any]) -> torch.Tensor | None:
    if "talker_hidden_states" not in payload:
        return None
    h = payload["talker_hidden_states"]
    if not isinstance(h, torch.Tensor):
        h = torch.as_tensor(h)
    if h.ndim != 2:
        raise ValueError(f"Expected talker_hidden_states [T,D], got {tuple(h.shape)}")
    return h.float()


def _ranges_from_mask(mask: np.ndarray) -> list[list[int]]:
    if mask.ndim != 1:
        raise ValueError(f"Expected 1D mask, got shape={mask.shape}")
    ranges: list[list[int]] = []
    start: int | None = None
    for idx, flag in enumerate(mask.tolist()):
        if flag and start is None:
            start = idx
        elif (not flag) and start is not None:
            ranges.append([start, idx - 1])
            start = None
    if start is not None:
        ranges.append([start, int(mask.shape[0]) - 1])
    return ranges


def _collect_valid_sample_dirs(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Root directory not found: {root}")
    sample_dirs = [p for p in root.iterdir() if p.is_dir()]
    sample_dirs.sort(key=lambda p: (0, int(p.name)) if p.name.isdigit() else (1, p.name))
    valid: list[Path] = []
    for sd in sample_dirs:
        if (sd / "output_hidden.pt").is_file():
            valid.append(sd)
    if not valid:
        raise FileNotFoundError(f"No valid samples under {root}. Need root/*/output_hidden.pt")
    return valid


def _load_mono_wav(path: Path) -> tuple[np.ndarray, int]:
    if not path.is_file():
        raise FileNotFoundError(f"WAV file not found: {path}")
    try:
        import soundfile as sf

        wav, sr = sf.read(str(path), always_2d=False, dtype="float32")
        arr = np.asarray(wav, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr.mean(axis=1)
        return arr.reshape(-1), int(sr)
    except Exception:
        with wave.open(str(path), "rb") as wf:
            sr = int(wf.getframerate())
            nchan = int(wf.getnchannels())
            sw = int(wf.getsampwidth())
            frames = wf.readframes(wf.getnframes())
        if sw == 2:
            arr = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        elif sw == 4:
            arr = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"Unsupported sample width {sw} bytes for {path}")
        if nchan > 1:
            arr = arr.reshape(-1, nchan).mean(axis=1)
        return arr.astype(np.float32), sr


def _step_abs_amplitude(wav: np.ndarray, sr: int, frame_rate_hz: float, t_total: int) -> np.ndarray:
    if t_total <= 0:
        return np.zeros((0,), dtype=np.float32)
    samples_per_step = max(1, int(round(float(sr) / float(frame_rate_hz))))
    out = np.zeros((t_total,), dtype=np.float32)
    abs_wav = np.abs(wav)
    for t in range(t_total):
        s = t * samples_per_step
        e = min((t + 1) * samples_per_step, abs_wav.shape[0])
        if s >= abs_wav.shape[0] or e <= s:
            out[t] = np.nan
        else:
            out[t] = float(abs_wav[s:e].mean())
    return out


def _ranges_to_mask(ranges: list[list[int]], t_total: int) -> np.ndarray:
    mask = np.zeros((t_total,), dtype=bool)
    for pair in ranges:
        if len(pair) != 2:
            continue
        start, end = int(pair[0]), int(pair[1])
        if end < 0 or start >= t_total:
            continue
        s = max(0, start)
        e = min(t_total - 1, end)
        if e >= s:
            mask[s : e + 1] = True
    return mask


def _resolve_audio_paths(sample_dir: Path, payload: dict[str, Any]) -> tuple[Path, Path]:
    input_wav = sample_dir / "input.wav"
    output_wav = sample_dir / "output.wav"

    payload_in = payload.get("input_wav")
    payload_out = payload.get("output_wav")
    if (not input_wav.is_file()) and isinstance(payload_in, str) and payload_in:
        p = Path(payload_in)
        if p.is_file():
            input_wav = p
    if (not output_wav.is_file()) and isinstance(payload_out, str) and payload_out:
        p = Path(payload_out)
        if p.is_file():
            output_wav = p

    return input_wav, output_wav


def _log_prob_of_targets(logits_2d: torch.Tensor, targets_1d: torch.Tensor) -> torch.Tensor:
    log_probs = torch.log_softmax(logits_2d.float(), dim=-1)
    clamped_targets = targets_1d.clamp(min=0)
    vals = log_probs.gather(1, clamped_targets.unsqueeze(1)).squeeze(1)
    vals = vals.masked_fill(targets_1d < 0, torch.nan)
    return vals


def _text_ll_for_targets_chunked(
    hidden_td: torch.Tensor,
    targets_t: torch.Tensor,
    *,
    lm_head: torch.nn.Module,
    norm_layer: torch.nn.Module | None,
    device: torch.device,
    proj_dtype: torch.dtype,
    chunk_size: int,
) -> torch.Tensor:
    steps = int(hidden_td.shape[0])
    out = torch.full((steps,), torch.nan, dtype=torch.float32)
    if steps == 0:
        return out

    for start in range(0, steps, chunk_size):
        end = min(steps, start + chunk_size)
        h = hidden_td[start:end].to(device=device, dtype=proj_dtype)
        tgt = targets_t[start:end].to(device=device, dtype=torch.long)

        safe_tgt = tgt.clamp(min=0)
        if norm_layer is not None:
            h = norm_layer(h)
        logits = lm_head(h).float()
        ll = logits.gather(1, safe_tgt.unsqueeze(1)).squeeze(1) - torch.logsumexp(logits, dim=-1)
        ll = ll.masked_fill(tgt < 0, torch.nan)
        out[start:end] = ll.detach().cpu()

    return out


def _compute_step_ll_mats(
    hidden_tld: torch.Tensor,
    input_ids_tk: torch.Tensor,
    output_ids_tk: torch.Tensor,
    talker_hidden_td: torch.Tensor | None,
    model: Any,
    *,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return output_ll[L,T] and input_ll[L,T-1]."""
    t_steps, n_layers, _ = hidden_tld.shape
    if input_ids_tk.shape[0] != t_steps or output_ids_tk.shape[0] != t_steps:
        raise ValueError("Length mismatch among hidden/input_ids/output_ids")

    device = model.device
    lm_head = model.lm_head
    norm_layer = getattr(model.text_model, "norm", None)

    text_out_target = output_ids_tk[:, 0].to(device=device)
    text_in_target_next = output_ids_tk[1:, 0].to(device=device)

    text_out_ll = torch.full((n_layers, t_steps), torch.nan, dtype=torch.float32)
    text_in_ll = torch.full((n_layers, max(0, t_steps - 1)), torch.nan, dtype=torch.float32)

    proj_dtype = lm_head.weight.dtype
    for layer_idx in range(n_layers):
        layer_hidden = hidden_tld[:, layer_idx, :]
        text_out_ll[layer_idx] = _text_ll_for_targets_chunked(
            layer_hidden,
            text_out_target,
            lm_head=lm_head,
            norm_layer=norm_layer,
            device=device,
            proj_dtype=proj_dtype,
            chunk_size=chunk_size,
        )
        if t_steps > 1:
            text_in_ll[layer_idx] = _text_ll_for_targets_chunked(
                layer_hidden[:-1],
                text_in_target_next,
                lm_head=lm_head,
                norm_layer=norm_layer,
                device=device,
                proj_dtype=proj_dtype,
                chunk_size=chunk_size,
            )

    audio_out_ll: torch.Tensor | None = None
    audio_in_ll: torch.Tensor | None = None
    if talker_hidden_td is not None and getattr(model, "audio_lm_head", None) is not None:
        ah = talker_hidden_td.to(device=device, dtype=model.audio_lm_head.weight.dtype)
        audio_logits = model.audio_lm_head(ah).float()
        audio_out_target = (
            output_ids_tk[:, 1].to(device=device)
            if output_ids_tk.shape[1] > 1
            else torch.full((t_steps,), -1, device=device, dtype=torch.long)
        )
        audio_out_ll = _log_prob_of_targets(audio_logits, audio_out_target).detach().cpu()

        if t_steps > 1:
            if output_ids_tk.shape[1] > 1:
                audio_in_target_next = output_ids_tk[1:, 1].to(device=device)
            else:
                audio_in_target_next = torch.full((t_steps - 1,), -1, device=device, dtype=torch.long)
            audio_in_ll = _log_prob_of_targets(audio_logits[:-1], audio_in_target_next).detach().cpu()

    out_ll = text_out_ll.clone()
    in_ll = text_in_ll.clone()

    if audio_out_ll is not None:
        audio_out_mat = audio_out_ll.unsqueeze(0).repeat(n_layers, 1)
        text_valid = ~torch.isnan(out_ll)
        audio_valid = ~torch.isnan(audio_out_mat)
        both = text_valid & audio_valid
        out_ll = torch.where(both, 0.5 * (out_ll + audio_out_mat), out_ll)
        out_ll = torch.where(~text_valid & audio_valid, audio_out_mat, out_ll)

    if audio_in_ll is not None:
        audio_in_mat = audio_in_ll.unsqueeze(0).repeat(n_layers, 1)
        text_valid = ~torch.isnan(in_ll)
        audio_valid = ~torch.isnan(audio_in_mat)
        both = text_valid & audio_valid
        in_ll = torch.where(both, 0.5 * (in_ll + audio_in_mat), in_ll)
        in_ll = torch.where(~text_valid & audio_valid, audio_in_mat, in_ll)

    return out_ll.numpy(), in_ll.numpy()


def _extract_input_text(sample_dir: Path) -> str:
    input_json = sample_dir / "input.json"
    if input_json.is_file():
        try:
            with input_json.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and isinstance(data.get("input"), str):
                return data["input"]
        except Exception:
            pass
    output_json = sample_dir / "output.json"
    if output_json.is_file():
        try:
            with output_json.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and isinstance(data.get("text"), str):
                return data["text"]
        except Exception:
            pass
    return ""


def _validate_layer_range(layer_start: int, layer_end: int, num_layers: int) -> list[int]:
    if layer_start < 0 or layer_end < 0:
        raise ValueError("layer_start and layer_end must be >= 0")
    if layer_start > layer_end:
        raise ValueError("layer_start must be <= layer_end")
    if layer_end >= num_layers:
        raise ValueError(
            f"Layer range [{layer_start}, {layer_end}] exceeds available layers [0, {num_layers - 1}]"
        )
    return list(range(layer_start, layer_end + 1))


def _save_ll_heatmap(
    ll_mat_lt: np.ndarray,
    layer_ids: list[int],
    out_png: Path,
    *,
    title: str,
) -> None:
    """Save LL heatmap where y=layer and x=step index."""
    if ll_mat_lt.ndim != 2:
        raise ValueError(f"Expected 2D LL matrix, got shape={ll_mat_lt.shape}")
    if ll_mat_lt.shape[0] != len(layer_ids):
        raise ValueError(
            f"Layer dimension mismatch: mat has {ll_mat_lt.shape[0]}, layer_ids has {len(layer_ids)}"
        )

    vals = ll_mat_lt[np.isfinite(ll_mat_lt)]
    if vals.size == 0:
        vmin, vmax = -1.0, 1.0
    else:
        vmin = float(np.percentile(vals, 5.0))
        vmax = float(np.percentile(vals, 95.0))
        if vmax <= vmin:
            vmax = vmin + 1e-6

    n_layers, n_steps = ll_mat_lt.shape
    fig_w = max(8.0, min(20.0, 0.06 * n_steps))
    fig_h = max(4.0, min(10.0, 0.34 * n_layers + 2.5))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=180)
    im = ax.imshow(
        ll_mat_lt,
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel("Step index")
    ax.set_ylabel("Layer")
    ax.set_title(title)

    ax.set_yticks(np.arange(n_layers, dtype=np.int32))
    ax.set_yticklabels([str(x) for x in layer_ids])

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Log-likelihood")

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def generate_input_jsons(
    root_dir: str,
    model_path: str,
    theta_listen: float,
    theta_speak: float,
    layer_start: int,
    layer_end: int,
    *,
    device: str = "cuda",
    dtype: str = "bfloat16",
    chunk_size: int = 16,
) -> None:
    from transformers import AutoModel

    root = Path(root_dir)
    sample_dirs = _collect_valid_sample_dirs(root)

    first_payload = _load_hidden_payload(sample_dirs[0] / "output_hidden.pt")
    hidden = _extract_hidden_layers(first_payload)
    num_layers = int(hidden.shape[1])
    layer_ids = _validate_layer_range(layer_start, layer_end, num_layers)

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype not in dtype_map:
        raise ValueError(f"Unsupported dtype '{dtype}'. Use one of: {sorted(dtype_map)}")

    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=False,
        dtype=dtype_map[dtype],
    ).to(device).eval()

    print(
        "[raon.mode_class_label] Generating input.json using LL thresholds "
        f"(theta_listen={theta_listen}, theta_speak={theta_speak}, "
        f"layers={layer_start}..{layer_end})"
    )

    ok = 0
    for sample_dir in sample_dirs:
        try:
            payload = _load_hidden_payload(sample_dir / "output_hidden.pt")
            hidden_tld = _extract_hidden_layers(payload)
            input_ids_tk, output_ids_tk = _extract_ids(payload)
            talker_hidden_td = _extract_talker_hidden(payload)
            out_ll, in_ll = _compute_step_ll_mats(
                hidden_tld,
                input_ids_tk,
                output_ids_tk,
                talker_hidden_td,
                model,
                chunk_size=chunk_size,
            )

            # Align listen/speak LL by shared step domain.
            # listen line is shifted and has T-1, speak line has T.
            t_common = min(int(in_ll.shape[1]), int(out_ll.shape[1]))
            if t_common <= 0:
                raise ValueError("No valid steps to classify (t_common <= 0)")
            ll_listen = in_ll[:, :t_common]
            ll_speak = out_ll[:, :t_common]

            layer_idx = np.asarray(layer_ids, dtype=np.int64)
            ll_listen_sel = ll_listen[layer_idx, :]
            ll_speak_sel = ll_speak[layer_idx, :]

            speak_mask = np.all(ll_speak_sel >= theta_speak, axis=0) & np.all(
                ll_listen_sel < theta_listen, axis=0
            )
            listen_mask = np.all(ll_listen_sel >= theta_listen, axis=0) & np.all(
                ll_speak_sel < theta_speak, axis=0
            )

            out_ll_listen_png = sample_dir / "ll_heatmap_listen.png"
            _save_ll_heatmap(
                ll_listen_sel,
                layer_ids,
                out_ll_listen_png,
                title=(
                    f"LL Heatmap (listen line) | layers={layer_ids[0]}..{layer_ids[-1]} | "
                    f"steps={t_common}"
                ),
            )

            out_ll_speak_png = sample_dir / "ll_heatmap_speak.png"
            _save_ll_heatmap(
                ll_speak_sel,
                layer_ids,
                out_ll_speak_png,
                title=(
                    f"LL Heatmap (speak line) | layers={layer_ids[0]}..{layer_ids[-1]} | "
                    f"steps={t_common}"
                ),
            )

            out = {
                "input": _extract_input_text(sample_dir),
                "modes": {
                    "listening": _ranges_from_mask(listen_mask),
                    "speaking": _ranges_from_mask(speak_mask),
                },
            }

            out_path = sample_dir / "input.json"
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(out, f, indent=2, ensure_ascii=False)

            ok += 1
            print(
                f"  [OK] {sample_dir.name}: steps={t_common}, "
                f"listen={int(listen_mask.sum())}, speak={int(speak_mask.sum())}, "
                f"heatmaps={out_ll_listen_png.name},{out_ll_speak_png.name}"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  [SKIP] {sample_dir.name}: {exc}")

    print(f"[raon.mode_class_label] Done. Wrote input.json for {ok}/{len(sample_dirs)} samples.")


def visualize_token_distribution(root_dir: str) -> None:
    """Generate token_dist.png per sample using step-level labels in input.json."""
    root = Path(root_dir)
    sample_dirs = _collect_valid_sample_dirs(root)

    print(f"[raon.mode_class_label] Generating token_dist.png under {root}")
    ok = 0
    for sample_dir in sample_dirs:
        try:
            payload = _load_hidden_payload(sample_dir / "output_hidden.pt")
            hidden_tld = _extract_hidden_layers(payload)
            t_total = int(hidden_tld.shape[0])
            frame_rate_hz = float(payload.get("frame_rate", 12.5))

            input_json = sample_dir / "input.json"
            if not input_json.is_file():
                raise FileNotFoundError(
                    f"Missing input.json in {sample_dir}; run generate-input first"
                )
            with input_json.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict) or "modes" not in data:
                raise ValueError(f"Invalid input.json format in {sample_dir}")
            modes = data["modes"]
            if not isinstance(modes, dict):
                raise ValueError(f"input.json modes must be an object in {sample_dir}")

            listen_mask = _ranges_to_mask(modes.get("listening", []), t_total)
            speak_mask = _ranges_to_mask(modes.get("speaking", []), t_total)

            input_wav_path, output_wav_path = _resolve_audio_paths(sample_dir, payload)

            in_amp = np.full((t_total,), np.nan, dtype=np.float32)
            out_amp = np.full((t_total,), np.nan, dtype=np.float32)

            if input_wav_path.is_file():
                input_wav, input_sr = _load_mono_wav(input_wav_path)
                in_amp = _step_abs_amplitude(input_wav, input_sr, frame_rate_hz, t_total)

            if output_wav_path.is_file():
                output_wav, output_sr = _load_mono_wav(output_wav_path)
                out_amp = _step_abs_amplitude(output_wav, output_sr, frame_rate_hz, t_total)

            xs = np.arange(t_total, dtype=np.int32)
            fig_w = max(11.0, min(26.0, t_total * 0.06))
            fig, ax = plt.subplots(figsize=(fig_w, 4.8), dpi=170)

            for t in range(t_total):
                if listen_mask[t]:
                    ax.axvspan(t - 0.5, t + 0.5, color="#5cb85c", alpha=0.22, linewidth=0)
                elif speak_mask[t]:
                    ax.axvspan(t - 0.5, t + 0.5, color="#d9534f", alpha=0.22, linewidth=0)

            ax.plot(xs, in_amp, color="#2ca02c", linewidth=1.1, alpha=0.95, label="|input.wav|")
            ax.plot(xs, out_amp, color="#1f77b4", linewidth=1.1, alpha=0.95, label="|output.wav|")

            ax.set_xlim(-0.5, t_total - 0.5)
            ax.set_xlabel("Step index")
            ax.set_ylabel("Absolute amplitude")
            ax.set_title(
                "Step classification and aligned audio activity "
                f"(listen={int(listen_mask.sum())}, speak={int(speak_mask.sum())}, T={t_total})"
            )
            ax.grid(True, axis="x", linestyle=":", linewidth=0.6, alpha=0.65)
            ax.legend(loc="upper right", fontsize=8)

            out_png = sample_dir / "token_dist.png"
            fig.tight_layout()
            fig.savefig(out_png, bbox_inches="tight")
            plt.close(fig)

            ok += 1
            print(f"  [OK] {sample_dir.name}: saved {out_png.name}")
        except Exception as exc:  # noqa: BLE001
            print(f"  [SKIP] {sample_dir.name}: {exc}")

    print(f"[raon.mode_class_label] Done. Generated {ok}/{len(sample_dirs)} token_dist.png files.")


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="raon.mode_class_label",
        description="Generate RAON input.json labels from LL thresholds and visualize step/audio distribution.",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    gen = sub.add_parser(
        "generate-input",
        help="Generate input.json with modes.listening and modes.speaking at step level",
    )
    gen.add_argument("--root-dir", type=str, required=True, help="Root dir containing <id>/output_hidden.pt")
    gen.add_argument("--model-path", type=str, default="KRAFTON/Raon-SpeechChat-9B", help="Model path/repo used to load unembedding heads")
    gen.add_argument("--theta-listen", type=float, required=True, help="LL threshold for listening line")
    gen.add_argument("--theta-speak", type=float, required=True, help="LL threshold for speaking line")
    gen.add_argument("--layer-start", type=int, required=True, help="First layer index (inclusive)")
    gen.add_argument("--layer-end", type=int, required=True, help="Last layer index (inclusive)")
    gen.add_argument("--device", type=str, default="cuda", help="Device for LL projection")
    gen.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"], help="Projection dtype")
    gen.add_argument("--chunk-size", type=int, default=16, help="Step chunk size for LL projection")

    vis = sub.add_parser(
        "visualize-token-dist",
        help="Generate token_dist.png for each sample dir from input.json + step-aligned audio",
    )
    vis.add_argument("--root-dir", type=str, required=True)

    args = ap.parse_args()

    if args.cmd == "generate-input":
        generate_input_jsons(
            root_dir=args.root_dir,
            model_path=args.model_path,
            theta_listen=float(args.theta_listen),
            theta_speak=float(args.theta_speak),
            layer_start=int(args.layer_start),
            layer_end=int(args.layer_end),
            device=str(args.device),
            dtype=str(args.dtype),
            chunk_size=int(args.chunk_size),
        )
    elif args.cmd == "visualize-token-dist":
        visualize_token_distribution(root_dir=args.root_dir)
    else:
        raise ValueError(f"Unsupported command: {args.cmd}")


if __name__ == "__main__":
    main()
