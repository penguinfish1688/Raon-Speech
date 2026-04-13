"""Plot anchor-aligned layer-wise log-likelihood heatmaps from RAON hidden payloads.

This follows Personaplex-style premature decoding semantics at step level:
- output line: log p(target at step n | layer-l distribution at step n)
- input line:  log p(target at step n+1 | layer-l distribution at step n)

Targets are computed from per-step saved token ids in ``output_hidden.pt``:
- text target: ``output_token_ids[:, 0]`` (current step) / ``output_token_ids[1:, 0]`` (next step)
- audio target: ``output_token_ids[:, 1]`` (current/next step)

For each (step, layer):
- if both text+audio targets exist, use mean(text_ll, audio_ll)
- if text target is missing, use audio_ll only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModel

DEFAULT_ANCHORS = ["question_start", "interrupt_start"]


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
        raise KeyError("Payload requires input_token_ids and output_token_ids for LL plotting.")
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


def _extract_centered_2d(mat: np.ndarray, center: int, span: int) -> np.ndarray:
    n_layers, n_steps = mat.shape
    out = np.full((n_layers, 2 * span + 1), np.nan, dtype=np.float32)
    start = center - span
    end = center + span
    src_l = max(0, start)
    src_r = min(n_steps - 1, end)
    if src_r < src_l:
        return out
    dst_l = src_l - start
    dst_r = dst_l + (src_r - src_l)
    out[:, dst_l : dst_r + 1] = mat[:, src_l : src_r + 1]
    return out


def _collect_valid_sample_dirs(root: Path) -> list[Path]:
    sample_dirs = [p for p in root.iterdir() if p.is_dir()]
    sample_dirs.sort(key=lambda p: int(p.name) if p.name.isdigit() else p.name)
    valid: list[Path] = []
    for sd in sample_dirs:
        # Explicitly ignore samples missing output_hidden.pt.
        if not (sd / "output_hidden.pt").is_file():
            continue
        if not (sd / "input_timing.json").is_file():
            continue
        valid.append(sd)
    return valid


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
    """Compute per-step log-likelihood for target token ids without materializing [T,V] globally."""
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
        logits = lm_head(h).float()  # [B, V]
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
    t_steps, n_layers, d = hidden_tld.shape
    if input_ids_tk.shape[0] != t_steps or output_ids_tk.shape[0] != t_steps:
        raise ValueError("Length mismatch among hidden/input_ids/output_ids")

    device = model.device
    lm_head = model.lm_head
    norm_layer = getattr(model.text_model, "norm", None)

    text_out_target = output_ids_tk[:, 0].to(device=device)
    # Use next-step predicted output token as the shifted target.
    # This is robust across sequence layouts (e.g., UTA vs non-UTA).
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

    # Audio LL is step-level (not per layer). Broadcast over layers when available.
    audio_out_ll: torch.Tensor | None = None
    audio_in_ll: torch.Tensor | None = None
    if talker_hidden_td is not None and getattr(model, "audio_lm_head", None) is not None:
        ah = talker_hidden_td.to(device=device, dtype=model.audio_lm_head.weight.dtype)
        audio_logits = model.audio_lm_head(ah).float()  # [T, V_audio]
        audio_out_target = output_ids_tk[:, 1].to(device=device) if output_ids_tk.shape[1] > 1 else torch.full((t_steps,), -1, device=device, dtype=torch.long)
        audio_out_ll = _log_prob_of_targets(audio_logits, audio_out_target).detach().cpu()  # [T]

        if t_steps > 1:
            if output_ids_tk.shape[1] > 1:
                audio_in_target_next = output_ids_tk[1:, 1].to(device=device)
            else:
                audio_in_target_next = torch.full((t_steps - 1,), -1, device=device, dtype=torch.long)
            audio_in_ll = _log_prob_of_targets(audio_logits[:-1], audio_in_target_next).detach().cpu()  # [T-1]

    # Combine text+audio like Personaplex: mean if both exist, else audio-only when text missing.
    out_ll = text_out_ll.clone()
    in_ll = text_in_ll.clone()

    if audio_out_ll is not None:
        audio_out_mat = audio_out_ll.unsqueeze(0).repeat(n_layers, 1)  # [L,T]
        text_valid = ~torch.isnan(out_ll)
        audio_valid = ~torch.isnan(audio_out_mat)
        both = text_valid & audio_valid
        out_ll = torch.where(both, 0.5 * (out_ll + audio_out_mat), out_ll)
        out_ll = torch.where(~text_valid & audio_valid, audio_out_mat, out_ll)

    if audio_in_ll is not None:
        audio_in_mat = audio_in_ll.unsqueeze(0).repeat(n_layers, 1)  # [L,T-1]
        text_valid = ~torch.isnan(in_ll)
        audio_valid = ~torch.isnan(audio_in_mat)
        both = text_valid & audio_valid
        in_ll = torch.where(both, 0.5 * (in_ll + audio_in_mat), in_ll)
        in_ll = torch.where(~text_valid & audio_valid, audio_in_mat, in_ll)

    return out_ll.numpy(), in_ll.numpy()


def _plot_heatmap(mat: np.ndarray, out_path: Path, *, anchor: str, mode_name: str, span: int, num_samples: int) -> None:
    vals = mat[np.isfinite(mat)]
    if vals.size == 0:
        raise ValueError(f"No finite values for {anchor}/{mode_name}")

    layer_lo = 10 if mat.shape[0] > 20 else 0
    layer_hi = min(21, mat.shape[0])
    scale_vals = mat[layer_lo:layer_hi, :]
    scale_vals = scale_vals[np.isfinite(scale_vals)]
    if scale_vals.size == 0:
        scale_vals = vals

    vmin = float(np.percentile(scale_vals, 5.0))
    vmax = float(np.percentile(scale_vals, 95.0))
    if vmax <= vmin:
        vmax = vmin + 1e-6

    rel = np.arange(-span, span + 1, dtype=np.int32)
    fig, ax = plt.subplots(figsize=(11.0, 6.2), dpi=180)
    img = ax.imshow(
        mat,
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
        extent=(float(rel[0]), float(rel[-1]), -0.5, float(mat.shape[0]) - 0.5),
    )
    ax.axvline(0.0, color="#333333", linestyle="--", linewidth=1.0)
    ax.set_xlabel(f"Relative step index to {anchor}")
    ax.set_ylabel("Layer")
    ax.set_title(f"LL Heatmap | {anchor} | {mode_name} | n={num_samples} | window=+/-{span}")
    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label("Log-likelihood")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_mode_ll_heatmap(
    root_dir: str,
    *,
    model_path: str,
    span: int = 35,
    anchors: list[str] | None = None,
    device: str = "cuda",
    dtype: str = "bfloat16",
    chunk_size: int = 16,
) -> None:
    root = Path(root_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Root directory not found: {root}")
    if span < 1:
        raise ValueError(f"span must be >= 1, got {span}")

    anchors = anchors or DEFAULT_ANCHORS
    sample_dirs = _collect_valid_sample_dirs(root)
    if not sample_dirs:
        raise FileNotFoundError(f"No valid samples under {root}. Need root/*/output_hidden.pt and input_timing.json")

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

    buckets: dict[str, dict[str, list[np.ndarray]]] = {a: {"input": [], "output": []} for a in anchors}

    for sd in sample_dirs:
        hidden_path = sd / "output_hidden.pt"
        timing_path = sd / "input_timing.json"
        try:
            payload = _load_hidden_payload(hidden_path)
            hidden_tld = _extract_hidden_layers(payload)
            input_ids_tk, output_ids_tk = _extract_ids(payload)
            talker_hidden_td = _extract_talker_hidden(payload)
            output_ll, input_ll = _compute_step_ll_mats(
                hidden_tld,
                input_ids_tk,
                output_ids_tk,
                talker_hidden_td,
                model,
                chunk_size=chunk_size,
            )

            frame_rate = float(payload.get("frame_rate", 12.5))
            with timing_path.open("r", encoding="utf-8") as f:
                timing = json.load(f)
            if not isinstance(timing, dict):
                continue

            for anchor in anchors:
                if anchor not in timing:
                    continue
                center = int(round(float(timing[anchor]) * frame_rate))
                if center < 0:
                    continue
                buckets[anchor]["output"].append(_extract_centered_2d(output_ll, center, span))
                buckets[anchor]["input"].append(_extract_centered_2d(input_ll, center, span))
        except Exception as exc:  # noqa: BLE001
            print(f"[plot-mode][WARN] Skip {sd}: {exc}")

    generated = 0
    for anchor in anchors:
        for mode_name in ("input", "output"):
            mats = buckets[anchor][mode_name]
            if not mats:
                print(f"[plot-mode][WARN] No valid data for anchor={anchor}, mode={mode_name}")
                continue

            avg = np.nanmean(np.stack(mats, axis=0), axis=0).astype(np.float32)
            out_png = root / f"mode_ll_heatmap_{anchor}_{mode_name}.png"
            _plot_heatmap(avg, out_png, anchor=anchor, mode_name=mode_name, span=span, num_samples=len(mats))

            out_json = root / f"mode_ll_heatmap_{anchor}_{mode_name}.json"
            payload = {
                "anchor": anchor,
                "mode": mode_name,
                "span": int(span),
                "num_samples": int(len(mats)),
                "shape": [int(avg.shape[0]), int(avg.shape[1])],
                "relative_step_index": np.arange(-span, span + 1, dtype=np.int32).tolist(),
                "avg_ll": avg.tolist(),
            }
            with out_json.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)

            print(f"[plot-mode] Saved {out_png}")
            print(f"[plot-mode] Saved {out_json}")
            generated += 1

    if generated == 0:
        raise RuntimeError(
            "No heatmaps generated. Ensure samples contain output_hidden.pt and input_timing.json "
            f"with anchors: {anchors}."
        )
    print(f"[plot-mode] Done. Generated {generated} heatmaps.")


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="plot_mode",
        description="Plot anchor-aligned layer-wise log-likelihood heatmaps from output_hidden.pt dataset.",
    )
    ap.add_argument("--root-dir", type=str, required=True, help="Root dir containing <id>/output_hidden.pt and <id>/input_timing.json")
    ap.add_argument("--model-path", type=str, default="KRAFTON/Raon-SpeechChat-9B", help="Model path/repo used to load unembedding heads")
    ap.add_argument("--device", type=str, default="cuda", help="Device for model forward")
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"], help="Model/head dtype for LL projection")
    ap.add_argument("--chunk-size", type=int, default=16, help="Step chunk size for LL projection to reduce VRAM")
    ap.add_argument("--span", type=int, default=35, help="Half-window size in step index around anchor")
    ap.add_argument("--anchors", type=str, nargs="+", default=DEFAULT_ANCHORS, help="Anchor keys from input_timing.json")
    args = ap.parse_args()

    plot_mode_ll_heatmap(
        root_dir=args.root_dir,
        model_path=args.model_path,
        span=int(args.span),
        anchors=[str(a) for a in args.anchors],
        device=args.device,
        dtype=args.dtype,
        chunk_size=int(args.chunk_size),
    )


if __name__ == "__main__":
    main()
