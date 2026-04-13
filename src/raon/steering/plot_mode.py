"""Plot anchor-aligned layer-wise KL heatmaps from RAON hidden payloads.

For each ``<root-dir>/*`` sample that contains:
- ``output_hidden.pt``
- ``input_timing.json``

we compute two KL traces for every text layer over decoding steps:

1. ``output`` mode:
	KL( softmax(layer[t]) || softmax(last_layer[t]) )
2. ``input`` mode:
	KL( softmax(layer[t]) || softmax(next_input_embedding[t+1]) )

Then, for each anchor key (default: ``question_start`` and ``interrupt_start``),
we align windows around the anchor and average across all valid samples,
producing one heatmap per (anchor, mode). With two anchors, that yields
four heatmaps.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch


DEFAULT_ANCHORS = ["question_start", "interrupt_start"]


def _load_hidden_payload(path: Path) -> dict[str, Any]:
	data = torch.load(str(path), map_location="cpu", weights_only=False)
	if not isinstance(data, dict):
		raise TypeError(f"Expected dict payload at {path}, got {type(data).__name__}")
	return data


def _extract_text_hidden_layers(payload: dict[str, Any]) -> torch.Tensor:
	"""Return hidden tensor as [T, L, D]."""
	if "text_hidden_layers" in payload:
		hidden = payload["text_hidden_layers"]
		if not isinstance(hidden, torch.Tensor):
			hidden = torch.as_tensor(hidden)
		if hidden.ndim != 3:
			raise ValueError(f"Expected text_hidden_layers [T,L,D], got {tuple(hidden.shape)}")
		return hidden.float()

	if "hidden_states" in payload:
		hidden = payload["hidden_states"]
		if not isinstance(hidden, torch.Tensor):
			hidden = torch.as_tensor(hidden)
		if hidden.ndim == 3:
			return hidden.float()
		if hidden.ndim == 2:
			return hidden.float().unsqueeze(1)
		raise ValueError(f"Expected hidden_states [T,D] or [T,L,D], got {tuple(hidden.shape)}")

	raise KeyError("Payload has neither 'text_hidden_layers' nor 'hidden_states'.")


def _extract_input_embeddings(payload: dict[str, Any]) -> torch.Tensor:
	"""Return input embedding tensor as [T, D]."""
	if "full_input_embeddings" in payload:
		emb = payload["full_input_embeddings"]
	elif "input_embeddings" in payload:
		emb = payload["input_embeddings"]
	elif "hidden_states" in payload:
		# Fallback: when only hidden_states are saved, use last-layer hidden as proxy embedding.
		h = payload["hidden_states"]
		if not isinstance(h, torch.Tensor):
			h = torch.as_tensor(h)
		if h.ndim == 3:
			emb = h[:, -1, :]
		elif h.ndim == 2:
			emb = h
		else:
			raise ValueError(f"Expected hidden_states [T,D] or [T,L,D], got {tuple(h.shape)}")
	else:
		raise KeyError(
			"Payload has neither 'full_input_embeddings' nor 'input_embeddings'."
		)

	if not isinstance(emb, torch.Tensor):
		emb = torch.as_tensor(emb)
	if emb.ndim != 2:
		raise ValueError(f"Expected input embeddings [T,D], got {tuple(emb.shape)}")
	return emb.float()


def _compute_layerwise_kls(
	hidden_tld: torch.Tensor,
	input_emb_td: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
	"""Compute output/input KL matrices.

	Returns:
		output_kl: [L, T]
		input_kl: [L, T-1] (or smaller if embedding length differs)
	"""
	if hidden_tld.ndim != 3:
		raise ValueError(f"Expected [T,L,D], got {tuple(hidden_tld.shape)}")
	if input_emb_td.ndim != 2:
		raise ValueError(f"Expected input embeddings [T,D], got {tuple(input_emb_td.shape)}")

	t_steps, _n_layers, _dim = hidden_tld.shape
	if t_steps < 1:
		raise ValueError("Need at least 1 time step")

	# output KL: KL(layer[t] || last[t])
	ref_output = hidden_tld  # [T, L, D]
	tgt_output = hidden_tld[:, -1, :]  # [T, D]
	p_out_log = torch.log_softmax(ref_output, dim=-1)  # [T, L, D]
	p_out = torch.softmax(ref_output, dim=-1)  # [T, L, D]
	q_out_log = torch.log_softmax(tgt_output, dim=-1)  # [T, D]
	output_kl_tl = torch.sum(p_out * (p_out_log - q_out_log[:, None, :]), dim=-1)  # [T, L]

	# input KL: KL(layer[t] || next_input_embedding[t+1])
	n_common = min(int(t_steps), int(input_emb_td.shape[0]))
	if n_common < 2:
		raise ValueError(
			"Need at least 2 aligned steps between hidden and input embeddings "
			f"(got hidden T={t_steps}, embed T={int(input_emb_td.shape[0])})"
		)
	ref_input = hidden_tld[: n_common - 1, :, :]  # [N-1, L, D]
	next_input = input_emb_td[1:n_common, :]  # [N-1, D]
	p_in_log = torch.log_softmax(ref_input, dim=-1)  # [N-1, L, D]
	p_in = torch.softmax(ref_input, dim=-1)  # [N-1, L, D]
	q_in_log = torch.log_softmax(next_input, dim=-1)  # [N-1, D]
	input_kl_tl = torch.sum(p_in * (p_in_log - q_in_log[:, None, :]), dim=-1)  # [N-1, L]

	return output_kl_tl.transpose(0, 1).cpu().numpy(), input_kl_tl.transpose(0, 1).cpu().numpy()


def _extract_centered_2d(mat: np.ndarray, center: int, span: int) -> np.ndarray:
	"""Extract [L, 2*span+1] centered at token center with NaN padding."""
	n_layers, n_tokens = mat.shape
	out = np.full((n_layers, 2 * span + 1), np.nan, dtype=np.float32)
	start = center - span
	end = center + span
	src_l = max(0, start)
	src_r = min(n_tokens - 1, end)
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
		if (sd / "output_hidden.pt").is_file() and (sd / "input_timing.json").is_file():
			valid.append(sd)
	return valid


def _plot_heatmap(
	mat: np.ndarray,
	out_path: Path,
	*,
	anchor: str,
	mode_name: str,
	span: int,
	num_samples: int,
) -> None:
	vals = mat[np.isfinite(mat)]
	if vals.size == 0:
		raise ValueError(f"No finite values for {anchor}/{mode_name}")

	# Personaplex-like robust scaling from middle layers when possible.
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
	ax.set_xlabel(f"Relative token index to {anchor}")
	ax.set_ylabel("Layer")
	ax.set_title(
		f"KL Heatmap | {anchor} | {mode_name} | n={num_samples} | window=+/-{span}"
	)
	cbar = fig.colorbar(img, ax=ax)
	cbar.set_label("KL divergence")
	fig.tight_layout()
	out_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(out_path, bbox_inches="tight")
	plt.close(fig)


def plot_mode_kl_heatmap(
	root_dir: str,
	*,
	span: int = 35,
	anchors: list[str] | None = None,
) -> None:
	root = Path(root_dir)
	if not root.is_dir():
		raise FileNotFoundError(f"Root directory not found: {root}")
	if span < 1:
		raise ValueError(f"span must be >= 1, got {span}")

	anchors = anchors or DEFAULT_ANCHORS
	sample_dirs = _collect_valid_sample_dirs(root)
	if not sample_dirs:
		raise FileNotFoundError(
			f"No valid samples under {root}. Need root/*/output_hidden.pt and input_timing.json"
		)

	# buckets[anchor][mode] -> list of [L, 2*span+1]
	buckets: dict[str, dict[str, list[np.ndarray]]] = {
		a: {"input": [], "output": []} for a in anchors
	}

	for sd in sample_dirs:
		hidden_path = sd / "output_hidden.pt"
		timing_path = sd / "input_timing.json"
		try:
			payload = _load_hidden_payload(hidden_path)
			hidden_tld = _extract_text_hidden_layers(payload)
			input_emb_td = _extract_input_embeddings(payload)
			output_kl, input_kl = _compute_layerwise_kls(hidden_tld, input_emb_td)

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
				buckets[anchor]["output"].append(_extract_centered_2d(output_kl, center, span))
				buckets[anchor]["input"].append(_extract_centered_2d(input_kl, center, span))
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

			# Personaplex-style naming: one png per anchor/mode under root.
			out_png = root / f"mode_kl_heatmap_{anchor}_{mode_name}.png"
			_plot_heatmap(
				avg,
				out_png,
				anchor=anchor,
				mode_name=mode_name,
				span=span,
				num_samples=len(mats),
			)

			out_json = root / f"mode_kl_heatmap_{anchor}_{mode_name}.json"
			payload = {
				"anchor": anchor,
				"mode": mode_name,
				"span": int(span),
				"num_samples": int(len(mats)),
				"shape": [int(avg.shape[0]), int(avg.shape[1])],
				"relative_token_index": np.arange(-span, span + 1, dtype=np.int32).tolist(),
				"avg_kl": avg.tolist(),
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
		description="Plot anchor-aligned layer-wise KL heatmaps from output_hidden.pt dataset.",
	)
	ap.add_argument(
		"--root-dir",
		type=str,
		required=True,
		help="Root dir containing <id>/output_hidden.pt and <id>/input_timing.json",
	)
	ap.add_argument(
		"--span",
		type=int,
		default=35,
		help="Half-window size in token steps around anchor (default: 35).",
	)
	ap.add_argument(
		"--anchors",
		type=str,
		nargs="+",
		default=DEFAULT_ANCHORS,
		help="Anchor keys from input_timing.json (default: question_start interrupt_start)",
	)
	args = ap.parse_args()

	plot_mode_kl_heatmap(
		root_dir=args.root_dir,
		span=int(args.span),
		anchors=[str(a) for a in args.anchors],
	)


if __name__ == "__main__":
	main()
