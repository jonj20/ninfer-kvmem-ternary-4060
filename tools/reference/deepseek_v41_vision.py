"""FP64 reference of DeepSeek-V4.1's vision tower (the ViT and the aligner into the text model) over
a llama.cpp `clip` GGUF with projector deepseek4v, and of its image preprocessing.

An independent transcription of the release's inference/vision.py and image_processor.py for golden
values: the patch projection with its bias; per block x += attention(RMS(x)) and x += MLP(RMS(x)),
the attention's fused QKV projection with bias, a 2D RoPE on each head of D values (pairs (i, i +
D/2); the first D/4 turn by the patch's row, the next D/4 by its column, at theta^(-4i / D)),
unmasked softmax attention with scale 1/sqrt(D) over the image's patches and the output projection
with bias, and the SiLU-gated MLP; then the final RMS norm. The aligner pads the patch grid with
zero rows and columns to a multiple of the downsample ratio r, takes each r x r cell's values in
F.unfold order (channel, y, x), projects them with bias, applies the erf GELU and projects again
with bias. Every value is FP64 over the GGUF's BF16/F32 values.

    python -m tools.reference.deepseek_v41_vision --mmproj MMPROJ.gguf \\
        --patches patches.npy --grid W H --out rows.npy

`--patches` holds the engine's input, BF16 words [H * W, 3 * patch * patch] (a .npy array, or the
raw words of a .bin file), each patch's values (channel, y, x) with x fastest and the patches row by
row; `--out` receives the aligner rows [ceil(H / r) * ceil(W / r), text hidden] in FP64 (.npy, or
raw for .bin). The tower's shape comes from the mmproj's metadata.

`--image FILE --config config.json` instead derives the patches from an image as the release's
image_processor does (needs Pillow): the grid plan (upscale to the minimum pixel count, round up to
patches, shrink to the token budget), ImageOps.pad with a bicubic resize onto gray 127, centered,
and the values (x / 255 - 0.5) / 0.5 rounded to BF16. `--patches-out` saves those patches.

`--release DIR` also runs the release's own ViT and Aligner (DIR/inference/vision.py, needs torch)
in FP32 with the mmproj's weights on the same patches and prints how far it lies from the FP64
rows: a check of this transcription.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from tools.convert.sources.gguf import GGUFFile


def rms(x: np.ndarray, eps: float, w: np.ndarray) -> np.ndarray:
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * w


def gelu_erf(x: np.ndarray) -> np.ndarray:
    erf = np.vectorize(math.erf, otypes=[np.float64])
    return 0.5 * x * (1.0 + erf(x / math.sqrt(2.0)))


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def bf16_words(words: np.ndarray) -> np.ndarray:
    return (words.astype(np.uint32) << 16).view(np.float32).astype(np.float64)


def to_bf16_words(values: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even BF16 words of FP32 values."""
    bits = values.astype(np.float32).view(np.uint32).astype(np.uint64)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def shape_of(mmproj: GGUFFile) -> dict:
    kv, v = mmproj.kv, "clip.vision."
    return {"layers": int(kv[v + "block_count"]), "hidden": int(kv[v + "embedding_length"]),
            "heads": int(kv[v + "attention.head_count"]),
            "patch": int(kv[v + "patch_size"]), "ratio": int(kv[v + "projector.scale_factor"]),
            "eps": float(kv[v + "attention.layer_norm_epsilon"]), "theta": 10000.0}


def tower(mmproj: GGUFFile, patches: np.ndarray, grid_w: int, grid_h: int) -> np.ndarray:
    """Aligner rows [ceil(grid_h / r) * ceil(grid_w / r), text hidden] of the patches
    [grid_h * grid_w, 3 * patch * patch]."""

    s = shape_of(mmproj)
    h, heads, eps, r = s["hidden"], s["heads"], s["eps"], s["ratio"]
    d = h // heads
    n = grid_w * grid_h
    if patches.shape[0] != n:
        raise ValueError("patch count does not match the grid")
    read = lambda name: mmproj.read_direct(name).astype(np.float64)

    x = patches @ read("v.patch_embd.weight").reshape(h, -1).T + read("v.patch_embd.bias")
    # RoPE angles [n, D/2]: D/4 frequencies by row, then the same D/4 by column.
    freq = 1.0 / s["theta"] ** (np.arange(0, d // 2, 2, dtype=np.float64) / (d // 2))
    rows, columns = np.arange(n) // grid_w, np.arange(n) % grid_w
    angle = np.concatenate([rows[:, None] * freq, columns[:, None] * freq], axis=1)
    cos, sin = np.cos(angle)[:, None, :], np.sin(angle)[:, None, :]

    def rope(t: np.ndarray) -> np.ndarray:  # [n, heads, d]
        a, b = t[..., : d // 2], t[..., d // 2:]
        return np.concatenate([a * cos - b * sin, b * cos + a * sin], axis=-1)

    for i in range(s["layers"]):
        g = f"v.blk.{i}."
        normed = rms(x, eps, read(g + "ln1.weight"))
        qkv = normed @ read(g + "attn_qkv.weight").T + read(g + "attn_qkv.bias")
        q, k, v = (t.reshape(n, heads, d) for t in np.split(qkv, 3, axis=-1))
        q, k = rope(q), rope(k)
        qh, kh, vh = (t.transpose(1, 0, 2) for t in (q, k, v))  # [heads, n, d]
        scores = qh @ kh.transpose(0, 2, 1) / math.sqrt(d)
        scores -= scores.max(axis=-1, keepdims=True)
        p = np.exp(scores)
        p /= p.sum(axis=-1, keepdims=True)
        o = (p @ vh).transpose(1, 0, 2).reshape(n, h)
        x = x + o @ read(g + "attn_out.weight").T + read(g + "attn_out.bias")
        normed = rms(x, eps, read(g + "ln2.weight"))
        gate = normed @ read(g + "ffn_gate.weight").T
        up = normed @ read(g + "ffn_up.weight").T
        x = x + (silu(gate) * up) @ read(g + "ffn_down.weight").T
    x = rms(x, eps, read("v.post_ln.weight"))

    llm_h, llm_w = -(-grid_h // r), -(-grid_w // r)
    padded = np.zeros((llm_h * r, llm_w * r, h))
    padded[:grid_h, :grid_w] = x.reshape(grid_h, grid_w, h)
    # F.unfold's order: channel, then the cell's row, then its column.
    cells = padded.reshape(llm_h, r, llm_w, r, h).transpose(0, 2, 4, 1, 3).reshape(-1, h * r * r)
    up = gelu_erf(cells @ read("mm.1.weight").T + read("mm.1.bias"))
    return up @ read("mm.2.weight").T + read("mm.2.bias")


def plan_grid(width: int, height: int, vc: dict) -> tuple[int, int, int, int]:
    """(n_llm_h, n_llm_w, pixel height, pixel width) of an image, as plan_image_grid computes it."""

    p, r, budget = vc["patch_size"], vc["downsample_ratio"], vc["max_image_tokens"]
    ratio = vc.get("max_wh_ratio")
    if ratio is not None and width > height * ratio:
        width = height * ratio
    if 0 < width * height < vc["min_pixels"]:
        scale = (vc["min_pixels"] / (width * height)) ** 0.5
        width, height = int(width * scale), int(height * scale)
    best_w, best_h = math.ceil(width / p) * p, math.ceil(height / p) * p
    grid = lambda bh, bw: (math.ceil((bh // p) / r), math.ceil((bw // p) / r))
    tokens = lambda gh, gw: gh * (gw + 1) + 2
    llm_h, llm_w = grid(best_h, best_w)
    if tokens(llm_h, llm_w) > budget:
        aspect = height / width
        max_w = math.sqrt((budget - 2) / aspect + 0.25) - 0.5
        max_h = max_w * aspect
        cell = p * r
        if max_w < 1.0:
            best_h, best_w = (budget - 2) // 2 * cell, cell
        elif max_h < 1.0:
            best_h, best_w = cell, (budget - 3) * cell
        else:
            beta = min(math.floor(max_w) * cell / width, math.floor(max_h) * cell / height)
            best_h = math.floor(height * beta / p) * p
            best_w = math.floor(width * beta / p) * p
        llm_h, llm_w = grid(best_h, best_w)
    return llm_h, llm_w, best_h, best_w


def image_patches(path: Path, vc: dict) -> tuple[np.ndarray, int, int]:
    """BF16 words [grid_h * grid_w, 3 * p * p] of an image and its grid (W, H)."""

    from PIL import Image, ImageOps

    p = vc["patch_size"]
    with Image.open(path) as source:
        image = source.convert("RGB")
    _, _, best_h, best_w = plan_grid(image.width, image.height, vc)
    ratio = vc.get("max_wh_ratio")
    if ratio is not None and image.width >= ratio * image.height:
        image = image.resize((best_w, best_h), Image.Resampling.BICUBIC)
    else:
        image = ImageOps.pad(image, (best_w, best_h), method=Image.Resampling.BICUBIC,
                             color=(127, 127, 127))
    x = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / np.float32(255)
    x = (x - np.float32(0.5)) / np.float32(0.5)
    gh, gw = best_h // p, best_w // p
    x = x.reshape(3, gh, p, gw, p).transpose(1, 3, 0, 2, 4).reshape(gh * gw, -1)
    return to_bf16_words(x), gw, gh


def release_tower(release: Path, mmproj: GGUFFile, patches: np.ndarray, grid_w: int,
                  grid_h: int) -> np.ndarray:
    """The release's ViT and Aligner in FP32 with the mmproj's weights."""

    import importlib.util
    from types import SimpleNamespace

    import torch

    spec = importlib.util.spec_from_file_location("ds_vision", release / "inference" / "vision.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    s = shape_of(mmproj)
    w1 = mmproj.read_direct("mm.1.weight")
    args = SimpleNamespace(vision_dim=s["hidden"], vision_n_heads=s["heads"],
                           vision_inter_dim=int(mmproj.kv["clip.vision.feed_forward_length"]),
                           vision_patch_size=s["patch"], vision_rope_theta=s["theta"],
                           vision_n_layers=s["layers"], vision_downsample_ratio=s["ratio"],
                           dim=w1.shape[0])
    vit, aligner = module.ViT(args).float().eval(), module.Aligner(args).float().eval()
    t = lambda name: torch.from_numpy(mmproj.read_direct(name).astype(np.float32))
    state = {"patch_embed.proj.weight": t("v.patch_embd.weight").reshape(s["hidden"], -1),
             "patch_embed.proj.bias": t("v.patch_embd.bias"), "norm.weight": t("v.post_ln.weight")}
    for i in range(s["layers"]):
        g, b = f"v.blk.{i}.", f"blocks.{i}."
        state |= {b + "norm1.weight": t(g + "ln1.weight"), b + "norm2.weight": t(g + "ln2.weight"),
                  b + "attn.wqkv.weight": t(g + "attn_qkv.weight"),
                  b + "attn.wqkv.bias": t(g + "attn_qkv.bias"),
                  b + "attn.wo.weight": t(g + "attn_out.weight"),
                  b + "attn.wo.bias": t(g + "attn_out.bias"),
                  b + "mlp.w1.weight": torch.cat([t(g + "ffn_gate.weight"), t(g + "ffn_up.weight")]),
                  b + "mlp.w2.weight": t(g + "ffn_down.weight")}
    vit.load_state_dict(state, strict=True)
    aligner.load_state_dict({"w1.weight": t("mm.1.weight"), "w1.bias": t("mm.1.bias"),
                             "w2.weight": t("mm.2.weight"), "w2.bias": t("mm.2.bias")}, strict=True)
    with torch.no_grad():
        x = torch.from_numpy(patches.astype(np.float32))
        rows = aligner(vit(x, grid_h, grid_w), grid_h, grid_w)
    return rows.double().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mmproj", required=True, type=Path)
    parser.add_argument("--patches", type=Path, default=None)
    parser.add_argument("--grid", type=int, nargs=2, metavar=("W", "H"))
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None, help="the release's config.json")
    parser.add_argument("--patches-out", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--release", type=Path, default=None, help="the release's directory")
    args = parser.parse_args()
    with GGUFFile(args.mmproj) as mmproj:
        features = 3 * shape_of(mmproj)["patch"] ** 2
        if args.image is not None:
            if args.config is None:
                parser.error("--image needs --config")
            vc = json.loads(args.config.read_text())["vision_config"]
            words, w, h = image_patches(args.image, vc)
            print(f"{args.image}: grid {w} x {h} patches")
            if args.patches_out is not None:
                words.astype("<u2").tofile(args.patches_out) if args.patches_out.suffix == ".bin" \
                    else np.save(args.patches_out, words)
        else:
            if args.patches is None or args.grid is None:
                parser.error("give --patches and --grid, or --image")
            words = (np.fromfile(args.patches, dtype="<u2").reshape(-1, features)
                     if args.patches.suffix == ".bin" else np.load(args.patches))
            w, h = args.grid
        patches = bf16_words(words)
        rows = tower(mmproj, patches, w, h)
        print(f"aligner rows {rows.shape}, mean |v| {np.abs(rows).mean():.6g}")
        if args.out is not None:
            # A .bin file receives the raw little-endian FP64 values (the C++ test reads them).
            rows.astype("<f8").tofile(args.out) if args.out.suffix == ".bin" else np.save(args.out, rows)
        if args.release is not None:
            ref = release_tower(args.release, mmproj, patches, w, h)
            diff = np.abs(ref - rows)
            print(f"release FP32 vs FP64: max |diff| {diff.max():.4g} ({diff.max() / np.abs(rows).max():.3g}"
                  f" of max |v|), cosine "
                  f"{(ref * rows).sum() / np.linalg.norm(ref) / np.linalg.norm(rows):.8f}")


if __name__ == "__main__":
    main()
