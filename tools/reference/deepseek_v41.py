"""FP64 reference forward of DeepSeek-V4.1-Flash (DeepseekV41ForCausalLM) over its split GGUF.

An independent transcription of the release's inference/model.py (DeepSeek's reference model) for
golden values, over the GGUF's exactly decoded blocks and the Engram GGUF's tables:

- the residual stream is four copies mixed by hyper-connections: every sublayer derives pre, post
  and Sinkhorn-normalised comb coefficients from the RMS-scaled flattened stream; a sublayer's
  input collapses the copies with the pre mix of the sublayer before it (the attention with the
  previous block's FFN mix, the FFN with this block's attention mix, the head with the last FFN's);
- latent attention: a low-rank, RMS-normed query of 64 heads of 512 over one key/value latent
  (keys are values), with an interleaved-pair RoPE on the last 64 values, a learned sink logit per
  head, and the output's RoPE undone before the grouped low-rank output projection. Every block
  attends over a sliding window of 128 positions including its own; a block with a compression
  ratio also attends over compressed latents: groups of `ratio` tokens pooled by a channelwise
  softmax gate (ratio 2) or projected token by token (ratio 1), produced by the block's KV source
  and shared until the next one. An index source scores them with its indexer (32 rectified heads
  of 128 against index keys from the latent, weighted per head) and keeps the best 512 visible to
  each query; the candidate source first keeps each query's best 2048 blocks of 8 latents (its
  newest block always), and the later index sources rank only inside those. Blocks between index
  sources reuse the last selection;
- window-only blocks rotate at theta 10000; the others at theta 160000 with YaRN (factor 16 over
  65,536 positions, beta 32 to 1), a latent at the position of its group's first token;
- Engram (blocks 1 and 14): the 2- to 4-grams ending at each position, over the GGUF's compressed
  token map, hashed per head into the table's rows; their rows project to one key per copy and a
  shared value, gated per copy by the signed square root of the stream's normalised match;
- the MoE: sqrt(softplus) router scores, the top 6 of 384 by score plus bias, their scores
  renormalised and scaled by 1.5, clamped SwiGLU experts (gate at most 10, up within +-10) and
  the shared expert; then the final norm and the head.

Every value is FP64. The release's model rounds what its attention keeps as the training did,
and so does this one unless `--no-roundings` is given: the window latents to FP8 (E4M3 per 32
values with a power-of-two scale), the indexer's queries and keys to FP4 (E2M1 per 32, power-of-two
scale), the compressed latents to FP4 (per 16, E4M3 scale) and the attention output to FP8 before
its projection. The FP8 and FP4 rounding of every projection's input (the release's GEMMs) is
inference arithmetic of other weights than the GGUF's and is not applied. The GGUF's Engram hash
constants are checked against the release's derivation (odd multipliers from a per-layer RNG,
distinct primes from 16,000,000 up).

    python -m tools.reference.deepseek_v41 \\
        --gguf DeepSeek-V4.1-Flash-1017-IQ1_M-00001-of-00004.gguf \\
        --engram DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf --tokens ids.json \\
        [--layers 4] [--threads 8] [--no-roundings] [--out reference.npz]

`ids.json` is the token ids to run (a JSON list). It prints the NLL of every target (token i + 1
given tokens 0..i) and their total; `--layers` runs only the first blocks (no head); `--out` keeps
every block's stream, the routed experts and the logits.
"""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from tools.convert.deepseek_v41 import GGUFShards
from tools.convert.sources.gguf import TYPE_BF16, TYPE_F16, TYPE_F32, GGUFFile

from .gemma4 import rms, target_nll

P = "deepseek41."
DIRECT_TYPES = (TYPE_F32, TYPE_F16, TYPE_BF16)


def dequantize(file: GGUFFile, name: str, row_begin: int = 0,
               row_end: int | None = None) -> np.ndarray:
    """Rows [row_begin, row_end) of a stored matrix (or stack, by flattened rows) as FP64."""

    info = file.info(name)
    if info.type_id in DIRECT_TYPES:
        values = file.read_direct(name).astype(np.float64)
        return values.reshape(-1, info.shape[-1])[row_begin:row_end]
    from gguf import GGMLQuantizationType
    from gguf.quants import dequantize as decode

    blocks = file.read_blocks(name, row_begin, row_end)
    values = decode(np.ascontiguousarray(blocks), GGMLQuantizationType(info.type_id))
    return np.asarray(values, dtype=np.float64).reshape(blocks.shape[0], -1)


class Weights:
    def __init__(self, gguf: GGUFShards):
        self.gguf = gguf

    def has(self, name: str) -> bool:
        return name in self.gguf.tensors

    def matrix(self, name: str, row_begin: int = 0, row_end: int | None = None) -> np.ndarray:
        return dequantize(self.gguf.file(name), name, row_begin, row_end)

    def vector(self, name: str) -> np.ndarray:
        return self.gguf.file(name).read_direct(name).astype(np.float64).reshape(-1)


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def softplus(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, x)


def round_e4m3(x: np.ndarray) -> np.ndarray:
    """To the nearest E4M3 value, ties to even (|x| <= 448)."""
    a = np.abs(x)
    _, e = np.frexp(a)
    step = np.ldexp(1.0, np.maximum(e - 1, -6) - 3)
    return np.copysign(np.rint(a / step) * step, x)


def round_e2m1(x: np.ndarray) -> np.ndarray:
    """To the nearest E2M1 value, ties to even (|x| <= 6)."""
    a = np.abs(x)
    _, e = np.frexp(a)
    step = np.ldexp(1.0, np.maximum(e - 1, 0) - 1)
    return np.copysign(np.minimum(np.rint(a / step) * step, 6.0), x)


def fp8_rounded(x: np.ndarray) -> np.ndarray:
    """The release's act_quant (ue8m0): per 32 values s = 2^ceil(log2(max(amax, 1e-4) / 448))."""
    groups = x.reshape(-1, 32)
    amax = np.maximum(np.abs(groups).max(-1, keepdims=True), 1e-4)
    scale = 2.0 ** np.ceil(np.log2(amax / 448))
    return (round_e4m3(np.clip(groups / scale, -448, 448)) * scale).reshape(x.shape)


def fp4_rounded(x: np.ndarray, block: int, e4m3_scale: bool) -> np.ndarray:
    """The release's fp4_act_quant: per `block` values an E4M3 scale e4m3(max(amax, 6 2^-9) / 6)
    or a power-of-two one 2^ceil(log2(max(amax, 6 2^-126) / 6))."""
    groups = x.reshape(-1, block)
    amax = np.abs(groups).max(-1, keepdims=True)
    if e4m3_scale:
        scale = round_e4m3(np.maximum(amax, 6 * 2.0**-9) / 6)
    else:
        scale = 2.0 ** np.ceil(np.log2(np.maximum(amax, 6 * 2.0**-126) / 6))
    return (round_e2m1(np.clip(groups / scale, -6, 6)) * scale).reshape(x.shape)


def rope_angles(positions: np.ndarray, dim: int, base: float, yarn: dict | None) -> np.ndarray:
    """[n, dim / 2] rotation angles: the release's precompute_freqs_cis, YaRN when given."""

    freqs = 1.0 / base ** (np.arange(0, dim, 2, dtype=np.float64) / dim)
    if yarn is not None:
        original, factor = yarn["original"], yarn["factor"]

        def corrected_dim(rotations: float) -> float:
            return dim * math.log(original / (rotations * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(corrected_dim(yarn["beta_fast"])), 0)
        high = min(math.ceil(corrected_dim(yarn["beta_slow"])), dim - 1)
        ramp = np.clip((np.arange(dim // 2, dtype=np.float64) - low) / max(high - low, 1e-3), 0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    return positions[:, None].astype(np.float64) * freqs[None, :]


def rotate(x: np.ndarray, angles: np.ndarray, inverse: bool = False) -> np.ndarray:
    """RoPE of the last 2 * angles.shape[-1] values of x [n, ..., d], adjacent pairs as complex
    numbers; `angles` is [n, pairs]."""

    pairs = angles.shape[-1]
    shape = (angles.shape[0],) + (1,) * (x.ndim - 2) + (pairs,)
    cos, sin = np.cos(angles).reshape(shape), np.sin(angles).reshape(shape)
    if inverse:
        sin = -sin
    out = x.copy()
    tail = x[..., -2 * pairs:]
    even, odd = tail[..., 0::2], tail[..., 1::2]
    out[..., -2 * pairs::2] = even * cos - odd * sin
    out[..., -2 * pairs + 1::2] = even * sin + odd * cos
    return out


def hc_mixes(x: np.ndarray, fn: np.ndarray, scale: np.ndarray, base: np.ndarray, hc: int,
             iterations: int, hc_eps: float, eps: float):
    """The pre [n, hc], post [n, hc] and comb [n, hc, hc] coefficients of stream x [n, hc, d]."""

    flat = x.reshape(x.shape[0], -1)
    mixes = (flat @ fn.T) / np.sqrt((flat * flat).mean(-1, keepdims=True) + eps)
    pre = sigmoid(mixes[:, :hc] * scale[0] + base[:hc]) + hc_eps
    post = 2 * sigmoid(mixes[:, hc:2 * hc] * scale[1] + base[hc:2 * hc])
    comb = (mixes[:, 2 * hc:] * scale[2] + base[2 * hc:]).reshape(-1, hc, hc)
    comb = np.exp(comb - comb.max(-1, keepdims=True))
    comb = comb / comb.sum(-1, keepdims=True) + hc_eps
    comb = comb / (comb.sum(-2, keepdims=True) + hc_eps)
    for _ in range(iterations - 1):
        comb = comb / (comb.sum(-1, keepdims=True) + hc_eps)
        comb = comb / (comb.sum(-2, keepdims=True) + hc_eps)
    return pre, post, comb


def hc_pre(x: np.ndarray, pre: np.ndarray) -> np.ndarray:
    return (pre[..., None] * x).sum(1)


def hc_post(x: np.ndarray, residual: np.ndarray, post: np.ndarray, comb: np.ndarray) -> np.ndarray:
    """Copy c is post[c] x + sum over copies a of comb[a, c] residual[a]."""

    return post[..., None] * x[:, None, :] + comb.transpose(0, 2, 1) @ residual


def next_prime(start: int, seen: set[int]) -> int:
    candidate = start + 1
    while candidate in seen or any(candidate % f == 0 for f in range(2, math.isqrt(candidate) + 1)):
        candidate += 1
    return candidate


def engram_constants(gguf: GGUFShards) -> dict:
    """The GGUF's hash constants, checked against the release's derivation."""

    kv = gguf.kv
    layers = list(kv[P + "engram.layer_ids"])
    n, heads = int(kv[P + "engram_max_ngram_size"]), int(kv[P + "engram_n_heads"])
    vocab, compressed = int(kv[P + "engram_vocab_size"]), int(kv[P + "engram_compressed_vocab_size"])
    token_map = gguf.integers("engram_token_map.weight", "<i4").astype(np.int64)
    multipliers = gguf.integers("engram_multipliers.weight", "<i8").reshape(len(layers), n)
    primes = gguf.integers("engram_primes.weight", "<i8").reshape(len(layers), n - 1, heads)
    offsets = gguf.integers("engram_offsets.weight", "<i8").reshape(len(layers), (n - 1) * heads)
    bound = max(1, (np.iinfo(np.int64).max // compressed) // 2)
    seen: set[int] = set()
    for index, layer in enumerate(layers):
        drawn = np.random.default_rng(10007 * layer).integers(0, bound, size=(n,), dtype=np.int64)
        if not np.array_equal(drawn * 2 + 1, multipliers[index]):
            raise ValueError(f"Engram layer {layer}: multipliers differ from the release's")
        for gram in range(n - 1):
            current = vocab - 1
            for head in range(heads):
                current = next_prime(current, seen)
                seen.add(current)
                if current != primes[index, gram, head]:
                    raise ValueError(f"Engram layer {layer}: primes differ from the release's")
        flat = primes[index].reshape(-1)
        if not np.array_equal(np.concatenate([[0], np.cumsum(flat[:-1])]), offsets[index]):
            raise ValueError(f"Engram layer {layer}: offsets differ from the primes' sums")
        if int(flat.sum()) != int(kv[P + "engram.num_embeddings"][index]):
            raise ValueError(f"Engram layer {layer}: the primes do not sum to its rows")
    pad = int(token_map[int(kv[P + "engram_pad_token_id"])])
    return {"layers": layers, "n": n, "token_map": token_map, "multipliers": multipliers,
            "primes": primes, "offsets": offsets, "pad": pad}


def engram_rows(constants: dict, index: int, ids: list[int]) -> np.ndarray:
    """[n, (max_ngram - 1) * heads] rows of Engram layer `index` that each position reads."""

    compressed = constants["token_map"][np.asarray(ids)]
    count, n = len(ids), constants["n"]
    tokens = np.empty((count, n), dtype=np.int64)
    for shift in range(n):
        at = np.arange(count) - shift
        tokens[:, shift] = np.where(at >= 0, compressed[np.maximum(at, 0)], constants["pad"])
    products = tokens * constants["multipliers"][index][None, :]
    rolling, hashes = products[:, 0], []
    for i in range(1, n):
        rolling = np.bitwise_xor(rolling, products[:, i])
        hashes.append(rolling[:, None] % constants["primes"][index, i - 1][None, :])
    return np.concatenate(hashes, -1) + constants["offsets"][index][None, :]


def forward(gguf: GGUFShards, table: GGUFFile, ids: list[int], layers: int | None = None,
            threads: int = 1, roundings: bool = True) -> dict[str, np.ndarray]:
    kv = gguf.kv
    fp8 = fp8_rounded if roundings else (lambda x: x)
    fp4 = fp4_rounded if roundings else (lambda x, block, e4m3_scale: x)
    w = Weights(gguf)
    blocks = int(kv[P + "block_count"]) if layers is None else layers
    dim = int(kv[P + "embedding_length"])
    heads = int(kv[P + "attention.head_count"])
    head_dim = int(kv[P + "attention.key_length"])
    rope_dim = int(kv[P + "rope.dimension_count"])
    groups = int(kv[P + "attention.output_group_count"])
    window = int(kv[P + "attention.sliding_window"])
    eps = float(kv[P + "attention.layer_norm_rms_epsilon"])
    ratios = list(kv[P + "attention.compress_ratios"])
    kv_sources = set(kv[P + "kv_source_layer_ids"])
    index_sources = set(kv[P + "index_source_layer_ids"])
    candidate_source = int(kv[P + "attention.candidate_source_layer_id"])
    candidate_blocks = int(kv[P + "attention.candidate_topk_blocks"])
    candidate_width = int(kv[P + "attention.candidate_block_size"])
    index_heads = int(kv[P + "attention.indexer.head_count"])
    index_dim = int(kv[P + "attention.indexer.key_length"])
    index_topk = int(kv[P + "attention.indexer.top_k"])
    hc = int(kv[P + "hyper_connection.count"])
    iterations = int(kv[P + "hyper_connection.sinkhorn_iterations"])
    hc_eps = float(kv[P + "hyper_connection.epsilon"])
    top_k = int(kv[P + "expert_used_count"])
    width = int(kv[P + "expert_feed_forward_length"])
    route_scale = float(kv[P + "expert_weights_scale"])
    clamp_routed = list(kv[P + "swiglu_clamp_exp"])
    clamp_shared = list(kv[P + "swiglu_clamp_shexp"])
    yarn = {"original": float(kv[P + "rope.scaling.original_context_length"]),
            "factor": float(kv[P + "rope.scaling.factor"]),
            "beta_fast": float(kv[P + "rope.scaling.yarn_beta_fast"]),
            "beta_slow": float(kv[P + "rope.scaling.yarn_beta_slow"])}
    window_base = float(kv[P + "rope.freq_base"])
    compress_base = float(kv[P + "attention.compress_rope_freq_base"])
    engram = engram_constants(gguf)

    n = len(ids)
    pos = np.arange(n)
    scale = head_dim ** -0.5
    window_mask = (pos[None, :] <= pos[:, None]) & (pos[:, None] - pos[None, :] < window)

    x = np.stack([w.matrix("token_embd.weight", t, t + 1)[0] for t in ids])
    h = np.repeat(x[:, None, :], hc, axis=1)
    pre_mix = np.zeros((n, hc))
    pre_mix[:, 0] = 1.0
    out: dict[str, np.ndarray] = {"embedding": x.astype(np.float32)}
    shared: dict[str, np.ndarray | int | None] = {"latent": None, "index_k": None,
                                                  "selected": None, "candidates": None}

    def ffn(y: np.ndarray, gate: np.ndarray, up: np.ndarray, down: np.ndarray, limit: float,
            weight: np.ndarray | float = 1.0) -> np.ndarray:
        g = np.minimum(y @ gate.T, limit)
        u = np.clip(y @ up.T, -limit, limit)
        return (weight * (silu(g) * u)) @ down.T

    for il in range(blocks):
        p = f"blk.{il}."
        ratio = ratios[il]
        if il in engram["layers"]:
            index = engram["layers"].index(il)
            rows = engram_rows(engram, index, ids)
            name = f"blk.{il}.engram_embd.weight"
            embeds = np.stack([np.concatenate([dequantize(table, name, int(r), int(r) + 1)[0]
                                               for r in row]) for row in rows])
            projected = embeds @ w.matrix(p + "engram_wkv.weight").T
            key = projected[:, :hc * dim].reshape(n, hc, dim)
            value = projected[:, hc * dim:]
            weight = w.matrix(p + "engram_q.weight") * w.matrix(p + "engram_k.weight")
            rstd = (1 / np.sqrt((h * h).mean(-1) + eps)) * (1 / np.sqrt((key * key).mean(-1) + eps))
            dot = (h * weight[None] * key).sum(-1) * rstd * dim ** -0.5
            gate = sigmoid(np.copysign(np.sqrt(np.maximum(np.abs(dot), 1e-6)), dot))
            h = h + gate[..., None] * value[:, None, :]
            out[f"engram-{il}"] = h.astype(np.float32)

        # Attention, between this block's attention mixes and the previous block's FFN pre mix.
        residual = h
        attn_pre, attn_post, attn_comb = hc_mixes(
            h, w.matrix(p + "hc_attn_fn.weight"), w.vector(p + "hc_attn_scale.weight"),
            w.vector(p + "hc_attn_base.weight"), hc, iterations, hc_eps, eps)
        a = rms(hc_pre(h, pre_mix), eps, w.vector(p + "attn_norm.weight"))
        angles = (rope_angles(pos, rope_dim, compress_base, yarn) if ratio
                  else rope_angles(pos, rope_dim, window_base, None))
        qr = rms(a @ w.matrix(p + "attn_q_a.weight").T, eps, w.vector(p + "attn_q_a_norm.weight"))
        q = rotate((qr @ w.matrix(p + "attn_q_b.weight").T).reshape(n, heads, head_dim), angles)
        latent_kv = fp8(rotate(rms(a @ w.matrix(p + "attn_kv.weight").T, eps,
                                   w.vector(p + "attn_kv_a_norm.weight")), angles))
        keys, masks = [latent_kv], [window_mask]
        if ratio:
            if il in kv_sources:
                ckv = a @ w.matrix(p + "attn_compressor_kv.weight").T
                if ratio > 1:
                    complete = n - n % ratio
                    score = (a @ w.matrix(p + "attn_compressor_gate.weight").T)[:complete]
                    score = score.reshape(-1, ratio, head_dim)
                    gate = np.exp(score - score.max(1, keepdims=True))
                    gate /= gate.sum(1, keepdims=True)
                    ckv = (ckv[:complete].reshape(-1, ratio, head_dim) * gate).sum(1)
                latent = rms(ckv, eps, w.vector(p + "attn_compressor_norm.weight"))
                group_pos = np.arange(latent.shape[0]) * ratio
                group_angles = rope_angles(group_pos, rope_dim, compress_base, yarn)
                index_k = rms(latent @ w.matrix(p + "indexer.attn_k.weight").T, eps,
                              w.vector(p + "indexer.k_norm.weight"))
                shared["index_k"] = fp4(rotate(index_k, group_angles), 32, False)
                shared["latent"] = fp4(rotate(latent, group_angles), 16, True)
                shared["ratio"] = ratio
            latent = shared["latent"]
            groups_n = latent.shape[0]
            visible = np.arange(groups_n)[None, :] < ((pos + 1) // ratio)[:, None]
            if il in index_sources:
                iq = fp4(rotate((qr @ w.matrix(p + "indexer.attn_q_b.weight").T)
                                .reshape(n, index_heads, index_dim), angles), 32, False)
                weights = (a @ w.matrix(p + "indexer.proj.weight").T) * \
                    (index_dim ** -0.5 * index_heads ** -0.5)
                index_score = iq @ shared["index_k"].T
                index_score = (np.maximum(index_score, 0) * weights[..., None]).sum(1)
                index_score = np.where(visible, index_score, -np.inf)
                if il == candidate_source:
                    shared["candidates"] = candidate_mask(index_score, (pos + 1) // ratio,
                                                          candidate_blocks, candidate_width)
                elif shared["candidates"] is not None and il > candidate_source:
                    index_score = np.where(shared["candidates"], index_score, -np.inf)
                selected = np.zeros_like(visible)
                keep = min(index_topk, groups_n)
                for s in range(n):
                    order = np.argsort(-index_score[s], kind="stable")[:keep]
                    selected[s, order] = True
                shared["selected"] = selected & visible
                out[f"selected-{il}"] = shared["selected"]
            keys.append(latent)
            masks.append(shared["selected"])
        # Keys are values; the sink joins each head's softmax denominator.
        sinks = w.vector(p + "attn_sinks.weight")
        kv_rows = np.concatenate(keys, 0)
        mask = np.concatenate(masks, 1)
        scores = (q @ kv_rows.T) * scale
        scores = np.where(mask[:, None, :], scores, -np.inf)
        top = np.maximum(scores.max(-1, keepdims=True), sinks[None, :, None])
        probabilities = np.exp(scores - top)
        denominator = probabilities.sum(-1, keepdims=True) + np.exp(sinks[None, :, None] - top)
        o = (probabilities / denominator) @ kv_rows
        o = fp8(rotate(o, angles, inverse=True)).reshape(n, groups, -1)
        wo_a = w.matrix(p + "attn_output_a.weight").reshape(groups, -1, o.shape[-1])
        o = (o.transpose(1, 0, 2) @ wo_a.transpose(0, 2, 1)).transpose(1, 0, 2).reshape(n, -1)
        attn = o @ w.matrix(p + "attn_output_b.weight").T
        h = hc_post(attn, residual, attn_post, attn_comb)
        out[f"attn_out-{il}"] = h.astype(np.float32)

        # The MoE, collapsed with this block's attention pre mix.
        residual = h
        ffn_pre, ffn_post, ffn_comb = hc_mixes(
            h, w.matrix(p + "hc_ffn_fn.weight"), w.vector(p + "hc_ffn_scale.weight"),
            w.vector(p + "hc_ffn_base.weight"), hc, iterations, hc_eps, eps)
        y = rms(hc_pre(h, attn_pre), eps, w.vector(p + "ffn_norm.weight"))
        scores = np.sqrt(softplus(y @ w.matrix(p + "ffn_gate_inp.weight").T))
        biased = scores + w.vector(p + "exp_probs_b.bias")[None, :]
        chosen = np.argsort(-biased, axis=-1, kind="stable")[:, :top_k]
        weights = np.take_along_axis(scores, chosen, -1)
        weights = weights / (weights.sum(-1, keepdims=True) + 1e-20) * route_scale

        def expert(e: int) -> tuple[np.ndarray, np.ndarray]:
            """The rows that chose expert e and their weighted outputs."""
            gate = w.matrix(p + "ffn_gate_exps.weight", e * width, (e + 1) * width)
            up = w.matrix(p + "ffn_up_exps.weight", e * width, (e + 1) * width)
            down = w.matrix(p + "ffn_down_exps.weight", e * dim, (e + 1) * dim)
            rows, slots = np.nonzero(chosen == e)
            return rows, ffn(y[rows], gate, up, down, clamp_routed[il],
                             weights[rows, slots][:, None])

        moe = ffn(y, w.matrix(p + "ffn_gate_shexp.weight"), w.matrix(p + "ffn_up_shexp.weight"),
                  w.matrix(p + "ffn_down_shexp.weight"), clamp_shared[il])
        with ThreadPoolExecutor(max_workers=max(1, threads)) as pool:
            for rows, values in pool.map(expert, sorted(set(chosen.reshape(-1).tolist()))):
                moe[rows] += values
        out[f"moe_ids-{il}"] = chosen
        out[f"moe_weights-{il}"] = weights
        h = hc_post(moe, residual, ffn_post, ffn_comb)
        pre_mix = ffn_pre
        out[f"l_out-{il}"] = h.astype(np.float32)

    if blocks == int(kv[P + "block_count"]):
        final = rms(hc_pre(h, pre_mix), eps, w.vector("output_norm.weight"))
        out["final"] = final
        vocab = gguf.info("output.weight").shape[0]
        logits = np.zeros((n, vocab))
        for first in range(0, vocab, 8192):
            last = min(vocab, first + 8192)
            logits[:, first:last] = final @ w.matrix("output.weight", first, last).T
        out["logits"] = logits
    return out


def candidate_mask(scores: np.ndarray, lengths: np.ndarray, count: int, width: int) -> np.ndarray:
    """Each query's best `count` blocks of `width` latents by their best score, its newest
    (partly filled) block always among them; unreachable blocks never."""

    positions = scores.shape[-1]
    padded = np.pad(scores, ((0, 0), (0, -positions % width)), constant_values=-np.inf)
    block_scores = padded.reshape(scores.shape[0], -1, width).max(-1)
    newest = (lengths - 1) // width
    block_scores[np.arange(scores.shape[0]), np.maximum(newest, 0)] = np.where(
        newest >= 0, np.inf, block_scores[np.arange(scores.shape[0]), 0])
    keep = np.zeros_like(block_scores, dtype=bool)
    for s in range(scores.shape[0]):
        order = np.argsort(-block_scores[s], kind="stable")[:count]
        keep[s, order] = block_scores[s, order] > -np.inf
    return np.repeat(keep, width, axis=-1)[:, :positions]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--gguf", required=True, type=Path, help="the first shard of the model")
    parser.add_argument("--engram", required=True, type=Path, help="the Engram tables' GGUF")
    parser.add_argument("--tokens", required=True, type=Path, help="JSON list of token ids")
    parser.add_argument("--layers", type=int, default=None, help="run only the first N blocks")
    parser.add_argument("--threads", type=int, default=1, help="experts decoded at once")
    parser.add_argument("--no-roundings", action="store_true",
                        help="keep the attention's caches and output in FP64")
    parser.add_argument("--out", type=Path, default=None, help="write the tensors to this .npz")
    args = parser.parse_args()
    ids = json.loads(args.tokens.read_text())
    with GGUFShards(args.gguf) as gguf:
        result = forward(gguf, GGUFFile(args.engram), ids, args.layers, args.threads,
                         not args.no_roundings)
    if "logits" in result:
        nll = target_nll(result["logits"], ids)
        for i, value in enumerate(nll):
            print(f"target {i + 1} (token {ids[i + 1]}): nll {value:.6f}")
        print(f"total nll {nll.sum():.6f} over {len(nll)} targets")
    if args.out is not None:
        np.savez(args.out, **result)


if __name__ == "__main__":
    main()
