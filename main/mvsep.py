from __future__ import annotations

import argparse
import contextlib
import gc
import math
import os
import random
import re
import signal
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

STEMS = ("vocals", "other")
AUDIO_EXTENSIONS = (".wav", ".flac")
VALIDATION_METRIC = "mean_full_track_sdr_v1"
CHECKPOINT_FORMAT_VERSION = 1
RUNTIME_CONFIG_FIELDS = frozenset({"use_checkpoint", "dropout", "drop_path"})


@dataclass
class ModelConfig:
    sample_rate: int = 44_100
    n_fft: int = 4096
    hop_length: int = 512
    win_length: int = 4096
    audio_channels: int = 2
    num_stems: int = len(STEMS)
    num_bands: int = 124
    dim: int = 256
    depth: int = 12
    num_heads: int = 8
    ffn_mult: float = 3.0
    filter_taps: int = 3
    dropout: float = 0.0
    drop_path: float = 0.08
    use_checkpoint: bool = True
    architecture: str = "bs124_gated_roformer"

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")
        if self.n_fft <= 0 or self.n_fft % 2 != 0:
            raise ValueError("n_fft must be a positive even integer.")
        if not 0 < self.win_length <= self.n_fft:
            raise ValueError("win_length must be in the range [1, n_fft].")
        if not 0 < self.hop_length <= self.win_length:
            raise ValueError("hop_length must be in the range [1, win_length].")
        if self.audio_channels != 2:
            raise ValueError("This trainer currently requires stereo audio_channels=2.")
        if self.num_stems != len(STEMS):
            raise ValueError(
                f"num_stems must match STEMS ({len(STEMS)}), got {self.num_stems}."
            )
        if self.dim <= 0 or self.depth <= 0:
            raise ValueError("dim and depth must be positive.")
        if self.num_heads <= 0 or self.dim % self.num_heads != 0:
            raise ValueError(
                f"dim ({self.dim}) must be divisible by num_heads ({self.num_heads})."
            )
        head_dim = self.dim // self.num_heads
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim ({head_dim}) must be even for RoPE.")
        if self.ffn_mult <= 0.0:
            raise ValueError("ffn_mult must be positive.")
        if self.filter_taps not in (1, 3, 5):
            raise ValueError("filter_taps must be one of {1, 3, 5}.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if not 0.0 <= self.drop_path < 1.0:
            raise ValueError("drop_path must be in [0, 1).")
        if self.num_bands != 124:
            raise ValueError("This architecture is intentionally fixed at 124 bands.")
        if self.architecture != "bs124_gated_roformer":
            raise ValueError(f"Unsupported architecture {self.architecture!r}.")


@dataclass
class LossConfig:
    waveform_weight: float = 1.0
    sdr_weight: float = 0.20
    main_stft_weight: float = 0.80
    mrstft_weight: float = 0.60
    midside_weight: float = 0.15
    silence_weight: float = 0.025
    mask_silence_weight: float = 0.025
    silence_margin: float = 1.5
    silence_floor_db: float = -65.0
    silence_active_db: float = -54.0


def seed_everything(seed: int) -> None:
    seed = int(seed) % (2**31 - 1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def training_worker_init(worker_id: int) -> None:
    seed_worker(worker_id)
    info = torch.utils.data.get_worker_info()
    if info is None:
        return
    dataset = info.dataset
    if hasattr(dataset, "segment_samples") and hasattr(dataset, "sample_rate"):
        _pink_noise(dataset.segment_samples, dataset.sample_rate)


def clean_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        key = key.replace("_orig_mod.", "").replace("._orig_mod", "")
        cleaned[key] = value
    return cleaned


def model_configs_compatible(
    saved_config: dict | None,
    current_config: ModelConfig,
) -> bool:
    if not isinstance(saved_config, dict):
        return False
    current = asdict(current_config)
    for key, current_value in current.items():
        if key in RUNTIME_CONFIG_FIELDS:
            continue
        if saved_config.get(key) != current_value:
            return False
    return True


@dataclass(frozen=True)
class StateLoadReport:
    matched: int
    incoming: int
    expected: int
    skipped: tuple[str, ...]
    missing: tuple[str, ...]

    @property
    def is_exact(self) -> bool:
        return (
            self.matched == self.expected
            and self.incoming == self.expected
            and not self.skipped
            and not self.missing
        )


def load_matching_state_dict(
    module: nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> StateLoadReport:
    current = module.state_dict()
    incoming = clean_state_dict(state_dict)
    matched: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    for key, value in incoming.items():
        if key in current and current[key].shape == value.shape:
            matched[key] = value
        else:
            skipped.append(key)
    missing = [key for key in current if key not in matched]
    module.load_state_dict(matched, strict=False)
    return StateLoadReport(
        matched=len(matched),
        incoming=len(incoming),
        expected=len(current),
        skipped=tuple(skipped),
        missing=tuple(missing),
    )


def db_to_gain(db: float) -> float:
    return 10.0 ** (db / 20.0)


def build_bs_bands(
    n_fft: int,
    num_bands: int,
) -> list[tuple[int, int]]:
    freq_bins = n_fft // 2 + 1
    if num_bands <= 0:
        raise ValueError("num_bands must be positive.")
    if num_bands > freq_bins:
        raise ValueError(
            f"Cannot create {num_bands} non-empty bands from {freq_bins} bins."
        )

    if n_fft == 4096 and num_bands == 124:
        # 2049 bins total across 124 psychoacoustic bands (7 exact-width groups,
        # max band width 128, zero power-of-two bucket padding waste).
        widths = (
            [2] * 48
            + [4] * 24
            + [12] * 16
            + [24] * 16
            + [48] * 15
            + [49]
            + [128] * 4
        )
    else:
        positions = torch.linspace(0.0, 1.0, num_bands + 1)
        boundaries = torch.round(positions.square() * freq_bins).long()
        boundaries[0] = 0
        boundaries[-1] = freq_bins
        for index in range(1, num_bands):
            minimum = int(boundaries[index - 1]) + 1
            maximum = freq_bins - (num_bands - index)
            boundaries[index] = boundaries[index].clamp(min=minimum, max=maximum)
        widths = [
            int(boundaries[index + 1] - boundaries[index])
            for index in range(num_bands)
        ]

    if len(widths) != num_bands or sum(widths) != freq_bins:
        raise RuntimeError(
            f"Invalid band layout: {len(widths)} bands cover {sum(widths)} bins; "
            f"expected {num_bands} bands covering {freq_bins} bins."
        )
    if any(width <= 0 for width in widths):
        raise RuntimeError("Band layout contains an empty band.")

    bands: list[tuple[int, int]] = []
    start = 0
    for width in widths:
        end = start + width
        bands.append((start, end))
        start = end

    coverage = torch.zeros(freq_bins, dtype=torch.int64)
    for start, end in bands:
        coverage[start:end] += 1
    if not torch.all(coverage == 1):
        raise RuntimeError("BS band construction must cover every bin exactly once.")
    return bands


def make_stft(
    audio: torch.Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: torch.Tensor,
) -> torch.Tensor:
    original_shape = audio.shape[:-1]
    audio_flat = audio.reshape(-1, audio.shape[-1]).float()
    spec = torch.stft(
        audio_flat,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window.float(),
        center=True,
        return_complex=True,
    )
    return spec.reshape(*original_shape, spec.shape[-2], spec.shape[-1])


def make_istft(
    spec: torch.Tensor,
    length: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: torch.Tensor,
) -> torch.Tensor:
    original_shape = spec.shape[:-2]
    spec_flat = spec.reshape(-1, spec.shape[-2], spec.shape[-1]).to(torch.complex64)
    audio = torch.istft(
        spec_flat,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window.float(),
        center=True,
        length=length,
    )
    return audio.reshape(*original_shape, length)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float = 10_000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE requires even head_dim, got {head_dim}.")
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cached_len = 0
        self.register_buffer("_cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("_sin_cached", torch.empty(0), persistent=False)

    def _get_cos_sin(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            not torch.compiler.is_compiling()
            and self._cached_len >= seq_len
            and self._cos_cached.device == device
            and self._cos_cached.dtype == dtype
        ):
            return (
                self._cos_cached[:, :seq_len, :, :],
                self._sin_cached[:, :seq_len, :, :],
            )
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(positions, self.inv_freq.to(device=device, dtype=torch.float32))
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=dtype)[None, :, None, :]
        sin = emb.sin().to(dtype=dtype)[None, :, None, :]
        if not torch.compiler.is_compiling():
            self._cos_cached = cos
            self._sin_cached = sin
            self._cached_len = seq_len
        return cos, sin

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # q, k are contiguous [B, L, H, head_dim]
        cos, sin = self._get_cos_sin(q.shape[1], q.device, q.dtype)
        q_rot = (q * cos) + (_rotate_half(q) * sin)
        k_rot = (k * cos) + (_rotate_half(k) * sin)
        return q_rot, k_rot


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


class GatedRoPEBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        ffn_mult: float = 3.0,
        rope_base: float = 10_000.0,
        dropout: float = 0.0,
        residual_scale: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout
        ffn_hidden = max(64, int(round(dim * ffn_mult / 64.0)) * 64)

        self.attn_norm = nn.RMSNorm(dim)
        self.qkvg_proj = nn.Linear(dim, dim * 4, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)
        self.rope = RotaryEmbedding(self.head_dim, base=rope_base)
        self.attn_out = nn.Linear(dim, dim, bias=False)

        self.ffn_norm = nn.RMSNorm(dim)
        self.ffn_in = nn.Linear(dim, ffn_hidden * 2, bias=False)
        self.ffn_out = nn.Linear(ffn_hidden, dim, bias=False)

        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self._init_weights(residual_scale)

    def _init_weights(self, residual_scale: float) -> None:
        in_std = 1.0 / math.sqrt(self.dim)
        for proj in (self.qkvg_proj, self.ffn_in):
            nn.init.normal_(proj.weight, std=in_std)
        nn.init.normal_(self.attn_out.weight, std=in_std * residual_scale)
        ffn_out_std = (1.0 / math.sqrt(self.ffn_out.in_features)) * residual_scale
        nn.init.normal_(self.ffn_out.weight, std=ffn_out_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_eff, seq_len, dim = x.shape

        h = self.attn_norm(x)
        q, k, v, attn_gate = self.qkvg_proj(h).chunk(4, dim=-1)
        q = self.q_norm(q.reshape(batch_eff, seq_len, self.num_heads, self.head_dim))
        k = self.k_norm(k.reshape(batch_eff, seq_len, self.num_heads, self.head_dim))
        v = v.reshape(batch_eff, seq_len, self.num_heads, self.head_dim)
        q, k = self.rope(q, k)

        attn_p = self.dropout if self.training else 0.0
        attn = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            dropout_p=attn_p,
        )
        attn = attn.transpose(1, 2).reshape(batch_eff, seq_len, dim)
        attn = attn * torch.sigmoid(attn_gate)
        x = x + self.drop(self.attn_out(attn))

        h = self.ffn_norm(x)
        ffn_gate, ffn_val = self.ffn_in(h).chunk(2, dim=-1)
        ffn = F.silu(ffn_gate) * ffn_val
        x = x + self.drop(self.ffn_out(self.drop(ffn)))
        return x


class DualPathEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.depth = config.depth
        self.drop_path = config.drop_path
        res_scale = 1.0 / math.sqrt(2.0 * config.depth)
        self.time_layers = nn.ModuleList(
            GatedRoPEBlock(
                dim=config.dim,
                num_heads=config.num_heads,
                ffn_mult=config.ffn_mult,
                rope_base=10_000.0,
                dropout=config.dropout,
                residual_scale=res_scale,
            )
            for _ in range(config.depth)
        )
        self.freq_layers = nn.ModuleList(
            GatedRoPEBlock(
                dim=config.dim,
                num_heads=config.num_heads,
                ffn_mult=config.ffn_mult,
                rope_base=1_000.0,
                dropout=config.dropout,
                residual_scale=res_scale,
            )
            for _ in range(config.depth)
        )
        self.num_unet_skips = config.depth // 2
        if self.num_unet_skips > 0:
            self.unet_skip_gates = nn.Parameter(
                torch.full((self.num_unet_skips, config.dim), 0.1)
            )
        else:
            self.register_parameter("unet_skip_gates", None)
        self.skip_proj = nn.Linear(config.dim, config.dim, bias=False)
        nn.init.eye_(self.skip_proj.weight)
        self.output_norm = nn.RMSNorm(config.dim)
        self.use_checkpoint = config.use_checkpoint

    def compile_layers(self, mode: str = "default") -> None:
        for unit in (*self.time_layers, *self.freq_layers):
            unit.compile(mode=mode, dynamic=False)

    @staticmethod
    def _run_module(
        module: nn.Module,
        x: torch.Tensor,
        use_checkpoint: bool,
    ) -> torch.Tensor:
        if not use_checkpoint:
            return module(x)
        return checkpoint(module, x, use_reentrant=False)

    def _drop_keep_mask(
        self,
        batch: int,
        drop_prob: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if not self.training or drop_prob <= 0.0:
            return None
        keep_prob = 1.0 - drop_prob
        mask = (torch.rand(batch, 1, 1, 1, device=device) < keep_prob).to(dtype)
        return mask / keep_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, frames, bands, dim = x.shape
        should_checkpoint = (
            self.use_checkpoint and torch.is_grad_enabled() and x.requires_grad
        )
        stem_skip = self.skip_proj(x)
        total_blocks = max(1, 2 * self.depth - 1)
        unet_skips: list[torch.Tensor] = []
        pop_start = self.depth - self.num_unet_skips

        for layer_idx, (time_layer, freq_layer) in enumerate(
            zip(self.time_layers, self.freq_layers)
        ):
            if layer_idx >= pop_start and self.unet_skip_gates is not None:
                skip_idx = layer_idx - pop_start
                gate = self.unet_skip_gates[skip_idx].to(
                    device=x.device, dtype=x.dtype
                )
                x = x + unet_skips.pop() * gate

            p_time = self.drop_path * (2 * layer_idx) / total_blocks
            keep_t = self._drop_keep_mask(batch, p_time, x.device, x.dtype)
            time_in = x.permute(0, 2, 1, 3).reshape(batch * bands, frames, dim)
            time_out = self._run_module(time_layer, time_in, should_checkpoint)
            time_out = time_out.reshape(batch, bands, frames, dim).permute(0, 2, 1, 3)
            x = time_out if keep_t is None else x + (time_out - x) * keep_t

            p_freq = self.drop_path * (2 * layer_idx + 1) / total_blocks
            keep_f = self._drop_keep_mask(batch, p_freq, x.device, x.dtype)
            freq_in = x.reshape(batch * frames, bands, dim)
            freq_out = self._run_module(freq_layer, freq_in, should_checkpoint)
            freq_out = freq_out.reshape(batch, frames, bands, dim)
            x = freq_out if keep_f is None else x + (freq_out - x) * keep_f

            if layer_idx < self.num_unet_skips:
                unet_skips.append(x)

        return self.output_norm(x + stem_skip)


class BandInputGroup(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        bands: Sequence[tuple[int, int]],
        band_ids: Sequence[int],
        band_width: int,
    ):
        super().__init__()
        self.band_width = band_width
        self.audio_channels = config.audio_channels
        self.feature_width = band_width * config.audio_channels * 2
        self.num_group_bands = len(band_ids)

        self.register_buffer(
            "band_ids", torch.tensor(band_ids, dtype=torch.long), persistent=False
        )
        freq_indices = torch.empty(
            self.num_group_bands, band_width, dtype=torch.long
        )
        for local_index, band_id in enumerate(band_ids):
            start, end = bands[band_id]
            width = end - start
            if width != band_width:
                raise ValueError(
                    f"Band {band_id} width {width} does not match group width {band_width}."
                )
            freq_indices[local_index] = torch.arange(start, end, dtype=torch.long)

        self.register_buffer("freq_indices", freq_indices, persistent=False)

        self.gamma = nn.Parameter(
            torch.ones(self.num_group_bands, self.feature_width)
        )
        self.weight = nn.Parameter(
            torch.empty(self.num_group_bands, self.feature_width, config.dim)
        )
        self.bias = nn.Parameter(
            torch.zeros(self.num_group_bands, config.dim)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = math.sqrt(6.0 / (self.feature_width + self.weight.shape[-1]))
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.bias)

    def forward(
        self,
        real_imag: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Compute normalization and quadratic stereo spatial cues in fp32
        gathered = real_imag[:, :, self.freq_indices].float()

        # Quadratic stereo spatial features per (B, T, band): Mid/Side ratio & L/R coherence
        left_r = gathered[..., 0, 0]
        left_i = gathered[..., 0, 1]
        right_r = gathered[..., 1, 0]
        right_i = gathered[..., 1, 1]

        inv_bins = 1.0 / float(self.band_width)
        mid_pow = (0.25 * inv_bins) * (
            (left_r + right_r).square() + (left_i + right_i).square()
        ).sum(dim=-1)
        side_pow = (0.25 * inv_bins) * (
            (left_r - right_r).square() + (left_i - right_i).square()
        ).sum(dim=-1)
        ms_log_ratio = (0.5 * torch.log((mid_pow + 1e-7) / (side_pow + 1e-7))).clamp(
            -8.0, 8.0
        )

        left_pow = (left_r.square() + left_i.square()).sum(dim=-1)
        right_pow = (right_r.square() + right_i.square()).sum(dim=-1)
        cross_r = (left_r * right_r + left_i * right_i).sum(dim=-1)
        cross_i = (left_i * right_r - left_r * right_i).sum(dim=-1)
        coh_denom = (left_pow * right_pow).clamp_min(0.0).sqrt() + 1e-7
        coh_real = (cross_r / coh_denom).clamp(-1.0, 1.0)
        coh_imag = (cross_i / coh_denom).clamp(-1.0, 1.0)
        spatial_cues = torch.stack((ms_log_ratio, coh_real, coh_imag), dim=-1)

        features = gathered.reshape(
            gathered.shape[0],
            gathered.shape[1],
            self.num_group_bands,
            self.feature_width,
        )
        mean_square = features.square().mean(dim=-1, keepdim=True)
        features = features * torch.rsqrt(mean_square + 1e-7)
        features = features * self.gamma[None, None].float()

        tokens = torch.einsum("btni,nid->btnd", features, self.weight) + self.bias
        return tokens, mean_square.squeeze(-1), spatial_cues


class BandSplit(nn.Module):
    SPATIAL_ENERGY_DIM = 5
    LOCAL_ANCHOR_FRAMES = 173  # ~2.0s @ hop=512, sr=44.1kHz (sliding-window consistent)

    def __init__(self, config: ModelConfig, bands: Sequence[tuple[int, int]]):
        super().__init__()
        self.dim = config.dim
        self.num_bands = len(bands)
        grouped_ids: dict[int, list[int]] = {}
        for band_id, (start, end) in enumerate(bands):
            band_width = end - start
            grouped_ids.setdefault(band_width, []).append(band_id)

        self.groups = nn.ModuleList(
            BandInputGroup(config, bands, ids, band_width)
            for band_width, ids in sorted(grouped_ids.items())
        )
        self.register_buffer(
            "band_feature_counts",
            torch.tensor(
                [(end - start) * config.audio_channels * 2 for start, end in bands],
                dtype=torch.float32,
            ),
            persistent=False,
        )

        freq_bins = config.n_fft // 2 + 1
        metadata = []
        for start, end in bands:
            center = 0.5 * (start + end - 1)
            width = end - start
            center_hz = center * config.sample_rate / config.n_fft
            width_hz = width * config.sample_rate / config.n_fft
            metadata.append(
                (
                    math.log1p(center_hz) / math.log1p(config.sample_rate / 2),
                    math.log1p(width_hz) / math.log1p(config.sample_rate / 2),
                    start / max(1, freq_bins - 1),
                    (end - 1) / max(1, freq_bins - 1),
                )
            )
        self.register_buffer(
            "band_metadata",
            torch.tensor(metadata, dtype=torch.float32),
            persistent=False,
        )
        self.metadata_proj = nn.Sequential(
            nn.Linear(4, config.dim),
            nn.SiLU(),
            nn.Linear(config.dim, config.dim, bias=False),
        )
        nn.init.normal_(self.metadata_proj[-1].weight, std=0.01)

        cue_hidden = 32
        self.cue_fc1_w = nn.Parameter(
            torch.empty(self.num_bands, self.SPATIAL_ENERGY_DIM, cue_hidden)
        )
        self.cue_fc1_b = nn.Parameter(torch.zeros(self.num_bands, cue_hidden))
        self.cue_fc2_w = nn.Parameter(
            torch.empty(self.num_bands, cue_hidden, config.dim * 2)
        )
        self.cue_fc2_b = nn.Parameter(torch.zeros(self.num_bands, config.dim * 2))
        nn.init.normal_(self.cue_fc1_w, std=1.0 / math.sqrt(self.SPATIAL_ENERGY_DIM))
        nn.init.normal_(self.cue_fc2_w, std=0.02)
        with torch.no_grad():
            self.cue_fc2_b[:, : config.dim].fill_(1.0)

    def _local_temporal_anchor(self, frame_log_rms: torch.Tensor) -> torch.Tensor:
        frames = frame_log_rms.shape[1]
        win = min(self.LOCAL_ANCHOR_FRAMES, frames)
        if win % 2 == 0:
            win = max(1, win - 1)
        if win <= 1:
            return frame_log_rms.mean(dim=1, keepdim=True)
        half = win // 2
        padded = F.pad(frame_log_rms.transpose(1, 2), (half, half), mode="reflect")
        return F.avg_pool1d(padded, kernel_size=win, stride=1).transpose(1, 2)

    def forward_real(self, real_imag: torch.Tensor) -> torch.Tensor:
        real_imag = real_imag.permute(0, 3, 2, 1, 4)
        batch, frames = real_imag.shape[0], real_imag.shape[1]

        first_tokens, first_power, first_spatial = self.groups[0](real_imag)
        output = first_tokens.new_zeros(batch, frames, self.num_bands, self.dim)
        band_power = first_power.new_zeros(batch, frames, self.num_bands)
        spatial_all = first_spatial.new_zeros(batch, frames, self.num_bands, 3)

        output.index_copy_(2, self.groups[0].band_ids, first_tokens)
        band_power.index_copy_(2, self.groups[0].band_ids, first_power)
        spatial_all.index_copy_(2, self.groups[0].band_ids, first_spatial)

        for group in self.groups[1:]:
            group_tokens, group_power, group_spatial = group(real_imag)
            output.index_copy_(
                2, group.band_ids, group_tokens.to(dtype=output.dtype)
            )
            band_power.index_copy_(2, group.band_ids, group_power)
            spatial_all.index_copy_(2, group.band_ids, group_spatial)

        log_rms = 0.5 * torch.log(band_power + 1e-7)
        feature_counts = self.band_feature_counts.to(dtype=band_power.dtype)
        frame_power = (
            band_power * feature_counts[None, None]
        ).sum(dim=2, keepdim=True) / feature_counts.sum()
        frame_log_rms = 0.5 * torch.log(frame_power + 1e-7)
        spectral_relative = (log_rms - frame_log_rms).clamp(-10.0, 10.0)
        local_anchor = self._local_temporal_anchor(frame_log_rms)
        temporal_relative = (frame_log_rms - local_anchor).clamp(-10.0, 10.0)

        cue_features = torch.cat(
            (
                spectral_relative.unsqueeze(-1),
                temporal_relative.expand_as(log_rms).unsqueeze(-1),
                spatial_all,
            ),
            dim=-1,
        )
        cue_hidden = F.silu(
            torch.einsum("btne,neh->btnh", cue_features, self.cue_fc1_w)
            + self.cue_fc1_b[None, None]
        )
        # FiLM multiplicative gate + additive shift attenuates unit-normed noise-floor bands
        cue_proj = (
            torch.einsum("btnh,nhd->btnd", cue_hidden, self.cue_fc2_w)
            + self.cue_fc2_b[None, None]
        )
        cue_scale, cue_shift = cue_proj.chunk(2, dim=-1)
        cue_gate = torch.sigmoid(cue_scale).to(dtype=output.dtype)
        metadata_tokens = self.metadata_proj(
            self.band_metadata.to(device=output.device, dtype=output.dtype)
        )[None, None]
        return output * cue_gate + cue_shift.to(dtype=output.dtype) + metadata_tokens


class BandMaskGroup(nn.Module):
    """Per-band MLP with temporal depthwise conv and adaptive SwiGLU bottleneck
    producing fp32 multi-tap cross-channel (C_out x C_in MIMO) complex deep filters.
    Full rank is preserved on narrow low-frequency harmonic bands while wide >2 kHz
    bands use a factorized bottleneck to prevent high-frequency dataset memorization.
    """

    @staticmethod
    def _select_bottleneck_dim(band_width: int, hidden_dim: int) -> int:
        if band_width <= 8:
            return 0
        if band_width <= 16:
            return min(hidden_dim, 160)
        if band_width <= 32:
            return min(hidden_dim, 128)
        if band_width <= 64:
            return min(hidden_dim, 96)
        return min(hidden_dim, 64)

    def __init__(
        self,
        config: ModelConfig,
        bands: Sequence[tuple[int, int]],
        band_ids: Sequence[int],
        band_width: int,
    ):
        super().__init__()
        self.audio_channels = config.audio_channels
        self.filter_taps = config.filter_taps
        self.band_width = band_width
        self.feature_width = (
            band_width
            * self.audio_channels
            * self.audio_channels
            * self.filter_taps
            * 2
        )
        self.num_group_bands = len(band_ids)

        self.register_buffer(
            "band_ids", torch.tensor(band_ids, dtype=torch.long), persistent=False
        )
        freq_indices = torch.empty(
            self.num_group_bands, band_width, dtype=torch.long
        )
        for local_index, band_id in enumerate(band_ids):
            start, end = bands[band_id]
            width = end - start
            if width != band_width:
                raise ValueError(
                    f"Band {band_id} width {width} does not match group width {band_width}."
                )
            freq_indices[local_index] = torch.arange(start, end, dtype=torch.long)

        self.register_buffer(
            "flat_freq_indices", freq_indices.reshape(-1), persistent=False
        )

        hidden_dim = config.dim
        self.norm1 = nn.RMSNorm(config.dim)
        self.fc1_weight = nn.Parameter(
            torch.empty(self.num_group_bands, config.dim, hidden_dim)
        )
        self.fc1_bias = nn.Parameter(
            torch.zeros(self.num_group_bands, hidden_dim)
        )

        # Group-shared temporal conv + second MLP layer for local phase continuity
        self.temporal_dwconv = nn.Conv1d(
            hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim
        )
        self.norm2 = nn.RMSNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.bottleneck_dim = self._select_bottleneck_dim(band_width, hidden_dim)
        if self.bottleneck_dim > 0:
            self.bottleneck_weight = nn.Parameter(
                torch.empty(
                    self.num_group_bands, hidden_dim, self.bottleneck_dim * 2
                )
            )
            self.bottleneck_bias = nn.Parameter(
                torch.zeros(self.num_group_bands, self.bottleneck_dim * 2)
            )
            self.output_weight = nn.Parameter(
                torch.empty(
                    self.num_group_bands, self.bottleneck_dim, self.feature_width
                )
            )
            self.output_bias = nn.Parameter(
                torch.zeros(self.num_group_bands, self.feature_width)
            )
            nn.init.normal_(
                self.bottleneck_weight, std=1.0 / math.sqrt(hidden_dim)
            )
            nn.init.normal_(
                self.output_weight, std=0.25 / math.sqrt(self.bottleneck_dim)
            )
        else:
            self.register_parameter("bottleneck_weight", None)
            self.register_parameter("bottleneck_bias", None)
            self.output_weight = nn.Parameter(
                torch.empty(
                    self.num_group_bands, hidden_dim, self.feature_width * 2
                )
            )
            self.output_bias = nn.Parameter(
                torch.zeros(self.num_group_bands, self.feature_width * 2)
            )
            nn.init.normal_(self.output_weight, std=0.25 / math.sqrt(hidden_dim))

        nn.init.normal_(self.fc1_weight, std=1.0 / math.sqrt(config.dim))
        nn.init.zeros_(self.temporal_dwconv.bias)
        nn.init.normal_(self.fc2.weight, std=1.0 / math.sqrt(hidden_dim))
        nn.init.zeros_(self.fc2.bias)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, frames, num_bands, _ = x.shape
        h = self.norm1(x)
        h = F.silu(
            torch.einsum("btnd,ndh->btnh", h, self.fc1_weight)
            + self.fc1_bias[None, None]
        )

        h_t = h.permute(0, 2, 3, 1).reshape(batch * num_bands, h.shape[-1], frames)
        h_t = self.temporal_dwconv(h_t).reshape(batch, num_bands, h.shape[-1], frames)
        h = h + h_t.permute(0, 3, 1, 2)
        h = h + F.silu(self.fc2(self.norm2(h)))

        # Compute final filter tap projections in fp32 to avoid bf16 -48 dB mantissa steps
        if self.bottleneck_dim > 0 and self.bottleneck_weight is not None:
            bn = torch.einsum("btnh,nhr->btnr", h, self.bottleneck_weight)
            bn = F.glu(bn.float() + self.bottleneck_bias[None, None].float(), dim=-1)
            raw = (
                torch.einsum("btnr,nrq->btnq", bn, self.output_weight.float())
                + self.output_bias[None, None].float()
            )
        else:
            raw = torch.einsum(
                "btnh,nhq->btnq", h.float(), self.output_weight.float()
            )
            raw = F.glu(raw + self.output_bias[None, None].float(), dim=-1)

        raw = raw.reshape(
            batch,
            frames,
            self.num_group_bands,
            self.band_width,
            self.audio_channels,
            self.audio_channels,
            self.filter_taps,
            2,
        )
        # -> [B, C_out, C_in, T, N_group * band_width, taps, 2]
        source = raw.permute(0, 4, 5, 1, 2, 3, 6, 7).reshape(
            batch,
            self.audio_channels,
            self.audio_channels,
            frames,
            self.num_group_bands * self.band_width,
            self.filter_taps,
            2,
        )
        return source, self.flat_freq_indices


class MaskHead(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        bands: Sequence[tuple[int, int]],
    ):
        super().__init__()
        self.dim = config.dim
        self.num_bands = len(bands)
        self.audio_channels = config.audio_channels
        self.filter_taps = config.filter_taps
        self.center_tap = config.filter_taps // 2
        self.freq_bins = config.n_fft // 2 + 1
        self.filter_channels = (
            self.audio_channels
            * self.audio_channels
            * self.filter_taps
            * 2
        )

        self.band_context_norm = nn.RMSNorm(config.dim)
        self.band_context_dw = nn.Conv1d(
            config.dim, config.dim, kernel_size=5, padding=2, groups=config.dim
        )
        self.band_context_pw = nn.Linear(config.dim, config.dim, bias=False)
        nn.init.zeros_(self.band_context_dw.bias)
        nn.init.zeros_(self.band_context_pw.weight)

        grouped_ids: dict[int, list[int]] = {}
        for band_id, (start, end) in enumerate(bands):
            band_width = end - start
            grouped_ids.setdefault(band_width, []).append(band_id)
        self.groups = nn.ModuleList(
            BandMaskGroup(config, bands, ids, band_width)
            for band_width, ids in sorted(grouped_ids.items())
        )

        self.seam_dw = nn.Conv1d(
            self.filter_channels,
            self.filter_channels,
            kernel_size=5,
            padding=2,
            groups=self.filter_channels,
        )
        self.seam_pw = nn.Conv1d(
            self.filter_channels, self.filter_channels, kernel_size=1
        )
        nn.init.zeros_(self.seam_dw.bias)
        nn.init.zeros_(self.seam_pw.weight)
        nn.init.zeros_(self.seam_pw.bias)

        coverage = torch.zeros(self.freq_bins, dtype=torch.float32)
        for start, end in bands:
            coverage[start:end] += 1.0
        if not torch.all(coverage == 1):
            raise ValueError("Every frequency bin must be covered by exactly one BS band.")

        tap_bias = torch.zeros(
            1,
            self.audio_channels,
            self.audio_channels,
            1,
            1,
            self.filter_taps,
            2,
            dtype=torch.float32,
        )
        for ch in range(self.audio_channels):
            tap_bias[0, ch, ch, 0, 0, self.center_tap, 0] = 0.5
        self.register_buffer("tap_bias", tap_bias, persistent=False)

    def forward_real(self, x: torch.Tensor) -> torch.Tensor:
        batch, frames = x.shape[0], x.shape[1]
        ctx = self.band_context_norm(x).reshape(
            batch * frames, self.num_bands, self.dim
        )
        ctx = F.silu(self.band_context_dw(ctx.transpose(1, 2))).transpose(1, 2)
        x = x + self.band_context_pw(ctx).reshape(
            batch, frames, self.num_bands, self.dim
        )

        first_src, first_idx = self.groups[0](x.index_select(2, self.groups[0].band_ids))
        output = first_src.new_zeros(
            batch,
            self.audio_channels,
            self.audio_channels,
            frames,
            self.freq_bins,
            self.filter_taps,
            2,
            dtype=torch.float32,
        )
        output.index_copy_(4, first_idx, first_src.float())

        for group in self.groups[1:]:
            group_x = x.index_select(2, group.band_ids)
            source, flat_indices = group(group_x)
            output.index_copy_(4, flat_indices, source.float())

        device_type = output.device.type
        no_autocast = (
            torch.amp.autocast(device_type=device_type, enabled=False)
            if device_type == "cuda"
            else contextlib.nullcontext()
        )
        with no_autocast:
            flat_f = output.permute(0, 3, 1, 2, 5, 6, 4).reshape(
                batch * frames, self.filter_channels, self.freq_bins
            )
            flat_f = flat_f + self.seam_pw(F.silu(self.seam_dw(flat_f)))
            refined = flat_f.reshape(
                batch,
                frames,
                self.audio_channels,
                self.audio_channels,
                self.filter_taps,
                2,
                self.freq_bins,
            ).permute(0, 2, 3, 6, 1, 4, 5).contiguous()
            return refined + self.tap_bias


def unfold_stft_taps(spec: torch.Tensor, taps: int) -> torch.Tensor:
    """Build [B, C, F, T, taps] zero-padded temporal neighborhood around each STFT frame."""
    if taps == 1:
        return spec.unsqueeze(-1)
    pad = taps // 2
    padded = F.pad(spec, (pad, pad))
    return padded.unfold(dimension=-1, size=taps, step=1)


class ResonantConvGateSeparator(nn.Module):
    """124-Band Gated-RoPE + 2x2 MIMO Multi-Tap Complex Deep-Filter Separator."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.bands = build_bs_bands(config.n_fft, config.num_bands)
        self.band_split = BandSplit(config, self.bands)
        self.encoder = DualPathEncoder(config)
        self.mask_head = MaskHead(config, self.bands)

    def forward_filters_real(
        self,
        mixture_real_imag: torch.Tensor,
    ) -> torch.Tensor:
        tokens = self.band_split.forward_real(mixture_real_imag)
        if tokens.shape[2] != self.config.num_bands:
            raise ValueError(
                f"Expected {self.config.num_bands} band tokens, got {tokens.shape[2]}."
            )
        tokens = self.encoder(tokens)
        return self.mask_head.forward_real(tokens)

    def forward(
        self,
        mixture_spec: torch.Tensor,
    ) -> torch.Tensor:
        """Returns multi-tap MIMO complex vocal filter [B, C_out, C_in, F, T, taps]."""
        mixture_spec_c64 = mixture_spec.to(torch.complex64)
        mixture_real_imag = torch.view_as_real(mixture_spec_c64)
        vocal_filters_ri = self.forward_filters_real(mixture_real_imag)
        return torch.view_as_complex(vocal_filters_ri)

    def estimate_specs(
        self,
        mixture_spec: torch.Tensor,
        mask_deadzone: float = 0.0,
    ) -> torch.Tensor:
        mixture_c64 = mixture_spec.to(torch.complex64)
        vocal_filters = self(mixture_c64)
        neighbors = unfold_stft_taps(mixture_c64, self.config.filter_taps)
        # 2x2 space-time MIMO Wiener filter: sum across input channels (dim=2) and taps (dim=-1)
        raw_vocal_est = (
            (vocal_filters * neighbors.unsqueeze(1))
            .sum(dim=(2, -1))
            .unsqueeze(1)
        )
        mix_ref = mixture_c64[:, None]
        if mask_deadzone > 0.0:
            mix_mag = mix_ref.abs()
            est_mag = raw_vocal_est.abs().clamp_min(1e-8)
            thresh = mask_deadzone * mix_mag
            gain = ((est_mag - thresh).clamp_min(0.0) / est_mag).clamp_max(1.0)
            vocal_est = raw_vocal_est * gain
        else:
            vocal_est = raw_vocal_est

        other_est = mix_ref - vocal_est
        return torch.cat((vocal_est, other_est), dim=1)


def power_compress_complex(
    spec: torch.Tensor,
    exp: float = 0.6,
    eps: float = 1e-10,
) -> torch.Tensor:
    """Smooth, phase-preserving power-law complex compression S * (|S|^2 + eps)^((exp-1)/2).
    Preserves Bayes-optimal Wiener phase shrinkage on uncertain bins while still
    providing clean phase-sensitive gradients to quiet mid/high harmonics.
    """
    power = spec.real.square() + spec.imag.square() + eps
    return spec * power.pow(0.5 * (exp - 1.0))


class MultiResolutionSTFTLoss(nn.Module):
    """Phase-consistent multi-resolution complex RMSE + power-compressed complex STFT loss."""

    def __init__(
        self,
        resolutions: Sequence[tuple[int, int, int]] = (
            (4096, 1024, 4096),
            (2048, 512, 2048),
            (1024, 256, 1024),
            (512, 128, 512),
            (256, 64, 256),
        ),
        compress_exp: float = 0.6,
    ):
        super().__init__()
        self.resolutions = tuple(resolutions)
        self.compress_exp = compress_exp
        for index, (_n_fft, _hop, win_length) in enumerate(self.resolutions):
            window = torch.hann_window(win_length)
            self.register_buffer(f"window_{index}", window, persistent=False)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        w_l1 = 0.12
        w_comp = 0.18
        w_rmse = 1.0 - w_l1 - w_comp

        batch = prediction.shape[0] if prediction.ndim >= 2 else 1
        pred_flat = prediction.reshape(-1, prediction.shape[-1]).float()
        target_flat = target.reshape(-1, target.shape[-1]).float()
        both_flat = torch.cat((pred_flat, target_flat), dim=0)
        total = pred_flat.new_tensor(0.0)

        for index, (n_fft, hop_length, win_length) in enumerate(self.resolutions):
            window = getattr(self, f"window_{index}")
            win_norm = (0.5 * window.sum()).clamp_min(1.0)
            both_spec = torch.stft(
                both_flat,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                center=True,
                return_complex=True,
            ) / win_norm
            pred_spec, target_spec = both_spec.chunk(2, dim=0)
            spec_f, spec_t = pred_spec.shape[-2], pred_spec.shape[-1]
            pred_spec_b = pred_spec.reshape(batch, -1, spec_f, spec_t)
            target_spec_b = target_spec.reshape(batch, -1, spec_f, spec_t)
            target_mag_b = target_spec_b.abs()

            # 1. Dominant Parseval/SDR-exact mixture-anchored complex RMSE (+ L1)
            diff_c = pred_spec_b - target_spec_b
            diff_sq = diff_c.real.square() + diff_c.imag.square()
            complex_scale = target_mag_b.mean(dim=(1, 2, 3)).detach().clamp_min(1e-3)
            complex_rmse = (
                diff_sq.mean(dim=(1, 2, 3)).add(1e-10).sqrt() / complex_scale
            ).mean()
            complex_l1 = (
                (diff_sq + 1e-12).sqrt().mean(dim=(1, 2, 3)) / complex_scale
            ).mean()

            # 2. Phase-preserving power-compressed complex RMSE (no phase-blind mag penalty)
            pred_comp = power_compress_complex(pred_spec_b, self.compress_exp)
            targ_comp = power_compress_complex(target_spec_b, self.compress_exp)
            comp_diff = pred_comp - targ_comp
            comp_sq = comp_diff.real.square() + comp_diff.imag.square()
            comp_scale = (
                (target_mag_b + 1e-5)
                .pow(self.compress_exp)
                .mean(dim=(1, 2, 3))
                .detach()
                .clamp_min(1e-3)
            )
            comp_complex_rmse = (
                comp_sq.mean(dim=(1, 2, 3)).add(1e-10).sqrt() / comp_scale
            ).mean()

            total = (
                total
                + w_rmse * complex_rmse
                + w_l1 * complex_l1
                + w_comp * comp_complex_rmse
            )

        return total / len(self.resolutions)


def loudness_anchored_wave_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    external_anchor: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dominant conditional-mean RMSE (Bayes-optimal for full-track SDR) plus
    L1 transient guidance, anchored to stereo mixture energy.
    """
    w_l1 = 0.15
    w_rmse = 1.0 - w_l1

    pred_f = prediction.float()
    targ_f = target.float()
    diff = pred_f - targ_f
    l1_err = diff.abs().mean(dim=-1)
    rmse_err = diff.square().mean(dim=-1).add(1e-10).sqrt()

    targ_rms = targ_f.square().mean(dim=-1).add(1e-10).sqrt()
    if external_anchor is not None:
        shape = (external_anchor.shape[0],) + (1,) * (targ_rms.ndim - 1)
        mix_anchor = external_anchor.reshape(shape)
    elif targ_rms.ndim > 1:
        reduce_dims = tuple(range(1, targ_rms.ndim))
        mix_anchor = targ_rms.amax(dim=reduce_dims, keepdim=True)
    else:
        mix_anchor = targ_rms
    # Hybrid anchor: 50% track-relative, 50% soft-floored global scale (0.04)
    soft_anchor = (
        0.5 * mix_anchor
        + 0.5 * torch.maximum(mix_anchor, mix_anchor.new_tensor(0.04))
    ).clamp_min(1e-3)
    stem_denom = torch.maximum(targ_rms, 0.25 * soft_anchor)

    return (w_rmse * (rmse_err / soft_anchor) + w_l1 * (l1_err / stem_denom)).mean()


def tamed_sdr_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    saturation_db: float = 28.0,
    rel_error_floor: float = 5e-4,
) -> torch.Tensor:
    """Scale-dependent SDR loss with mixture-anchored error floor so quiet-vocal
    crops cannot explode 1/||e||^2 gradients or starve loud, high-leverage choruses.
    """
    pred_f = prediction.float()
    targ_f = target.float()
    error_power = (pred_f - targ_f).square().mean(dim=(-2, -1))
    target_power = targ_f.square().mean(dim=(-2, -1))

    if target_power.ndim > 1:
        ref_power = target_power.amax(dim=-1, keepdim=True)
    else:
        ref_power = target_power

    # Only score stems that genuinely contain active audio in this segment
    valid = (target_power > 1e-6) & (target_power > 5e-3 * ref_power)
    if not valid.any():
        return pred_f.new_tensor(0.0)

    anchor_power = torch.maximum(target_power, 0.10 * ref_power)
    floored_error = error_power + rel_error_floor * anchor_power + 1e-8
    ratio_db = 10.0 * torch.log10((target_power + 1e-8) / floored_error)
    ratio_db = saturation_db * torch.tanh(ratio_db / saturation_db)
    valid_f = valid.to(pred_f.dtype)
    if ratio_db.ndim > 1:
        per_item = -(ratio_db * valid_f).sum(dim=-1) / valid_f.sum(dim=-1).clamp_min(1.0)
        return per_item.mean()
    return -ratio_db[valid].mean()


def mid_side(audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mid = (audio[..., 0, :] + audio[..., 1, :]) * 0.5
    side = (audio[..., 0, :] - audio[..., 1, :]) * 0.5
    return mid, side


SILENCE_FRAME = 1024


def _frame_mean(values: torch.Tensor, frame: int) -> torch.Tensor:
    pad = (-values.shape[-1]) % frame
    if pad:
        values = F.pad(values, (0, pad))
    return values.reshape(*values.shape[:-1], -1, frame).mean(dim=-1)


def _stem_frame_power(audio: torch.Tensor, frame: int = SILENCE_FRAME) -> torch.Tensor:
    """Mean power per frame of a [B, C, T] stem, averaged over channels."""
    return _frame_mean(audio.float().square().mean(dim=-2), frame)


def vocal_silence_weight(
    target_vocal_power: torch.Tensor,
    mixture_power: torch.Tensor,
    floor_db: float = -65.0,
    active_db: float = -54.0,
) -> torch.Tensor:
    """Weight ~1 only where the target vocal is genuinely silent, and 0 on active
    or quiet vocals (breaths, whispers, reverb tails).
    """
    relative_db = 10.0 * torch.log10(
        (target_vocal_power + 1e-12) / (mixture_power + 1e-12)
    )
    span = max(active_db - floor_db, 1e-6)
    ramp = ((relative_db - floor_db) / span).clamp(0.0, 1.0)
    weight = 1.0 - ramp
    peak_power = mixture_power.amax(dim=-1, keepdim=True).clamp_min(1e-12)
    audible = mixture_power > 1e-4 * peak_power
    return weight * audible.to(weight.dtype)


def silence_residue_loss(
    predicted: torch.Tensor,
    reference: torch.Tensor,
    weight: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    excess = (predicted - margin * reference).clamp_min(0.0)
    weight = weight.expand_as(excess)
    if excess.ndim > 1:
        reduce_dims = tuple(range(1, excess.ndim))
        total_weight = weight.sum(dim=reduce_dims).clamp_min(1e-6)
        per_item = (excess * weight).sum(dim=reduce_dims) / total_weight
        return per_item.mean()
    total_weight = weight.sum().clamp_min(1e-6)
    return (excess * weight).sum() / total_weight


class SeparationLoss(nn.Module):
    def __init__(self, model_config: ModelConfig, loss_config: LossConfig):
        super().__init__()
        self.model_config = model_config
        self.loss_config = loss_config
        self.mrstft = MultiResolutionSTFTLoss()
        self.activity_threshold = 1e-4
        window = torch.hann_window(model_config.win_length)
        self.register_buffer("window", window, persistent=False)

    def compute_from_estimates(
        self,
        estimates: torch.Tensor,
        mixture_spec: torch.Tensor,
        target_audio: torch.Tensor,
        mixture_audio: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # Always compute losses in full fp32 precision for exact phase/SDR gradients
        estimates = estimates.to(torch.complex64)
        mixture_spec = mixture_spec.to(torch.complex64)
        target_audio = target_audio.float()
        if mixture_audio is not None:
            mixture_audio = mixture_audio.float()

        pred_audio = make_istft(
            estimates,
            length=target_audio.shape[-1],
            n_fft=self.model_config.n_fft,
            hop_length=self.model_config.hop_length,
            win_length=self.model_config.win_length,
            window=self.window,
        )
        target_specs = make_stft(
            target_audio,
            n_fft=self.model_config.n_fft,
            hop_length=self.model_config.hop_length,
            win_length=self.model_config.win_length,
            window=self.window,
        )

        wave_loss = loudness_anchored_wave_loss(pred_audio, target_audio)
        mrstft_loss = self.mrstft(pred_audio, target_audio)

        win_norm = (0.5 * self.window.sum()).clamp_min(1.0)
        est_normed = estimates / win_norm
        targ_normed = target_specs / win_norm
        target_mag = targ_normed.abs()

        item_dims = tuple(range(1, target_mag.ndim))
        spec_scale = target_mag.mean(dim=item_dims).detach().clamp_min(1e-3)
        diff_c = est_normed - targ_normed
        diff_sq = diff_c.real.square() + diff_c.imag.square()
        main_rmse = (
            diff_sq.mean(dim=item_dims).add(1e-10).sqrt() / spec_scale
        ).mean()
        main_l1 = (
            (diff_sq + 1e-12).sqrt().mean(dim=item_dims) / spec_scale
        ).mean()

        est_comp = power_compress_complex(est_normed, exp=0.6)
        targ_comp = power_compress_complex(targ_normed, exp=0.6)
        comp_diff = est_comp - targ_comp
        comp_sq = comp_diff.real.square() + comp_diff.imag.square()
        comp_scale = (
            (target_mag + 1e-5)
            .pow(0.6)
            .mean(dim=item_dims)
            .detach()
            .clamp_min(1e-3)
        )
        main_comp_complex = (
            comp_sq.mean(dim=item_dims).add(1e-10).sqrt() / comp_scale
        ).mean()

        w_l1 = 0.12
        w_comp = 0.18
        w_rmse = 1.0 - w_l1 - w_comp
        main_stft_loss = (
            w_rmse * main_rmse + w_l1 * main_l1 + w_comp * main_comp_complex
        )

        sdr_loss = tamed_sdr_loss(pred_audio, target_audio)
        stereo_anchor = (
            target_audio.square()
            .mean(dim=-1)
            .add(1e-10)
            .sqrt()
            .amax(dim=(1, 2))
            .detach()
        )
        pred_mid, pred_side = mid_side(pred_audio)
        true_mid, true_side = mid_side(target_audio)
        midside_loss = 0.5 * (
            loudness_anchored_wave_loss(
                pred_mid, true_mid, external_anchor=stereo_anchor
            )
            + loudness_anchored_wave_loss(
                pred_side, true_side, external_anchor=stereo_anchor
            )
        )

        cfg = self.loss_config
        mixture_stem = (
            mixture_audio if mixture_audio is not None else target_audio.sum(dim=1)
        )
        silence_frame_weight = vocal_silence_weight(
            _stem_frame_power(target_audio[:, 0]),
            _stem_frame_power(mixture_stem),
            cfg.silence_floor_db,
            cfg.silence_active_db,
        )

        sample_weight = torch.repeat_interleave(
            silence_frame_weight, SILENCE_FRAME, dim=-1
        )[..., : pred_audio.shape[-1]]
        wave_silence_scale = (
            mixture_stem.abs()
            .mean(dim=(-2, -1), keepdim=True)
            .squeeze(-1)
            .clamp_min(1e-3)
            .detach()
        )
        silence_wave = silence_residue_loss(
            pred_audio[:, 0].abs().mean(dim=1) / wave_silence_scale,
            target_audio[:, 0].abs().mean(dim=1) / wave_silence_scale,
            sample_weight,
            cfg.silence_margin,
        )

        # Mixture-normalized spectral vocal residue during vocal-silent frames:
        # algebraically equivalent to energy-weighted mask residue (|M|*|X|/E[|X|] = |V_est|/E[|X|]),
        # free of 1/|X_c| division spikes on hard-panned stereo or multi-tap deep filters.
        pred_vocal_mag = estimates[:, 0].abs().mean(dim=1)
        true_vocal_mag = target_specs[:, 0].abs().mean(dim=1)
        spec_silence_scale = (
            mixture_spec.abs()
            .mean(dim=(1, 2, 3), keepdim=True)
            .squeeze(1)
            .clamp_min(1e-4)
            .detach()
        )
        mask_frame_weight = F.interpolate(
            silence_frame_weight.unsqueeze(1),
            size=pred_vocal_mag.shape[-1],
            mode="nearest",
        ).squeeze(1)
        silence_mask = silence_residue_loss(
            pred_vocal_mag / spec_silence_scale,
            true_vocal_mag / spec_silence_scale,
            mask_frame_weight[..., None, :],
            cfg.silence_margin,
        )

        total = (
            cfg.waveform_weight * wave_loss
            + cfg.main_stft_weight * main_stft_loss
            + cfg.mrstft_weight * mrstft_loss
            + cfg.sdr_weight * sdr_loss
            + cfg.midside_weight * midside_loss
            + cfg.silence_weight * silence_wave
            + cfg.mask_silence_weight * silence_mask
        )

        with torch.no_grad():
            pred_vocal_rms = pred_audio[:, 0].square().mean(dim=(-2, -1)).sqrt()
            true_vocal_rms = target_audio[:, 0].square().mean(dim=(-2, -1)).sqrt()
            active_segments = true_vocal_rms >= self.activity_threshold
            if active_segments.any():
                vocal_level_db = (
                    20.0
                    * torch.log10(
                        (pred_vocal_rms[active_segments] + 1e-8)
                        / (true_vocal_rms[active_segments] + 1e-8)
                    )
                ).mean()
            else:
                vocal_level_db = pred_audio.new_tensor(0.0)
            vocal_mask_mag = (
                estimates[:, 0].abs().sum(dim=(1, 2, 3))
                / mixture_spec.abs().sum(dim=(1, 2, 3)).clamp_min(1e-6)
            ).mean()

        metrics = {
            "wave": wave_loss.detach(),
            "main_stft": main_stft_loss.detach(),
            "mrstft": mrstft_loss.detach(),
            "silence_wave": silence_wave.detach(),
            "silence_mask": silence_mask.detach(),
            "vocal_level_db": vocal_level_db.detach(),
            "vocal_mask_mag": vocal_mask_mag.detach(),
        }
        return total, metrics


@dataclass(frozen=True)
class AudioInfo:
    path: str
    frames: int
    sample_rate: int

    @property
    def duration(self) -> float:
        return self.frames / self.sample_rate


def find_optional_audio_file(directory: str, stem: str) -> str | None:
    by_lower_name = {name.lower(): name for name in os.listdir(directory)}
    for extension in AUDIO_EXTENSIONS:
        candidate = f"{stem}{extension}"
        actual = by_lower_name.get(candidate.lower())
        if actual is not None:
            return os.path.join(directory, actual)
    return None


def resolve_target_paths(directory: str) -> dict[str, tuple[str, ...]] | None:
    vocals = find_optional_audio_file(directory, "vocals")
    if vocals is None:
        return None

    bass = find_optional_audio_file(directory, "bass")
    drums = find_optional_audio_file(directory, "drums")
    other = find_optional_audio_file(directory, "other")
    if bass is not None and drums is not None and other is not None:
        accompaniment = (bass, drums, other)
    elif bass is not None or drums is not None:
        return None
    else:
        accompaniment_path = (
            find_optional_audio_file(directory, "instrumental")
            or find_optional_audio_file(directory, "accompaniment")
            or other
        )
        if accompaniment_path is None:
            return None
        accompaniment = (accompaniment_path,)
    return {"vocals": (vocals,), "other": accompaniment}


def fft_convolve(signal_tensor: torch.Tensor, impulse: torch.Tensor) -> torch.Tensor:
    total = signal_tensor.shape[-1] + impulse.shape[-1] - 1
    fft_size = 1 << (total - 1).bit_length()
    signal_spec = torch.fft.rfft(signal_tensor, fft_size, dim=-1)
    impulse_spec = torch.fft.rfft(impulse.to(signal_tensor.dtype), fft_size, dim=-1)
    convolved = torch.fft.irfft(
        signal_spec * impulse_spec.unsqueeze(0), fft_size, dim=-1
    )
    return convolved[..., : signal_tensor.shape[-1]]


def make_reverb_impulse(sample_rate: int, max_length: int) -> torch.Tensor:
    length = min(int(random.uniform(0.25, 1.1) * sample_rate), max_length)
    pre_delay = int(random.uniform(0.0, 0.04) * sample_rate)
    tail = max(1, length - pre_delay)
    time = torch.arange(tail, dtype=torch.float32)
    tau = random.uniform(0.07, 0.28) * sample_rate
    impulse = torch.randn(tail, dtype=torch.float32) * torch.exp(-time / tau)
    if pre_delay > 0:
        impulse = torch.cat((torch.zeros(pre_delay, dtype=torch.float32), impulse))
    n = impulse.shape[-1]
    freqs = torch.fft.rfftfreq(n, 1.0 / sample_rate).clamp_min(1.0)
    hp_cutoff = random.uniform(180.0, 340.0)
    lp_cutoff = random.uniform(3_500.0, 8_000.0)
    hp_response = 1.0 / (1.0 + (hp_cutoff / freqs).square())
    lp_response = (1.0 + (freqs / lp_cutoff) ** 2).rsqrt()
    response = hp_response * lp_response
    response[0] = 0.0
    impulse = torch.fft.irfft(torch.fft.rfft(impulse, n) * response, n)
    return impulse / impulse.square().sum().sqrt().clamp_min(1e-8)


def apply_reverb(audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
    impulse = make_reverb_impulse(sample_rate, audio.shape[-1])
    wet = fft_convolve(audio, impulse)
    mix = random.uniform(0.06, 0.18)
    return (1.0 - mix) * audio + mix * wet


def pitch_shift_plan(
    semitones: float,
    target_samples: int,
) -> tuple[int, int, int]:
    """Compute rational resampling factors (orig_freq, new_freq) and pre-load sample
    count so pitched-up crops read enough real forward audio without reflect-padding.
    """
    if abs(semitones) < 0.08:
        return 64, 64, target_samples
    grid = 64
    orig_freq = grid
    new_freq = max(16, int(round(grid * (2.0 ** (-semitones / 12.0)))))
    if new_freq == orig_freq:
        return orig_freq, orig_freq, target_samples
    if new_freq < orig_freq:
        needed = int(math.ceil(target_samples * orig_freq / new_freq)) + 64
    else:
        needed = target_samples
    return orig_freq, new_freq, needed


def apply_rational_pitch_shift(
    audio: torch.Tensor,
    orig_freq: int,
    new_freq: int,
    target_samples: int,
) -> torch.Tensor:
    """Bandlimited Kaiser-windowed sinc varispeed pitch shift with zero time-reversal."""
    if orig_freq != new_freq:
        lead_shape = audio.shape[:-1]
        flat = audio.reshape(-1, audio.shape[-1])
        shifted = torchaudio.functional.resample(
            flat,
            orig_freq=orig_freq,
            new_freq=new_freq,
            lowpass_filter_width=24,
        )
        audio = shifted.reshape(*lead_shape, -1)

    if audio.shape[-1] >= target_samples:
        return audio[..., :target_samples].contiguous()

    pad = target_samples - audio.shape[-1]
    fade_len = min(512, audio.shape[-1] // 8)
    if fade_len > 1:
        phase = torch.linspace(
            0.0, math.pi, fade_len, device=audio.device, dtype=audio.dtype
        )
        ramp = 0.5 * (1.0 + torch.cos(phase))
        audio = audio.clone()
        audio[..., -fade_len:] *= ramp
    return F.pad(audio, (0, pad))


def random_spectral_eq(
    audio: torch.Tensor,
    sample_rate: int,
) -> torch.Tensor:
    """Symmetric zero-phase parametric EQ (low-shelf, bell, high-shelf) without
    any asymmetric vocal high-pass cutoff that would bias <100 Hz fundamentals.
    """
    n = audio.shape[-1]
    if n < 64:
        return audio
    freqs = torch.fft.rfftfreq(n, 1.0 / sample_rate).clamp_min(1.0)
    log_f = torch.log(freqs)
    response = torch.ones_like(freqs)

    def bell(center_hz: float, q: float, gain_db: float) -> None:
        nonlocal response
        sigma = math.log(2.0) / max(q, 0.25)
        response = response * (
            10.0
            ** (
                (gain_db / 20.0)
                * torch.exp(-0.5 * ((log_f - math.log(center_hz)) / sigma) ** 2)
            )
        )

    def high_shelf(center_hz: float, width_decades: float, gain_db: float) -> None:
        nonlocal response
        response = response * (
            10.0
            ** (
                (gain_db / 20.0)
                * 0.5
                * (
                    1.0
                    + torch.tanh(
                        (log_f - math.log(center_hz))
                        / (math.log(10.0) * width_decades)
                    )
                )
            )
        )

    def low_shelf(center_hz: float, width_decades: float, gain_db: float) -> None:
        nonlocal response
        response = response * (
            10.0
            ** (
                (gain_db / 20.0)
                * 0.5
                * (
                    1.0
                    - torch.tanh(
                        (log_f - math.log(center_hz))
                        / (math.log(10.0) * width_decades)
                    )
                )
            )
        )

    for _ in range(random.randint(1, 3)):
        choice = random.random()
        if choice < 0.45:
            bell(
                random.uniform(180.0, 7_500.0),
                random.uniform(0.7, 2.5),
                random.uniform(-3.5, 3.5),
            )
        elif choice < 0.75:
            high_shelf(
                random.uniform(3_500.0, 10_000.0),
                random.uniform(0.5, 1.1),
                random.uniform(-3.5, 3.5),
            )
        else:
            low_shelf(
                random.uniform(70.0, 260.0),
                random.uniform(0.5, 1.0),
                random.uniform(-3.0, 3.0),
            )

    spec = torch.fft.rfft(audio, n, dim=-1)
    filtered_spec = spec * response.to(spec.dtype)
    mid_mask = (freqs >= 200.0) & (freqs <= 8_000.0)
    if mid_mask.any():
        orig_mid = spec[..., mid_mask].abs().square().mean().sqrt().clamp_min(1e-8)
        filt_mid = (
            filtered_spec[..., mid_mask].abs().square().mean().sqrt().clamp_min(1e-8)
        )
        gain = float((orig_mid / filt_mid).clamp(0.75, 1.30))
    else:
        gain = 1.0
    filtered = torch.fft.irfft(filtered_spec, n, dim=-1)
    return filtered * gain


_PINK_NOISE_CACHE: dict[int, torch.Tensor] = {}


def _pink_noise(length: int, sample_rate: int) -> torch.Tensor:
    if length <= 0:
        return torch.zeros(0)
    needed = max(2**20, 4 * length)
    buffer = _PINK_NOISE_CACHE.get(sample_rate)
    if buffer is None or buffer.numel() < needed:
        total = needed
        freqs = torch.fft.rfftfreq(total, 1.0 / sample_rate).clamp_min(1.0)
        amp = 1.0 / freqs.sqrt()
        spec = torch.complex(
            torch.randn(total // 2 + 1, dtype=torch.float32),
            torch.randn(total // 2 + 1, dtype=torch.float32),
        )
        spec = spec * amp
        spec[0] = 0.0
        buffer = torch.fft.irfft(spec, total)
        buffer = buffer / buffer.square().mean().sqrt().clamp_min(1e-8)
        _PINK_NOISE_CACHE.clear()
        _PINK_NOISE_CACHE[sample_rate] = buffer
    offset = random.randint(0, buffer.shape[-1] - length)
    return buffer[offset : offset + length].clone()


def add_noise_floor(audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
    length = audio.shape[-1]
    level_db = random.uniform(-68.0, -50.0)
    level = 10.0 ** (level_db / 20.0) * audio.square().mean().sqrt().clamp_min(1e-8)
    noise_l = _pink_noise(length, sample_rate)
    noise_r = _pink_noise(length, sample_rate)
    stereo_noise = torch.stack((noise_l, noise_r), dim=0).to(
        device=audio.device, dtype=audio.dtype
    )
    return audio + level * stereo_noise


def apply_saturation(audio: torch.Tensor) -> torch.Tensor:
    drive = random.uniform(1.08, 1.35)
    mix = random.uniform(0.15, 0.50)
    wet = torch.tanh(drive * audio) / math.tanh(drive)
    return mix * wet + (1.0 - mix) * audio


def _frame_rms(audio: torch.Tensor, frame: int = 512) -> torch.Tensor:
    power = audio.square().mean(dim=0)
    n = power.shape[-1]
    pad = (-n) % frame
    if pad:
        power = F.pad(power, (0, pad))
    return power.reshape(-1, frame).mean(dim=1).sqrt().clamp_min(1e-8)


def apply_bus_compression(targets: torch.Tensor, frame: int = 512) -> torch.Tensor:
    """Shared time-varying master-bus gain compression preserving exact mixture
    additivity: compress(vocal) + compress(other) == compress(vocal + other).
    """
    mixture = targets.sum(dim=0)
    env = _frame_rms(mixture, frame)
    env_db = 20.0 * torch.log10(env)
    threshold_db = random.uniform(-28.0, -15.0)
    ratio = random.uniform(1.5, 3.5)
    mix = random.uniform(0.25, 0.65)
    gain_db = torch.clamp((env_db - threshold_db) * (1.0 - 1.0 / ratio), max=0.0)
    alpha = random.uniform(0.35, 0.65)
    one_minus_alpha = 1.0 - alpha
    gain_list = gain_db.tolist()
    acc = gain_list[0]
    for i in range(len(gain_list)):
        acc = alpha * acc + one_minus_alpha * gain_list[i]
        gain_list[i] = acc
    smoothed = torch.tensor(gain_list, dtype=gain_db.dtype, device=gain_db.device)
    gain = torch.pow(10.0, (mix * smoothed) / 20.0)
    gain_full = F.interpolate(
        gain.view(1, 1, -1),
        size=mixture.shape[-1],
        mode="linear",
        align_corners=False,
    ).view(-1)
    return targets * gain_full.view(1, 1, -1)


class StemDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        sample_rate: int = 44_100,
        segment_samples: int = 352_800,
        virtual_size: int = 50_000,
        remix_probability: float = 0.75,
        min_activity_rms: float = 1e-4,
    ):
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.segment_samples = segment_samples
        self.segment_seconds = segment_samples / sample_rate
        self.virtual_size = virtual_size
        self.remix_probability = remix_probability
        self.min_activity_rms = min_activity_rms
        self.tracks: list[dict[str, tuple[AudioInfo, ...]]] = []
        self.track_cum_weights: list[float] = []
        self.multistem_other_tracks: list[dict[str, tuple[AudioInfo, ...]]] = []

        track_dirs = [
            os.path.join(root_dir, name)
            for name in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, name))
        ]
        print("Scanning track metadata...")
        running_weight = 0.0
        for track_dir in tqdm(track_dirs, desc="Caching tracks"):
            resolved = resolve_target_paths(track_dir)
            if resolved is None:
                continue
            track: dict[str, tuple[AudioInfo, ...]] = {}
            for stem, paths in resolved.items():
                infos = tuple(sf.info(path) for path in paths)
                track[stem] = tuple(
                    AudioInfo(path, info.frames, info.samplerate)
                    for path, info in zip(paths, infos)
                )
            self.tracks.append(track)
            common_duration = min(
                self._target_duration(track[stem]) for stem in STEMS
            )
            running_weight += max(common_duration, self.segment_seconds) ** 0.75
            self.track_cum_weights.append(running_weight)
            if len(track["other"]) > 1:
                self.multistem_other_tracks.append(track)

        if not self.tracks:
            raise RuntimeError(f"No complete {STEMS} tracks found under {root_dir!r}.")
        print(f"Cached {len(self.tracks)} complete tracks")

    def __len__(self) -> int:
        return self.virtual_size

    def _choose_track(self) -> dict[str, tuple[AudioInfo, ...]]:
        return random.choices(
            self.tracks, cum_weights=self.track_cum_weights, k=1
        )[0]

    def _load_segment(
        self,
        info: AudioInfo,
        start_seconds: float,
        num_samples: int | None = None,
    ) -> torch.Tensor:
        desired_samples = (
            self.segment_samples if num_samples is None else num_samples
        )
        desired_seconds = desired_samples / self.sample_rate
        source_start = int(round(start_seconds * info.sample_rate))
        source_frames = int(math.ceil(desired_seconds * info.sample_rate))
        source_start = max(0, min(source_start, max(0, info.frames - 1)))
        audio_np, source_sr = sf.read(
            info.path,
            start=source_start,
            frames=source_frames,
            dtype="float32",
            always_2d=True,
        )
        audio = torch.from_numpy(audio_np.T)
        audio = torch.nan_to_num(audio)
        if audio.shape[0] == 1:
            audio = audio.repeat(2, 1)
        elif audio.shape[0] > 2:
            audio = audio[:2]

        if source_sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, source_sr, self.sample_rate)

        if audio.shape[-1] < desired_samples:
            audio = F.pad(audio, (0, desired_samples - audio.shape[-1]))
        else:
            audio = audio[..., :desired_samples]
        return audio.contiguous()

    def _load_target(
        self,
        infos: Sequence[AudioInfo],
        start_seconds: float,
        num_samples: int | None = None,
        jitter_components: bool = False,
    ) -> torch.Tensor:
        components = [
            self._load_segment(info, start_seconds, num_samples=num_samples)
            for info in infos
        ]
        if jitter_components and len(components) > 1:
            components = [
                comp * db_to_gain(random.uniform(-2.5, 2.5))
                for comp in components
            ]
        return torch.stack(components).sum(dim=0)

    @staticmethod
    def _target_duration(infos: Sequence[AudioInfo]) -> float:
        return min(info.duration for info in infos)

    def _random_start(
        self,
        infos: Sequence[AudioInfo],
        needed_seconds: float | None = None,
    ) -> float:
        span = self.segment_seconds if needed_seconds is None else needed_seconds
        duration = self._target_duration(infos)
        return random.uniform(0.0, max(0.0, duration - span))

    def _sample_active_vocal(self, semitones: float) -> torch.Tensor:
        orig_f, new_f, needed_samples = pitch_shift_plan(
            semitones, self.segment_samples
        )
        needed_sec = needed_samples / self.sample_rate
        vocal_crop: torch.Tensor | None = None
        for _ in range(4):
            v_track = self._choose_track()
            v_infos = v_track["vocals"]
            raw = self._load_target(
                v_infos,
                self._random_start(v_infos, needed_sec),
                num_samples=needed_samples,
            )
            vocal_crop = apply_rational_pitch_shift(
                raw, orig_f, new_f, self.segment_samples
            )
            if vocal_crop.square().mean().sqrt() >= self.min_activity_rms:
                break
        assert vocal_crop is not None
        return vocal_crop

    def _sample_targets(self) -> torch.Tensor:
        pitch_prob = 0.35

        if random.random() < self.remix_probability:
            v_shift = (
                random.uniform(-2.0, 2.0) if random.random() < pitch_prob else 0.0
            )
            vocal_crop = self._sample_active_vocal(v_shift)

            # Multi-vocal harmony / backing-vocal simulation so overlapping voices
            # are retained in vocals rather than leaked into accompaniment
            if random.random() < 0.15:
                h_shift = (
                    random.uniform(-2.0, 2.0)
                    if random.random() < pitch_prob
                    else 0.0
                )
                harmony = self._sample_active_vocal(h_shift)
                h_gain = db_to_gain(random.uniform(-12.0, -3.5))
                h_pan = random.uniform(-0.35, 0.35)
                harmony = torch.stack(
                    (
                        harmony[0] * math.sqrt(1.0 - h_pan),
                        harmony[1] * math.sqrt(1.0 + h_pan),
                    ),
                    dim=0,
                )
                vocal_crop = vocal_crop + h_gain * harmony

            o_shift = (
                random.uniform(-2.0, 2.0) if random.random() < pitch_prob else 0.0
            )
            orig_f, new_f, needed_samples = pitch_shift_plan(
                o_shift, self.segment_samples
            )
            needed_sec = needed_samples / self.sample_rate
            o_track = self._choose_track()
            o_infos = o_track["other"]

            if len(o_infos) > 1 and len(self.multistem_other_tracks) > 1 and random.random() < 0.6:
                parts: list[torch.Tensor] = []
                for comp_idx in range(len(o_infos)):
                    donor = random.choice(self.multistem_other_tracks)
                    donor_infos = donor["other"]
                    info = donor_infos[min(comp_idx, len(donor_infos) - 1)]
                    comp = self._load_segment(
                        info,
                        self._random_start((info,), needed_sec),
                        num_samples=needed_samples,
                    )
                    parts.append(comp * db_to_gain(random.uniform(-2.5, 2.5)))
                raw_other = torch.stack(parts).sum(dim=0)
            else:
                raw_other = self._load_target(
                    o_infos,
                    self._random_start(o_infos, needed_sec),
                    num_samples=needed_samples,
                    jitter_components=True,
                )
            other_crop = apply_rational_pitch_shift(
                raw_other, orig_f, new_f, self.segment_samples
            )
            return torch.stack((vocal_crop, other_crop))

        shared_shift = (
            random.uniform(-2.0, 2.0) if random.random() < pitch_prob else 0.0
        )
        orig_f, new_f, needed_samples = pitch_shift_plan(
            shared_shift, self.segment_samples
        )
        needed_sec = needed_samples / self.sample_rate

        targets_pair: torch.Tensor | None = None
        for _ in range(3):
            track = self._choose_track()
            common_duration = min(
                self._target_duration(track[stem]) for stem in STEMS
            )
            start = random.uniform(
                0.0, max(0.0, common_duration - needed_sec)
            )
            raw_pair = torch.stack(
                [
                    self._load_target(
                        track[stem], start, num_samples=needed_samples
                    )
                    for stem in STEMS
                ]
            )
            targets_pair = apply_rational_pitch_shift(
                raw_pair, orig_f, new_f, self.segment_samples
            )
            if targets_pair.sum(dim=0).square().mean().sqrt() >= self.min_activity_rms:
                break
        assert targets_pair is not None
        return targets_pair

    @staticmethod
    def _augment(
        targets: torch.Tensor,
        sample_rate: int,
    ) -> torch.Tensor:
        gains_db = torch.empty(targets.shape[0]).uniform_(-6.0, 4.0)
        gains = torch.pow(10.0, gains_db / 20.0).view(-1, 1, 1)
        targets = targets * gains

        for stem_index in range(targets.shape[0]):
            if random.random() < 0.5:
                targets[stem_index] = -targets[stem_index]
            width = (
                random.uniform(0.35, 1.35)
                if stem_index == 0
                else random.uniform(0.70, 1.30)
            )
            mid = (targets[stem_index, 0] + targets[stem_index, 1]) * 0.5
            side = (targets[stem_index, 0] - targets[stem_index, 1]) * 0.5 * width
            # Subtle stereo pan jitter on individual stems before summing to mixture
            pan = random.uniform(-0.12, 0.12)
            left_gain = math.sqrt(1.0 - pan)
            right_gain = math.sqrt(1.0 + pan)
            targets[stem_index, 0] = (mid + side) * left_gain
            targets[stem_index, 1] = (mid - side) * right_gain

        if random.random() < 0.5:
            targets = targets.flip(dims=(1,))

        # Clean, unmodified low-end preservation: NO unconditional vocal high-pass!
        vocal = targets[0]
        other_rms = targets[1].square().mean().sqrt()
        if random.random() < 0.03 and other_rms > 2e-3:
            vocal = torch.zeros_like(vocal)
        elif random.random() < 0.14:
            vocal = vocal.clone()
            samples = vocal.shape[-1]
            for _ in range(random.randint(1, 2)):
                span = random.randint(max(64, samples // 40), max(65, samples // 3))
                start = random.randint(0, max(0, samples - span))
                end = min(samples, start + span)
                ramp = min(1024, max(0, (end - start) // 8))
                envelope = vocal.new_ones(samples)
                envelope[start:end] = 0.0
                if ramp > 1:
                    fade = torch.linspace(1.0, 0.0, ramp, device=vocal.device)
                    left = max(0, start - ramp)
                    if left < start:
                        envelope[left:start] = torch.minimum(
                            envelope[left:start], fade[-(start - left) :]
                        )
                    right = min(samples, end + ramp)
                    if end < right:
                        envelope[end:right] = torch.minimum(
                            envelope[end:right], fade[: right - end].flip(0)
                        )
                vocal.mul_(envelope)

        if random.random() < 0.20:
            vocal = random_spectral_eq(vocal, sample_rate)
        if random.random() < 0.10:
            vocal = apply_saturation(vocal)
        if random.random() < 0.10:
            vocal = apply_reverb(vocal, sample_rate)
        targets[0] = vocal

        other = targets[1]
        if random.random() < 0.20:
            other = random_spectral_eq(other, sample_rate)
        if random.random() < 0.10:
            other = apply_saturation(other)
        if random.random() < 0.10:
            other = apply_reverb(other, sample_rate)
        if random.random() < 0.08:
            other = add_noise_floor(other, sample_rate)
        targets[1] = other

        if random.random() < 0.16:
            targets = apply_bus_compression(targets)

        global_gain = db_to_gain(random.uniform(-4.0, 3.0))
        targets = targets * global_gain
        peak = targets.sum(dim=0).abs().amax()
        if peak > 1.0:
            targets = targets * (0.98 / peak)
        return targets

    def __getitem__(self, _: int) -> tuple[torch.Tensor, torch.Tensor]:
        last_error: Exception | None = None
        for _attempt in range(20):
            try:
                targets = self._augment(
                    self._sample_targets(), self.sample_rate
                )
                mixture = targets.sum(dim=0)
                if mixture.square().mean().sqrt() < self.min_activity_rms:
                    continue
                return mixture, targets
            except Exception as error:
                last_error = error
        raise RuntimeError(f"Unable to load a valid training example: {last_error}")


class WeightedDataset(Dataset):
    def __init__(
        self,
        datasets: Sequence[Dataset],
        weights: Sequence[float],
    ):
        if not datasets:
            raise ValueError("WeightedDataset needs at least one source dataset.")
        if len(datasets) != len(weights):
            raise ValueError(
                f"Got {len(datasets)} datasets but {len(weights)} weights."
            )
        if any(weight <= 0.0 for weight in weights):
            raise ValueError("WeightedDataset weights must be positive.")
        self.datasets = list(datasets)
        total = float(sum(weights))
        self.probabilities = [float(weight) / total for weight in weights]
        first = self.datasets[0]
        self.sample_rate = first.sample_rate
        self.segment_samples = first.segment_samples

    def __len__(self) -> int:
        return sum(len(dataset) for dataset in self.datasets)

    def __getitem__(self, _: int) -> tuple[torch.Tensor, torch.Tensor]:
        choice = random.choices(
            range(len(self.datasets)), weights=self.probabilities, k=1
        )[0]
        dataset = self.datasets[choice]
        return dataset[random.randrange(len(dataset))]


def parse_weighted_dirs(args: argparse.Namespace) -> list[tuple[str, float]]:
    if args.data_dir:
        return [(args.data_dir, 1.0)]
    dirs = [part.strip() for part in args.data_dirs.split(",") if part.strip()]
    raw_weights = [
        part.strip() for part in args.data_weights.split(",") if part.strip()
    ]
    if len(dirs) != len(raw_weights):
        raise ValueError(
            f"Got {len(dirs)} --data_dirs entries but {len(raw_weights)} "
            "--data_weights entries; the counts must match."
        )
    try:
        weights = [float(text) for text in raw_weights]
    except ValueError:
        raise ValueError(
            "--data_weights must be comma-separated numbers, got "
            f"{args.data_weights!r}."
        ) from None
    if any(weight <= 0.0 for weight in weights):
        raise ValueError("--data_weights entries must be positive.")
    return list(zip(dirs, weights))


def build_training_dataset(
    args: argparse.Namespace,
    config: ModelConfig,
) -> Dataset:
    sources = parse_weighted_dirs(args)
    stem_datasets = [
        StemDataset(
            root_dir=root_dir,
            sample_rate=config.sample_rate,
            segment_samples=args.segment_samples,
            virtual_size=args.dataset_size,
            remix_probability=args.remix_probability,
        )
        for root_dir, _weight in sources
    ]
    if len(stem_datasets) == 1:
        print(f"Training from data directory: {sources[0][0]}")
        return stem_datasets[0]
    total_weight = sum(weight for _root_dir, weight in sources)
    summary = ", ".join(
        f"{root_dir} (weight {weight:g}, "
        f"{weight / total_weight * 100.0:.1f}% of samples)"
        for root_dir, weight in sources
    )
    print(f"Training from weighted data sources: {summary}")
    return WeightedDataset(
        stem_datasets,
        [weight for _root_dir, weight in sources],
    )


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9995):
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must be in [0, 1).")
        self.model = model
        self.decay = decay
        self.updates = 0
        self.shadow = {
            name: parameter.detach().float().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.backup: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self) -> None:
        self.updates += 1
        warm_decay = (1.0 + self.updates) / (10.0 + self.updates)
        decay = min(self.decay, warm_decay)
        trainable = [
            (self.shadow[name], parameter.detach().float())
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        ]
        if trainable:
            shadow_tensors, parameter_tensors = zip(*trainable)
            torch._foreach_lerp_(shadow_tensors, parameter_tensors, 1.0 - decay)

    @torch.no_grad()
    def apply_shadow(self) -> None:
        if self.backup:
            raise RuntimeError("EMA shadow weights are already applied.")
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad:
                self.backup[name] = parameter.detach().clone()
                parameter.copy_(self.shadow[name].to(dtype=parameter.dtype))

    @torch.no_grad()
    def restore(self) -> None:
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad and name in self.backup:
                parameter.copy_(self.backup[name])
        self.backup = {}

    @contextlib.contextmanager
    def average_parameters(self):
        self.apply_shadow()
        try:
            yield
        finally:
            self.restore()

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.shadow

    def load_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        updates: int = 0,
    ) -> StateLoadReport:
        report = load_matching_state_dict(_EMAStateView(self.shadow), state_dict)
        self.updates = max(0, int(updates))
        return report


class _EMAStateView(nn.Module):
    def __init__(self, shadow: dict[str, torch.Tensor]):
        super().__init__()
        self.shadow = shadow

    def state_dict(self, *args, **kwargs):
        del args, kwargs
        return self.shadow.copy()

    def load_state_dict(self, state_dict, strict=False):
        del strict
        for name, value in state_dict.items():
            self.shadow[name].copy_(value.float())
        return None


NO_DECAY_KEYWORDS = ("bias", "norm", "gamma", "gate")


def split_decay_parameters(
    model: nn.Module,
    weight_decay: float,
) -> list[dict]:
    decay_params: list[nn.Parameter] = []
    no_decay_params: list[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lower = name.lower()
        if (
            param.ndim < 2
            or lower.endswith("_b")
            or any(token in lower for token in NO_DECAY_KEYWORDS)
        ):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def build_optimizer(
    model: nn.Module,
    weight_decay: float,
    slice_p: int,
) -> torch.optim.Optimizer:
    from prodigyopt import Prodigy

    param_groups = split_decay_parameters(model, weight_decay)
    return Prodigy(
        param_groups,
        lr=1.0,
        betas=(0.9, 0.98),
        weight_decay=weight_decay,
        slice_p=slice_p,
    )


def peek_checkpoint_step(checkpoint_path: str | None) -> int:
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        return 0
    match = re.search(r"step_(\d+)", Path(checkpoint_path).name)
    if match:
        return int(match.group(1))
    try:
        data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        try:
            return int(data.get("step", 0))
        finally:
            del data
    except Exception:
        return 0


def find_latest_checkpoint(folder: str = "ckpts") -> str | None:
    paths = list(Path(folder).glob("checkpoint_step_*.pt"))
    if not paths:
        return None

    def step(path: Path) -> int:
        match = re.search(r"step_(\d+)", path.name)
        return int(match.group(1)) if match else 0

    return str(max(paths, key=step))


def find_latest_compatible_checkpoint(
    config: ModelConfig,
    folder: str = "ckpts",
) -> str | None:
    paths = list(Path(folder).glob("checkpoint_step_*.pt"))

    def step(path: Path) -> int:
        match = re.search(r"step_(\d+)", path.name)
        return int(match.group(1)) if match else 0

    for path in sorted(paths, key=step, reverse=True):
        try:
            checkpoint_data = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as error:
            print(f"Ignoring unreadable checkpoint {path}: {error}")
            continue
        try:
            if (
                checkpoint_data.get("checkpoint_format_version", 0)
                >= CHECKPOINT_FORMAT_VERSION
                and model_configs_compatible(checkpoint_data.get("model_config"), config)
            ):
                return str(path)
        finally:
            del checkpoint_data
            gc.collect()
    return None


def checkpoint_sdr_from_path(path: str | Path) -> float | None:
    match = re.search(r"sdr_(-?\d+(?:\.\d+)?)\.pt$", Path(path).name)
    return float(match.group(1)) if match else None


def find_best_checkpoint(
    folder: str = "best_ckpts",
    config: ModelConfig | None = None,
    validation_metric: str | None = None,
) -> str | None:
    scored: list[tuple[float, Path]] = []
    for path in Path(folder).glob("*.pt"):
        score = checkpoint_sdr_from_path(path)
        if score is None:
            continue
        if config is not None or validation_metric is not None:
            try:
                checkpoint_data = torch.load(
                    path, map_location="cpu", weights_only=False
                )
            except Exception as error:
                print(f"Ignoring unreadable best checkpoint {path}: {error}")
                continue
            try:
                if config is not None and (
                    checkpoint_data.get("checkpoint_format_version", 0)
                    < CHECKPOINT_FORMAT_VERSION
                    or not model_configs_compatible(
                        checkpoint_data.get("model_config"), config
                    )
                ):
                    continue
                if (
                    validation_metric is not None
                    and checkpoint_data.get("validation_metric") != validation_metric
                ):
                    continue
            finally:
                del checkpoint_data
                gc.collect()
        scored.append((score, path))
    return str(max(scored, key=lambda item: item[0])[1]) if scored else None


def save_checkpoint(
    path: str,
    model: ResonantConvGateSeparator,
    ema: EMA,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    step: int,
    best_sdr: float,
    avg_loss: float,
    extra_metrics: dict | None = None,
) -> None:
    payload = {
        "step": step,
        "model_state_dict": clean_state_dict(model.state_dict()),
        "ema_state_dict": clean_state_dict(ema.state_dict()),
        "ema_updates": ema.updates,
        "optimizer_state_dict": optimizer.state_dict(),
        "optimizer_class": optimizer.__class__.__name__,
        "scaler_state_dict": scaler.state_dict(),
        "best_sdr": best_sdr,
        "validation_metric": VALIDATION_METRIC,
        "avg_loss": avg_loss,
        "stems": STEMS,
        "model_config": asdict(model.config),
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
    }
    if extra_metrics:
        payload.update(extra_metrics)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def prune_old_checkpoints(
    folder: str,
    keep: int = 3,
    config: ModelConfig | None = None,
    by_sdr: bool = False,
) -> None:
    if keep <= 0:
        return
    paths: list[Path] = []
    for path in Path(folder).glob("*.pt"):
        if config is not None:
            try:
                checkpoint_data = torch.load(
                    path, map_location="cpu", weights_only=False
                )
            except Exception:
                continue
            try:
                if (
                    checkpoint_data.get("checkpoint_format_version", 0)
                    < CHECKPOINT_FORMAT_VERSION
                    or not model_configs_compatible(
                        checkpoint_data.get("model_config"), config
                    )
                ):
                    continue
            finally:
                del checkpoint_data
                gc.collect()
        paths.append(path)

    def sort_key(path: Path) -> tuple[float, int, float]:
        match = re.search(r"step_(\d+)", path.name)
        step_num = int(match.group(1)) if match else 0
        mtime = path.stat().st_mtime
        if by_sdr:
            sdr = checkpoint_sdr_from_path(path)
            return (sdr if sdr is not None else -float("inf"), step_num, mtime)
        return (0.0, step_num, mtime)

    paths.sort(key=sort_key)
    if len(paths) > keep:
        for path in paths[:-keep]:
            path.unlink(missing_ok=True)


def crossfade_window(
    length: int,
    overlap: int,
    fade_in: bool,
    fade_out: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    window = torch.ones(length, device=device, dtype=dtype)
    fade_length = min(overlap, length)
    if fade_length <= 0:
        return window
    phase = torch.linspace(0.0, math.pi / 2.0, fade_length, device=device, dtype=dtype)
    fade = torch.sin(phase).square()
    if fade_in:
        window[:fade_length] *= fade
    if fade_out:
        window[-fade_length:] *= fade.flip(0)
    return window


def chunk_starts(total_length: int, chunk_size: int, overlap: int) -> list[int]:
    if total_length <= chunk_size:
        return [0]
    step = chunk_size - overlap
    if step <= 0:
        raise ValueError("Overlap must be smaller than chunk size.")
    starts = list(range(0, total_length - chunk_size + 1, step))
    final_start = total_length - chunk_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


@torch.inference_mode()
def separate_tensor(
    model: ResonantConvGateSeparator,
    mixture: torch.Tensor,
    chunk_size: int,
    overlap: int,
    device: torch.device,
    precision: str = "bf16",
    show_progress: bool = False,
    mask_deadzone: float = 0.0,
) -> list[torch.Tensor]:
    if mixture.ndim != 2:
        raise ValueError("Mixture must be [channels, samples].")
    if mixture.shape[0] == 1:
        mixture = mixture.repeat(2, 1)
    if mixture.shape[0] != model.config.audio_channels:
        raise ValueError(
            f"Expected {model.config.audio_channels} channels, got {mixture.shape[0]}."
        )

    mixture = mixture.to(device=device, dtype=torch.float32)
    orig_length = mixture.shape[-1]
    # Reflect-pad multi-chunk tracks so song start and end sit inside bidirectional context
    edge_pad = (
        min(overlap // 2, orig_length - 1)
        if orig_length > chunk_size and overlap > 0
        else 0
    )
    if edge_pad > 0:
        padded_mix = F.pad(mixture.unsqueeze(0), (edge_pad, edge_pad), mode="reflect").squeeze(0)
    else:
        padded_mix = mixture

    total_length = padded_mix.shape[-1]
    starts = chunk_starts(total_length, chunk_size, overlap)
    output = torch.zeros(
        model.config.num_stems,
        model.config.audio_channels,
        total_length,
        device=device,
        dtype=torch.float32,
    )
    weight_sum = torch.zeros(total_length, device=device, dtype=torch.float32)
    stft_window = torch.hann_window(model.config.win_length, device=device)

    iterator: Iterable[int] = starts
    if show_progress:
        iterator = tqdm(starts, desc="Separating", leave=False)

    for start in iterator:
        usable = min(chunk_size, total_length - start)
        chunk = padded_mix[:, start : start + usable]
        if usable < chunk_size:
            pad = chunk_size - usable
            if usable > 1:
                reflect = min(pad, usable - 1)
                chunk = F.pad(chunk.unsqueeze(0), (0, reflect), mode="reflect").squeeze(0)
                if reflect < pad:
                    chunk = F.pad(chunk, (0, pad - reflect))
            else:
                chunk = F.pad(chunk, (0, pad))

        spec = make_stft(
            chunk.unsqueeze(0),
            n_fft=model.config.n_fft,
            hop_length=model.config.hop_length,
            win_length=model.config.win_length,
            window=stft_window,
        )
        with autocast_context(device, precision):
            estimated_specs = model.estimate_specs(
                spec, mask_deadzone=mask_deadzone
            )

        estimated = make_istft(
            estimated_specs,
            length=chunk_size,
            n_fft=model.config.n_fft,
            hop_length=model.config.hop_length,
            win_length=model.config.win_length,
            window=stft_window,
        ).squeeze(0)

        is_first = start == 0
        is_last = start + chunk_size >= total_length
        window = crossfade_window(
            chunk_size,
            overlap,
            fade_in=not is_first,
            fade_out=not is_last,
            device=device,
            dtype=estimated.dtype,
        )[:usable]
        output[..., start : start + usable] += estimated[..., :usable] * window
        weight_sum[start : start + usable] += window

    output = output / weight_sum.clamp_min(1e-8)
    if edge_pad > 0:
        output = output[..., edge_pad : edge_pad + orig_length]

    if mask_deadzone <= 0.0:
        residual = mixture - output.sum(dim=0)
        output[1] = output[1] + residual
    return [output[index] for index in range(model.config.num_stems)]


def calculate_track_sdr(
    prediction: torch.Tensor,
    target: torch.Tensor,
    accumulation_samples: int,
    activity_threshold: float = 1e-4,
) -> float | None:
    signal_power = torch.zeros((), device=target.device, dtype=torch.float64)
    error_power = torch.zeros((), device=target.device, dtype=torch.float64)
    for start in range(0, target.shape[-1], accumulation_samples):
        end = min(start + accumulation_samples, target.shape[-1])
        target_block = target[..., start:end].double()
        prediction_block = prediction[..., start:end].double()
        signal_power += target_block.square().sum()
        error_power += (prediction_block - target_block).square().sum()

    target_rms = (signal_power / target.numel()).sqrt()
    if target_rms < activity_threshold:
        return None
    score = float(
        10.0 * torch.log10((signal_power + 1e-12) / (error_power + 1e-12))
    )
    return score if math.isfinite(score) else None


@torch.inference_mode()
def calculate_vocal_residue_db(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mixture: torch.Tensor,
    frame: int = SILENCE_FRAME,
    active_db: float = -45.0,
) -> float | None:
    pred_power = _stem_frame_power(prediction, frame)
    target_power = _stem_frame_power(target, frame)
    mixture_power = _stem_frame_power(mixture, frame)
    weight = vocal_silence_weight(
        target_power, mixture_power, floor_db=-65.0, active_db=active_db
    )
    total_weight = weight.sum()
    if float(total_weight) <= 0.0:
        return None
    ratio_db = 10.0 * torch.log10(
        (pred_power + 1e-10) / (mixture_power + 1e-10)
    ).clamp_min(-120.0)
    return float((ratio_db * weight).sum() / total_weight)


@torch.inference_mode()
def calculate_other_spill_db(
    prediction: torch.Tensor,
    target: torch.Tensor,
    vocal: torch.Tensor,
    mixture: torch.Tensor,
    frame: int = SILENCE_FRAME,
    active_db: float = -25.0,
) -> float | None:
    mixture_power = _stem_frame_power(mixture, frame)
    audible = (
        mixture_power > 1e-4 * mixture_power.amax().clamp_min(1e-12)
    ).to(mixture_power.dtype)
    active = (
        1.0
        - vocal_silence_weight(
            _stem_frame_power(vocal, frame),
            mixture_power,
            floor_db=-60.0,
            active_db=active_db,
        )
    ) * audible
    total_weight = active.sum()
    if float(total_weight) <= 0.0:
        return None
    excess = (
        _stem_frame_power(prediction, frame) - _stem_frame_power(target, frame)
    ).clamp_min(0.0)
    ratio = (excess * active).sum() / ((mixture_power * active).sum() + 1e-10)
    return float(10.0 * torch.log10(ratio.clamp_min(1e-10)))


@torch.inference_mode()
def validate(
    model: ResonantConvGateSeparator,
    test_dir: str,
    device: torch.device,
    chunk_size: int,
    overlap: int,
    precision: str,
    mask_deadzone: float = 0.0,
) -> tuple[
    list[float], float | None, float | None, float | None, float | None
]:
    model.eval()
    track_dirs = (
        [
            os.path.join(test_dir, name)
            for name in sorted(os.listdir(test_dir))
            if os.path.isdir(os.path.join(test_dir, name))
        ]
        if os.path.isdir(test_dir)
        else []
    )
    if not track_dirs:
        print(f"No validation tracks found under {test_dir!r}.")
        return [0.0 for _ in STEMS], None, None, None, None

    per_stem_track_scores: list[list[float]] = [[] for _ in STEMS]
    residue_scores: list[float] = []
    target_residue_scores: list[float] = []
    spill_scores: list[float] = []
    valid_tracks = 0
    progress = tqdm(track_dirs, desc="Validating", leave=False)

    for track_dir in progress:
        try:
            resolved = resolve_target_paths(track_dir)
            if resolved is None:
                raise FileNotFoundError(
                    "Track does not contain a complete two-stem or MUSDB target set."
                )
            targets: list[torch.Tensor] = []
            for stem in STEMS:
                components: list[torch.Tensor] = []
                for path in resolved[stem]:
                    audio_np, sample_rate = sf.read(
                        path,
                        dtype="float32",
                        always_2d=True,
                    )
                    audio = torch.nan_to_num(torch.from_numpy(audio_np.T))
                    if audio.shape[0] == 1:
                        audio = audio.repeat(2, 1)
                    audio = audio[:2]
                    if sample_rate != model.config.sample_rate:
                        audio = torchaudio.functional.resample(
                            audio, sample_rate, model.config.sample_rate
                        )
                    components.append(audio)
                component_length = min(audio.shape[-1] for audio in components)
                targets.append(
                    torch.stack(
                        [audio[..., :component_length] for audio in components]
                    ).sum(dim=0)
                )

            length = min(target.shape[-1] for target in targets)
            targets = [target[..., :length].to(device) for target in targets]
            mixture = torch.stack(targets).sum(dim=0)
            predictions = separate_tensor(
                model,
                mixture,
                chunk_size=chunk_size,
                overlap=overlap,
                device=device,
                precision=precision,
                show_progress=False,
                mask_deadzone=mask_deadzone,
            )
            scores = [
                calculate_track_sdr(
                    pred,
                    target,
                    accumulation_samples=chunk_size,
                )
                for pred, target in zip(predictions, targets)
            ]
            residue = calculate_vocal_residue_db(
                predictions[0], targets[0], mixture
            )
            target_residue = calculate_vocal_residue_db(
                targets[0], targets[0], mixture
            )
            spill = calculate_other_spill_db(
                predictions[1], targets[1], targets[0], mixture
            )
            if residue is not None:
                residue_scores.append(residue)
            if target_residue is not None:
                target_residue_scores.append(target_residue)
            if spill is not None:
                spill_scores.append(spill)
            for index, score in enumerate(scores):
                if score is not None:
                    per_stem_track_scores[index].append(score)
            valid_tracks += 1
            leak_text = f" | leak: {residue:.1f} dB" if residue is not None else ""
            if spill is not None:
                leak_text += f" | spill: {spill:.1f} dB"
            progress.set_postfix_str(
                " | ".join(
                    f"{stem}: {score:.3f}" if score is not None else f"{stem}: inactive"
                    for stem, score in zip(STEMS, scores)
                )
                + leak_text
            )
            del predictions, mixture, targets
        except Exception as error:
            print(f"\nSkipping {track_dir}: {error}")

    if valid_tracks == 0 or not any(per_stem_track_scores):
        return [0.0 for _ in STEMS], None, None, None, None
    means = [
        sum(scores) / len(scores) if scores else float("nan")
        for scores in per_stem_track_scores
    ]
    all_scores = [
        score
        for scores in per_stem_track_scores
        for score in scores
        if math.isfinite(score)
    ]
    combined = sum(all_scores) / len(all_scores) if all_scores else None
    residue_mean = (
        sum(residue_scores) / len(residue_scores) if residue_scores else None
    )
    target_residue_mean = (
        sum(target_residue_scores) / len(target_residue_scores)
        if target_residue_scores
        else None
    )
    spill_mean = sum(spill_scores) / len(spill_scores) if spill_scores else None
    return means, combined, residue_mean, target_residue_mean, spill_mean


def train(
    model: ResonantConvGateSeparator,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_module: SeparationLoss,
    device: torch.device,
    args: argparse.Namespace,
    checkpoint_path: str | None,
) -> None:
    model.to(device)
    loss_module.to(device)
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=device.type == "cuda" and args.precision == "fp16",
    )
    step = 0
    best_sdr = -float("inf")
    avg_loss = 0.0
    checkpoint_data: dict | None = None

    if checkpoint_path:
        checkpoint_data = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        saved_config = checkpoint_data.get("model_config")
        if not model_configs_compatible(saved_config, model.config):
            raise RuntimeError(
                "Checkpoint architecture mismatch: saved model_config does not "
                "match current ModelConfig."
            )
        model_state = checkpoint_data.get("model_state_dict", checkpoint_data)
        report = load_matching_state_dict(model, model_state)
        if not report.is_exact:
            raise RuntimeError(
                "Checkpoint does not exactly match the model: "
                f"matched {report.matched}/{report.expected} tensors, "
                f"{len(report.skipped)} incoming tensors incompatible, "
                f"{len(report.missing)} required tensors missing."
            )
        print(f"Loaded all {report.matched} model tensors from {checkpoint_path}.")

    ema = EMA(model, decay=args.ema_decay)

    if checkpoint_data is not None:
        if "ema_state_dict" in checkpoint_data:
            ema_report = ema.load_state_dict(
                checkpoint_data["ema_state_dict"],
                updates=int(checkpoint_data.get("ema_updates", 0)),
            )
            if not ema_report.is_exact:
                raise RuntimeError(
                    "Exact model checkpoint has an incomplete EMA state: "
                    f"{ema_report.matched}/{ema_report.expected} tensors matched."
                )
            print(f"Loaded complete EMA state at update {ema.updates}.")

        step = int(checkpoint_data.get("step", 0))
        best_sdr = float(checkpoint_data.get("best_sdr", best_sdr))
        avg_loss = float(checkpoint_data.get("avg_loss", 0.0))

        expected_opt_class = optimizer.__class__.__name__
        if args.reset_optimizer:
            print(
                f"Loaded exact weights; --reset_optimizer starts a fresh "
                f"{expected_opt_class} while keeping EMA and step={step}."
            )
        else:
            saved_optimizer_class = checkpoint_data.get("optimizer_class")
            if saved_optimizer_class != expected_opt_class:
                raise RuntimeError(
                    f"Checkpoint optimizer is {saved_optimizer_class!r}, expected "
                    f"{expected_opt_class!r}. Pass --reset_optimizer to switch optimizers."
                )
            if "optimizer_state_dict" not in checkpoint_data:
                raise RuntimeError("Continuation checkpoint has no optimizer state.")
            optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
            for group in optimizer.param_groups:
                group["lr"] = 1.0
            if len(optimizer.param_groups) >= 2:
                optimizer.param_groups[0]["weight_decay"] = args.weight_decay
                optimizer.param_groups[1]["weight_decay"] = 0.0

        saved_scaler_state = checkpoint_data.get("scaler_state_dict")
        if scaler.is_enabled() and saved_scaler_state:
            scaler.load_state_dict(saved_scaler_state)
        del checkpoint_data
        checkpoint_data = None
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        best_checkpoint = find_best_checkpoint(
            "best_ckpts",
            model.config,
            validation_metric=VALIDATION_METRIC,
        )
        best_checkpoint_sdr = (
            checkpoint_sdr_from_path(best_checkpoint)
            if best_checkpoint is not None
            else None
        )
        if best_checkpoint_sdr is not None and best_checkpoint_sdr > best_sdr:
            best_sdr = best_checkpoint_sdr
            print(
                f"Recovered newer best SDR {best_sdr:.4f} dB from {best_checkpoint}."
            )
        optimizer_status = "Fresh" if args.reset_optimizer else "Resuming"
        print(
            f"{optimizer_status} {optimizer.__class__.__name__} at step {step} "
            f"(lr=1.0, weight_decay={args.weight_decay:.2e}, grad_accum={args.grad_accumulation})."
        )

    if args.compile:
        model.encoder.compile_layers(mode="default")
        print(
            f"Compiled {len(model.encoder.time_layers) + len(model.encoder.freq_layers)} "
            "Gated-RoPE axial blocks (FlashAttention + SwiGLU, zero FFT graph breaks)."
        )

    stft_window = torch.hann_window(model.config.win_length, device=device)
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(
        initial=step,
        total=args.max_steps if args.max_steps > 0 else None,
        dynamic_ncols=True,
        disable=None,
        mininterval=0.5,
        bar_format="{desc} | {n_fmt} steps [{elapsed}, {rate_fmt}{postfix}]",
    )
    data_iterator = iter(dataloader)
    stop_requested = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stop_requested
        if stop_requested:
            raise KeyboardInterrupt
        stop_requested = True
        progress.write(
            "Stop requested; finishing the current optimizer step and saving. "
            "Press Ctrl+C again to force an immediate exit."
        )

    previous_sigint_handler = signal.signal(signal.SIGINT, request_stop)

    while not stop_requested:
        if args.max_steps > 0 and step >= args.max_steps:
            progress.write(f"Reached --max_steps={args.max_steps}; finishing training.")
            break

        model.train()
        accumulated_loss_tensor = torch.zeros((), device=device)
        latest_metrics: dict[str, torch.Tensor] = {}
        completed_accumulation = True

        for _micro_step in range(args.grad_accumulation):
            try:
                mixture_audio, target_audio = next(data_iterator)
            except StopIteration:
                data_iterator = iter(dataloader)
                mixture_audio, target_audio = next(data_iterator)

            mixture_audio = mixture_audio.to(device, non_blocking=True)
            target_audio = target_audio.to(device, non_blocking=True)
            mixture_spec = make_stft(
                mixture_audio,
                n_fft=model.config.n_fft,
                hop_length=model.config.hop_length,
                win_length=model.config.win_length,
                window=stft_window,
            )

            with autocast_context(device, args.precision):
                estimates = model.estimate_specs(
                    mixture_spec, mask_deadzone=0.0
                )
            loss, latest_metrics = loss_module.compute_from_estimates(
                estimates,
                mixture_spec,
                target_audio,
                mixture_audio,
            )
            scaled_loss = loss / args.grad_accumulation

            if not torch.isfinite(loss):
                print(f"Non-finite loss at step {step}; discarding gradients.")
                optimizer.zero_grad(set_to_none=True)
                accumulated_loss_tensor.zero_()
                completed_accumulation = False
                break

            scaler.scale(scaled_loss).backward()
            accumulated_loss_tensor.add_(loss.detach())
            del estimates, loss, scaled_loss

        if not completed_accumulation:
            continue

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.grad_clip, error_if_nonfinite=False
        )
        if not torch.isfinite(grad_norm):
            print(f"Non-finite gradient norm at step {step}; skipping optimizer step.")
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
            continue

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        ema.update()

        step += 1
        accumulated_loss = float(
            accumulated_loss_tensor.div_(args.grad_accumulation)
        )
        avg_loss = (
            accumulated_loss
            if step == 1
            else 0.995 * avg_loss + 0.005 * accumulated_loss
        )
        current_lr = (
            optimizer.param_groups[0].get("d", 1.0) * optimizer.param_groups[0]["lr"]
        )
        progress.set_description(
            f"Step {step} | loss {accumulated_loss:.4f} | avg {avg_loss:.4f} "
            f"| lr {current_lr:.2e} | grad {float(grad_norm):.2f} "
            f"| best {best_sdr:.4f}",
            refresh=False,
        )
        if latest_metrics:
            (
                wave,
                main_stft,
                mrstft,
                vocal_db,
                mask_mag,
                residue_wave,
                residue_mask,
            ) = torch.stack(
                (
                    latest_metrics["wave"],
                    latest_metrics["main_stft"],
                    latest_metrics["mrstft"],
                    latest_metrics["vocal_level_db"],
                    latest_metrics["vocal_mask_mag"],
                    latest_metrics["silence_wave"],
                    latest_metrics["silence_mask"],
                )
            ).float().cpu().tolist()
            progress.set_postfix(
                wave=f"{wave:.3f}",
                stft=f"{main_stft:.3f}",
                mr=f"{mrstft:.3f}",
                vdb=f"{vocal_db:+.1f}",
                vmask=f"{mask_mag:.3f}",
                sil=f"{residue_wave:.3f}/{residue_mask:.3f}",
                refresh=False,
            )
        progress.update(1)

        if step % args.checkpoint_steps == 0:
            regular_path = f"ckpts/checkpoint_step_{step}.pt"
            save_checkpoint(
                regular_path,
                model,
                ema,
                optimizer,
                scaler,
                step,
                best_sdr,
                avg_loss,
            )
            prune_old_checkpoints("ckpts", keep=3, config=model.config)

            with ema.average_parameters():
                compiler_context = (
                    torch.compiler.set_stance("force_eager")
                    if args.compile and hasattr(torch.compiler, "set_stance")
                    else contextlib.nullcontext()
                )
                with compiler_context:
                    (
                        stem_scores,
                        combined_sdr,
                        vocal_residue_db,
                        target_residue_db,
                        other_spill_db,
                    ) = validate(
                        model,
                        args.test_dir,
                        device,
                        chunk_size=args.segment_samples,
                        overlap=args.inference_overlap,
                        precision=args.precision,
                        mask_deadzone=0.0,
                    )

            previous_best = best_sdr
            improved = combined_sdr is not None and combined_sdr > previous_best
            if combined_sdr is None:
                print("\nValidation produced no valid tracks; best SDR was not changed.")
            else:
                score_text = ", ".join(
                    f"{stem}: {score:.4f} dB"
                    for stem, score in zip(STEMS, stem_scores)
                )
                residue_text = (
                    f", vocal residue: {vocal_residue_db:.1f} dB"
                    if vocal_residue_db is not None
                    else ""
                )
                if target_residue_db is not None:
                    residue_text += f" (target {target_residue_db:.1f} dB)"
                if other_spill_db is not None:
                    residue_text += f", instrumental spill: {other_spill_db:.1f} dB"
                print(
                    f"\nValidation step {step} "
                    f"(EMA, mean full-track SDR): "
                    f"{score_text}, combined: {combined_sdr:.4f} dB"
                    f"{residue_text}"
                )
                if improved:
                    best_sdr = combined_sdr

            if improved:
                save_checkpoint(
                    regular_path,
                    model,
                    ema,
                    optimizer,
                    scaler,
                    step,
                    best_sdr,
                    avg_loss,
                )

            if improved and combined_sdr is not None:
                best_path = (
                    f"best_ckpts/checkpoint_step_{step}_sdr_{combined_sdr:.4f}.pt"
                )
                save_checkpoint(
                    best_path,
                    model,
                    ema,
                    optimizer,
                    scaler,
                    step,
                    best_sdr,
                    avg_loss,
                    extra_metrics={
                        "vocal_residue_db": vocal_residue_db,
                        "target_residue_db": target_residue_db,
                        "other_spill_db": other_spill_db,
                        "mask_deadzone": args.mask_deadzone,
                    },
                )
                prune_old_checkpoints(
                    "best_ckpts", keep=2, config=model.config, by_sdr=True
                )
                print(f"New best checkpoint: {best_path}\n")
            elif combined_sdr is not None:
                print(f"Best combined SDR remains {best_sdr:.4f} dB.\n")
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    progress.close()
    signal.signal(signal.SIGINT, previous_sigint_handler)
    try:
        if "data_iterator" in locals() and hasattr(data_iterator, "_shutdown_workers"):
            data_iterator._shutdown_workers()
    except Exception:
        pass
    try:
        del data_iterator
    except Exception:
        pass
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if step > 0:
        stopped_path = f"ckpts/checkpoint_step_{step}.pt"
        save_checkpoint(
            stopped_path,
            model,
            ema,
            optimizer,
            scaler,
            step,
            best_sdr,
            avg_loss,
        )
        prune_old_checkpoints("ckpts", keep=3, config=model.config)
        print(f"Training stopped cleanly; checkpoint saved to {stopped_path}.")


def model_config_from_args(args: argparse.Namespace) -> ModelConfig:
    return ModelConfig(
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        audio_channels=2,
        num_stems=len(STEMS),
        num_bands=124,
        dim=args.model_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        ffn_mult=args.ffn_mult,
        filter_taps=args.filter_taps,
        dropout=args.dropout,
        drop_path=args.drop_path,
        use_checkpoint=args.ckpt,
    )


def inspect_checkpoint_config(
    checkpoint_path: str,
    fallback: ModelConfig,
) -> ModelConfig:
    checkpoint_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    try:
        saved_stems = tuple(checkpoint_data.get("stems", STEMS))
        if saved_stems != STEMS:
            raise ValueError(
                f"Checkpoint stems {saved_stems} do not match this script's STEMS {STEMS}."
            )
        config_data = checkpoint_data.get("model_config")
        if not config_data:
            return fallback
        valid_fields = ModelConfig.__dataclass_fields__.keys()
        filtered = {
            key: value for key, value in config_data.items() if key in valid_fields
        }
        return ModelConfig(**filtered)
    finally:
        del checkpoint_data
        gc.collect()


def load_inference_weights(
    model: ResonantConvGateSeparator,
    checkpoint_path: str,
) -> None:
    checkpoint_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    try:
        state = checkpoint_data.get("ema_state_dict") or checkpoint_data.get(
            "model_state_dict"
        )
        if state is None:
            state = checkpoint_data
        report = load_matching_state_dict(model, state)
        if not report.is_exact:
            raise RuntimeError(
                "Checkpoint architecture mismatch: "
                f"loaded {report.matched}/{report.expected} required tensors; "
                f"{len(report.skipped)} incoming tensors were incompatible and "
                f"{len(report.missing)} required tensors were missing."
            )
        print(f"Loaded EMA/model weights from {checkpoint_path}.")
    finally:
        del checkpoint_data
        gc.collect()


def read_input_audio(path: str, sample_rate: int) -> torch.Tensor:
    audio_np, source_sr = sf.read(path, dtype="float32", always_2d=True)
    audio = torch.from_numpy(audio_np.T)
    audio = torch.nan_to_num(audio)
    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)
    elif audio.shape[0] > 2:
        audio = audio[:2]
    if source_sr != sample_rate:
        audio = torchaudio.functional.resample(audio, source_sr, sample_rate)
    return audio


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="124-band Gated-RoPE + 2x2 MIMO 3-Tap Deep-Filter MVSEP Separator"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train", action="store_true")
    mode.add_argument("--infer", action="store_true")

    parser.add_argument("--data_dirs", type=str, default="train1,train2")
    parser.add_argument("--data_weights", type=str, default="1,1")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--test_dir", type=str, default="test")
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--latest", action="store_true")
    parser.add_argument("--reset_optimizer", action="store_true")
    parser.add_argument("--fresh", action="store_true")

    parser.add_argument("--sample_rate", type=int, default=44_100)
    parser.add_argument("--n_fft", type=int, default=4096)
    parser.add_argument("--hop_length", type=int, default=512)
    parser.add_argument("--win_length", type=int, default=4096)
    parser.add_argument("--segment_seconds", type=float, default=8.0)
    parser.add_argument(
        "--inference_overlap_seconds",
        type=float,
        default=None,
        help="Inference/validation overlap in seconds (default: 0.75 * --segment_seconds).",
    )
    parser.add_argument("--model_dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument(
        "--num_heads",
        type=int,
        default=None,
        help="Attention heads (default: model_dim // 32 for 32-d FlashAttention heads).",
    )
    parser.add_argument("--ffn_mult", type=float, default=3.0)
    parser.add_argument("--filter_taps", type=int, default=3, choices=(1, 3, 5))
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--drop_path", type=float, default=0.08)
    parser.add_argument(
        "--ckpt", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--compile", action="store_true")

    parser.add_argument(
        "--max_steps",
        type=int,
        default=0,
        help="Total optimizer steps (0 = infinite/open-ended).",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accumulation", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--dataset_size", type=int, default=50_000)
    parser.add_argument("--remix_probability", type=float, default=0.75)
    parser.add_argument("--checkpoint_steps", type=int, default=8_000)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--slice_p", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--ema_decay", type=float, default=0.9996)
    parser.add_argument(
        "--precision", choices=("bf16", "fp16", "fp32"), default="bf16"
    )
    parser.add_argument(
        "--mask_deadzone",
        type=float,
        default=0.0,
        help=(
            "Optional inference soft-threshold on estimated vocal magnitude relative "
            "to mixture. Default 0.0 avoids active-bin shrinkage and preserves max SDR."
        ),
    )
    parser.add_argument("--silence_weight", type=float, default=0.025)
    parser.add_argument("--mask_silence_weight", type=float, default=0.025)
    parser.add_argument("--silence_margin", type=float, default=1.5)
    parser.add_argument("--silence_floor_db", type=float, default=-65.0)
    parser.add_argument("--silence_active_db", type=float, default=-54.0)
    parser.add_argument("--sdr_weight", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=1337)
    return parser


def validate_runtime_args(args: argparse.Namespace) -> None:
    if args.num_heads is None:
        args.num_heads = (
            max(1, args.model_dim // 32) if args.model_dim % 32 == 0 else 8
        )
    positive_integer_fields = (
        "batch_size",
        "grad_accumulation",
        "dataset_size",
        "checkpoint_steps",
        "num_heads",
    )
    for field in positive_integer_fields:
        if getattr(args, field) <= 0:
            raise ValueError(f"--{field} must be positive.")
    if args.num_workers < 0:
        raise ValueError("--num_workers cannot be negative.")
    if args.max_steps < 0:
        raise ValueError("--max_steps cannot be negative.")
    if not 0.0 <= args.drop_path < 1.0:
        raise ValueError("--drop_path must be in [0, 1).")
    if args.ffn_mult <= 0.0:
        raise ValueError("--ffn_mult must be positive.")
    if args.grad_clip <= 0.0:
        raise ValueError("--grad_clip must be positive.")
    if args.weight_decay < 0.0:
        raise ValueError("--weight_decay cannot be negative.")
    if args.slice_p <= 0:
        raise ValueError("--slice_p must be positive.")
    if not 0.0 <= args.remix_probability <= 1.0:
        raise ValueError("--remix_probability must be in [0, 1].")
    if not 0.0 <= args.ema_decay < 1.0:
        raise ValueError("--ema_decay must be in [0, 1).")
    if args.segment_seconds <= 0.0:
        raise ValueError("--segment_seconds must be positive.")
    if args.inference_overlap_seconds is None:
        args.inference_overlap_seconds = 0.75 * args.segment_seconds
    if args.inference_overlap_seconds < 0.0:
        raise ValueError("--inference_overlap_seconds cannot be negative.")
    if args.dataset_size < args.batch_size:
        raise ValueError("--dataset_size must be at least --batch_size.")
    if args.mask_deadzone < 0.0 or args.mask_deadzone >= 1.0:
        raise ValueError("--mask_deadzone must be in [0, 1).")
    if args.silence_weight < 0.0 or args.mask_silence_weight < 0.0:
        raise ValueError("Silence loss weights cannot be negative.")
    if args.sdr_weight < 0.0:
        raise ValueError("--sdr_weight cannot be negative.")
    if args.silence_floor_db >= args.silence_active_db:
        raise ValueError(
            "--silence_floor_db must be below --silence_active_db."
        )


def main() -> None:
    args = build_parser().parse_args()
    validate_runtime_args(args)
    os.makedirs("ckpts", exist_ok=True)
    os.makedirs("best_ckpts", exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    args.segment_samples = int(round(args.segment_seconds * args.sample_rate))
    args.inference_overlap = int(
        round(args.inference_overlap_seconds * args.sample_rate)
    )
    if args.segment_samples < args.win_length:
        raise ValueError(
            "Training/inference segments must contain at least one full STFT window."
        )
    if args.inference_overlap >= args.segment_samples:
        raise ValueError("Inference overlap must be smaller than the segment length.")

    config = model_config_from_args(args)
    checkpoint_path = args.checkpoint_path
    if checkpoint_path is None:
        if args.train and not args.fresh:
            checkpoint_path = find_latest_compatible_checkpoint(config, "ckpts")
            if checkpoint_path is None and find_latest_checkpoint("ckpts") is not None:
                print(
                    f"No compatible v{CHECKPOINT_FORMAT_VERSION} auto-resume checkpoint "
                    "was found in ckpts/. Starting fresh run of the new architecture."
                )
            elif checkpoint_path is not None:
                print(
                    f"Auto-resuming training from {checkpoint_path}. "
                    "Use --fresh to start a new run instead."
                )
        elif args.infer:
            checkpoint_path = (
                find_latest_compatible_checkpoint(config, "ckpts")
                if args.latest
                else find_best_checkpoint("best_ckpts", config)
            )

    # Offset RNG seed by resumed step so auto-resumed runs never replay step-0 data
    resume_step = peek_checkpoint_step(checkpoint_path) if args.train else 0
    effective_seed = (args.seed + resume_step * 100_003) % (2**31 - 1)
    seed_everything(effective_seed)

    if args.infer and checkpoint_path:
        config = inspect_checkpoint_config(checkpoint_path, config)
        args.segment_samples = int(round(args.segment_seconds * config.sample_rate))
        args.inference_overlap = int(
            round(args.inference_overlap_seconds * config.sample_rate)
        )
        if args.segment_samples < config.win_length:
            raise ValueError(
                "Inference segment must contain at least one checkpoint STFT window."
            )
        if args.inference_overlap >= args.segment_samples:
            raise ValueError("Inference overlap must be smaller than segment length.")

    if device.type == "cuda" and args.precision == "bf16":
        if not torch.cuda.is_bf16_supported():
            print("CUDA device lacks BF16 support; falling back to FP16.")
            args.precision = "fp16"

    model = ResonantConvGateSeparator(config)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    encoder_count = sum(
        parameter.numel() for parameter in model.encoder.parameters()
    )
    mask_count = sum(
        parameter.numel() for parameter in model.mask_head.parameters()
    )
    print(
        f"124-band Gated-RoPE + 2x2 MIMO {config.filter_taps}-tap DeepFilter parameters: "
        f"{parameter_count / 1e6:.2f}M (encoder: {encoder_count / 1e6:.2f}M, "
        f"mask head: {mask_count / 1e6:.2f}M)"
    )

    if args.train:
        if checkpoint_path:
            print(f"Checkpoint selected: {checkpoint_path} (RNG seed offset by step {resume_step})")
        dataset = build_training_dataset(args, config)
        generator = torch.Generator()
        generator.manual_seed(effective_seed)
        use_pin_memory = device.type == "cuda"
        use_persistent = args.num_workers > 0
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=use_pin_memory,
            persistent_workers=use_persistent,
            prefetch_factor=2 if args.num_workers > 0 else None,
            worker_init_fn=training_worker_init,
            generator=generator,
            drop_last=True,
        )
        optimizer = build_optimizer(
            model,
            weight_decay=args.weight_decay,
            slice_p=args.slice_p,
        )
        loss_module = SeparationLoss(
            config,
            LossConfig(
                sdr_weight=args.sdr_weight,
                silence_weight=args.silence_weight,
                mask_silence_weight=args.mask_silence_weight,
                silence_margin=args.silence_margin,
                silence_floor_db=args.silence_floor_db,
                silence_active_db=args.silence_active_db,
            ),
        )
        train(
            model,
            dataloader,
            optimizer,
            loss_module,
            device,
            args,
            checkpoint_path,
        )
        return

    if not args.input_file:
        raise ValueError("--input_file is required for inference.")
    if not checkpoint_path:
        raise FileNotFoundError("No checkpoint was supplied or found.")

    load_inference_weights(model, checkpoint_path)
    model.to(device).eval()
    if args.compile:
        model.encoder.compile_layers(mode="default")
    mixture = read_input_audio(args.input_file, config.sample_rate)
    predictions = separate_tensor(
        model,
        mixture,
        chunk_size=args.segment_samples,
        overlap=args.inference_overlap,
        device=device,
        precision=args.precision,
        show_progress=True,
        mask_deadzone=args.mask_deadzone,
    )
    input_prefix = Path(args.input_file).stem
    for stem, prediction in zip(STEMS, predictions):
        output_path = os.path.join(args.output_dir, f"{input_prefix}_{stem}.wav")
        sf.write(
            output_path,
            prediction.cpu().numpy().T,
            config.sample_rate,
            subtype="FLOAT",
        )
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
