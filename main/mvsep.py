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
from prodigyopt import Prodigy
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


STEMS = ("vocals", "other")
AUDIO_EXTENSIONS = (".wav", ".flac")
VALIDATION_METRIC = "mean_full_track_sdr_v1"
CHECKPOINT_FORMAT_VERSION = 8


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass
class ModelConfig:
    sample_rate: int = 44_100
    n_fft: int = 2048
    hop_length: int = 512
    win_length: int = 2048
    audio_channels: int = 2
    num_stems: int = len(STEMS)
    num_bands: int = 124
    dim: int = 384
    depth: int = 14
    heads: int = 8
    dropout: float = 0.0
    use_checkpoint: bool = True
    architecture: str = "bs124_roformer_axial_v6_direct_mask"

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
        if self.dim <= 0 or self.depth <= 0 or self.heads <= 0:
            raise ValueError("dim, depth, and heads must be positive.")
        if self.dim % self.heads != 0:
            raise ValueError("dim must be divisible by heads.")
        if (self.dim // self.heads) % 2 != 0:
            raise ValueError("The attention head dimension must be even for RoPE.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if self.num_bands != 124:
            raise ValueError("This architecture is intentionally fixed at exactly 124 bands.")
        if self.architecture != "bs124_roformer_axial_v6_direct_mask":
            raise ValueError(
                "Unsupported architecture "
                f"{self.architecture!r}; expected "
                "bs124_roformer_axial_v6_direct_mask."
            )


@dataclass
class LossConfig:
    waveform_weight: float = 1.0
    main_stft_weight: float = 0.65
    mrstft_weight: float = 0.9
    mask_weight: float = 0.15
    sdr_weight: float = 0.30
    midside_weight: float = 0.05


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
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


RUNTIME_CONFIG_FIELDS = frozenset({"use_checkpoint"})


def model_configs_compatible(
    saved_config: dict | None,
    current_config: ModelConfig,
) -> bool:
    """Compare checkpoint-sensitive model fields, excluding runtime switches."""
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
    """Load keys whose names and shapes match and report exact coverage.

    Callers must check ``is_exact``: a partial load means the checkpoint does
    not match this architecture and must be rejected.
    """
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
    """Build the regular disjoint BS-RoFormer frequency bands.

    The 2048-FFT / 124-band preset refines the established 62-band BS-RoFormer
    layout by splitting every original band in two. Every real-STFT bin is
    assigned to exactly one band; there is no Mel spacing, overlap, duplicated
    coverage, or cross-band mask averaging.
    """
    freq_bins = n_fft // 2 + 1
    if num_bands <= 0:
        raise ValueError("num_bands must be positive.")
    if num_bands > freq_bins:
        raise ValueError(
            f"Cannot create {num_bands} non-empty bands from {freq_bins} bins."
        )

    if n_fft == 2048 and num_bands in (62, 124):
        base_widths = (
            [2] * 24
            + [4] * 12
            + [12] * 8
            + [24] * 8
            + [48] * 8
            + [128, 129]
        )
        if num_bands == 124:
            widths = [
                part
                for width in base_widths
                for part in (width // 2, width - width // 2)
            ]
        else:
            widths = base_widths
    else:
        # Deterministic non-Mel fallback: many narrow low-frequency bands and
        # progressively wider high-frequency bands, with strict disjoint coverage.
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
    """STFT for tensors shaped [..., samples]."""
    original_shape = audio.shape[:-1]
    audio_flat = audio.reshape(-1, audio.shape[-1]).float()
    spec = torch.stft(
        audio_flat,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
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
    """ISTFT for tensors shaped [..., frequency, frames]."""
    original_shape = spec.shape[:-2]
    spec_flat = spec.reshape(-1, spec.shape[-2], spec.shape[-1]).to(torch.complex64)
    audio = torch.istft(
        spec_flat,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        length=length,
    )
    return audio.reshape(*original_shape, length)


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float = 10_000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE head dimension must be even.")
        inv_freq = base ** (-torch.arange(0, head_dim, 2).float() / head_dim)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(length, device=device, dtype=self.inv_freq.dtype)
        angles = torch.outer(positions, self.inv_freq)
        angles = torch.cat((angles, angles), dim=-1)
        return angles.cos().to(dtype=dtype), angles.sin().to(dtype=dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return x * cos + rotate_half(x) * sin


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.in_proj = nn.Linear(dim, hidden_dim * 2, bias=False)
        self.out_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.in_proj(x).chunk(2, dim=-1)
        return self.dropout(self.out_proj(F.silu(gate) * value))


class RoPEAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        if dim % heads != 0:
            raise ValueError("Model dimension must be divisible by the number of heads.")
        self.heads = heads
        self.head_dim = dim // heads
        self.dropout = dropout

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.out_dropout = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        batch, length, dim = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        cos, sin = self.rope(length, x.device, q.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        attention_dropout = self.dropout if self.training else 0.0
        # PyTorch SDPA selects the best available kernel per call (flash,
        # cuDNN, memory-efficient, or math fallback) with no manual tuning.
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=attention_dropout,
            is_causal=False,
        )
        out = out.transpose(1, 2).reshape(batch, length, dim)
        return self.out_dropout(self.out_proj(out))


class TransformerUnit(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dropout: float,
    ):
        super().__init__()
        self.attn_norm = nn.RMSNorm(dim)
        self.attn = RoPEAttention(
            dim,
            heads,
            dropout=dropout,
        )
        hidden_dim = int(math.ceil((dim * 2.5) / 64.0) * 64)
        self.ff_norm = nn.RMSNorm(dim)
        self.ff = SwiGLU(dim, hidden_dim, dropout=dropout)
        # Zero-init the FF output projection so the feed-forward residual path
        # starts neutral and is learned on top of the attention function.
        nn.init.zeros_(self.ff.out_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        return x + self.ff(self.ff_norm(x))


class DualPathEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        unit_kwargs = dict(
            dim=config.dim,
            heads=config.heads,
            dropout=config.dropout,
        )
        self.time_layers = nn.ModuleList(
            TransformerUnit(**unit_kwargs) for _ in range(config.depth)
        )
        self.freq_layers = nn.ModuleList(
            TransformerUnit(**unit_kwargs) for _ in range(config.depth)
        )
        self.output_norm = nn.RMSNorm(config.dim)
        self.use_checkpoint = config.use_checkpoint

    def compile_layers(self, mode: str = "default") -> None:
        if mode != "default":
            raise ValueError("compile_layers currently supports only the default mode.")
        for unit in (*self.time_layers, *self.freq_layers):
            unit.compile(
                dynamic=True,
                options={"comprehensive_padding": False},
            )

    @staticmethod
    def _run_module(
        module: nn.Module,
        x: torch.Tensor,
        use_checkpoint: bool,
    ) -> torch.Tensor:
        if not use_checkpoint:
            return module(x)
        return checkpoint(module, x, use_reentrant=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, frames, bands, dim]
        batch, frames, bands, dim = x.shape
        should_checkpoint = self.use_checkpoint and self.training

        for time_layer, freq_layer in zip(self.time_layers, self.freq_layers):
            time_x = x.permute(0, 2, 1, 3).reshape(batch * bands, frames, dim)
            time_x = self._run_module(time_layer, time_x, should_checkpoint)
            x = time_x.reshape(batch, bands, frames, dim).permute(0, 2, 1, 3)

            freq_x = x.reshape(batch * frames, bands, dim)
            freq_x = self._run_module(freq_layer, freq_x, should_checkpoint)
            x = freq_x.reshape(batch, frames, bands, dim)

        return self.output_norm(x)


def next_power_of_two(value: int) -> int:
    return 1 if value <= 1 else 1 << (value - 1).bit_length()


class BandInputGroup(nn.Module):
    """A batch of bands padded only to the next power-of-two width."""

    def __init__(
        self,
        config: ModelConfig,
        bands: Sequence[tuple[int, int]],
        band_ids: Sequence[int],
        bucket_width: int,
    ):
        super().__init__()
        self.feature_width = bucket_width * config.audio_channels * 2
        self.num_group_bands = len(band_ids)

        self.register_buffer(
            "band_ids", torch.tensor(band_ids, dtype=torch.long), persistent=False
        )
        freq_indices = torch.zeros(
            self.num_group_bands, bucket_width, dtype=torch.long
        )
        freq_valid = torch.zeros(
            self.num_group_bands, bucket_width, dtype=torch.bool
        )
        valid_feature_counts = torch.zeros(
            self.num_group_bands, dtype=torch.float32
        )
        for local_index, band_id in enumerate(band_ids):
            start, end = bands[band_id]
            width = end - start
            freq_indices[local_index, :width] = torch.arange(start, end)
            freq_valid[local_index, :width] = True
            valid_feature_counts[local_index] = width * config.audio_channels * 2

        feature_valid = (
            freq_valid[:, :, None, None]
            .expand(-1, -1, config.audio_channels, 2)
            .reshape(self.num_group_bands, self.feature_width)
        )
        self.register_buffer("freq_indices", freq_indices, persistent=False)
        self.register_buffer("feature_valid", feature_valid, persistent=False)
        self.register_buffer(
            "valid_feature_counts", valid_feature_counts, persistent=False
        )

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
        with torch.no_grad():
            self.weight.zero_()
            for band in range(self.num_group_bands):
                fan_in = int(self.valid_feature_counts[band].item())
                bound = math.sqrt(6.0 / (fan_in + self.weight.shape[-1]))
                self.weight[band, :fan_in].uniform_(-bound, bound)
            self.gamma.masked_fill_(~self.feature_valid, 0.0)

    def forward(self, real_imag: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # real_imag: [B, T, F, C, 2]
        gathered = real_imag[:, :, self.freq_indices]
        features = gathered.reshape(
            gathered.shape[0],
            gathered.shape[1],
            self.num_group_bands,
            self.feature_width,
        )
        features = features * self.feature_valid[None, None]
        mean_square = features.square().sum(dim=-1, keepdim=True)
        mean_square = mean_square / self.valid_feature_counts[None, None, :, None]
        features = features * torch.rsqrt(mean_square + 1e-5)
        features = features * self.gamma[None, None]
        tokens = torch.einsum("btni,nid->btnd", features, self.weight) + self.bias
        return tokens, mean_square.squeeze(-1)


class BandSplit(nn.Module):
    """Project each disjoint complex stereo BS band into one token."""

    def __init__(self, config: ModelConfig, bands: Sequence[tuple[int, int]]):
        super().__init__()
        self.num_bands = len(bands)
        grouped_ids: dict[int, list[int]] = {}
        for band_id, (start, end) in enumerate(bands):
            bucket = next_power_of_two(end - start)
            grouped_ids.setdefault(bucket, []).append(band_id)

        self.groups = nn.ModuleList(
            BandInputGroup(config, bands, ids, bucket)
            for bucket, ids in sorted(grouped_ids.items())
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
            "band_metadata", torch.tensor(metadata, dtype=torch.float32), persistent=False
        )
        self.metadata_proj = nn.Sequential(
            nn.Linear(4, config.dim),
            nn.SiLU(),
            nn.Linear(config.dim, config.dim, bias=False),
        )
        # Keep the new constant band metadata path neutral at initialization so
        # the learned band projections stay dominant at the start of training.
        nn.init.zeros_(self.metadata_proj[-1].weight)

        self.energy_weight = nn.Parameter(torch.empty(self.num_bands, 2, config.dim))
        nn.init.normal_(self.energy_weight, std=0.02)

    def forward_real(self, real_imag: torch.Tensor) -> torch.Tensor:
        real_imag = real_imag.permute(0, 3, 2, 1, 4)  # [B, T, F, C, 2]
        output = real_imag.new_zeros(
            real_imag.shape[0],
            real_imag.shape[1],
            self.num_bands,
            self.groups[0].weight.shape[-1],
        )
        band_power = real_imag.new_zeros(
            real_imag.shape[0], real_imag.shape[1], self.num_bands
        )
        for group in self.groups:
            group_tokens, group_power = group(real_imag)
            output = output.index_copy(2, group.band_ids, group_tokens)
            band_power = band_power.index_copy(2, group.band_ids, group_power)

        log_rms = 0.5 * torch.log(band_power + 1e-5)
        feature_counts = self.band_feature_counts.to(dtype=band_power.dtype)
        frame_power = (
            band_power * feature_counts[None, None]
        ).sum(dim=2, keepdim=True) / feature_counts.sum()
        frame_log_rms = 0.5 * torch.log(frame_power + 1e-5)
        spectral_relative = log_rms - frame_log_rms
        temporal_relative = frame_log_rms - frame_log_rms.mean(dim=1, keepdim=True)
        energy_features = torch.stack(
            (spectral_relative, temporal_relative.expand_as(log_rms)), dim=-1
        ).clamp_(-8.0, 8.0)
        energy_tokens = torch.einsum(
            "btne,ned->btnd", energy_features, self.energy_weight
        )
        metadata_tokens = self.metadata_proj(
            self.band_metadata.to(device=output.device, dtype=output.dtype)
        )[None, None]
        return output + energy_tokens.to(dtype=output.dtype) + metadata_tokens

    def forward(self, mixture_spec: torch.Tensor) -> torch.Tensor:
        return self.forward_real(torch.view_as_real(mixture_spec.to(torch.complex64)))


class BandMaskGroup(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        bands: Sequence[tuple[int, int]],
        band_ids: Sequence[int],
        bucket_width: int,
    ):
        super().__init__()
        self.num_predicted_stems = 1
        self.audio_channels = config.audio_channels
        self.bucket_width = bucket_width
        self.feature_width = bucket_width * config.audio_channels * 2
        self.num_group_bands = len(band_ids)

        self.register_buffer(
            "band_ids", torch.tensor(band_ids, dtype=torch.long), persistent=False
        )
        freq_indices = torch.zeros(
            self.num_group_bands, bucket_width, dtype=torch.long
        )
        freq_valid = torch.zeros(
            self.num_group_bands, bucket_width, dtype=torch.bool
        )
        for local_index, band_id in enumerate(band_ids):
            start, end = bands[band_id]
            width = end - start
            freq_indices[local_index, :width] = torch.arange(start, end)
            freq_valid[local_index, :width] = True

        feature_valid = (
            freq_valid[:, :, None, None]
            .expand(-1, -1, config.audio_channels, 2)
            .reshape(self.num_group_bands, self.feature_width)
        )
        self.register_buffer("freq_indices", freq_indices, persistent=False)
        self.register_buffer("feature_valid", feature_valid, persistent=False)

        # Each band token is decoded to its complex stereo mask by a single
        # band-specific gated linear projection. There is no shared predictor
        # network on top of the transformer; the encoder representation is
        # mapped straight to mask coefficients.
        output_width = self.num_predicted_stems * self.feature_width
        self.output_weight = nn.Parameter(
            torch.empty(self.num_group_bands, config.dim, output_width * 2)
        )
        self.output_bias = nn.Parameter(
            torch.zeros(self.num_group_bands, output_width * 2)
        )
        nn.init.normal_(self.output_weight, std=1e-3)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, T, group_bands, D]
        raw = torch.einsum("btnd,ndq->btnq", x, self.output_weight)
        raw = F.glu(raw + self.output_bias[None, None], dim=-1)
        raw = raw.reshape(
            x.shape[0],
            x.shape[1],
            self.num_group_bands,
            self.num_predicted_stems,
            self.feature_width,
        )
        raw = raw * self.feature_valid[None, None, :, None]
        raw = raw.reshape(
            x.shape[0],
            x.shape[1],
            self.num_group_bands,
            self.num_predicted_stems,
            self.bucket_width,
            self.audio_channels,
            2,
        )
        source = raw.permute(0, 3, 5, 1, 2, 4, 6).reshape(
            x.shape[0],
            self.num_predicted_stems,
            self.audio_channels,
            x.shape[1],
            self.num_group_bands * self.bucket_width,
            2,
        )
        return source, self.freq_indices.reshape(-1)


class MaskHead(nn.Module):
    """Decode final transformer tokens into the foreground vocal mask.

    Mask prediction happens entirely inside the transformer's output stage:
    each normalized band token is mapped to its band's complex stereo mask by
    a per-band gated linear projection, with no dedicated predictor network
    between the encoder and the mask output.
    """

    def __init__(
        self,
        config: ModelConfig,
        bands: Sequence[tuple[int, int]],
    ):
        super().__init__()
        self.audio_channels = config.audio_channels
        self.freq_bins = config.n_fft // 2 + 1

        grouped_ids: dict[int, list[int]] = {}
        for band_id, (start, end) in enumerate(bands):
            bucket = next_power_of_two(end - start)
            grouped_ids.setdefault(bucket, []).append(band_id)
        self.groups = nn.ModuleList(
            BandMaskGroup(config, bands, ids, bucket)
            for bucket, ids in sorted(grouped_ids.items())
        )

        coverage = torch.zeros(self.freq_bins, dtype=torch.float32)
        for start, end in bands:
            coverage[start:end] += 1.0
        if not torch.all(coverage == 1):
            raise ValueError("Every frequency bin must be covered by exactly one BS band.")

    def forward_real(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, bands, D]
        output = x.new_zeros(
            x.shape[0], 1, self.audio_channels, x.shape[1], self.freq_bins, 2
        )
        for group in self.groups:
            group_x = x.index_select(2, group.band_ids)
            source, flat_indices = group(group_x)
            scatter_index = flat_indices.view(1, 1, 1, 1, -1, 1).expand(
                x.shape[0], 1, self.audio_channels, x.shape[1], -1, 2
            )
            output.scatter_add_(dim=4, index=scatter_index, src=source)

        output = output.permute(0, 1, 2, 4, 3, 5).contiguous().float()
        # Keep a neutral two-source prior. It does not force leakage: the learned
        # raw mask can move all the way to zero, while retaining 0.5 keeps the
        # vocal head centered at the outset of training.
        mask_bias = output.new_tensor((0.5, 0.0))
        return output + mask_bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.view_as_complex(self.forward_real(x))



class BSRoFormerSeparator(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.bands = build_bs_bands(config.n_fft, config.num_bands)
        self.band_split = BandSplit(config, self.bands)
        self.encoder = DualPathEncoder(config)
        self.mask_head = MaskHead(config, self.bands)

    def forward_real(self, mixture_real_imag: torch.Tensor) -> torch.Tensor:
        tokens = self.band_split.forward_real(mixture_real_imag)
        tokens = self.encoder(tokens)
        vocal_mask = self.mask_head.forward_real(tokens)

        # No vocal activity gate: the separator's foreground mask is used directly.
        # The accompaniment remains the exact residual complement, so reconstruction
        # consistency never injects residual mixture energy back into the vocal stem.
        one = torch.zeros_like(vocal_mask)
        one[..., 0] = 1.0
        other_mask = one - vocal_mask
        return torch.cat((vocal_mask, other_mask), dim=1)

    def forward(self, mixture_spec: torch.Tensor) -> torch.Tensor:
        mixture_real_imag = torch.view_as_real(mixture_spec.to(torch.complex64))
        masks_real_imag = self.forward_real(mixture_real_imag)
        return torch.view_as_complex(masks_real_imag)

    def estimate_specs(
        self, mixture_spec: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        masks = self(mixture_spec)
        estimates = masks * mixture_spec[:, None]
        # Masks are complementary by construction. Route only floating-point
        # reconstruction residue to the accompaniment so vocals are never polluted.
        residual = mixture_spec - estimates.sum(dim=1)
        estimates[:, 1] = estimates[:, 1] + residual
        return estimates, masks


# -----------------------------------------------------------------------------
# Losses
# -----------------------------------------------------------------------------


class MultiResolutionSTFTLoss(nn.Module):
    def __init__(
        self,
        resolutions: Sequence[tuple[int, int, int]] = (
            (2048, 147, 2048),
            (1024, 147, 1024),
            (512, 147, 512),
            (256, 147, 256),
        ),
        activity_threshold: float = 1e-4,
    ):
        super().__init__()
        self.resolutions = tuple(resolutions)
        self.activity_threshold = activity_threshold
        for index, (_, _, win_length) in enumerate(self.resolutions):
            self.register_buffer(
                f"window_{index}",
                torch.hann_window(win_length),
                persistent=False,
            )

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_flat = prediction.reshape(-1, prediction.shape[-1]).float()
        target_flat = target.reshape(-1, target.shape[-1]).float()
        active_targets = (
            target_flat.square().mean(dim=1).sqrt() >= self.activity_threshold
        )
        total = pred_flat.new_tensor(0.0)

        for index, (n_fft, hop_length, win_length) in enumerate(self.resolutions):
            window = getattr(self, f"window_{index}")
            pred_spec = torch.stft(
                pred_flat,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                center=True,
                return_complex=True,
            )
            target_spec = torch.stft(
                target_flat,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                center=True,
                return_complex=True,
            )
            pred_mag = pred_spec.abs()
            target_mag = target_spec.abs()

            diff_norm = torch.linalg.vector_norm(
                (pred_mag - target_mag).flatten(1), dim=1
            )
            target_norm = torch.linalg.vector_norm(target_mag.flatten(1), dim=1)
            # Spectral convergence is a relative error and is undefined for a
            # silent target. Dividing leakage by a tiny epsilon made silent stem
            # channels produce losses in the hundreds of millions. The absolute
            # log-magnitude and complex terms below still train those channels
            # toward silence.
            active_diff_norm = diff_norm[active_targets]
            active_target_norm = target_norm[active_targets]
            spectral_convergence = (
                active_diff_norm / active_target_norm.clamp_min(1e-6)
            ).sum() / active_targets.count_nonzero().clamp_min(1)

            log_magnitude = F.l1_loss(
                torch.log1p(pred_mag),
                torch.log1p(target_mag),
            )
            complex_normalizer = target_mag.mean().detach().clamp_min(1e-4)
            complex_loss = (pred_spec - target_spec).abs().mean() / complex_normalizer
            total = total + spectral_convergence + log_magnitude + 0.25 * complex_loss

        return total / len(self.resolutions)


def normalized_l1(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Relative L1 without letting fully silent stems dominate the batch.

    A fixed 1e-4 denominator makes a deliberately zeroed vocal target hundreds of
    times more important than an ordinary active source.  That creates a strong
    collapse incentive for a foreground/residual separator: predicting no vocals
    anywhere cheaply solves those rare examples.  Floor each source denominator
    at 5% of the strongest target level in the same example instead.  Active
    sources keep their normal relative scaling while silent sources remain
    supervised, just not catastrophically overweighted.
    """
    error = (prediction - target).abs().mean(dim=-1)
    scale = target.abs().mean(dim=-1)
    if scale.ndim > 1:
        reduce_dims = tuple(range(1, scale.ndim))
        reference = scale.amax(dim=reduce_dims, keepdim=True)
    else:
        reference = scale
    scale_floor = (0.05 * reference).clamp_min(1e-4)
    return (error / torch.maximum(scale, scale_floor)).mean()


def scale_dependent_sdr_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    error_power = (prediction - target).square().mean(dim=(-2, -1))
    target_power = target.square().mean(dim=(-2, -1))
    valid = target_power > 1e-7
    ratio_db = 10.0 * torch.log10(
        (target_power + 1e-8) / (error_power + 1e-8)
    )
    ratio_db = ratio_db.clamp(-50.0, 50.0)
    if valid.any():
        return -ratio_db[valid].mean()
    return prediction.new_tensor(0.0)


def mid_side(audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mid = (audio[..., 0, :] + audio[..., 1, :]) * 0.5
    side = (audio[..., 0, :] - audio[..., 1, :]) * 0.5
    return mid, side


def frame_mean_square(
    audio: torch.Tensor,
    win_length: int,
    hop_length: int,
) -> torch.Tensor:
    """Frame-local mean-square envelope aligned to center=True STFT frames.

    ``audio`` is [..., samples] and the result is [..., n_frames]; leading
    dimensions are preserved so a batch of mixes is pooled independently (the
    previous mean over dim -2 merged batch entries together).

    Keeping this in the power domain avoids the singular derivative of sqrt(0),
    which matters because silence augmentation intentionally creates exact zeros.
    """
    flat = audio.square().reshape(-1, audio.shape[-1]).unsqueeze(1)
    pooled = F.avg_pool1d(
        flat,
        kernel_size=win_length,
        stride=hop_length,
        padding=win_length // 2,
        count_include_pad=False,
    )
    return pooled.squeeze(1).reshape(*audio.shape[:-1], pooled.shape[-1])


class SeparationLoss(nn.Module):
    def __init__(self, model_config: ModelConfig, loss_config: LossConfig):
        super().__init__()
        self.model_config = model_config
        self.loss_config = loss_config
        self.mrstft = MultiResolutionSTFTLoss()
        self.activity_threshold = 1e-4
        self.register_buffer(
            "window", torch.hann_window(model_config.win_length), persistent=False
        )

    def forward(
        self,
        model: BSRoFormerSeparator,
        mixture_spec: torch.Tensor,
        target_audio: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        masks = model(mixture_spec)
        estimates = masks * mixture_spec[:, None]
        residual = mixture_spec - estimates.sum(dim=1)
        estimates[:, 1] = estimates[:, 1] + residual

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

        wave_loss = normalized_l1(pred_audio, target_audio)
        mrstft_loss = self.mrstft(pred_audio, target_audio)

        target_mag = target_specs.abs()
        spec_normalizer = target_mag.mean().detach().clamp_min(1e-4)
        main_complex = (estimates - target_specs).abs().mean() / spec_normalizer
        main_logmag = F.l1_loss(torch.log1p(estimates.abs()), torch.log1p(target_mag))
        main_stft_loss = main_complex + main_logmag

        mix_power = mixture_spec.abs().square()
        ideal_masks = (
            target_specs * mixture_spec[:, None].conj() / (mix_power[:, None] + 1e-5)
        )
        ideal_mag = ideal_masks.abs().clamp_max(8.0)
        ideal_masks = torch.polar(ideal_mag, torch.angle(ideal_masks))
        tf_weight = mixture_spec.abs()
        tf_weight = tf_weight / tf_weight.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-4)
        tf_weight = tf_weight.clamp(max=5.0)
        effective_masks = (
            estimates * mixture_spec[:, None].conj() / (mix_power[:, None] + 1e-5)
        )
        mask_loss = ((effective_masks - ideal_masks).abs() * tf_weight[:, None]).mean()

        sdr_loss = scale_dependent_sdr_loss(pred_audio, target_audio)
        pred_mid, pred_side = mid_side(pred_audio)
        true_mid, true_side = mid_side(target_audio)
        midside_loss = 0.5 * (
            normalized_l1(pred_mid, true_mid) + normalized_l1(pred_side, true_side)
        )

        cfg = self.loss_config
        total = (
            cfg.waveform_weight * wave_loss
            + cfg.main_stft_weight * main_stft_loss
            + cfg.mrstft_weight * mrstft_loss
            + cfg.mask_weight * mask_loss
            + cfg.sdr_weight * sdr_loss
            + cfg.midside_weight * midside_loss
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
            vocal_mask_mag = masks[:, 0].abs().mean()

        metrics = {
            "wave": wave_loss.detach(),
            "main_stft": main_stft_loss.detach(),
            "mrstft": mrstft_loss.detach(),
            "mask": mask_loss.detach(),
            "sdr_loss": sdr_loss.detach(),
            "midside": midside_loss.detach(),
            "vocal_level_db": vocal_level_db.detach(),
            "vocal_mask_mag": vocal_mask_mag.detach(),
        }
        return total, metrics


# -----------------------------------------------------------------------------
# Dataset and augmentation
# -----------------------------------------------------------------------------


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
    """Resolve a track to the logical two-stem training contract.

    Native two-stem datasets may provide ``vocals`` plus ``other``,
    ``instrumental``, or ``accompaniment``. Standard MUSDB tracks instead store
    bass, drums, and other separately; those three files must be summed or the
    training mixture would silently omit most of the accompaniment.
    """
    vocals = find_optional_audio_file(directory, "vocals")
    if vocals is None:
        return None

    bass = find_optional_audio_file(directory, "bass")
    drums = find_optional_audio_file(directory, "drums")
    other = find_optional_audio_file(directory, "other")
    if bass is not None and drums is not None and other is not None:
        accompaniment = (bass, drums, other)
    elif bass is not None or drums is not None:
        # A partially populated MUSDB track is unsafe: falling back to other
        # would create a mixture with missing accompaniment components.
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


# -----------------------------------------------------------------------------
# Wet / backwards / wide vocal augmentation
# -----------------------------------------------------------------------------


def fft_convolve(signal: torch.Tensor, impulse: torch.Tensor) -> torch.Tensor:
    """Convolve [channels, samples] signal with a 1-D impulse via FFT."""
    total = signal.shape[-1] + impulse.shape[-1] - 1
    fft_size = 1
    while fft_size < total:
        fft_size <<= 1
    signal_spec = torch.fft.rfft(signal, fft_size, dim=-1)
    impulse_spec = torch.fft.rfft(impulse.to(signal.dtype), fft_size, dim=-1)
    convolved = torch.fft.irfft(
        signal_spec * impulse_spec.unsqueeze(0), fft_size, dim=-1
    )
    return convolved[..., : signal.shape[-1]]


def make_reverb_impulse(sample_rate: int, max_length: int) -> torch.Tensor:
    """Random exponentially decaying noise IR with a short pre-delay.

    The tail is low-passed at a random 3-9 kHz corner.  A white-noise IR
    sounds like a bright hiss and trains the separator to smear broadband
    energy around the voice; real rooms roll the highs off.
    """
    length = min(int(random.uniform(0.4, 2.0) * sample_rate), max_length)
    pre_delay = int(random.uniform(0.0, 0.12) * sample_rate)
    tail = max(1, length - pre_delay)
    time = torch.arange(tail, dtype=torch.float32)
    tau = random.uniform(0.08, 0.4) * sample_rate
    impulse = torch.randn(tail, dtype=torch.float32) * torch.exp(-time / tau)
    if pre_delay > 0:
        impulse = torch.cat((torch.zeros(pre_delay, dtype=torch.float32), impulse))
    n = impulse.shape[-1]
    freqs = torch.fft.rfftfreq(n, 1.0 / sample_rate).clamp_min(1.0)
    cutoff = random.uniform(3_000.0, 7_000.0)
    response = (1.0 + (freqs / cutoff) ** 2).rsqrt()
    impulse = torch.fft.irfft(torch.fft.rfft(impulse, n) * response, n)
    return impulse / impulse.square().sum().sqrt().clamp_min(1e-8)


def apply_reverb(audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Mix a synthetic reverb tail into a stem so wet vocals stay extractable.

    The wet mix is deliberately modest: a loud diffuse tail in the vocal
    target overlaps the instrumental's high band (cymbals, hats) for seconds
    after the voice stops, and a mask separator answers by leaving its mask
    open there -- the "static instrumental riding the vocals" artifact.
    """
    impulse = make_reverb_impulse(sample_rate, audio.shape[-1])
    wet = fft_convolve(audio, impulse)
    mix = random.uniform(0.1, 0.3)
    return (1.0 - mix) * audio + mix * wet


def apply_reverse_echo(audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Add a reversed, swelling vocal window that flows into the following voice.

    This mimics the production trick where a reversed reverb of the upcoming
    phrase swells up right before the downbeat.  The reversed copy is part of
    the generated vocal target, so the model learns to attribute backwards-
    smeared vocal energy to the vocals instead of leaking it to accompaniment.
    """
    samples = audio.shape[-1]
    window = min(int(random.uniform(0.4, 1.6) * sample_rate), samples // 2)
    if random.random() < 0.75:
        # Bias the window end toward a frame with vocal energy so the swell
        # flows into the voice, like a reverse-reverb pre-echo.
        energy = audio.square().mean(dim=0)
        weight = energy.clamp_min(1e-6).sqrt()
        weight = weight * 0.7 + weight.mean() * 0.3
        candidates = weight[window:]
        total_weight = candidates.sum()
        if total_weight > 0:
            end = (
                int(torch.multinomial(candidates.double() / total_weight, 1).item())
                + window
            )
        else:
            end = random.randint(window, samples)
    else:
        end = random.randint(window, samples)
    start = end - window

    echo = audio[:, start:end].flip(dims=(-1,))
    if random.random() < 0.6:
        impulse = make_reverb_impulse(sample_rate, window)
        echo = fft_convolve(echo, impulse)
    swell = torch.linspace(0.05, 1.0, window) ** random.uniform(1.0, 2.5)
    gain = random.uniform(0.12, 0.4)
    augmented = audio.clone()
    augmented[:, start:end] += gain * swell[None] * echo
    return augmented


def apply_air_boost(audio: torch.Tensor) -> torch.Tensor:
    """Random first-order pre-emphasis: a gentle high-frequency "air" lift.

    The coefficient is kept moderate: y[n] = x[n] - c*x[n-1] is a high-freq
    differentiator, and at c=0.95 it pushes +6 dB at Nyquist while turning any
    residual stem noise into audible hiss.
    """
    coefficient = random.uniform(0.3, 0.6)
    boosted = audio.clone()
    boosted[:, 1:] = audio[:, 1:] - coefficient * audio[:, :-1]
    return boosted


def make_synth_lead(sample_rate: int, duration: int) -> torch.Tensor:
    """A vocal-ish sustained synth tone: harmonics, vibrato, slow envelope.

    Clean harmonic timbre with no formant structure and no breath noise --
    the opposite of a real voice.  Placed in the accompaniment, it teaches the
    model that a prominent sustained pitched tone is not automatically vocals.
    """
    f0 = math.exp(random.uniform(math.log(80.0), math.log(880.0)))
    t = torch.arange(duration, dtype=torch.float32) / sample_rate
    vibrato_depth = random.uniform(0.0, 0.004)
    vibrato_rate = random.uniform(3.0, 6.5)
    freq = f0 * (1.0 + vibrato_depth * torch.sin(2 * math.pi * vibrato_rate * t))
    phase = 2 * math.pi * torch.cumsum(freq, dim=0) / sample_rate
    harmonics = random.randint(8, 28)
    harmonics = min(harmonics, int(sample_rate / (2 * f0)) - 1)
    harmonics = max(1, harmonics)
    # Rolloff kept >= 1.0: slower rolloffs make the lead buzzy, and a bright
    # harmonic stack in the accompaniment is exactly what leaks into the
    # vocal output as high-frequency shimmer.
    rolloff = random.uniform(1.0, 1.5)
    wave = torch.zeros(duration, dtype=torch.float32)
    for k in range(1, harmonics + 1):
        amp = (k ** -rolloff) * random.uniform(0.7, 1.3)
        wave += amp * torch.sin(k * phase)
    attack = int(random.uniform(0.05, 0.4) * sample_rate)
    release = int(random.uniform(0.1, 0.6) * sample_rate)
    env = torch.ones(duration)
    if attack > 1:
        env[:attack] = torch.linspace(0.0, 1.0, attack)
    if release > 1:
        env[-release:] = torch.linspace(1.0, 0.0, release)
    tremolo = 1.0 + random.uniform(0.05, 0.2) * torch.sin(
        2 * math.pi * random.uniform(2.0, 5.0) * t
    )
    return wave * env * tremolo


def add_synth_lead(
    other: torch.Tensor,
    vocal: torch.Tensor,
    sample_rate: int,
) -> torch.Tensor:
    """Drop a prominent vocal-sounding synth lead into the accompaniment.

    Placement is biased toward windows where the vocal is quiet, recreating
    the hard case of a vocal-like lead synth playing on its own (e.g. the
    intro of a pop track).
    """
    samples = other.shape[-1]
    duration = min(int(random.uniform(1.5, 6.0) * sample_rate), samples)
    if random.random() < 0.6 and duration < samples:
        energy = vocal.square().mean(dim=0)
        weight = energy.clamp_min(1e-6).sqrt()
        weight = weight * 0.7 + weight.mean() * 0.3
        weight = weight.max() - weight + weight.mean() * 0.5
        candidates = weight[: samples - duration]
        total_weight = candidates.sum()
        if total_weight > 0:
            start = int(torch.multinomial(candidates.double() / total_weight, 1).item())
        else:
            start = random.randint(0, samples - duration)
    else:
        start = random.randint(0, samples - duration)

    lead = make_synth_lead(sample_rate, duration)
    lead = lead / lead.square().mean().sqrt().clamp_min(1e-8)
    rms_scale = max(
        float(other.square().mean().sqrt()) * random.uniform(0.25, 2.0), 0.02
    )
    width = random.uniform(0.0, 0.6)
    augmented = other.clone()
    augmented[0, start : start + duration] += rms_scale * lead * (1.0 + width)
    augmented[1, start : start + duration] += rms_scale * lead * (1.0 - width)
    return augmented


VOWEL_FORMANTS = (
    (800.0, 1150.0, 2900.0),  # ah
    (400.0, 2200.0, 2900.0),  # eh
    (300.0, 2100.0, 2800.0),  # ee
    (500.0, 800.0, 2800.0),  # oh
    (350.0, 700.0, 2700.0),  # oo
)


def make_sustained_vowel(sample_rate: int, duration: int) -> torch.Tensor:
    """A long synthetic sung vowel: glottal source through formant filters.

    Strong formant structure, vibrato, and breath noise -- the cues that
    separate a real sustained voice from a synth.  Placed in the vocal stem,
    it teaches the model to keep long sustained notes (like a held "oo")
    voiced and free of leaked background.
    """
    f0 = math.exp(random.uniform(math.log(90.0), math.log(300.0)))
    f1, f2, f3 = random.choice(VOWEL_FORMANTS)
    f1 *= random.uniform(0.85, 1.15)
    f2 *= random.uniform(0.85, 1.15)
    f3 *= random.uniform(0.85, 1.15)
    t = torch.arange(duration, dtype=torch.float32) / sample_rate
    vibrato_depth = random.uniform(0.004, 0.012)
    vibrato_rate = random.uniform(4.0, 6.5)
    freq = f0 * (1.0 + vibrato_depth * torch.sin(2 * math.pi * vibrato_rate * t))
    phase = 2 * math.pi * torch.cumsum(freq, dim=0) / sample_rate

    # Glottal source spectrum: harmonics at k * f0 with 1/k amplitudes and
    # random phases (phase is irrelevant; the model reproduces it via masking).
    # Harmonics are capped at ~8 kHz: a sustained vowel that manufactures
    # energy in the air band teaches the separator to emit high-frequency
    # shimmer around the voice.
    n = duration
    spec = torch.zeros(n // 2 + 1, dtype=torch.complex64)
    kmax = min(int(sample_rate / (2 * f0)), 160, int(8_000.0 / f0))
    for k in range(1, kmax + 1):
        bin_index = int(round(k * f0 * n / sample_rate))
        if bin_index <= n // 2:
            angle = random.uniform(0.0, 2 * math.pi)
            spec[bin_index] = torch.polar(
                torch.tensor(1.0 / k, dtype=torch.float32),
                torch.tensor(angle, dtype=torch.float32),
            )
    freqs = torch.fft.rfftfreq(n, 1.0 / sample_rate)
    for formant in (f1, f2, f3):
        q = random.uniform(5.0, 10.0)
        ratio = freqs / formant
        # clamp the 1/ratio term: the glottal source has no DC energy, so
        # clamping the DC bin is harmless and avoids inf/nan at freq=0.
        ratio_inv = 1.0 / ratio.clamp_min(1e-6)
        response = 1.0 / (1.0 + 1j * q * (ratio - ratio_inv))
        spec = spec * response
    vowel = torch.fft.irfft(spec, n)

    # Quiet breath noise and a slow sung envelope with tremolo.  Breath is
    # pink and quiet: white hiss in the vocal target is unrecoverable by the
    # mask separator (it reads as static), and real breath is darker than 1/f.
    breath = _pink_noise(n, sample_rate)
    breath *= random.uniform(0.003, 0.010) * vowel.square().mean().sqrt().clamp_min(1e-8)
    vowel = vowel + breath
    attack = int(random.uniform(0.03, 0.15) * sample_rate)
    release = int(random.uniform(0.15, 0.8) * sample_rate)
    env = torch.ones(n)
    if attack > 1:
        env[:attack] = torch.linspace(0.0, 1.0, attack)
    if release > 1:
        env[-release:] = torch.linspace(1.0, 0.0, release)
    tremolo = 1.0 + random.uniform(0.03, 0.12) * torch.sin(
        2 * math.pi * random.uniform(2.0, 4.5) * t
    )
    return vowel * env * tremolo


def add_sustained_vowel(vocal: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Add a long synthetic sustained vowel to the vocal stem."""
    samples = vocal.shape[-1]
    duration = min(int(random.uniform(1.5, 6.0) * sample_rate), samples)
    start = random.randint(0, samples - duration)
    vowel = make_sustained_vowel(sample_rate, duration)
    vowel = vowel / vowel.square().mean().sqrt().clamp_min(1e-8)
    level = max(
        float(vocal.square().mean().sqrt()) * random.uniform(0.4, 1.5), 0.01
    )
    width = random.uniform(0.0, 0.8)
    augmented = vocal.clone()
    augmented[0, start : start + duration] += level * vowel * (1.0 + width)
    augmented[1, start : start + duration] += level * vowel * (1.0 - width)
    return augmented


# -----------------------------------------------------------------------------
# Broad-spectrum augmentation: pitch, EQ, noise, drive, dynamics, delay, space
# -----------------------------------------------------------------------------
# Everything here is cheap for an 8 s crop: vectorized ops, a handful of FFTs
# only when triggered, and small frame-level loops.  Every effect preserves the
# sum-to-mixture contract by construction (stems are augmented, then summed).


def fft_resample(audio: torch.Tensor, factor: float) -> torch.Tensor:
    """Bandlimited linear-phase resampling, RMS-preserving, [..., samples].

    Linear-phase resampling via FFT is the cheap stand-in for a polyphase FIR:
    one forward and one inverse FFT, all vectorized.  Callers truncate or pad
    the result back to the fixed segment length.
    """
    n = audio.shape[-1]
    new_len = max(1, int(round(n * factor)))
    spec = torch.fft.rfft(audio, n, dim=-1)
    half = new_len // 2 + 1
    if half <= spec.shape[-1]:
        new_spec = spec[..., :half]
    else:
        new_spec = F.pad(spec, (0, half - spec.shape[-1]))
    shifted = torch.fft.irfft(new_spec, new_len, dim=-1)
    rms = audio.square().mean().sqrt().clamp_min(1e-8)
    shifted_rms = shifted.square().mean().sqrt().clamp_min(1e-8)
    return shifted * (rms / shifted_rms)


def pitch_shift_crop(targets: torch.Tensor) -> torch.Tensor:
    """Pitch-shift the whole crop by resampling (+-2 semitones).

    Applied to every stem so the key stays internally consistent (a real track
    does not detune the vocal against the band).  Resampling also moves the
    tempo, mirroring how different recordings of the same song sit in
    different keys *and* tempos; the resampled crop is truncated or zero-padded
    back to the fixed segment length.
    """
    factor = 2.0 ** (random.uniform(-2.0, 2.0) / 12.0)
    n = targets.shape[-1]
    shifted = fft_resample(targets, factor)
    if shifted.shape[-1] >= n:
        return shifted[..., :n]
    return F.pad(shifted, (0, n - shifted.shape[-1]))


def random_spectral_eq(audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Random smooth EQ curve in the FFT domain (shelves, bells, rolloffs).

    Covers the real-mix spectral shape family: muffled radio vocals, bright
    airy pop, telephone band-pass, nasal notches, thin low-cut mixes.  The
    response is real and non-negative, so phase is untouched and the filter is
    zero-phase; at these intensities it reads as "different microphone / EQ",
    not as an artifact.  Loudness is restored so EQ changes timbre, not level.
    """
    n = audio.shape[-1]
    freqs = torch.fft.rfftfreq(n, 1.0 / sample_rate).clamp_min(1.0)
    log_f = torch.log(freqs)
    response = torch.ones_like(freqs)

    def bell(center_hz: float, q: float, gain_db: float) -> None:
        nonlocal response
        sigma = math.log(2.0) / max(q, 0.25)
        response *= 10.0 ** (
            gain_db / 20.0 * torch.exp(-0.5 * ((log_f - math.log(center_hz)) / sigma) ** 2)
        )

    def shelf(center_hz: float, width_decades: float, gain_db: float) -> None:
        nonlocal response
        response *= 10.0 ** (
            gain_db / 20.0
            * 0.5
            * (1.0 + torch.tanh((log_f - math.log(center_hz)) / (math.log(10.0) * width_decades)))
        )

    def lowpass(corner_hz: float, order: float = 2.0) -> None:
        nonlocal response
        response *= (1.0 + (freqs / corner_hz) ** (2.0 * order)).rsqrt()

    def highpass(corner_hz: float, order: float = 2.0) -> None:
        nonlocal response
        response *= (1.0 + (corner_hz / freqs) ** (2.0 * order)).rsqrt()

    for _ in range(random.randint(1, 3)):
        choice = random.random()
        if choice < 0.25:
            bell(
                random.uniform(200.0, 8000.0),
                random.uniform(0.5, 4.0),
                random.uniform(-10.0, 6.0),
            )
        elif choice < 0.5:
            shelf(random.uniform(200.0, 600.0), random.uniform(0.5, 1.5), random.uniform(-8.0, 6.0))
        elif choice < 0.75:
            shelf(random.uniform(2500.0, 9000.0), random.uniform(0.5, 1.5), random.uniform(-8.0, 6.0))
        else:
            corner = random.uniform(150.0, 2500.0)
            order = random.uniform(1.0, 3.0)
            if random.random() < 0.5:
                lowpass(corner, order)
            else:
                highpass(corner, order)
    spec = torch.fft.rfft(audio, n, dim=-1)
    filtered = torch.fft.irfft(spec * response.to(spec.dtype), n, dim=-1)
    rms = audio.square().mean().sqrt().clamp_min(1e-8)
    filtered_rms = filtered.square().mean().sqrt().clamp_min(1e-8)
    return filtered * (rms / filtered_rms)


_PINK_NOISE_CACHE: dict[int, torch.Tensor] = {}


def _pink_noise(length: int, sample_rate: int) -> torch.Tensor:
    """Bounded per-sample-rate pink-noise buffer.

    The previous implementation keyed the cache by (length, sample_rate).
    make_sustained_vowel() passes a random duration, so every new duration
    inserted a new multi-MB buffer into each DataLoader worker.

    This version keeps only one buffer per sample rate and slices random
    windows from it.
    """
    if length <= 0:
        return torch.zeros(0)
    needed = max(2 ** 20, 4 * length)
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
    # Clone so returned tensors do not keep views into the cached buffer.
    return buffer[offset : offset + length].clone()


def add_noise_floor(audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Add a pink-noise floor (tape hiss / room tone) at -56..-40 dB."""
    length = audio.shape[-1]
    level_db = random.uniform(-56.0, -40.0)
    level = 10.0 ** (level_db / 20.0) * audio.square().mean().sqrt().clamp_min(1e-8)
    return audio + level * _pink_noise(length, sample_rate)


def apply_saturation(audio: torch.Tensor) -> torch.Tensor:
    """Soft-knee tanh drive (tube / tape) with a dry/wet mix.

    Drive stays low.  tanh intermodulation manufactures energy above the
    voice's natural bandwidth (measured +15-25 dB in the >8 kHz band at
    drive 3) and the mask separator smears that back out as fuzz riding the
    vocal.  Drive 1.2-2 is gentle tube warmth: harmonics appear, but the
    top end is not manufactured from nothing.
    """
    drive = random.uniform(1.2, 2.0)
    mix = random.uniform(0.3, 1.0)
    wet = torch.tanh(drive * audio) / math.tanh(drive)
    return mix * wet + (1.0 - mix) * audio


def _frame_rms(audio: torch.Tensor, frame: int = 512) -> torch.Tensor:
    """Per-frame RMS envelope of [channels, samples], returns [n_frames]."""
    power = audio.square().mean(dim=0)
    n = power.shape[-1]
    pad = frame - (n % frame)
    if pad < frame:
        power = F.pad(power, (0, pad))
    return power.reshape(-1, frame).mean(dim=1).sqrt().clamp_min(1e-8)


def apply_bus_compression(targets: torch.Tensor, frame: int = 512) -> torch.Tensor:
    """Side-chain "mix bus" compression applied identically to every stem.

    The gain envelope is derived from the mixture, exactly like a real bus
    compressor, and the same gain is applied to the stems and the mixture so
    the sum-to-mixture contract survives.  This is the last big realism gap in
    a pipeline that otherwise only varies levels with linear gains: real mixes
    are heavily compressed, with pumping and ducking that flat gains never
    reproduce.  Frame-level envelope + 1-pole smoothing keeps it vectorized.
    """
    mixture = targets.sum(dim=0)
    env = _frame_rms(mixture, frame)
    env_db = 20.0 * torch.log10(env)
    threshold_db = random.uniform(-38.0, -12.0)
    ratio = random.uniform(2.0, 8.0)
    mix = random.uniform(0.3, 1.0)
    gain_db = torch.clamp((env_db - threshold_db) * (1.0 - 1.0 / ratio), max=0.0)
    alpha = random.uniform(0.25, 0.6)
    smoothed = torch.empty_like(gain_db)
    acc = gain_db[0].clone()
    for i in range(gain_db.shape[0]):
        acc = alpha * acc + (1.0 - alpha) * gain_db[i]
        smoothed[i] = acc
    gain = torch.pow(10.0, (mix * smoothed) / 20.0)
    gain_full = F.interpolate(
        gain.view(1, 1, -1),
        size=mixture.shape[-1],
        mode="linear",
        align_corners=False,
    ).view(-1)
    return targets * gain_full.view(1, 1, -1)


def _delay(audio: torch.Tensor, samples: int) -> torch.Tensor:
    """Zero-padded delay of [..., samples]; safe for out-of-range delays."""
    if samples <= 0 or samples >= audio.shape[-1]:
        return torch.zeros_like(audio)
    out = torch.zeros_like(audio)
    out[..., samples:] = audio[..., :-samples]
    return out


def apply_echo(audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Multi-tap slapback / delay echo (FIR: no feedback recursion, no pre-echo)."""
    taps = random.randint(1, 3)
    delay_ms = random.uniform(40.0, 320.0)
    gain = random.uniform(0.15, 0.45)
    out = audio.clone()
    for tap in range(1, taps + 1):
        d = int(
            delay_ms * tap * (1.0 + random.uniform(-0.1, 0.1)) / 1000.0 * sample_rate
        )
        out = out + gain / tap * _delay(audio, d)
    return out


def apply_chorus(audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Static chorus / Haas decorrelation: short per-channel delays.

    A real chorus sweeps a delay LFO; a static short delay with a darkened
    copy is a faithful-enough stand-in that costs only a few shifted adds.
    """
    out = audio.clone()
    for ch in range(audio.shape[0]):
        d = int(random.uniform(5.0, 30.0) / 1000.0 * sample_rate)
        g = random.uniform(0.15, 0.4)
        delayed = _delay(audio[ch], d)
        # Darken the delayed copy with a cheap 3-tap moving average.
        delayed = (delayed + torch.roll(delayed, 1) + torch.roll(delayed, 2)) / 3.0
        out[ch] = audio[ch] + g * delayed
    return out


def apply_pan_sweep(audio: torch.Tensor) -> torch.Tensor:
    """Slow mid/side pan movement: the stereo image drifts over the segment."""
    n = audio.shape[-1]
    mid = (audio[0] + audio[1]) * 0.5
    side = (audio[0] - audio[1]) * 0.5
    start = random.uniform(0.5, 1.2)
    end = random.uniform(0.5, 1.2)
    ramp = torch.linspace(start, end, n)
    out = audio.clone()
    out[0] = mid + side * ramp
    out[1] = mid - side * ramp
    return out


def apply_vocal_tremolo(audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Slow amplitude modulation on the vocal stem (sung tremolo)."""
    n = audio.shape[-1]
    t = torch.arange(n, dtype=torch.float32) / sample_rate
    depth = random.uniform(0.03, 0.12)
    rate = random.uniform(1.0, 5.0)
    phase = random.uniform(0.0, 2.0 * math.pi)
    mod = 1.0 + depth * torch.sin(2.0 * math.pi * rate * t + phase)
    return audio * mod.view(1, -1)


def apply_vocal_doubling(vocal: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Double-tracked vocal: a slightly detuned, few-ms-delayed copy layered in.

    Pop vocals are double-tracked constantly; a resampled copy (+-1.5 %) plus
    a short delay gives the thickened chorus sound without a second take.
    """
    factor = 1.0 + random.uniform(-0.015, 0.015)
    n = vocal.shape[-1]
    copy = fft_resample(vocal, factor)
    if copy.shape[-1] >= n:
        copy = copy[..., :n]
    else:
        copy = F.pad(copy, (0, n - copy.shape[-1]))
    d = int(random.uniform(0.0, 15.0) / 1000.0 * sample_rate)
    gain = random.uniform(0.4, 0.9)
    return vocal + gain * _delay(copy, d)


# -----------------------------------------------------------------------------
# Lightweight zero-FFT augmentations (all O(n) vectorized, ~0.2-0.5 ms / 8s)
# -----------------------------------------------------------------------------

def apply_cheap_lowpass(audio: torch.Tensor) -> torch.Tensor:
    """3-7 tap box low-pass (moving average) via reflect padding.

    Cheap stand-in for a muffled mic / dull tape. No FFT, one avg_pool1d.
    """
    k = random.choice([3, 5, 7])
    # avg_pool1d is highly optimized; reflect pad avoids edge clicks.
    padded = F.pad(audio.unsqueeze(0), (k // 2, k // 2), mode="reflect").squeeze(0)
    # F.avg_pool1d expects [N,C,L]; treat channels as batch
    filtered = F.avg_pool1d(padded.unsqueeze(0), kernel_size=k, stride=1).squeeze(0)
    # blend so it reads as tone, not total loss of air
    mix = random.uniform(0.5, 1.0)
    return mix * filtered + (1.0 - mix) * audio


def apply_cheap_highpass(audio: torch.Tensor) -> torch.Tensor:
    """Thin high-pass: original minus 3-tap moving average.

    Simulates cheap laptop mic / aggressive low-cut. Zero FFT.
    """
    low = F.avg_pool1d(
        F.pad(audio.unsqueeze(0), (1, 1), mode="reflect").squeeze(0).unsqueeze(0),
        kernel_size=3, stride=1,
    ).squeeze(0)
    hp = audio - low
    mix = random.uniform(0.4, 0.9)
    return mix * hp + (1.0 - mix) * audio


def apply_telephone_bandlimit(audio: torch.Tensor) -> torch.Tensor:
    """Narrow telephone band: high-pass + low-pass via two box filters."""
    # high-pass first, then dull the top - both via avg_pool, still no FFT
    hp = audio - F.avg_pool1d(
        F.pad(audio.unsqueeze(0), (2, 2), mode="reflect").squeeze(0).unsqueeze(0),
        kernel_size=5, stride=1,
    ).squeeze(0)
    lp = F.avg_pool1d(F.pad(hp.unsqueeze(0), (2, 2), mode="reflect").squeeze(0).unsqueeze(0), kernel_size=5, stride=1).squeeze(0)
    mix = random.uniform(0.6, 1.0)
    return mix * lp + (1.0 - mix) * audio


def apply_bitcrush(audio: torch.Tensor) -> torch.Tensor:
    """Uniform quantization noise: 10-14 bit crush. One round(), no FFT."""
    bits = random.randint(10, 14)
    scale = float(1 << bits)
    # peak-normalized so step size is relative, not absolute
    peak = audio.abs().amax().clamp_min(1e-4)
    norm = audio / peak
    crushed = torch.round(norm * scale) / scale
    return crushed * peak


def apply_soft_clip(audio: torch.Tensor) -> torch.Tensor:
    """Hard digital clipping at 0.5..0.95 (cheap limiter simulation)."""
    gain = random.uniform(1.2, 2.2)
    thresh = random.uniform(0.5, 0.95)
    return torch.clamp(audio * gain, min=-thresh, max=thresh) / gain


def apply_mono_fold(audio: torch.Tensor) -> torch.Tensor:
    """Fold to mono: copy one channel to both (mono radio / phone)."""
    idx = random.randint(0, 1)
    mono = audio[idx : idx + 1].repeat(2, 1)
    return mono


def apply_micro_delay(audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Per-channel micro delay +-12 ms via _delay (no FFT, just shift).

    Simulates mic-bleed timing / small alignment drift.
    """
    out = audio.clone()
    for ch in range(audio.shape[0]):
        if random.random() < 0.7:
            d = int(random.uniform(-12.0, 12.0) / 1000.0 * sample_rate)
            if d == 0:
                continue
            if d > 0:
                out[ch] = _delay(audio[ch : ch + 1], d).squeeze(0)
            else:
                # negative delay = advance (roll left then zero pad tail)
                adv = -d
                tmp = torch.zeros_like(audio[ch])
                if adv < audio.shape[-1]:
                    tmp[..., :-adv] = audio[ch, adv:]
                out[ch] = tmp
    return out


def apply_static_pan(audio: torch.Tensor) -> torch.Tensor:
    """Constant pan law gain: independent L/R scalar (one multiply)."""
    g_l = random.uniform(0.7, 1.3)
    g_r = random.uniform(0.7, 1.3)
    out = audio.clone()
    out[0] *= g_l
    out[1] *= g_r
    # keep loudness roughly constant
    out *= (2.0 / (g_l + g_r))
    return out


def apply_fade_edges(audio: torch.Tensor) -> torch.Tensor:
    """Short linear fade in/out 5-40 ms at a random edge."""
    n = audio.shape[-1]
    out = audio.clone()
    if random.random() < 0.5:
        fade_len = int(random.uniform(0.005, 0.040) * 44100)  # sample_rate agnostic enough
        fade_len = min(fade_len, n // 4)
        ramp = torch.linspace(0.0, 1.0, fade_len, device=audio.device)
        out[:, :fade_len] *= ramp
    else:
        fade_len = int(random.uniform(0.005, 0.040) * 44100)
        fade_len = min(fade_len, n // 4)
        ramp = torch.linspace(1.0, 0.0, fade_len, device=audio.device)
        out[:, -fade_len:] *= ramp
    return out


def apply_channel_gain_jitter(audio: torch.Tensor) -> torch.Tensor:
    """Independent per-channel gain +-1.5 dB (cable / preamp trim)."""
    g = torch.empty(2, 1, device=audio.device).uniform_(0.84, 1.19)  # +-1.5 dB
    return audio * g


class StemDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        sample_rate: int = 44_100,
        segment_samples: int = 352_800,
        virtual_size: int = 50_000,
        remix_probability: float = 0.5,
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

        track_dirs = [
            os.path.join(root_dir, name)
            for name in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, name))
        ]
        print("Scanning track metadata...")
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

        if not self.tracks:
            raise RuntimeError(f"No complete {STEMS} tracks found under {root_dir!r}.")
        print(
            f"Cached {len(self.tracks)} complete tracks"
        )

    def __len__(self) -> int:
        return self.virtual_size

    def _load_segment(self, info: AudioInfo, start_seconds: float) -> torch.Tensor:
        source_start = int(round(start_seconds * info.sample_rate))
        source_frames = int(math.ceil(self.segment_seconds * info.sample_rate))
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

        if audio.shape[-1] < self.segment_samples:
            audio = F.pad(audio, (0, self.segment_samples - audio.shape[-1]))
        else:
            audio = audio[..., : self.segment_samples]
        return audio.contiguous()

    def _load_target(
        self,
        infos: Sequence[AudioInfo],
        start_seconds: float,
    ) -> torch.Tensor:
        components = [self._load_segment(info, start_seconds) for info in infos]
        return torch.stack(components).sum(dim=0)

    @staticmethod
    def _target_duration(infos: Sequence[AudioInfo]) -> float:
        return min(info.duration for info in infos)

    def _random_start(self, infos: Sequence[AudioInfo]) -> float:
        duration = self._target_duration(infos)
        return random.uniform(0.0, max(0.0, duration - self.segment_seconds))

    def _sample_targets(self) -> torch.Tensor:
        targets: list[torch.Tensor] = []
        if random.random() < self.remix_probability:
            for stem in STEMS:
                track = random.choice(self.tracks)
                infos = track[stem]
                targets.append(self._load_target(infos, self._random_start(infos)))
        else:
            track = random.choice(self.tracks)
            common_duration = min(
                self._target_duration(track[stem]) for stem in STEMS
            )
            start = random.uniform(0.0, max(0.0, common_duration - self.segment_seconds))
            for stem in STEMS:
                targets.append(self._load_target(track[stem], start))
        return torch.stack(targets)

    @staticmethod
    def _augment(targets: torch.Tensor, sample_rate: int) -> torch.Tensor:
        gains_db = torch.empty(targets.shape[0]).uniform_(-8.0, 4.0)
        gains = torch.pow(10.0, gains_db / 20.0).view(-1, 1, 1)
        targets = targets * gains

        for stem_index in range(targets.shape[0]):
            if random.random() < 0.5:
                targets[stem_index] = -targets[stem_index]
            if stem_index == 0:
                # Vocals may be pushed far out to the sides or squashed to
                # near-mono (phone / live / mono-synth mixes); accompaniment
                # keeps a more conservative width range.
                if random.random() < 0.3:
                    width = random.uniform(0.0, 0.4)
                else:
                    width = random.uniform(0.5, 1.6)
            else:
                width = random.uniform(0.75, 1.25)
            mid = (targets[stem_index, 0] + targets[stem_index, 1]) * 0.5
            side = (targets[stem_index, 0] - targets[stem_index, 1]) * 0.5 * width
            targets[stem_index, 0] = mid + side
            targets[stem_index, 1] = mid - side

        # Whole-crop key/tempo variation, before any per-stem effects.
        if random.random() < 0.20:
            targets = pitch_shift_crop(targets)

        # --- Lightweight global color (all O(n), no FFT) ---
        # These are intentionally low-probability and vectorized so 8 workers
        # stay well under CPU budget. Each is a single multiply / shift / pool.
        if random.random() < 0.08:
            # independent L/R trim on whole mix (keeps sum-to-mixture)
            g = torch.empty(targets.shape[0], 2, 1).uniform_(0.84, 1.19)
            targets = targets * g
        if random.random() < 0.06:
            # whole-crop mono fold (mono compatibility)
            c = random.randint(0, 1)
            targets = targets[:, c : c + 1, :].repeat(1, 2, 1) if random.random() < 0.5 else targets
        if random.random() < 0.08:
            # whole-crop micro-alignment drift +-6 ms (same shift on all stems)
            d = int(random.uniform(-6.0, 6.0) / 1000.0 * sample_rate)
            if d != 0:
                if d > 0:
                    targets = torch.stack([_delay(t, d) for t in targets])
                else:
                    adv = -d
                    tmp = torch.zeros_like(targets)
                    if adv < targets.shape[-1]:
                        tmp[..., :-adv] = targets[..., adv:]
                    targets = tmp

        # Add occasional *local* vocal gaps, but never manufacture an entire
        # six-second no-vocal example.  Whole-segment erasure combined with
        # relative source losses strongly rewards the degenerate solution
        # "vocals = 0, other = mixture".  Natural fully silent crops are still
        # preserved by the dataset, while these short gaps teach low leakage.
        vocal = targets[0]
        if random.random() < 0.15:
            samples = vocal.shape[-1]
            for _ in range(random.randint(1, 2)):
                span = random.randint(max(64, samples // 50), max(65, samples // 10))
                start = random.randint(0, max(0, samples - span))
                end = min(samples, start + span)
                ramp = min(512, max(0, (end - start) // 8))
                envelope = vocal.new_ones(samples)
                envelope[start:end] = 0.0
                if ramp > 1:
                    fade = torch.linspace(1.0, 0.0, ramp, device=vocal.device)
                    left = max(0, start - ramp)
                    if left < start:
                        envelope[left:start] = torch.minimum(
                            envelope[left:start], fade[-(start-left):]
                        )
                    right = min(samples, end + ramp)
                    if end < right:
                        envelope[end:right] = torch.minimum(
                            envelope[end:right], fade[: right-end].flip(0)
                        )
                vocal.mul_(envelope)

        # --- Vocal signal chain ---
        # Signal-flow order: timbre (EQ, drive) first, then time-based effects
        # (chorus, echo, reverb), then level effects (doubling, tremolo) and
        # the synthetic vowel.  The mixture is built after augmentation, so the
        # model sees every effect as part of the vocal stem it must recover.
        # Note: no noise floor here.  Independent per-stem hiss is unrecoverable
        # by a mask separator (two uncorrelated broadband noises cannot be told
        # apart), so the model answered with a high-frequency static riding the
        # voice.  Hiss is a recording-chain property: it lives in the
        # accompaniment, and the model should learn to push it there, keeping
        # the vocal clean.
        vocal = targets[0]
        if random.random() < 0.25:
            vocal = random_spectral_eq(vocal, sample_rate)
        if random.random() < 0.09:
            vocal = apply_cheap_lowpass(vocal)
        if random.random() < 0.09:
            vocal = apply_cheap_highpass(vocal)
        if random.random() < 0.06:
            vocal = apply_telephone_bandlimit(vocal)
        if random.random() < 0.25:
            vocal = apply_saturation(vocal)
        if random.random() < 0.08:
            vocal = apply_soft_clip(vocal)
        if random.random() < 0.07:
            vocal = apply_bitcrush(vocal)
        if random.random() < 0.20:
            vocal = apply_chorus(vocal, sample_rate)
        if random.random() < 0.25:
            vocal = apply_echo(vocal, sample_rate)
        if random.random() < 0.25:
            vocal = apply_reverb(vocal, sample_rate)
        if random.random() < 0.20:
            vocal = apply_reverse_echo(vocal, sample_rate)
        if random.random() < 0.20:
            vocal = apply_air_boost(vocal)
        if random.random() < 0.10:
            vocal = apply_micro_delay(vocal, sample_rate)
        if random.random() < 0.10:
            vocal = apply_static_pan(vocal)
        if random.random() < 0.08:
            vocal = apply_channel_gain_jitter(vocal)
        if random.random() < 0.07:
            vocal = apply_fade_edges(vocal)
        if random.random() < 0.15:
            vocal = apply_vocal_doubling(vocal, sample_rate)
        if random.random() < 0.15:
            vocal = apply_vocal_tremolo(vocal, sample_rate)
        if random.random() < 0.22:
            vocal = add_sustained_vowel(vocal, sample_rate)
        if random.random() < 0.06:
            vocal = apply_mono_fold(vocal)
        targets[0] = vocal

        # --- Accompaniment signal chain ---
        # Wet instruments are just as common as wet vocals; EQ, drive, noise
        # and pan movement widen the instrument timbre space too.  The noise
        # floor lives here: a single correlated hiss in the instrumental is
        # physically believable (tape hiss / room tone) and the model learns
        # to route it to the accompaniment instead of smearing it on the
        # vocal as static.
        other = targets[1]
        if random.random() < 0.30:
            other = random_spectral_eq(other, sample_rate)
        if random.random() < 0.10:
            other = apply_cheap_lowpass(other)
        if random.random() < 0.10:
            other = apply_cheap_highpass(other)
        if random.random() < 0.07:
            other = apply_telephone_bandlimit(other)
        if random.random() < 0.20:
            other = apply_saturation(other)
        if random.random() < 0.09:
            other = apply_soft_clip(other)
        if random.random() < 0.07:
            other = apply_bitcrush(other)
        if random.random() < 0.10:
            other = apply_micro_delay(other, sample_rate)
        if random.random() < 0.12:
            other = apply_channel_gain_jitter(other)
        if random.random() < 0.07:
            other = apply_fade_edges(other)
        if random.random() < 0.25:
            other = apply_reverb(other, sample_rate)
        if random.random() < 0.15:
            other = apply_pan_sweep(other)
        if random.random() < 0.12:
            other = apply_static_pan(other)
        if random.random() < 0.30:
            other = add_synth_lead(other, vocal, sample_rate)
        if random.random() < 0.20:
            other = add_noise_floor(other, sample_rate)
        targets[1] = other

        # Mix-bus dynamics: the same gain curve on every stem, so real-world
        # compression/pumping is present without breaking sum-to-mixture.
        if random.random() < 0.25:
            targets = apply_bus_compression(targets)

        if random.random() < 0.5:
            targets = targets.flip(dims=(1,))
        if random.random() < 0.5:
            # Time reversal: free (no FFT), and a strong regularizer -- the
            # network cannot lean on direction-of-time cues.
            targets = targets.flip(dims=(2,))

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
                targets = self._augment(self._sample_targets(), self.sample_rate)
                mixture = targets.sum(dim=0)
                if mixture.square().mean().sqrt() < self.min_activity_rms:
                    continue
                return mixture, targets
            except Exception as error:  # corrupted files should not kill workers
                last_error = error
        raise RuntimeError(f"Unable to load a valid training example: {last_error}")


# -----------------------------------------------------------------------------
# EMA, optimizer, checkpointing
# -----------------------------------------------------------------------------


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must be in [0, 1).")
        self.model = model
        self.decay = decay
        self.updates = 0
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.backup: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self) -> None:
        self.updates += 1
        # Warm the decay up so early validation is not dominated by the random
        # initialization. It approaches the requested long-horizon decay.
        warm_decay = (1.0 + self.updates) / (10.0 + self.updates)
        decay = min(self.decay, warm_decay)
        trainable = [
            (self.shadow[name], parameter.detach())
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
                parameter.copy_(self.shadow[name])

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
    """Minimal adapter that lets EMA tensors use the strict state-load report."""

    def __init__(self, shadow: dict[str, torch.Tensor]):
        super().__init__()
        self.shadow = shadow

    def state_dict(self, *args, **kwargs):  # type: ignore[override]
        del args, kwargs
        return self.shadow.copy()

    def load_state_dict(self, state_dict, strict=False):  # type: ignore[override]
        del strict
        for name, value in state_dict.items():
            self.shadow[name].copy_(value)
        return None


def build_optimizer(
    model: nn.Module,
    weight_decay: float,
    slice_p: int,
) -> Prodigy:
    return Prodigy(
        model.parameters(),
        lr=1.0,
        weight_decay=weight_decay,
        slice_p=slice_p,
    )


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
    model: BSRoFormerSeparator,
    ema: EMA,
    optimizer: Prodigy,
    scaler: torch.amp.GradScaler,
    step: int,
    best_sdr: float,
    avg_loss: float,
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
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def prune_old_checkpoints(
    folder: str,
    keep: int = 3,
    config: ModelConfig | None = None,
) -> None:
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
    paths.sort(key=lambda path: path.stat().st_mtime)
    for path in paths[:-keep]:
        path.unlink(missing_ok=True)


# -----------------------------------------------------------------------------
# Inference and validation
# -----------------------------------------------------------------------------


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
        window[:fade_length] = fade
    if fade_out:
        window[-fade_length:] = fade.flip(0)
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
    model: BSRoFormerSeparator,
    mixture: torch.Tensor,
    chunk_size: int,
    overlap: int,
    device: torch.device,
    precision: str = "bf16",
    show_progress: bool = False,
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
    total_length = mixture.shape[-1]
    starts = chunk_starts(total_length, chunk_size, overlap)
    output = torch.zeros(
        model.config.num_stems,
        model.config.audio_channels,
        total_length,
        device=device,
    )
    weight_sum = torch.zeros(total_length, device=device)
    stft_window = torch.hann_window(model.config.win_length, device=device)

    iterator: Iterable[int] = starts
    if show_progress:
        iterator = tqdm(starts, desc="Separating", leave=False)

    for start in iterator:
        usable = min(chunk_size, total_length - start)
        chunk = mixture[:, start : start + usable]
        if usable < chunk_size:
            pad = chunk_size - usable
            if usable > 1:
                reflect = min(pad, usable - 1)
                chunk = F.pad(chunk, (0, reflect), mode="reflect")
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
            estimated_specs, _ = model.estimate_specs(spec)
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

    # Enforce exact waveform mixture consistency after overlap-add. Do not
    # clamp; clipping predictions changes SDR and belongs only at export.
    residual = mixture - output.sum(dim=0)
    output[1] = output[1] + residual
    return [output[index] for index in range(model.config.num_stems)]


def calculate_track_sdr(
    prediction: torch.Tensor,
    target: torch.Tensor,
    accumulation_samples: int,
    activity_threshold: float = 1e-4,
) -> float | None:
    """Scale-dependent SDR over the entire track."""
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
def validate(
    model: BSRoFormerSeparator,
    test_dir: str,
    device: torch.device,
    chunk_size: int,
    overlap: int,
    precision: str,
) -> tuple[list[float], float | None]:
    model.eval()
    track_dirs = [
        os.path.join(test_dir, name)
        for name in os.listdir(test_dir)
        if os.path.isdir(os.path.join(test_dir, name))
    ] if os.path.isdir(test_dir) else []
    if not track_dirs:
        print(f"No validation tracks found under {test_dir!r}.")
        return [0.0 for _ in STEMS], None

    per_stem_track_scores: list[list[float]] = [[] for _ in STEMS]
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
                    if sample_rate != model.config.sample_rate:
                        raise ValueError(
                            f"Validation file {path} is {sample_rate} Hz, expected "
                            f"{model.config.sample_rate} Hz."
                        )
                    audio = torch.nan_to_num(torch.from_numpy(audio_np.T))
                    if audio.shape[0] == 1:
                        audio = audio.repeat(2, 1)
                    components.append(audio[:2])
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
            )
            scores = [
                calculate_track_sdr(
                    pred,
                    target,
                    accumulation_samples=chunk_size,
                )
                for pred, target in zip(predictions, targets)
            ]
            for index, score in enumerate(scores):
                if score is not None:
                    per_stem_track_scores[index].append(score)
            valid_tracks += 1
            progress.set_postfix_str(
                " | ".join(
                    f"{stem}: {score:.3f}" if score is not None else f"{stem}: inactive"
                    for stem, score in zip(STEMS, scores)
                )
            )
        except Exception as error:
            print(f"\nSkipping {track_dir}: {error}")

    if valid_tracks == 0 or not any(per_stem_track_scores):
        return [0.0 for _ in STEMS], None
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
    return means, combined


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------


def train(
    model: BSRoFormerSeparator,
    dataloader: DataLoader,
    optimizer: Prodigy,
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
                "Checkpoint architecture mismatch: the saved model_config does not "
                "match the current ModelConfig. Only exact-continuation checkpoints "
                "of the current architecture are supported."
            )
        # A true continuation must resume the raw trainable weights alongside
        # optimizer/EMA state, not replace them with averaged weights.
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

    # Initialize after model loading so any loaded checkpoint also seeds EMA.
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

        if args.reset_optimizer:
            print(
                "Loaded exact raw weights, but --reset_optimizer starts a fresh "
                "Prodigy optimizer while keeping the EMA and step timeline."
            )
        else:
            saved_optimizer_class = checkpoint_data.get("optimizer_class")
            if saved_optimizer_class != "Prodigy":
                raise RuntimeError(
                    "Continuation checkpoint must contain a Prodigy optimizer; "
                    f"found {saved_optimizer_class or 'no optimizer class metadata'}. "
                    "Use --reset_optimizer to discard its optimizer state explicitly."
                )
            if "optimizer_state_dict" not in checkpoint_data:
                raise RuntimeError("Continuation checkpoint has no optimizer state.")
            optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
            resumed_slice_p = int(optimizer.param_groups[0]["slice_p"])
            if resumed_slice_p != args.slice_p:
                print(
                    f"Preserving checkpoint slice_p={resumed_slice_p}; changing "
                    "it would be incompatible with the saved Prodigy state."
                )
            for group in optimizer.param_groups:
                group["lr"] = 1.0
                group["weight_decay"] = args.weight_decay

        if "scaler_state_dict" in checkpoint_data:
            scaler.load_state_dict(checkpoint_data["scaler_state_dict"])
        # Free the large checkpoint dict (model+ema+optimizer ~1GB on CPU)
        # to avoid overnight RSS growth. Without this the dict stays alive
        # for the entire training run.
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
                f"Recovered newer best SDR {best_sdr:.4f} dB from "
                f"{best_checkpoint}."
            )
        optimizer_status = "Fresh" if args.reset_optimizer else "Resuming"
        active_slice_p = int(optimizer.param_groups[0]["slice_p"])
        print(
            f"{optimizer_status} Prodigy optimizer at checkpoint step {step} "
            f"with lr=1.0, weight_decay={args.weight_decay:.2e}, and "
            f"slice_p={active_slice_p}."
        )

    if args.compile:
        model.encoder.compile_layers(mode="default")
        print(
            f"Compiled {len(model.encoder.time_layers) + len(model.encoder.freq_layers)} "
            "transformer units; activation checkpoint boundaries remain eager."
        )

    stft_window = torch.hann_window(model.config.win_length, device=device)
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(
        initial=step,
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
        model.train()
        accumulated_loss_tensor = torch.zeros((), device=device)
        latest_metrics: dict[str, torch.Tensor] = {}
        completed_accumulation = True

        for _micro_step in range(args.grad_accumulation):
            try:
                mixture_audio, target_audio = next(data_iterator)
            except StopIteration:
                # Properly discard the exhausted iterator so its prefetch
                # queue + pin_memory buffers are freed. Overwriting without
                # del caused the old queue to stay alive and leak CPU RAM.
                try:
                    if hasattr(data_iterator, "_shutdown_workers"):
                        data_iterator._shutdown_workers()  # type: ignore[attr-defined]
                except Exception:
                    pass
                del data_iterator
                gc.collect()
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
                loss, latest_metrics = loss_module(
                    model,
                    mixture_spec,
                    target_audio,
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
        current_lr = optimizer.param_groups[0]["lr"]
        progress.set_description(
            f"Step {step} | loss {accumulated_loss:.4f} | avg {avg_loss:.4f} "
            f"| lr {current_lr:.2e} | grad {float(grad_norm):.2f} "
            f"| best {best_sdr:.4f}",
            refresh=False,
        )
        if latest_metrics:
            wave, main_stft, mrstft, vocal_db, mask_mag = torch.stack(
                (
                    latest_metrics["wave"],
                    latest_metrics["main_stft"],
                    latest_metrics["mrstft"],
                    latest_metrics["vocal_level_db"],
                    latest_metrics["vocal_mask_mag"],
                )
            ).float().cpu().tolist()
            progress.set_postfix(
                wave=f"{wave:.3f}",
                stft=f"{main_stft:.3f}",
                mr=f"{mrstft:.3f}",
                vdb=f"{vocal_db:+.1f}",
                vmask=f"{mask_mag:.3f}",
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
                # The transformer units are compiled for training with gradients
                # enabled.  Validation runs under inference_mode, which requires
                # different Dynamo guards and can exhaust the shared recompile
                # cache for TransformerUnit.forward.  Keep infrequent validation
                # eager so it cannot evict or disable the compiled training path.
                compiler_context = (
                    torch.compiler.set_stance("force_eager")
                    if args.compile
                    else contextlib.nullcontext()
                )
                with compiler_context:
                    stem_scores, combined_sdr = validate(
                        model,
                        args.test_dir,
                        device,
                        chunk_size=args.segment_samples,
                        overlap=args.inference_overlap,
                        precision=args.precision,
                    )

            improved = combined_sdr is not None and combined_sdr > best_sdr
            if combined_sdr is None:
                print("\nValidation produced no valid tracks; best SDR was not changed.")
            else:
                score_text = ", ".join(
                    f"{stem}: {score:.4f} dB"
                    for stem, score in zip(STEMS, stem_scores)
                )
                print(
                    f"\nValidation step {step} "
                    "(EMA, mean full-track SDR): "
                    f"{score_text}, combined: {combined_sdr:.4f} dB"
                )
                if improved:
                    best_sdr = combined_sdr

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
                )
                prune_old_checkpoints(
                    "best_ckpts", keep=1, config=model.config
                )
                print(f"New best checkpoint: {best_path}\n")
            elif combined_sdr is not None:
                print(f"Best combined SDR remains {best_sdr:.4f} dB.\n")
            # Validation loads full tracks + checkpoints on CPU; ensure
            # memory is returned to OS instead of accumulating overnight.
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    progress.close()
    signal.signal(signal.SIGINT, previous_sigint_handler)
    # Ensure DataLoader workers + pin_memory thread are terminated and
    # not counted as leaked RAM in Task Manager.
    try:
        if "data_iterator" in locals() and hasattr(data_iterator, "_shutdown_workers"):
            data_iterator._shutdown_workers()  # type: ignore[attr-defined]
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


# -----------------------------------------------------------------------------
# Command line entry point
# -----------------------------------------------------------------------------


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
        heads=args.heads,
        dropout=args.dropout,
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
        filtered = {key: value for key, value in config_data.items() if key in valid_fields}
        return ModelConfig(**filtered)
    finally:
        del checkpoint_data
        gc.collect()


def load_inference_weights(
    model: BSRoFormerSeparator,
    checkpoint_path: str,
) -> None:
    checkpoint_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    try:
        state = checkpoint_data.get("ema_state_dict") or checkpoint_data.get("model_state_dict")
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
        description=(
            "124-band regular BS-RoFormer with axial attention, "
            "foreground-residual separation, and silence-focused training"
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train", action="store_true")
    mode.add_argument("--infer", action="store_true")

    parser.add_argument("--data_dir", type=str, default="train")
    parser.add_argument("--test_dir", type=str, default="test")
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--latest", action="store_true")
    parser.add_argument("--reset_optimizer", action="store_true")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Start a new run instead of auto-resuming the latest compatible checkpoint.",
    )

    parser.add_argument("--sample_rate", type=int, default=44_100)
    parser.add_argument("--n_fft", type=int, default=2048)
    parser.add_argument("--hop_length", type=int, default=512)
    parser.add_argument("--win_length", type=int, default=2048)
    parser.add_argument("--segment_seconds", type=float, default=8)
    parser.add_argument(
        "--inference_overlap_seconds",
        type=float,
        default=3,
        help=(
            "Chunk overlap in seconds."
        ),
    )

    parser.add_argument("--model_dim", type=int, default=384)
    parser.add_argument("--depth", type=int, default=14)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--ckpt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accumulation", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--dataset_size", type=int, default=50_000)
    parser.add_argument("--remix_probability", type=float, default=0.5)
    parser.add_argument("--checkpoint_steps", type=int, default=4_000)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument(
        "--slice_p",
        type=int,
        default=1,
        help="Prodigy memory-saving update slicing; use 11 when memory is limited.",
    )
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--seed", type=int, default=1337)
    return parser


def validate_runtime_args(args: argparse.Namespace) -> None:
    positive_integer_fields = (
        "batch_size",
        "grad_accumulation",
        "dataset_size",
        "checkpoint_steps",
    )
    for field in positive_integer_fields:
        if getattr(args, field) <= 0:
            raise ValueError(f"--{field} must be positive.")
    if args.num_workers < 0:
        raise ValueError("--num_workers cannot be negative.")
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
    if args.inference_overlap_seconds < 0.0:
        raise ValueError("--inference_overlap_seconds cannot be negative.")
    if args.dataset_size < args.batch_size:
        raise ValueError("--dataset_size must be at least --batch_size.")


def main() -> None:
    args = build_parser().parse_args()
    validate_runtime_args(args)
    seed_everything(args.seed)
    os.makedirs("ckpts", exist_ok=True)
    os.makedirs("best_ckpts", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

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
                    "No compatible auto-resume checkpoint was found. Starting fresh; "
                    "use --checkpoint_path to resume an exact checkpoint."
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

    if args.infer and checkpoint_path:
        config = inspect_checkpoint_config(checkpoint_path, config)
        # Keep chunk timing tied to the checkpoint sample rate.
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

    model = BSRoFormerSeparator(config)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"124-band axial RoFormer parameters: {parameter_count / 1e6:.2f}M")

    if args.train:
        if checkpoint_path:
            print(f"Checkpoint selected: {checkpoint_path}")
        dataset = StemDataset(
            root_dir=args.data_dir,
            sample_rate=config.sample_rate,
            segment_samples=args.segment_samples,
            virtual_size=args.dataset_size,
            remix_probability=args.remix_probability,
        )
        generator = torch.Generator()
        generator.manual_seed(args.seed)
        # pin_memory + persistent_workers on Windows (spawn) is the #1
        # cause of overnight CPU RAM leaks: the pin_memory thread caches
        # cudaHostAlloc blocks forever and persistent workers keep their
        # prefetch queues alive even after the iterator is overwritten.
        # Disable both for stable long runs. Re-enable only if you
        # explicitly need the ~5% speedup and can monitor RAM.
        use_pin_memory = False
        use_persistent = False
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
        loss_module = SeparationLoss(config, LossConfig())
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
    mixture = read_input_audio(args.input_file, config.sample_rate)
    predictions = separate_tensor(
        model,
        mixture,
        chunk_size=args.segment_samples,
        overlap=args.inference_overlap,
        device=device,
        precision=args.precision,
        show_progress=True,
    )
    for stem, prediction in zip(STEMS, predictions):
        output_path = os.path.join("outputs", f"{stem}.wav")
        # WAV encoders may clip at export, but internal inference remains unclamped.
        sf.write(output_path, prediction.cpu().numpy().T, config.sample_rate, subtype="FLOAT")
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()