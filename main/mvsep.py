from __future__ import annotations

import argparse
import contextlib
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
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from flash_attn import flash_attn_func as external_flash_attn_func
except (ImportError, OSError) as error:
    external_flash_attn_func = None
    FLASH_ATTN_IMPORT_ERROR: Exception | None = error
else:
    FLASH_ATTN_IMPORT_ERROR = None


STEMS = ("vocals", "other")
AUDIO_EXTENSIONS = (".wav", ".flac")
VALIDATION_METRIC = "mean_full_track_sdr_v1"


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
    depth: int = 12
    heads: int = 8
    ff_mult: float = 8.0 / 3.0
    dropout: float = 0.0
    layer_scale_init: float = 0.1
    use_checkpoint: bool = True
    architecture: str = "bs_roformer_124"

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
        if self.ff_mult <= 0.0:
            raise ValueError("ff_mult must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if self.layer_scale_init < 0.0:
            raise ValueError("layer_scale_init must be non-negative.")
        if self.architecture != "bs_roformer_124":
            raise ValueError(
                f"Unsupported architecture {self.architecture!r}; expected bs_roformer_124."
            )


@dataclass
class LossConfig:
    waveform_weight: float = 1.0
    main_stft_weight: float = 0.5
    mrstft_weight: float = 1.0
    mask_weight: float = 0.1
    sdr_weight: float = 0.05
    midside_weight: float = 0.1


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

    Partial loading is useful for transfer learning, but it must never be
    mistaken for an exact continuation checkpoint.
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
    """Build disjoint BS-RoFormer frequency bands.

    The default 124-band / 2048-FFT preset refines the commonly used handcrafted
    62-band BS-RoFormer layout by splitting each band in two. It covers all 1025
    bins of a real 2048-point STFT with widths

        48x1, 24x2, 16x6, 16x12, 16x24, 3x64, 65.

    For non-default settings, a deterministic power-law layout is used. It is
    still a strict band split: no overlap, no duplicated bins, and no mask
    averaging.
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
            widths = [part for width in base_widths for part in (width // 2, width - width // 2)]
        else:
            widths = base_widths
    else:
        # A handcrafted-style fallback with many narrow low-frequency bands
        # and progressively wider high-frequency bands. This is deliberately
        # not a Mel filterbank and never overlaps bins.
        positions = torch.linspace(0.0, 1.0, num_bands + 1)
        boundaries = torch.round((positions.square()) * freq_bins).long()
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
            f"Invalid band layout: {len(widths)} bands cover {sum(widths)} "
            f"bins, expected {num_bands} bands covering {freq_bins} bins."
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


class GatedRoPEAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dropout: float = 0.0,
        use_value_residual: bool = True,
    ):
        super().__init__()
        if dim % heads != 0:
            raise ValueError("Model dimension must be divisible by the number of heads.")
        self.heads = heads
        self.head_dim = dim // heads
        self.dropout = dropout
        self.use_value_residual = use_value_residual
        # This is a runtime performance setting, not part of the model or its
        # checkpoint compatibility. ``fused`` prevents an unnoticed fallback
        # to the much slower quadratic-memory math implementation on CUDA.
        self.attention_backend = "fused"

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.head_gate = nn.Linear(dim, heads)
        self.value_mix = nn.Linear(dim, heads) if use_value_residual else None
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.out_dropout = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        value_residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, length, dim = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        original_v = v

        if value_residual is not None and self.value_mix is not None:
            mix = torch.sigmoid(self.value_mix(x)).transpose(1, 2).unsqueeze(-1)
            v = torch.lerp(v, value_residual, mix)

        cos, sin = self.rope(length, x.device, q.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        attention_dropout = self.dropout if self.training else 0.0
        use_external_flash = (
            external_flash_attn_func is not None
            and q.device.type == "cuda"
            and q.dtype in (torch.float16, torch.bfloat16)
            and self.attention_backend in ("fused", "flash")
        )
        if use_external_flash:
            # flash-attn uses [batch, sequence, heads, head_dim], whereas
            # PyTorch SDPA below uses [batch, heads, sequence, head_dim].
            out = external_flash_attn_func(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                dropout_p=attention_dropout,
                causal=False,
            )
            out = out.transpose(1, 2)
        else:
            if q.device.type != "cuda" or self.attention_backend == "auto":
                attention_context = contextlib.nullcontext()
            elif self.attention_backend == "fused":
                # Prefer built-in Flash when available, then cuDNN and
                # memory-efficient SDPA. Deliberately omit the math fallback.
                attention_context = sdpa_kernel(
                    [
                        SDPBackend.FLASH_ATTENTION,
                        SDPBackend.CUDNN_ATTENTION,
                        SDPBackend.EFFICIENT_ATTENTION,
                    ],
                    set_priority=True,
                )
            elif self.attention_backend == "flash":
                attention_context = sdpa_kernel(SDPBackend.FLASH_ATTENTION)
            else:
                attention_context = sdpa_kernel(SDPBackend.MATH)
            with attention_context:
                out = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    dropout_p=attention_dropout,
                    is_causal=False,
                )
        gates = torch.sigmoid(self.head_gate(x)).transpose(1, 2).unsqueeze(-1)
        out = out * gates
        out = out.transpose(1, 2).reshape(batch, length, dim)
        return self.out_dropout(self.out_proj(out)), original_v


class TransformerUnit(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        ff_mult: float,
        dropout: float,
        layer_scale_init: float,
    ):
        super().__init__()
        # 8/3 gives SwiGLU approximately the parameter count of a standard
        # 4x GELU FFN. Round the product, then align it for Tensor Cores.
        hidden_dim = round_up_to_multiple(round(dim * ff_mult), 64)

        self.attn_norm = nn.RMSNorm(dim)
        self.attn = GatedRoPEAttention(dim, heads, dropout=dropout)
        self.ff_norm = nn.RMSNorm(dim)
        self.ff = SwiGLU(dim, hidden_dim, dropout=dropout)
        self.attn_scale = nn.Parameter(torch.full((dim,), layer_scale_init))
        self.ff_scale = nn.Parameter(torch.full((dim,), layer_scale_init))

    def forward(
        self,
        x: torch.Tensor,
        value_residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attn_out, original_v = self.attn(self.attn_norm(x), value_residual)
        x = x + attn_out * self.attn_scale
        x = x + self.ff(self.ff_norm(x)) * self.ff_scale
        return x, original_v


class DualPathEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        unit_kwargs = dict(
            dim=config.dim,
            heads=config.heads,
            ff_mult=config.ff_mult,
            dropout=config.dropout,
            layer_scale_init=config.layer_scale_init,
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
        """Compile transformer units without capturing checkpoint boundaries.

        Compiling the complete separator graph makes TorchInductor capture the
        activation-checkpointing loop as part of one large graph.  Keeping the
        checkpoint calls eager and compiling only the work they wrap preserves
        checkpoint recomputation and its expected peak-memory behavior.
        """
        if mode != "default":
            raise ValueError("compile_layers currently supports only the default mode.")
        for unit in (*self.time_layers, *self.freq_layers):
            # Inductor's comprehensive padding can give saved RMSNorm tensors a
            # padded sequence stride (for example 1056 for 1034 frames). During
            # non-reentrant checkpoint recomputation it may instead produce the
            # ordinary contiguous stride, causing the compiled backward to reject
            # an otherwise identical tensor. Keep compilation enabled, but avoid
            # that checkpoint-incompatible internal layout optimization.
            unit.compile(options={"comprehensive_padding": False})

    @staticmethod
    def _run_unit(
        unit: TransformerUnit,
        x: torch.Tensor,
        value_residual: torch.Tensor | None,
        use_checkpoint: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not use_checkpoint:
            return unit(x, value_residual)
        if value_residual is None:
            return checkpoint(lambda z: unit(z, None), x, use_reentrant=False)
        return checkpoint(unit, x, value_residual, use_reentrant=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, frames, bands, dim]
        batch, frames, bands, dim = x.shape
        time_value_residual: torch.Tensor | None = None
        freq_value_residual: torch.Tensor | None = None
        should_checkpoint = self.use_checkpoint and self.training

        for time_layer, freq_layer in zip(self.time_layers, self.freq_layers):
            time_x = x.permute(0, 2, 1, 3).reshape(batch * bands, frames, dim)
            time_x, first_time_values = self._run_unit(
                time_layer,
                time_x,
                time_value_residual,
                should_checkpoint,
            )
            if time_value_residual is None:
                time_value_residual = first_time_values
            x = time_x.reshape(batch, bands, frames, dim).permute(0, 2, 1, 3)

            freq_x = x.reshape(batch * frames, bands, dim)
            freq_x, first_freq_values = self._run_unit(
                freq_layer,
                freq_x,
                freq_value_residual,
                should_checkpoint,
            )
            if freq_value_residual is None:
                freq_value_residual = first_freq_values
            x = freq_x.reshape(batch, frames, bands, dim)

        return self.output_norm(x)


def next_power_of_two(value: int) -> int:
    return 1 if value <= 1 else 1 << (value - 1).bit_length()


def round_up_to_multiple(value: float, multiple: int) -> int:
    if multiple <= 0:
        raise ValueError("multiple must be positive.")
    return int(math.ceil(value / multiple) * multiple)


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

    def forward(self, real_imag: torch.Tensor) -> torch.Tensor:
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
        return torch.einsum("btni,nid->btnd", features, self.weight) + self.bias


class BandSplit(nn.Module):
    """Project each disjoint complex stereo band into one token."""

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

    def forward_real(self, real_imag: torch.Tensor) -> torch.Tensor:
        """Project an STFT represented by a trailing real/imaginary dimension."""
        real_imag = real_imag.permute(0, 3, 2, 1, 4)  # [B, T, F, C, 2]
        output = real_imag.new_zeros(
            real_imag.shape[0],
            real_imag.shape[1],
            self.num_bands,
            self.groups[0].weight.shape[-1],
        )
        for group in self.groups:
            group_tokens = group(real_imag)
            output = output.index_copy(2, group.band_ids, group_tokens)
        return output

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
        self.num_stems = config.num_stems
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

        # A band-specific nonlinear mask head is materially more expressive than
        # projecting every band directly from the shared encoder representation.
        # Keeping the hidden width at ``dim`` controls the parameter cost of the
        # 124-band layout while still giving every band its own two-layer MLP.
        hidden_width = config.dim
        output_width = config.num_stems * self.feature_width
        self.hidden_weight = nn.Parameter(
            torch.empty(self.num_group_bands, config.dim, hidden_width)
        )
        self.hidden_bias = nn.Parameter(
            torch.zeros(self.num_group_bands, hidden_width)
        )
        self.output_weight = nn.Parameter(
            torch.empty(self.num_group_bands, hidden_width, output_width * 2)
        )
        self.output_bias = nn.Parameter(
            torch.zeros(self.num_group_bands, output_width * 2)
        )
        for band in range(self.num_group_bands):
            nn.init.xavier_uniform_(self.hidden_weight[band])
        nn.init.normal_(self.output_weight, std=1e-3)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, T, group_bands, D]
        hidden = torch.einsum("btnd,ndh->btnh", x, self.hidden_weight)
        hidden = torch.tanh(hidden + self.hidden_bias[None, None])
        raw = torch.einsum("btnh,nhq->btnq", hidden, self.output_weight)
        raw = F.glu(raw + self.output_bias[None, None], dim=-1)
        raw = raw.reshape(
            x.shape[0],
            x.shape[1],
            self.num_group_bands,
            self.num_stems,
            self.feature_width,
        )
        raw = raw * self.feature_valid[None, None, :, None]
        raw = raw.reshape(
            x.shape[0],
            x.shape[1],
            self.num_group_bands,
            self.num_stems,
            self.bucket_width,
            self.audio_channels,
            2,
        )
        source = raw.permute(0, 3, 5, 1, 2, 4, 6).reshape(
            x.shape[0],
            self.num_stems,
            self.audio_channels,
            x.shape[1],
            self.num_group_bands * self.bucket_width,
            2,
        )
        return source, self.freq_indices.reshape(-1)


class BandMaskEstimator(nn.Module):
    """Estimate one complex mask for every bin in each disjoint band."""

    def __init__(
        self,
        config: ModelConfig,
        bands: Sequence[tuple[int, int]],
        band_split: BandSplit,
    ):
        super().__init__()
        del band_split  # groups are reconstructed from the same immutable layout
        self.num_stems = config.num_stems
        self.audio_channels = config.audio_channels
        self.freq_bins = config.n_fft // 2 + 1
        self.num_bands = len(bands)

        grouped_ids: dict[int, list[int]] = {}
        for band_id, (start, end) in enumerate(bands):
            bucket = next_power_of_two(end - start)
            grouped_ids.setdefault(bucket, []).append(band_id)
        self.groups = nn.ModuleList(
            BandMaskGroup(config, bands, ids, bucket)
            for bucket, ids in sorted(grouped_ids.items())
        )

        coverage = torch.zeros(self.freq_bins, dtype=torch.int64)
        for start, end in bands:
            coverage[start:end] += 1
        if not torch.all(coverage == 1):
            raise ValueError("BandMaskEstimator requires a disjoint full-band layout.")

        hidden_dim = round_up_to_multiple(config.dim * 2.0, 64)
        self.norm = nn.RMSNorm(config.dim)
        self.shared_mlp = SwiGLU(
            config.dim, hidden_dim, dropout=config.dropout
        )
        self.mask_residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward_real(self, x: torch.Tensor) -> torch.Tensor:
        """Return masks represented by a trailing real/imaginary dimension."""
        # x: [B, T, bands, D]
        x = x + self.shared_mlp(self.norm(x))
        output = x.new_zeros(
            x.shape[0],
            self.num_stems,
            self.audio_channels,
            x.shape[1],
            self.freq_bins,
            2,
        )
        for group in self.groups:
            group_x = x.index_select(2, group.band_ids)
            source, flat_indices = group(group_x)
            scatter_index = flat_indices.view(1, 1, 1, 1, -1, 1).expand(
                x.shape[0],
                self.num_stems,
                self.audio_channels,
                x.shape[1],
                -1,
                2,
            )
            output.scatter_add_(dim=4, index=scatter_index, src=source)

        output = output.permute(0, 1, 2, 4, 3, 5).contiguous().float()
        output = output * self.mask_residual_scale.float()
        mask_bias = output.new_tensor((1.0 / self.num_stems, 0.0))
        return output + mask_bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.view_as_complex(self.forward_real(x))


class BSRoFormerSeparator(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.bands = build_bs_bands(
            config.n_fft,
            config.num_bands,
        )
        self.band_split = BandSplit(config, self.bands)
        self.encoder = DualPathEncoder(config)
        self.mask_estimator = BandMaskEstimator(
            config,
            self.bands,
            self.band_split,
        )

    def forward_real(self, mixture_real_imag: torch.Tensor) -> torch.Tensor:
        """Run the separator with real-valued graph inputs and outputs.

        Keeping complex tensors outside this method lets TorchInductor compile the
        compute-heavy model on backends that cannot generate Triton signatures for
        complex dtypes.
        """
        tokens = self.band_split.forward_real(mixture_real_imag)
        tokens = self.encoder(tokens)
        return self.mask_estimator.forward_real(tokens)

    def forward(self, mixture_spec: torch.Tensor) -> torch.Tensor:
        mixture_real_imag = torch.view_as_real(mixture_spec.to(torch.complex64))
        masks_real_imag = self.forward_real(mixture_real_imag)
        return torch.view_as_complex(masks_real_imag)

    def estimate_specs(
        self,
        mixture_spec: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        masks = self(mixture_spec)
        estimates = masks * mixture_spec[:, None]
        residual = mixture_spec - estimates.sum(dim=1)
        power = estimates.abs().square().clamp_min(1e-8)
        weights = power / power.sum(dim=1, keepdim=True).clamp_min(1e-8)
        estimates = estimates + weights * residual[:, None]
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
    error = (prediction - target).abs().mean(dim=-1)
    scale = target.abs().mean(dim=-1).clamp_min(1e-4)
    return (error / scale).mean()


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
    ratio_db = ratio_db.clamp(-30.0, 30.0)
    if valid.any():
        return -ratio_db[valid].mean()
    return prediction.new_tensor(0.0)


def mid_side(audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mid = (audio[..., 0, :] + audio[..., 1, :]) * 0.5
    side = (audio[..., 0, :] - audio[..., 1, :]) * 0.5
    return mid, side


class SeparationLoss(nn.Module):
    def __init__(self, model_config: ModelConfig, loss_config: LossConfig):
        super().__init__()
        self.model_config = model_config
        self.loss_config = loss_config
        self.mrstft = MultiResolutionSTFTLoss()
        self.register_buffer(
            "window",
            torch.hann_window(model_config.win_length),
            persistent=False,
        )

    def forward(
        self,
        model: BSRoFormerSeparator,
        mixture_spec: torch.Tensor,
        target_audio: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # target_audio: [B, S, C, samples]
        masks = model(mixture_spec)
        estimates = masks * mixture_spec[:, None]
        residual = mixture_spec - estimates.sum(dim=1)
        power = estimates.abs().square().clamp_min(1e-8)
        weights = power / power.sum(dim=1, keepdim=True).clamp_min(1e-8)
        estimates = estimates + weights * residual[:, None]

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
        main_logmag = F.l1_loss(
            torch.log1p(estimates.abs()),
            torch.log1p(target_mag),
        )
        main_stft_loss = main_complex + main_logmag

        mix_power = mixture_spec.abs().square()
        ideal_masks = (
            target_specs * mixture_spec[:, None].conj()
            / (mix_power[:, None] + 1e-5)
        )
        ideal_mag = ideal_masks.abs().clamp_max(8.0)
        ideal_phase = torch.angle(ideal_masks)
        ideal_masks = torch.polar(ideal_mag, ideal_phase)
        tf_weight = mixture_spec.abs()
        tf_weight = tf_weight / tf_weight.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-4)
        tf_weight = tf_weight.clamp(max=10.0)
        mask_loss = (
            (masks - ideal_masks).abs() * tf_weight[:, None]
        ).mean()

        sdr_loss = scale_dependent_sdr_loss(pred_audio, target_audio)
        pred_mid, pred_side = mid_side(pred_audio)
        true_mid, true_side = mid_side(target_audio)
        midside_loss = 0.5 * (
            normalized_l1(pred_mid, true_mid)
            + normalized_l1(pred_side, true_side)
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
        metrics = {
            "wave": wave_loss.detach(),
            "main_stft": main_stft_loss.detach(),
            "mrstft": mrstft_loss.detach(),
            "mask": mask_loss.detach(),
            "sdr_loss": sdr_loss.detach(),
            "midside": midside_loss.detach(),
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
        musdb_tracks = 0
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
            musdb_tracks += int(len(track["other"]) == 3)

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
    def _augment(targets: torch.Tensor) -> torch.Tensor:
        # Independent source loudness is the most useful augmentation for remix
        # generalization. Targets and mixture remain perfectly consistent.
        gains_db = torch.empty(targets.shape[0]).uniform_(-8.0, 4.0)
        gains = torch.pow(10.0, gains_db / 20.0).view(-1, 1, 1)
        targets = targets * gains

        for stem_index in range(targets.shape[0]):
            if random.random() < 0.5:
                targets[stem_index] = -targets[stem_index]
            width = random.uniform(0.75, 1.25)
            mid = (targets[stem_index, 0] + targets[stem_index, 1]) * 0.5
            side = (targets[stem_index, 0] - targets[stem_index, 1]) * 0.5 * width
            targets[stem_index, 0] = mid + side
            targets[stem_index, 1] = mid - side

        if random.random() < 0.5:
            targets = targets.flip(dims=(1,))

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
                targets = self._augment(self._sample_targets())
                mixture = targets.sum(dim=0)
                if mixture.square().mean().sqrt() < self.min_activity_rms:
                    continue
                return mixture, targets
            except Exception as error:  # corrupted files should not kill workers
                last_error = error
        raise RuntimeError(f"Unable to load a valid training example: {last_error}")


# -----------------------------------------------------------------------------
# EMA, optimizer, scheduler, checkpointing
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
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    decay_params: list[nn.Parameter] = []
    no_decay_params: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        lowered = name.lower()
        use_no_decay = (
            parameter.ndim < 2
            or name.endswith(".bias")
            or "norm" in lowered
            or name.endswith(".gamma")
            or name.endswith("_scale")
        )
        (no_decay_params if use_no_decay else decay_params).append(parameter)

    parameter_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        parameter_groups,
        lr=lr,
        betas=(0.9, 0.95),
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Warm up once, then hold the learning rate for an unbounded run."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return max(1e-8, (step + 1) / max(1, warmup_steps))
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def rebase_learning_rate(
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    lr: float,
) -> None:
    """Apply a new base LR without resetting the scheduler's resumed position."""
    if len(optimizer.param_groups) != len(scheduler.lr_lambdas):
        raise RuntimeError(
            "Optimizer parameter groups do not match the learning-rate scheduler."
        )

    scheduler.base_lrs = [lr for _ in optimizer.param_groups]
    if scheduler.last_epoch < 0:
        scheduled_lrs = scheduler.base_lrs.copy()
    else:
        scheduled_lrs = [
            lr * lr_lambda(scheduler.last_epoch)
            for lr_lambda in scheduler.lr_lambdas
        ]

    for parameter_group, scheduled_lr in zip(
        optimizer.param_groups, scheduled_lrs
    ):
        parameter_group["initial_lr"] = lr
        parameter_group["lr"] = scheduled_lr
    scheduler._last_lr = scheduled_lrs  # Keep its serialized/public state consistent.


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
        if (
            checkpoint_data.get("checkpoint_format_version", 0) >= 3
            and model_configs_compatible(checkpoint_data.get("model_config"), config)
        ):
            return str(path)
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
            if config is not None and (
                checkpoint_data.get("checkpoint_format_version", 0) < 3
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
        scored.append((score, path))
    return str(max(scored, key=lambda item: item[0])[1]) if scored else None


def save_checkpoint(
    path: str,
    model: BSRoFormerSeparator,
    ema: EMA,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
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
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_sdr": best_sdr,
        "validation_metric": VALIDATION_METRIC,
        "avg_loss": avg_loss,
        "stems": STEMS,
        "model_config": asdict(model.config),
        "checkpoint_format_version": 3,
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
            if (
                checkpoint_data.get("checkpoint_format_version", 0) < 3
                or not model_configs_compatible(
                    checkpoint_data.get("model_config"), config
                )
            ):
                continue
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
    power = output.abs().square().clamp_min(1e-8)
    weights = power / power.sum(dim=0, keepdim=True).clamp_min(1e-8)
    output = output + weights * residual.unsqueeze(0)
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
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
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
    exact_continuation = False

    if checkpoint_path:
        checkpoint_data = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        model_state = checkpoint_data.get("model_state_dict", checkpoint_data)
        report = load_matching_state_dict(model, model_state)
        print(
            f"Loaded {report.matched}/{report.expected} model tensors from "
            f"{checkpoint_path} ({len(report.skipped)} skipped, "
            f"{len(report.missing)} missing)."
        )
        saved_config = checkpoint_data.get("model_config")
        exact_continuation = (
            model_configs_compatible(saved_config, model.config) and report.is_exact
        )

    # Initialize after model loading so a transfer checkpoint also seeds EMA.
    ema = EMA(model, decay=args.ema_decay)

    if checkpoint_data is not None and exact_continuation:
        if args.reset_optimizer:
            print(
                "Loaded exact raw weights, but --reset_optimizer starts fresh "
                "optimizer, scheduler, EMA, and global-step timelines."
            )
        else:
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
            if "optimizer_state_dict" in checkpoint_data:
                saved_optimizer_class = checkpoint_data.get("optimizer_class")
                current_optimizer_class = optimizer.__class__.__name__
                if (
                    saved_optimizer_class is not None
                    and saved_optimizer_class != current_optimizer_class
                ):
                    raise RuntimeError(
                        f"Checkpoint optimizer is {saved_optimizer_class}, but the "
                        f"requested optimizer is {current_optimizer_class}. Use "
                        "--reset_optimizer when changing optimizer types."
                    )
                requested_weight_decays = [
                    float(group["weight_decay"]) for group in optimizer.param_groups
                ]
                optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
                for group, requested_weight_decay in zip(
                    optimizer.param_groups, requested_weight_decays
                ):
                    group["weight_decay"] = requested_weight_decay
            else:
                raise RuntimeError("Continuation checkpoint has no optimizer state.")
            if "scheduler_state_dict" in checkpoint_data:
                scheduler.load_state_dict(checkpoint_data["scheduler_state_dict"])
            else:
                raise RuntimeError("Continuation checkpoint has no scheduler state.")
            rebase_learning_rate(optimizer, scheduler, args.lr)
            if "scaler_state_dict" in checkpoint_data:
                scaler.load_state_dict(checkpoint_data["scaler_state_dict"])
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
            print(
                f"Resuming at optimizer step {step} with --lr={args.lr:.2e} "
                f"(scheduled LR {optimizer.param_groups[0]['lr']:.2e}) and "
                f"--weight_decay={args.weight_decay:.2e}."
            )
    elif checkpoint_data is not None:
        print(
            "Checkpoint is not an exact continuation. Using shape-matched weights "
            "as a transfer initialization with a fresh optimizer, scheduler, EMA "
            "timeline, and global step."
        )

    if args.compile:
        model.encoder.compile_layers(mode="default")
        print(
            f"Compiled {len(model.encoder.time_layers) + len(model.encoder.freq_layers)} "
            "transformer units; activation checkpoint boundaries remain eager."
        )

    train_model = model

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
                    train_model,  # type: ignore[arg-type]
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
        scheduler.step()
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
            wave, main_stft, mrstft = torch.stack(
                (
                    latest_metrics["wave"],
                    latest_metrics["main_stft"],
                    latest_metrics["mrstft"],
                )
            ).float().cpu().tolist()
            progress.set_postfix(
                wave=f"{wave:.3f}",
                stft=f"{main_stft:.3f}",
                mr=f"{mrstft:.3f}",
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
                scheduler,
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
                    scheduler,
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

    progress.close()
    signal.signal(signal.SIGINT, previous_sigint_handler)
    if step > 0:
        stopped_path = f"ckpts/checkpoint_step_{step}.pt"
        save_checkpoint(
            stopped_path,
            model,
            ema,
            optimizer,
            scheduler,
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
        ff_mult=args.ff_mult,
        dropout=args.dropout,
        layer_scale_init=args.layer_scale_init,
        use_checkpoint=args.ckpt,
    )


def inspect_checkpoint_config(
    checkpoint_path: str,
    fallback: ModelConfig,
) -> ModelConfig:
    checkpoint_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
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


def load_inference_weights(
    model: BSRoFormerSeparator,
    checkpoint_path: str,
) -> None:
    checkpoint_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
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
            "124-band, 2048-point STFT, non-overlapping BS-RoFormer "
            "vocals/accompaniment separator"
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
    parser.add_argument("--segment_seconds", type=float, default=6.0)
    parser.add_argument("--inference_overlap_seconds", type=float, default=2.0)

    parser.add_argument("--model_dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--ff_mult", type=float, default=8.0 / 3.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--layer_scale_init", type=float, default=0.1)
    parser.add_argument("--ckpt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--attention_backend",
        choices=("fused", "flash", "auto", "math"),
        default="fused",
        help=(
            "CUDA attention backend. 'fused' tries the external flash-attn package, "
            "PyTorch Flash, cuDNN, then memory-efficient attention with no math "
            "fallback; 'flash' requires external or PyTorch Flash Attention."
        ),
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accumulation", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--dataset_size", type=int, default=50_000)
    parser.add_argument("--remix_probability", type=float, default=0.5)
    parser.add_argument("--checkpoint_steps", type=int, default=4_000)
    parser.add_argument("--warmup_steps", type=int, default=4_000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
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
    if args.warmup_steps < 0:
        raise ValueError("--warmup_steps cannot be negative.")
    if args.grad_clip <= 0.0:
        raise ValueError("--grad_clip must be positive.")
    if args.lr <= 0.0:
        raise ValueError("--lr must be positive.")
    if args.weight_decay < 0.0:
        raise ValueError("--weight_decay cannot be negative.")
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
                    "use --checkpoint_path explicitly for shape-matched transfer."
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
    for module in model.modules():
        if isinstance(module, GatedRoPEAttention):
            module.attention_backend = args.attention_backend
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"BS-RoFormer parameters: {parameter_count / 1e6:.2f}M")
    if device.type == "cuda":
        flash_available = getattr(
            torch.backends.cuda, "is_flash_attention_available", lambda: True
        )()
        if args.attention_backend == "flash":
            if external_flash_attn_func is not None:
                print("CUDA attention backend: external flash-attn package (required).")
            elif not flash_available:
                import_detail = (
                    f" Import error: {FLASH_ATTN_IMPORT_ERROR}"
                    if FLASH_ATTN_IMPORT_ERROR is not None
                    else ""
                )
                raise RuntimeError(
                    "Neither the external flash-attn package nor this PyTorch build "
                    f"provides Flash Attention.{import_detail}"
                )
            else:
                print("CUDA attention backend: built-in PyTorch Flash Attention.")
        elif args.attention_backend == "fused":
            if external_flash_attn_func is not None:
                print(
                    "CUDA attention backend: external flash-attn package "
                    "(cuDNN/memory-efficient fallback available)."
                )
            elif flash_available:
                print(
                    "CUDA attention backend: fused (PyTorch Flash, cuDNN, then "
                    "memory-efficient; math disabled)."
                )
            else:
                print(
                    "Flash Attention is unavailable in this PyTorch build; using "
                    "fused cuDNN/memory-efficient attention with math disabled."
                )
        else:
            print(f"CUDA attention backend: {args.attention_backend}.")

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
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
            worker_init_fn=seed_worker,
            generator=generator,
            drop_last=True,
        )
        optimizer = build_optimizer(
            model,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        scheduler = build_scheduler(
            optimizer,
            warmup_steps=args.warmup_steps,
        )
        loss_module = SeparationLoss(config, LossConfig())
        train(
            model,
            dataloader,
            optimizer,
            scheduler,
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
