from __future__ import annotations

import argparse
import gc
import os
import re
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


SONG_PATTERN = re.compile(
    r"^(song_(?P<number>\d{3}))_mixture\.(?:wav|flac)$",
    re.IGNORECASE,
)
MVSEP_TO_MODEL_STEM = {"vocals": "vocals", "instrum": "other"}
OUTPUT_FORMATS = {
    "wav": ("WAV", "FLOAT"),
    "flac": ("FLAC", "PCM_24"),
}


@dataclass(frozen=True)
class Mixture:
    number: int
    prefix: str
    path: Path


def discover_mixtures(dataset_dir: Path, expected_songs: int) -> list[Mixture]:
    """Find and validate the official ``song_NNN_mixture`` files."""
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    by_number: dict[int, Mixture] = {}
    for path in dataset_dir.rglob("*"):
        if not path.is_file():
            continue
        match = SONG_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        number = int(match.group("number"))
        if number in by_number:
            raise ValueError(
                f"Duplicate song_{number:03d}_mixture files: "
                f"{by_number[number].path} and {path}"
            )
        by_number[number] = Mixture(
            number=number,
            prefix=f"song_{number:03d}",
            path=path,
        )

    mixtures = sorted(by_number.values(), key=lambda item: item.number)
    if len(mixtures) != expected_songs:
        raise ValueError(
            f"Expected {expected_songs} files named song_NNN_mixture.wav/.flac "
            f"under {dataset_dir}, found {len(mixtures)}."
        )
    return mixtures


def expected_archive_names(
    mixtures: Sequence[Mixture], stems: Sequence[str], extension: str
) -> set[str]:
    return {
        f"{mixture.prefix}_{stem}.{extension}"
        for mixture in mixtures
        for stem in stems
    }


def validate_archive(
    archive_path: Path,
    mixtures: Sequence[Mixture],
    stems: Sequence[str],
    extension: str,
) -> None:
    """Reject missing, extra, duplicate, or nested submission entries."""
    expected = expected_archive_names(mixtures, stems, extension)
    with zipfile.ZipFile(archive_path, "r") as archive:
        names = archive.namelist()
        bad_paths = [name for name in names if Path(name).name != name]
        if bad_paths:
            raise RuntimeError(f"Archive contains nested paths: {bad_paths[:3]}")
        if len(names) != len(set(names)):
            raise RuntimeError(f"Archive contains duplicate entries: {archive_path}")
        actual = set(names)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise RuntimeError(
                f"Invalid archive; missing={missing[:3]}, extra={extra[:3]}"
            )
        bad_entry = archive.testzip()
        if bad_entry is not None:
            raise RuntimeError(f"Corrupt ZIP entry: {bad_entry}")


def load_mvsep_module() -> tuple[Any, Path, Path]:
    """Import the sibling/parent ``mvsep.py`` training module.

    The evaluator first looks next to itself, then one directory above it. This
    supports both a flat project layout and a ``tools/`` subdirectory layout
    without requiring a separate model-script CLI option.
    """
    evaluator_dir = Path(__file__).resolve().parent
    candidates = (evaluator_dir / "mvsep.py", evaluator_dir.parent / "mvsep.py")
    model_script = next((path for path in candidates if path.is_file()), None)
    if model_script is None:
        searched = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"Could not locate mvsep.py. Searched: {searched}")

    model_script = model_script.resolve()
    main_dir = model_script.parent
    if str(main_dir) not in sys.path:
        sys.path.insert(0, str(main_dir))

    import mvsep  # pylint: disable=import-outside-toplevel

    return mvsep, main_dir, model_script


def _checkpoint_step_from_name(path: Path) -> int:
    match = re.search(r"step_(\d+)", path.name)
    return int(match.group(1)) if match else -1


def checkpoint_matches_current_architecture(path: Path, mvsep: Any) -> bool:
    """Accept checkpoints from this architecture without assuming default width/depth."""
    try:
        data = mvsep.torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        print(f"Ignoring unreadable checkpoint {path}: {error}")
        return False

    try:
        if not isinstance(data, dict):
            return False
        config = data.get("model_config")
        if not isinstance(config, dict):
            return False

        # Mirror mvsep's own checkpoint pickers: require at least the current
        # CHECKPOINT_FORMAT_VERSION (v13+ for bs124_conformer_rope_deepfilter_v2)
        # as well as a valid ModelConfig and exact ResonantConvGateSeparator
        # state-dict match so stale checkpoints are skipped during auto-selection
        # rather than failing inside load_inference_weights.
        expected_format = getattr(mvsep, "CHECKPOINT_FORMAT_VERSION", None)
        if (
            expected_format is not None
            and data.get("checkpoint_format_version", 0) < expected_format
        ):
            return False

        expected = mvsep.ModelConfig(use_checkpoint=False)
        if config.get("architecture") != expected.architecture:
            return False
        if config.get("num_bands", expected.num_bands) != expected.num_bands:
            return False
        if tuple(data.get("stems", ())) != tuple(mvsep.STEMS):
            return False

        expected_metric = getattr(mvsep, "VALIDATION_METRIC", None)
        saved_metric = data.get("validation_metric")
        if expected_metric is not None and saved_metric not in (None, expected_metric):
            return False

        state = data.get("ema_state_dict") or data.get("model_state_dict")
        if not isinstance(state, dict) or not state:
            return False

        valid_fields = mvsep.ModelConfig.__dataclass_fields__.keys()
        filtered = {
            key: value for key, value in config.items() if key in valid_fields
        }
        filtered["use_checkpoint"] = False
        try:
            parsed_config = mvsep.ModelConfig(**filtered)
        except (TypeError, ValueError):
            return False

        if parsed_config.hop_length % 64 != 0:
            return False

        probe_model = mvsep.ResonantConvGateSeparator(parsed_config)
        expected_state = probe_model.state_dict()
        incoming_state = mvsep.clean_state_dict(state)
        if len(incoming_state) != len(expected_state):
            return False
        if any(
            key not in incoming_state
            or incoming_state[key].shape != expected_tensor.shape
            for key, expected_tensor in expected_state.items()
        ):
            return False
        return True
    finally:
        del data
        gc.collect()


def compatible_checkpoints(folder: Path, mvsep: Any, pattern: str) -> list[Path]:
    return [
        path
        for path in folder.glob(pattern)
        if path.is_file() and checkpoint_matches_current_architecture(path, mvsep)
    ]


def resolve_checkpoint(args: argparse.Namespace, main_dir: Path, mvsep: Any) -> Path:
    if args.checkpoint is not None:
        checkpoint = args.checkpoint.expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint}")
        return checkpoint

    if args.latest:
        raw_candidates = [
            path
            for path in (main_dir / "ckpts").glob("checkpoint_step_*.pt")
            if path.is_file()
        ]
        raw_candidates.sort(
            key=lambda path: (_checkpoint_step_from_name(path), path.stat().st_mtime),
            reverse=True,
        )
        for candidate in raw_candidates:
            if checkpoint_matches_current_architecture(candidate, mvsep):
                return candidate.resolve()
        selection = "latest compatible ckpts checkpoint"
    else:
        scored: list[tuple[float, int, float, Path]] = []
        for path in (main_dir / "best_ckpts").glob("*.pt"):
            if not path.is_file():
                continue
            score = mvsep.checkpoint_sdr_from_path(path)
            if score is not None:
                scored.append(
                    (
                        float(score),
                        _checkpoint_step_from_name(path),
                        path.stat().st_mtime,
                        path,
                    )
                )
        scored.sort(key=lambda item: item[:3], reverse=True)
        for _score, _step, _mtime, candidate in scored:
            if checkpoint_matches_current_architecture(candidate, mvsep):
                return candidate.resolve()
        selection = "best compatible checkpoint"

    raise FileNotFoundError(
        f"Could not find the {selection}. Pass its path with --checkpoint."
    )


def load_model(args: argparse.Namespace) -> tuple[Any, Any, Any, Path, Path]:
    """Load the reviewed separator module and its inference checkpoint."""
    mvsep, main_dir, model_script = load_mvsep_module()
    if tuple(mvsep.STEMS) != ("vocals", "other"):
        raise RuntimeError(
            "This tool expects mvsep.STEMS to be ('vocals', 'other'); got "
            f"{tuple(mvsep.STEMS)!r}. Update the MVSEP submission mapping first."
        )

    checkpoint = resolve_checkpoint(args, main_dir, mvsep)
    fallback = mvsep.ModelConfig(use_checkpoint=False)
    config = mvsep.inspect_checkpoint_config(str(checkpoint), fallback)
    config.use_checkpoint = False

    if hasattr(args, "segment_seconds") and hasattr(args, "overlap_seconds"):
        chunk_size = int(round(args.segment_seconds * config.sample_rate))
        overlap = int(round(args.overlap_seconds * config.sample_rate))
        if chunk_size < config.win_length:
            raise ValueError(
                "--segment-seconds is too short for the checkpoint STFT window."
            )
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError(
                "--overlap-seconds must be non-negative and shorter than the segment."
            )

    torch = mvsep.torch
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device if args.device is not None else default_device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    elif args.precision != "fp32":
        print(
            f"{device.type.upper()} inference uses FP32; ignoring --precision={args.precision}."
        )
        args.precision = "fp32"

    if device.type == "cuda" and args.precision == "bf16":
        if not torch.cuda.is_bf16_supported():
            print("CUDA device lacks BF16 support; falling back to FP16.")
            args.precision = "fp16"

    model = mvsep.ResonantConvGateSeparator(config)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    encoder_count = sum(
        parameter.numel() for parameter in model.encoder.parameters()
    )
    print(
        f"{config.num_bands}-band Conformer-RoPE + {config.filter_taps}-tap DeepFilter "
        f"parameters: {parameter_count / 1e6:.2f}M "
        f"(encoder backbone: {encoder_count / 1e6:.2f}M)"
    )
    mvsep.load_inference_weights(model, str(checkpoint))
    model.to(device).eval()
    if getattr(args, "compile", False):
        model.encoder.compile_layers(mode="default")
    return mvsep, model, device, checkpoint, model_script


def temporary_archive_path(output_dir: Path) -> Path:
    handle = tempfile.NamedTemporaryFile(
        dir=output_dir,
        prefix=".multisong_",
        suffix=".zip.tmp",
        delete=False,
    )
    handle.close()
    return Path(handle.name)


def write_prediction(
    archive: zipfile.ZipFile,
    prediction: Any,
    sample_rate: int,
    entry_name: str,
    extension: str,
    temporary_dir: Path,
    soundfile: Any,
) -> None:
    format_name, subtype = OUTPUT_FORMATS[extension]
    temporary = tempfile.NamedTemporaryFile(
        dir=temporary_dir,
        suffix=f".{extension}",
        delete=False,
    )
    temporary_path = Path(temporary.name)
    temporary.close()
    try:
        soundfile.write(
            temporary_path,
            prediction.detach().float().cpu().numpy().T,
            sample_rate,
            format=format_name,
            subtype=subtype,
        )
        archive.write(temporary_path, arcname=entry_name)
    finally:
        temporary_path.unlink(missing_ok=True)


def process_dataset(args: argparse.Namespace) -> list[Path]:
    mixtures = discover_mixtures(args.dataset_dir, args.expected_songs)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    final_path = output_dir / f"{args.archive_prefix}.zip"
    if final_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Refusing to replace existing archive: {final_path}. "
            "Pass --overwrite to replace it."
        )

    mvsep, model, device, checkpoint, model_script = load_model(args)
    config = model.config
    chunk_size = int(round(args.segment_seconds * config.sample_rate))
    overlap = int(round(args.overlap_seconds * config.sample_rate))
    if chunk_size < config.win_length:
        raise ValueError("--segment-seconds is too short for the checkpoint STFT window.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("--overlap-seconds must be non-negative and shorter than the segment.")
    mask_deadzone = float(getattr(args, "mask_deadzone", 0.0))

    print(f"Dataset: {len(mixtures)} mixtures from {args.dataset_dir.resolve()}")
    print(f"Model script: {model_script}")
    print(f"Checkpoint: {checkpoint}")
    print(
        f"Inference: device={device}, precision={args.precision}, "
        f"segment={args.segment_seconds:g}s, overlap={args.overlap_seconds:g}s, "
        f"mask_deadzone={mask_deadzone:g}, compile={getattr(args, 'compile', False)}"
    )
    print(f"Output stems: {', '.join(args.stems)}")

    temporary_archive = temporary_archive_path(output_dir)
    temporary_audio_dir = Path(tempfile.gettempdir())
    completed: list[Path] = []
    try:
        with zipfile.ZipFile(
            temporary_archive,
            mode="w",
            compression=zipfile.ZIP_STORED,
            allowZip64=True,
        ) as archive:
            progress: Iterable[Mixture] = mvsep.tqdm(mixtures, desc="Multisong")
            for mixture in progress:
                audio = mvsep.read_input_audio(str(mixture.path), config.sample_rate)
                predictions = mvsep.separate_tensor(
                    model,
                    audio,
                    chunk_size=chunk_size,
                    overlap=overlap,
                    device=device,
                    precision=args.precision,
                    show_progress=False,
                    mask_deadzone=mask_deadzone,
                )
                by_model_stem = dict(zip(mvsep.STEMS, predictions))
                selected_predictions: dict[str, Any] = {}
                for mvsep_stem in args.stems:
                    model_stem = MVSEP_TO_MODEL_STEM[mvsep_stem]
                    prediction = by_model_stem[model_stem]
                    if not mvsep.torch.isfinite(prediction).all():
                        raise RuntimeError(
                            f"Non-finite samples in {mixture.path} ({mvsep_stem})."
                        )
                    selected_predictions[mvsep_stem] = prediction.detach().float().cpu()
                del by_model_stem, predictions, audio

                for mvsep_stem, cpu_prediction in selected_predictions.items():
                    entry_name = f"{mixture.prefix}_{mvsep_stem}.{args.format}"
                    write_prediction(
                        archive,
                        cpu_prediction,
                        config.sample_rate,
                        entry_name,
                        args.format,
                        temporary_audio_dir,
                        mvsep.sf,
                    )
                del selected_predictions

        validate_archive(
            temporary_archive,
            mixtures,
            stems=args.stems,
            extension=args.format,
        )
        os.replace(temporary_archive, final_path)
        completed.append(final_path)
    except BaseException:
        temporary_archive.unlink(missing_ok=True)
        raise

    return completed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Process the official MVSEP Multisong mixtures and create one flat "
            "ZIP containing all selected leaderboard stems."
        )
    )
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Extracted dataset directory containing song_NNN_mixture.wav/.flac files.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint to evaluate (default: highest-SDR compatible checkpoint in best_ckpts).",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Use the latest compatible ckpts/checkpoint_step_*.pt instead of best_ckpts.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--archive-prefix", default="multisong")
    parser.add_argument(
        "--stems",
        nargs="+",
        choices=tuple(MVSEP_TO_MODEL_STEM),
        default=list(MVSEP_TO_MODEL_STEM),
        help="Include these stems in the archive (default: vocals instrum).",
    )
    parser.add_argument(
        "--format",
        choices=tuple(OUTPUT_FORMATS),
        default="flac",
        help=(
            "Output format inside the ZIP (default: flac). FLAC uses lossless "
            "compression after quantizing to PCM_24; WAV/FLOAT preserves "
            "unquantized model output."
        ),
    )
    parser.add_argument("--segment-seconds", type=float, default=8.0)
    parser.add_argument("--overlap-seconds", type=float, default=4.0)
    parser.add_argument(
        "--mask-deadzone",
        type=float,
        default=0.0,
        help=(
            "Optional inference soft-threshold on estimated vocal magnitude relative "
            "to mixture in [0, 1). Default 0.0 avoids active-bin shrinkage and "
            "preserves exact mixture consistency and maximum SDR."
        ),
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile Conformer-RoPE encoder layers with torch.compile.",
    )
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device", default=None, help="Torch device, for example cuda or cpu.")
    parser.add_argument(
        "--expected-songs",
        type=int,
        default=100,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.expected_songs <= 0:
        raise ValueError("--expected-songs must be positive.")
    if args.segment_seconds <= 0.0:
        raise ValueError("--segment-seconds must be positive.")
    if args.overlap_seconds < 0.0:
        raise ValueError("--overlap-seconds cannot be negative.")
    if args.overlap_seconds >= args.segment_seconds:
        raise ValueError("--overlap-seconds must be smaller than --segment-seconds.")
    if args.mask_deadzone < 0.0 or args.mask_deadzone >= 1.0:
        raise ValueError("--mask-deadzone must be in [0, 1).")
    if len(args.stems) != len(set(args.stems)):
        raise ValueError("--stems cannot contain duplicates.")

    outputs = process_dataset(args)
    for path in outputs:
        print(f"Created {path} ({path.stat().st_size / (1024**3):.2f} GiB)")


if __name__ == "__main__":
    main()
