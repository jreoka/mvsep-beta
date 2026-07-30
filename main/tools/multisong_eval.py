from __future__ import annotations

import argparse
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


def resolve_checkpoint(args: argparse.Namespace, main_dir: Path, mvsep: Any) -> Path:
    if args.checkpoint is not None:
        checkpoint = args.checkpoint.expanduser().resolve()
    elif args.latest:
        found = mvsep.find_latest_checkpoint(str(main_dir / "ckpts"))
        checkpoint = Path(found).resolve() if found else Path()
    else:
        found = mvsep.find_best_checkpoint(str(main_dir / "best_ckpts"))
        checkpoint = Path(found).resolve() if found else Path()

    if not checkpoint.is_file():
        selection = "latest ckpts checkpoint" if args.latest else "best checkpoint"
        raise FileNotFoundError(
            f"Could not find the {selection}. Pass its path with --checkpoint."
        )
    return checkpoint


def load_model(args: argparse.Namespace) -> tuple[Any, Any, Any, Path]:
    """Import the sibling training module and load its inference checkpoint."""
    main_dir = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(main_dir))
    import mvsep  # pylint: disable=import-outside-toplevel

    checkpoint = resolve_checkpoint(args, main_dir, mvsep)
    fallback = mvsep.ModelConfig(use_checkpoint=False)
    config = mvsep.inspect_checkpoint_config(str(checkpoint), fallback)
    config.use_checkpoint = False

    if tuple(mvsep.STEMS) != ("vocals", "other"):
        raise RuntimeError(
            "This tool expects mvsep.STEMS to be ('vocals', 'other'); got "
            f"{tuple(mvsep.STEMS)!r}. Update the MVSEP submission mapping first."
        )

    torch = mvsep.torch
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device if args.device is not None else default_device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    elif args.precision != "fp32":
        print(f"{device.type.upper()} inference uses FP32; ignoring --precision={args.precision}.")
        args.precision = "fp32"

    if device.type == "cuda" and args.precision == "bf16":
        if not torch.cuda.is_bf16_supported():
            print("CUDA device lacks BF16 support; falling back to FP16.")
            args.precision = "fp16"

    model = mvsep.BSRoFormerSeparator(config)
    for module in model.modules():
        if isinstance(module, mvsep.PersistentMemoryRoPEAttention):
            module.attention_backend = args.attention_backend
    mvsep.load_inference_weights(model, str(checkpoint))
    model.to(device).eval()
    return mvsep, model, device, checkpoint


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

    mvsep, model, device, checkpoint = load_model(args)
    config = model.config
    chunk_size = int(round(args.segment_seconds * config.sample_rate))
    overlap = int(round(args.overlap_seconds * config.sample_rate))
    if chunk_size < config.win_length:
        raise ValueError("--segment-seconds is too short for the checkpoint STFT window.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("--overlap-seconds must be non-negative and shorter than the segment.")

    print(f"Dataset: {len(mixtures)} mixtures from {args.dataset_dir.resolve()}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Output stems: {', '.join(args.stems)}")

    temporary_archive = temporary_archive_path(output_dir)
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
                )
                by_model_stem = dict(zip(mvsep.STEMS, predictions))
                for mvsep_stem in args.stems:
                    model_stem = MVSEP_TO_MODEL_STEM[mvsep_stem]
                    prediction = by_model_stem[model_stem]
                    if not mvsep.torch.isfinite(prediction).all():
                        raise RuntimeError(
                            f"Non-finite samples in {mixture.path} ({mvsep_stem})."
                        )
                    entry_name = f"{mixture.prefix}_{mvsep_stem}.{args.format}"
                    write_prediction(
                        archive,
                        prediction,
                        config.sample_rate,
                        entry_name,
                        args.format,
                        Path(tempfile.gettempdir()),
                        mvsep.sf,
                    )

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
        help="Extracted dataset directory containing song_NNN_mixture.wav files.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint to evaluate (default: highest-SDR checkpoint in best_ckpts).",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Use the latest ckpts/checkpoint_step_*.pt instead of best_ckpts.",
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
    parser.add_argument("--segment-seconds", type=float, default=6.0)
    parser.add_argument("--overlap-seconds", type=float, default=2.0)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device", default=None, help="Torch device, for example cuda or cpu.")
    parser.add_argument(
        "--attention-backend",
        choices=("fused", "flash", "auto", "math"),
        default="fused",
    )
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
    if args.segment_seconds <= 0:
        raise ValueError("--segment-seconds must be positive.")
    if len(args.stems) != len(set(args.stems)):
        raise ValueError("--stems cannot contain duplicates.")

    outputs = process_dataset(args)
    for path in outputs:
        print(f"Created {path} ({path.stat().st_size / (1024**3):.2f} GiB)")


if __name__ == "__main__":
    main()
