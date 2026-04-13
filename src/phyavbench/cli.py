from __future__ import annotations

import argparse
import csv
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from phyavbench.utils.pipeline import (
    collect_audio_files,
    extract_audio_with_ffmpeg,
    extract_clap_embeddings_multi_target,
    extract_imagebind_embeddings_multi_target,
)
from phyavbench.utils.scoring import (
    EmbeddingMap,
    compute_cprs_score,
    cprs,
    load_embedding_directories,
    load_generated_embeddings,
    load_ground_truth_embeddings,
    render_combined_markdown_report,
    render_markdown_report,
    write_markdown_report,
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="phyavbench",
        description="CLI skeleton for extraction and CPRS scoring pipelines.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser(
        "extract",
        help="Extract audio from all mp4 files in a directory and run embedding pipelines.",
    )
    extract_parser.add_argument(
        "--video-dir",
        default=None,
        help="Path to an input directory containing mp4 files.",
    )
    extract_parser.add_argument(
        "--audio-dir",
        default=None,
        help="Path to input audio directory. If set, audio extraction step is skipped.",
    )
    extract_parser.add_argument(
        "--audio-output-dir",
        help="Directory for extracted audio files. Defaults to audio next to the input directory.",
    )
    extract_parser.add_argument(
        "--embedding-output-dir",
        help="Directory for embedding outputs. Defaults to audio_embedding next to the input directory.",
    )
    extract_parser.add_argument(
        "-b",
        "--batch-size",
        type=_positive_int,
        default=60,
        help="Batch size used for embedding extraction (default: 60).",
    )
    extract_parser.add_argument(
        "--model",
        choices=["clap", "imagebind", "all"],
        default="all",
        help="Embedding model to run (default: all).",
    )
    extract_parser.set_defaults(handler=_handle_extract_command)

    score_parser = subparsers.add_parser(
        "score",
        help="Compute CPRS from predicted and ground-truth embedding directories.",
    )
    score_parser.add_argument(
        "embedding_dir",
        help="Directory containing predicted embeddings.",
    )
    score_parser.add_argument(
        "ground_truth_embedding_dir",
        help="Directory containing ground-truth embeddings.",
    )
    score_parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory where markdown score report will be written (default: output).",
    )
    score_parser.add_argument(
        "--report-name",
        default=None,
        help="Optional markdown report filename. Default: cprs_result.md",
    )
    score_parser.add_argument(
        "--model",
        choices=["clap", "imagebind", "all"],
        default="all",
        help="Embedding model to score (default: all).",
    )
    score_parser.set_defaults(handler=_handle_score_command)

    batch_parser = subparsers.add_parser(
        "batch-score",
        help=(
            "Run extraction and CPRS scoring for multiple generation directories "
            "with shared CLAP/ImageBind model loading."
        ),
    )
    batch_parser.add_argument(
        "--base-data-dir",
        required=True,
        help="Base directory containing model subdirectories.",
    )
    batch_parser.add_argument(
        "--gen-dirs",
        nargs="+",
        required=True,
        help="Model directory names under --base-data-dir.",
    )
    batch_parser.add_argument(
        "--ground-truth-embedding-dir",
        required=True,
        help="Directory containing ground-truth embeddings (supports clap/imagebind subdirs).",
    )
    batch_parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory where merged markdown report will be written (default: output).",
    )
    batch_parser.add_argument(
        "--report-name",
        default="cprs_result.md",
        help="Merged markdown report filename (default: cprs_result.md).",
    )
    batch_parser.add_argument(
        "-b",
        "--batch-size",
        type=_positive_int,
        default=60,
        help="Batch size used for embedding extraction (default: 60).",
    )
    batch_parser.add_argument(
        "--model",
        choices=["clap", "imagebind", "all"],
        default="all",
        help="Embedding model to run (default: all).",
    )
    batch_parser.set_defaults(handler=_handle_batch_score_command)

    clean_parser = subparsers.add_parser(
        "clean",
        help=("Delete generated artifacts under model directories: audio/, audio_embedding/, and cprs.md."),
    )
    clean_parser.add_argument(
        "--base-data-dir",
        required=True,
        help="Base directory containing model subdirectories.",
    )
    clean_parser.add_argument(
        "--gen-dirs",
        nargs="+",
        required=True,
        help="Model directory names under --base-data-dir.",
    )
    clean_parser.set_defaults(handler=_handle_clean_command)

    return parser


def resolve_extract_output_dirs(
    video_dir: Path,
    audio_output_dir: Optional[str],
    embedding_output_dir: Optional[str],
) -> Tuple[Path, Path]:
    base_dir = video_dir.parent

    resolved_audio_dir = Path(audio_output_dir) if audio_output_dir else base_dir / "audio"
    resolved_embedding_dir = Path(embedding_output_dir) if embedding_output_dir else base_dir / "audio_embedding"
    return resolved_audio_dir, resolved_embedding_dir


def _collect_audio_files(audio_dir: Path) -> List[Path]:
    return collect_audio_files(audio_dir)


def resolve_score_output_markdown(output_dir: str, report_name: Optional[str]) -> Path:
    out_dir = Path(output_dir)
    filename = report_name or "cprs_result.md"
    return out_dir / filename


def _resolve_score_targets(embedding_dir: Path) -> List[Tuple[str, Path]]:
    return _resolve_score_targets_for_model(embedding_dir=embedding_dir, model="all")


def _selected_sections(model: str) -> List[str]:
    if model == "clap":
        return ["CLAP"]
    if model == "imagebind":
        return ["IMAGEBIND"]
    return ["CLAP", "IMAGEBIND"]


def _resolve_score_targets_for_model(
    embedding_dir: Path,
    model: str,
) -> List[Tuple[str, Path]]:
    # Only support model/audio_embedding/clap and model/audio_embedding/imagebind.
    # Nested clap/audio_embeddings or imagebind/audio_embeddings are intentionally not supported.
    targets: List[Tuple[str, Path]] = []
    for section_name in _selected_sections(model):
        section_dir = embedding_dir / section_name.lower()
        if not section_dir.exists() or not section_dir.is_dir():
            raise FileNotFoundError(f"Missing embedding directory for selected model section: {section_dir}")
        targets.append((section_name, section_dir))

    if not targets:
        raise FileNotFoundError(
            "No supported embedding directories found under "
            f"{embedding_dir}. Expected one of: "
            f"{embedding_dir / 'clap'} or {embedding_dir / 'imagebind'}"
        )

    return targets


def _resolve_ground_truth_section_dir(ground_truth_root: Path, section_name: str) -> Path:
    section_dir = ground_truth_root / section_name.lower()
    if not section_dir.exists() or not section_dir.is_dir():
        raise FileNotFoundError(f"Missing ground-truth section directory for selected model section: {section_dir}")
    return section_dir


def _infer_model_name_from_embedding_dir(embedding_dir: Path) -> str:
    if embedding_dir.name == "audio_embedding" and embedding_dir.parent != embedding_dir:
        return embedding_dir.parent.name
    return embedding_dir.name


def _handle_extract_command(args: argparse.Namespace) -> int:
    audio_files: List[Path]

    if args.audio_dir:
        audio_dir = Path(args.audio_dir)
        if not audio_dir.exists() or not audio_dir.is_dir():
            raise FileNotFoundError(f"Input audio directory not found or not a directory: {audio_dir}")
        audio_files = _collect_audio_files(audio_dir)
        if not audio_files:
            raise ValueError(f"No audio files found in input directory: {audio_dir}")

        if args.embedding_output_dir:
            embedding_dir = Path(args.embedding_output_dir)
        else:
            embedding_dir = audio_dir.parent / "audio_embedding"

        print(f"[extract] audio directory: {audio_dir}")
        print("[extract] skip audio extraction because --audio-dir is provided")
        if args.audio_output_dir:
            print("[extract] warning: --audio-output-dir is ignored when --audio-dir is provided")
    else:
        if not args.video_dir:
            raise ValueError("Either --video-dir or --audio-dir must be provided")

        video_dir = Path(args.video_dir)
        if not video_dir.exists() or not video_dir.is_dir():
            raise FileNotFoundError(f"Input video directory not found or not a directory: {video_dir}")
        audio_dir, embedding_dir = resolve_extract_output_dirs(
            video_dir=video_dir,
            audio_output_dir=args.audio_output_dir,
            embedding_output_dir=args.embedding_output_dir,
        )

        print(f"[extract] video directory: {video_dir}")
        print(f"[extract] audio output dir: {audio_dir}")
        existing_audio_by_stem: Dict[str, Path] = {}
        if audio_dir.exists() and audio_dir.is_dir():
            existing_audio = _collect_audio_files(audio_dir)
            existing_audio_by_stem = {audio_file.stem: audio_file for audio_file in existing_audio}

        video_files = sorted(path for path in video_dir.iterdir() if path.is_file() and path.suffix.lower() == ".mp4")
        missing_videos = [video_file for video_file in video_files if video_file.stem not in existing_audio_by_stem]

        if missing_videos:
            print(f"[extract] extracting missing audio files: {len(missing_videos)}")
            extract_audio_with_ffmpeg(
                video_dir=video_dir,
                audio_output_dir=audio_dir,
                video_files=missing_videos,
                skip_existing=True,
            )
        else:
            print(f"[extract] all audio already exists, skip extraction: {len(video_files)}")

        audio_files = _collect_audio_files(audio_dir)
        if not audio_files:
            raise ValueError(f"No audio files found after extraction step: {audio_dir}")

    print(f"[extract] embedding output dir: {embedding_dir}")
    print(f"[extract] model: {args.model}")

    if args.model in {"clap", "all"}:
        clap_targets = [
            (audio_file, embedding_dir)
            for audio_file in audio_files
            if not (embedding_dir / "clap" / f"{audio_file.stem}.npy").exists()
        ]
        if clap_targets:
            print(f"[extract] missing CLAP embeddings: {len(clap_targets)}")
            extract_clap_embeddings_multi_target(
                audio_embedding_targets=clap_targets,
                batch_size=args.batch_size,
            )
        else:
            print("[extract] all CLAP embeddings already exist, skip model loading")

    if args.model in {"imagebind", "all"}:
        imagebind_targets = [
            (audio_file, embedding_dir)
            for audio_file in audio_files
            if not (embedding_dir / "imagebind" / f"{audio_file.stem}.npy").exists()
        ]
        if imagebind_targets:
            print(f"[extract] missing ImageBind embeddings: {len(imagebind_targets)}")
            extract_imagebind_embeddings_multi_target(
                audio_embedding_targets=imagebind_targets,
                batch_size=args.batch_size,
            )
        else:
            print("[extract] all ImageBind embeddings already exist, skip model loading")

    return 0


def _handle_score_command(args: argparse.Namespace) -> int:
    embedding_dir = Path(args.embedding_dir)
    ground_truth_dir = Path(args.ground_truth_embedding_dir)
    model_name = _infer_model_name_from_embedding_dir(embedding_dir)
    markdown_path = resolve_score_output_markdown(
        output_dir=args.output_dir,
        report_name=args.report_name,
    )

    print(f"[score] embedding dir: {embedding_dir}")
    print(f"[score] ground truth dir: {ground_truth_dir}")
    print(f"[score] markdown output: {markdown_path}")
    print(f"[score] model: {args.model}")

    sections: List[Tuple[str, Path, dict[str, float]]] = []
    raw_rows_by_section: Dict[str, List[Dict[str, float | str]]] = {
        section_name: [] for section_name in _selected_sections(args.model)
    }
    for section_name, predicted_dir in _resolve_score_targets_for_model(
        embedding_dir=embedding_dir,
        model=args.model,
    ):
        section_ground_truth_dir = _resolve_ground_truth_section_dir(
            ground_truth_root=ground_truth_dir,
            section_name=section_name,
        )
        predicted, ground_truth = load_embedding_directories(
            embedding_dir=predicted_dir,
            ground_truth_embedding_dir=section_ground_truth_dir,
        )
        metrics = compute_cprs_score(predicted_embeddings=predicted, ground_truth_embeddings=ground_truth)
        sections.append((section_name, predicted_dir, metrics))
        raw_rows_by_section[section_name].extend(
            _compute_raw_cprs_rows(
                model_name=model_name,
                predicted_embeddings=predicted,
                ground_truth_embeddings=ground_truth,
            )
        )
        print(f"CPRS score [{section_name}]: {metrics['mean_cprs']:.6f}")

    if len(sections) == 1:
        _, predicted_dir, metrics = sections[0]
        markdown = render_markdown_report(
            metrics=metrics,
            embedding_dir=predicted_dir,
            ground_truth_embedding_dir=ground_truth_dir,
            generated_at=datetime.now(timezone.utc),
        )
    else:
        markdown = render_combined_markdown_report(
            sections=sections,
            embedding_root=embedding_dir,
            ground_truth_embedding_dir=ground_truth_dir,
            generated_at=datetime.now(timezone.utc),
        )
    write_markdown_report(markdown_path=markdown_path, content=markdown)
    print(f"[score] report written: {markdown_path}")

    for section_name, rows in raw_rows_by_section.items():
        csv_path = markdown_path.parent / f"{section_name.lower()}_cprs_raw.csv"
        _write_raw_cprs_csv(csv_path, rows)
        print(f"[score] raw CSV written: {csv_path}")

    return 0


def _prepare_audio_for_model_dir(model_root: Path) -> List[Path]:
    audio_dir = model_root / "audio"
    video_dir = model_root / "video"
    existing_audio_by_stem: Dict[str, Path] = {}
    if audio_dir.exists() and audio_dir.is_dir():
        existing_audio = _collect_audio_files(audio_dir)
        existing_audio_by_stem = {audio_file.stem: audio_file for audio_file in existing_audio}

    video_files: List[Path] = []
    if video_dir.exists() and video_dir.is_dir():
        video_files = sorted(path for path in video_dir.iterdir() if path.is_file() and path.suffix.lower() == ".mp4")

    if not video_files and not existing_audio_by_stem:
        raise FileNotFoundError(f"Missing both usable audio dir and video dir under model root: {model_root}")

    missing_videos = [video_file for video_file in video_files if video_file.stem not in existing_audio_by_stem]
    if missing_videos:
        print(f"[batch-score] extracting missing audio files: {len(missing_videos)} from {video_dir}")
        extract_audio_with_ffmpeg(
            video_dir=video_dir,
            audio_output_dir=audio_dir,
            video_files=missing_videos,
            skip_existing=True,
        )
    elif video_files:
        print(f"[batch-score] all audio already exists, skip extraction: {len(video_files)} in {audio_dir}")

    resolved_audio = _collect_audio_files(audio_dir) if audio_dir.exists() and audio_dir.is_dir() else []
    if not resolved_audio:
        raise ValueError(f"No audio files found for model root: {model_root}")

    print(f"[batch-score] ready audio files: {len(resolved_audio)} in {audio_dir}")
    return resolved_audio


def _render_sorted_batch_report(
    rows: List[Dict[str, float | str]],
    *,
    sections: List[str],
    base_data_dir: Path,
    ground_truth_embedding_dir: Path,
    generated_at: datetime,
) -> str:
    def _render_section(section_name: str) -> List[str]:
        section_rows = [row for row in rows if str(row["section"]).upper() == section_name.upper()]
        section_rows = sorted(section_rows, key=lambda row: float(row["mean_cprs"]), reverse=True)

        section_lines: List[str] = []
        section_lines.append(f"### {section_name.upper()}")
        section_lines.append("")
        if not section_rows:
            section_lines.append("No rows.")
            section_lines.append("")
            return section_lines

        section_lines.append("| Rank | Model | CPRS | Std | Matched Pairs |")
        section_lines.append("|------|-------|------|-----|---------------|")
        for rank, row in enumerate(section_rows, start=1):
            section_lines.append(
                "| "
                f"{rank} | "
                f"{row['gen_dir']} | "
                f"{float(row['mean_cprs']):.6f} | "
                f"{float(row['std_cprs']):.6f} | "
                f"{int(float(row['n_pairs']))} |"
            )
        section_lines.append("")
        return section_lines

    lines: List[str] = []
    lines.append("# CPRS Batch Report")
    lines.append("")
    lines.append(f"**Generated:** {generated_at.isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("## Configuration")
    lines.append(f"- Base data directory: `{base_data_dir}`")
    lines.append(f"- Ground-truth directory: `{ground_truth_embedding_dir}`")
    lines.append(f"- Rows: {len(rows)}")
    lines.append("")
    lines.append("## Sorted CPRS")
    lines.append("")
    for section_name in sections:
        lines.extend(_render_section(section_name))

    return "\n".join(lines)


def _compute_raw_cprs_rows(
    *,
    model_name: str,
    predicted_embeddings: EmbeddingMap,
    ground_truth_embeddings: EmbeddingMap,
    k: float = 5.0,
) -> List[Dict[str, float | str]]:
    rows: List[Dict[str, float | str]] = []
    for sample_id, emb_gen in predicted_embeddings.items():
        if sample_id not in ground_truth_embeddings:
            continue

        g = emb_gen.reshape(-1)
        r = ground_truth_embeddings[sample_id].reshape(-1)
        cprs_val, cos_sim, proj_coeff, proj_gauss = cprs(g, r, k=k)
        rows.append(
            {
                "model": model_name,
                "sample_id": sample_id,
                "cprs": cprs_val,
                "cos": cos_sim,
                "proj_coeff": proj_coeff,
                "proj_gauss": proj_gauss,
                "|proj_coeff-1|": abs(proj_coeff - 1.0),
            }
        )

    rows.sort(key=lambda row: (str(row["model"]), str(row["sample_id"])))
    return rows


def _write_raw_cprs_csv(csv_path: Path, rows: List[Dict[str, float | str]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "model",
        "sample_id",
        "cprs",
        "cos",
        "proj_coeff",
        "proj_gauss",
        "|proj_coeff-1|",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def _handle_batch_score_command(args: argparse.Namespace) -> int:
    base_data_dir = Path(args.base_data_dir)
    ground_truth_dir = Path(args.ground_truth_embedding_dir)
    markdown_path = resolve_score_output_markdown(
        output_dir=args.output_dir,
        report_name=args.report_name,
    )

    if not base_data_dir.exists() or not base_data_dir.is_dir():
        raise FileNotFoundError(f"Base data directory not found or not a directory: {base_data_dir}")
    if not ground_truth_dir.exists() or not ground_truth_dir.is_dir():
        raise FileNotFoundError(f"Ground-truth directory not found or not a directory: {ground_truth_dir}")

    print(f"[batch-score] base data dir: {base_data_dir}")
    print(f"[batch-score] ground truth dir: {ground_truth_dir}")
    print(f"[batch-score] model dirs: {', '.join(args.gen_dirs)}")
    print(f"[batch-score] model: {args.model}")

    selected_sections = _selected_sections(args.model)
    clap_targets: List[Tuple[Path, Path]] = []
    imagebind_targets: List[Tuple[Path, Path]] = []
    total_audio_files = 0
    for gen_dir in args.gen_dirs:
        model_root = base_data_dir / gen_dir
        if not model_root.exists() or not model_root.is_dir():
            print(f"[batch-score] skip missing model root: {model_root}")
            continue

        audio_files = _prepare_audio_for_model_dir(model_root=model_root)
        embedding_root = model_root / "audio_embedding"
        total_audio_files += len(audio_files)
        for audio_file in audio_files:
            if "CLAP" in selected_sections:
                clap_npy = embedding_root / "clap" / f"{audio_file.stem}.npy"
                if not clap_npy.exists() or not clap_npy.is_file():
                    clap_targets.append((audio_file, embedding_root))
            if "IMAGEBIND" in selected_sections:
                imagebind_npy = embedding_root / "imagebind" / f"{audio_file.stem}.npy"
                if not imagebind_npy.exists() or not imagebind_npy.is_file():
                    imagebind_targets.append((audio_file, embedding_root))

    if total_audio_files == 0:
        raise ValueError("No valid audio files found across all provided model directories")

    print(f"[batch-score] total audio files considered: {total_audio_files}")
    if args.model in {"clap", "all"} and clap_targets:
        print(f"[batch-score] missing CLAP embeddings: {len(clap_targets)}")
        extract_clap_embeddings_multi_target(
            audio_embedding_targets=clap_targets,
            batch_size=args.batch_size,
        )
    elif args.model in {"clap", "all"}:
        print("[batch-score] all CLAP embeddings already exist, skip model loading")

    if args.model in {"imagebind", "all"} and imagebind_targets:
        print(f"[batch-score] missing ImageBind embeddings: {len(imagebind_targets)}")
        extract_imagebind_embeddings_multi_target(
            audio_embedding_targets=imagebind_targets,
            batch_size=args.batch_size,
        )
    elif args.model in {"imagebind", "all"}:
        print("[batch-score] all ImageBind embeddings already exist, skip model loading")

    rows: List[Dict[str, float | str]] = []
    raw_rows_by_section: Dict[str, List[Dict[str, float | str]]] = {
        section_name: [] for section_name in selected_sections
    }
    ground_truth_by_section: Dict[str, EmbeddingMap] = {}
    for section_name in selected_sections:
        section_ground_truth_dir = _resolve_ground_truth_section_dir(
            ground_truth_root=ground_truth_dir,
            section_name=section_name,
        )
        ground_truth_by_section[section_name] = load_ground_truth_embeddings(section_ground_truth_dir)

    for gen_dir in args.gen_dirs:
        embedding_root = base_data_dir / gen_dir / "audio_embedding"
        if not embedding_root.exists() or not embedding_root.is_dir():
            continue

        for section_name, predicted_dir in _resolve_score_targets_for_model(
            embedding_dir=embedding_root,
            model=args.model,
        ):
            predicted = load_generated_embeddings(predicted_dir)
            ground_truth = ground_truth_by_section[section_name]
            metrics = compute_cprs_score(
                predicted_embeddings=predicted,
                ground_truth_embeddings=ground_truth,
            )
            rows.append(
                {
                    "gen_dir": gen_dir,
                    "section": section_name,
                    "mean_cprs": metrics["mean_cprs"],
                    "std_cprs": metrics["std_cprs"],
                    "n_pairs": metrics["n_pairs"],
                }
            )
            raw_rows_by_section[section_name].extend(
                _compute_raw_cprs_rows(
                    model_name=gen_dir,
                    predicted_embeddings=predicted,
                    ground_truth_embeddings=ground_truth,
                )
            )
            print(f"[batch-score] CPRS [{gen_dir}][{section_name}]: {metrics['mean_cprs']:.6f}")

    if not rows:
        raise ValueError("No CPRS rows were produced. Check embeddings and ground-truth inputs.")

    markdown = _render_sorted_batch_report(
        rows=rows,
        sections=selected_sections,
        base_data_dir=base_data_dir,
        ground_truth_embedding_dir=ground_truth_dir,
        generated_at=datetime.now(timezone.utc),
    )
    write_markdown_report(markdown_path=markdown_path, content=markdown)

    print(f"[batch-score] merged report written: {markdown_path}")

    if "CLAP" in raw_rows_by_section:
        clap_csv_path = markdown_path.parent / "clap_cprs_raw.csv"
        _write_raw_cprs_csv(clap_csv_path, raw_rows_by_section["CLAP"])
        print(f"[batch-score] raw CSV written: {clap_csv_path}")

    if "IMAGEBIND" in raw_rows_by_section:
        imagebind_csv_path = markdown_path.parent / "imagebind_cprs_raw.csv"
        _write_raw_cprs_csv(imagebind_csv_path, raw_rows_by_section["IMAGEBIND"])
        print(f"[batch-score] raw CSV written: {imagebind_csv_path}")

    return 0


def _clean_model_root(model_root: Path) -> Dict[str, bool]:
    targets = {
        "audio": model_root / "audio",
        "audio_embedding": model_root / "audio_embedding",
        "cprs_md": model_root / "cprs.md",
    }
    deleted = {"audio": False, "audio_embedding": False, "cprs_md": False}

    if targets["audio"].exists():
        shutil.rmtree(targets["audio"])
        deleted["audio"] = True
    if targets["audio_embedding"].exists():
        shutil.rmtree(targets["audio_embedding"])
        deleted["audio_embedding"] = True
    if targets["cprs_md"].exists():
        targets["cprs_md"].unlink()
        deleted["cprs_md"] = True

    return deleted


def _handle_clean_command(args: argparse.Namespace) -> int:
    base_data_dir = Path(args.base_data_dir)
    if not base_data_dir.exists() or not base_data_dir.is_dir():
        raise FileNotFoundError(f"Base data directory not found or not a directory: {base_data_dir}")

    print(f"[clean] base data dir: {base_data_dir}")
    deleted_audio = 0
    deleted_embedding = 0
    deleted_reports = 0

    for gen_dir in args.gen_dirs:
        model_root = base_data_dir / gen_dir
        if not model_root.exists() or not model_root.is_dir():
            print(f"[clean] skip missing model root: {model_root}")
            continue

        deleted = _clean_model_root(model_root)
        deleted_audio += int(deleted["audio"])
        deleted_embedding += int(deleted["audio_embedding"])
        deleted_reports += int(deleted["cprs_md"])
        print(
            "[clean] "
            f"model={gen_dir} "
            f"audio={'deleted' if deleted['audio'] else 'absent'} "
            f"audio_embedding={'deleted' if deleted['audio_embedding'] else 'absent'} "
            f"cprs.md={'deleted' if deleted['cprs_md'] else 'absent'}"
        )

    print(f"[clean] summary: audio={deleted_audio}, audio_embedding={deleted_embedding}, cprs.md={deleted_reports}")
    return 0


def cli(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "extract":
        has_video_dir = bool(args.video_dir)
        has_audio_dir = bool(args.audio_dir)
        if has_video_dir == has_audio_dir:
            parser.error("extract requires exactly one input source: --video-dir or --audio-dir")

    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 2
    return int(handler(args))
