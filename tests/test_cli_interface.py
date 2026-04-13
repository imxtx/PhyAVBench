import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

# Allow importing from src-layout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phyavbench.cli import (
    _build_parser,
    _compute_raw_cprs_rows,
    _handle_batch_score_command,
    _handle_clean_command,
    _handle_extract_command,
    _handle_score_command,
    _render_sorted_batch_report,
    _resolve_ground_truth_section_dir,
    _resolve_score_targets,
    _resolve_score_targets_for_model,
    _write_raw_cprs_csv,
    cli,
    resolve_extract_output_dirs,
    resolve_score_output_markdown,
)


def test_resolve_extract_output_dirs_uses_defaults_next_to_input() -> None:
    video_dir = Path("/tmp/demo_videos")

    audio_dir, embedding_dir = resolve_extract_output_dirs(
        video_dir=video_dir,
        audio_output_dir=None,
        embedding_output_dir=None,
    )

    assert audio_dir == Path("/tmp/audio")
    assert embedding_dir == Path("/tmp/audio_embedding")


def test_resolve_extract_output_dirs_honors_user_paths() -> None:
    video_dir = Path("/tmp/demo_videos")

    audio_dir, embedding_dir = resolve_extract_output_dirs(
        video_dir=video_dir,
        audio_output_dir="/tmp/custom_audio",
        embedding_output_dir="/tmp/custom_embedding",
    )

    assert audio_dir == Path("/tmp/custom_audio")
    assert embedding_dir == Path("/tmp/custom_embedding")


def test_resolve_score_output_markdown_uses_default_name() -> None:
    markdown_path = resolve_score_output_markdown(output_dir="output", report_name=None)
    assert markdown_path == Path("output/cprs_result.md")


def test_resolve_score_output_markdown_honors_report_name() -> None:
    markdown_path = resolve_score_output_markdown(
        output_dir="results",
        report_name="run_01.md",
    )
    assert markdown_path == Path("results/run_01.md")


def test_resolve_score_targets_accepts_only_clap_imagebind_under_root(
    tmp_path: Path,
) -> None:
    embedding_root = tmp_path / "audio_embedding"
    (embedding_root / "clap").mkdir(parents=True)
    (embedding_root / "imagebind").mkdir(parents=True)
    (embedding_root / "other").mkdir(parents=True)

    targets = _resolve_score_targets(embedding_root)

    assert targets == [
        ("CLAP", embedding_root / "clap"),
        ("IMAGEBIND", embedding_root / "imagebind"),
    ]


def test_resolve_score_targets_nested_layout_still_resolves_direct_dirs(
    tmp_path: Path,
) -> None:
    embedding_root = tmp_path / "audio_embedding"
    (embedding_root / "clap" / "audio_embeddings").mkdir(parents=True)
    (embedding_root / "imagebind" / "audio_embeddings").mkdir(parents=True)

    targets = _resolve_score_targets(embedding_root)
    assert targets == [
        ("CLAP", embedding_root / "clap"),
        ("IMAGEBIND", embedding_root / "imagebind"),
    ]


def test_resolve_score_targets_rejects_unsupported_layout(tmp_path: Path) -> None:
    embedding_root = tmp_path / "audio_embedding"
    (embedding_root / "other").mkdir(parents=True)

    with pytest.raises(FileNotFoundError):
        _resolve_score_targets(embedding_root)


def test_resolve_ground_truth_section_dir_requires_section_subdir(
    tmp_path: Path,
) -> None:
    gt_root = tmp_path / "gt"
    (gt_root / "clap").mkdir(parents=True)

    assert _resolve_ground_truth_section_dir(gt_root, "CLAP") == gt_root / "clap"
    with pytest.raises(FileNotFoundError):
        _resolve_ground_truth_section_dir(gt_root, "IMAGEBIND")


def test_resolve_score_targets_for_model_filters_sections(tmp_path: Path) -> None:
    embedding_root = tmp_path / "audio_embedding"
    (embedding_root / "clap").mkdir(parents=True)
    (embedding_root / "imagebind").mkdir(parents=True)

    clap_targets = _resolve_score_targets_for_model(embedding_root, "clap")
    imagebind_targets = _resolve_score_targets_for_model(embedding_root, "imagebind")
    all_targets = _resolve_score_targets_for_model(embedding_root, "all")

    assert clap_targets == [("CLAP", embedding_root / "clap")]
    assert imagebind_targets == [("IMAGEBIND", embedding_root / "imagebind")]
    assert all_targets == [
        ("CLAP", embedding_root / "clap"),
        ("IMAGEBIND", embedding_root / "imagebind"),
    ]


def test_handle_score_command_uses_section_gt_dirs(tmp_path: Path) -> None:
    embedding_root = tmp_path / "audio_embedding"
    gt_root = tmp_path / "gt"
    output_dir = tmp_path / "out"
    (gt_root / "clap").mkdir(parents=True)
    (gt_root / "imagebind").mkdir(parents=True)

    args = _build_parser().parse_args(
        [
            "score",
            str(embedding_root),
            str(gt_root),
            "--output-dir",
            str(output_dir),
            "--report-name",
            "cprs.md",
            "--model",
            "all",
        ]
    )

    dummy_metrics = {
        "mean_cprs": 0.8,
        "std_cprs": 0.1,
        "mean_cosine": 0.7,
        "std_cosine": 0.1,
        "mean_proj_coeff": 0.9,
        "std_proj_coeff": 0.1,
        "mean_proj_gauss": 0.8,
        "std_proj_gauss": 0.1,
        "proj_deviation": 0.1,
        "n_pairs": 10.0,
    }

    with patch(
        "phyavbench.cli._resolve_score_targets_for_model",
        return_value=[
            ("CLAP", Path("/pred/clap")),
            ("IMAGEBIND", Path("/pred/imagebind")),
        ],
    ), patch("phyavbench.cli.load_embedding_directories", return_value=({}, {})) as mocked_load, patch(
        "phyavbench.cli.compute_cprs_score", return_value=dummy_metrics
    ), patch("phyavbench.cli.render_combined_markdown_report", return_value="# report"), patch(
        "phyavbench.cli.write_markdown_report"
    ):
        _handle_score_command(args)

    gt_dirs = [call.kwargs["ground_truth_embedding_dir"] for call in mocked_load.call_args_list]
    assert gt_dirs == [gt_root / "clap", gt_root / "imagebind"]


def test_handle_score_command_writes_raw_csv_for_single_model(tmp_path: Path) -> None:
    model_root = tmp_path / "veo3.1"
    embedding_root = model_root / "audio_embedding"
    gt_root = tmp_path / "gt"
    output_dir = tmp_path / "out"
    (embedding_root / "clap").mkdir(parents=True)
    (gt_root / "clap").mkdir(parents=True)

    args = _build_parser().parse_args(
        [
            "score",
            str(embedding_root),
            str(gt_root),
            "--output-dir",
            str(output_dir),
            "--report-name",
            "cprs.md",
            "--model",
            "clap",
        ]
    )

    dummy_metrics = {
        "mean_cprs": 0.8,
        "std_cprs": 0.1,
        "mean_cosine": 0.7,
        "std_cosine": 0.1,
        "mean_proj_coeff": 0.9,
        "std_proj_coeff": 0.1,
        "mean_proj_gauss": 0.8,
        "std_proj_gauss": 0.1,
        "proj_deviation": 0.1,
        "n_pairs": 1.0,
    }

    raw_rows = [
        {
            "model": "audio_embedding",
            "sample_id": "sample1",
            "cprs": 0.95,
            "cos": 0.96,
            "proj_coeff": 0.97,
            "proj_gauss": 0.98,
            "|proj_coeff-1|": 0.03,
        }
    ]

    with patch(
        "phyavbench.cli._resolve_score_targets_for_model",
        return_value=[("CLAP", embedding_root / "clap")],
    ), patch("phyavbench.cli.load_embedding_directories", return_value=({}, {})), patch(
        "phyavbench.cli.compute_cprs_score", return_value=dummy_metrics
    ), patch("phyavbench.cli.render_markdown_report", return_value="# report"), patch(
        "phyavbench.cli.write_markdown_report"
    ), patch("phyavbench.cli._compute_raw_cprs_rows", return_value=raw_rows) as mocked_raw_rows, patch(
        "phyavbench.cli._write_raw_cprs_csv"
    ) as mocked_write_csv:
        _handle_score_command(args)

    assert mocked_raw_rows.call_args.kwargs["model_name"] == "veo3.1"
    mocked_write_csv.assert_called_once()
    csv_path = mocked_write_csv.call_args.args[0]
    assert csv_path == output_dir / "clap_cprs_raw.csv"
    assert mocked_write_csv.call_args.args[1] == raw_rows


def test_extract_parser_default_batch_size() -> None:
    parser = _build_parser()
    args = parser.parse_args(["extract", "--video-dir", "/tmp/videos"])
    assert args.batch_size == 60
    assert args.model == "all"


def test_extract_parser_custom_batch_size_long_option() -> None:
    parser = _build_parser()
    args = parser.parse_args(["extract", "--video-dir", "/tmp/videos", "--batch-size", "24"])
    assert args.batch_size == 24


def test_extract_parser_custom_batch_size_short_option() -> None:
    parser = _build_parser()
    args = parser.parse_args(["extract", "--video-dir", "/tmp/videos", "-b", "32"])
    assert args.batch_size == 32


def test_extract_parser_rejects_non_positive_batch_size() -> None:
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["extract", "--video-dir", "/tmp/videos", "--batch-size", "0"])


def test_extract_parser_accepts_audio_dir_and_model() -> None:
    parser = _build_parser()
    args = parser.parse_args(["extract", "--audio-dir", "/tmp/audio", "--model", "all"])
    assert args.audio_dir == "/tmp/audio"
    assert args.model == "all"


def test_extract_handler_audio_dir_skips_audio_extraction(tmp_path: Path) -> None:
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    wav = audio_dir / "x.wav"
    wav.write_bytes(b"audio")

    parser = _build_parser()
    args = parser.parse_args(["extract", "--audio-dir", str(audio_dir), "--model", "clap"])

    with patch("phyavbench.cli.extract_audio_with_ffmpeg") as mocked_extract_audio, patch(
        "phyavbench.cli.extract_clap_embeddings_multi_target"
    ) as mocked_clap:
        _handle_extract_command(args)

    mocked_extract_audio.assert_not_called()
    mocked_clap.assert_called_once()


def test_extract_handler_audio_dir_default_embedding_output(tmp_path: Path) -> None:
    audio_dir = tmp_path / "audio_in"
    audio_dir.mkdir(parents=True, exist_ok=True)
    wav = audio_dir / "x.wav"
    wav.write_bytes(b"audio")

    parser = _build_parser()
    args = parser.parse_args(["extract", "--audio-dir", str(audio_dir), "--model", "clap"])

    with patch("phyavbench.cli.extract_clap_embeddings_multi_target") as mocked_clap:
        _handle_extract_command(args)

    call_kwargs = mocked_clap.call_args.kwargs
    targets = call_kwargs["audio_embedding_targets"]
    assert targets == [(audio_dir / "x.wav", audio_dir.parent / "audio_embedding")]


def test_extract_handler_audio_dir_skips_existing_embeddings(tmp_path: Path) -> None:
    audio_dir = tmp_path / "audio"
    embedding_dir = tmp_path / "audio_embedding"
    clap_dir = embedding_dir / "clap"
    audio_dir.mkdir(parents=True, exist_ok=True)
    clap_dir.mkdir(parents=True, exist_ok=True)
    wav = audio_dir / "x.wav"
    wav.write_bytes(b"audio")
    (clap_dir / "x.npy").write_bytes(b"npy")

    parser = _build_parser()
    args = parser.parse_args(
        [
            "extract",
            "--audio-dir",
            str(audio_dir),
            "--embedding-output-dir",
            str(embedding_dir),
            "--model",
            "clap",
        ]
    )

    with patch("phyavbench.cli.extract_clap_embeddings_multi_target") as mocked_clap:
        _handle_extract_command(args)

    mocked_clap.assert_not_called()


def test_extract_cli_requires_one_input_source() -> None:
    with pytest.raises(SystemExit):
        cli(["extract"])


def test_extract_cli_rejects_both_input_sources() -> None:
    with pytest.raises(SystemExit):
        cli(["extract", "--video-dir", "/tmp/videos", "--audio-dir", "/tmp/audio"])


def test_batch_score_parser_accepts_required_args() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "batch-score",
            "--base-data-dir",
            "/tmp/data",
            "--gen-dirs",
            "a",
            "b",
            "--ground-truth-embedding-dir",
            "/tmp/gt",
        ]
    )

    assert args.base_data_dir == "/tmp/data"
    assert args.gen_dirs == ["a", "b"]
    assert args.ground_truth_embedding_dir == "/tmp/gt"
    assert args.report_name == "cprs_result.md"
    assert args.model == "all"


def test_handle_batch_score_skips_model_loading_and_loads_gt_once(tmp_path: Path) -> None:
    base_data_dir = tmp_path / "test_data"
    gt_root = tmp_path / "gt"
    output_dir = tmp_path / "out"

    for section in ["clap", "imagebind"]:
        (gt_root / section).mkdir(parents=True)

    for gen_dir in ["m1", "m2"]:
        model_root = base_data_dir / gen_dir
        audio_dir = model_root / "audio"
        (audio_dir).mkdir(parents=True)
        (audio_dir / "sample_a.wav").write_bytes(b"audio-a")
        (audio_dir / "sample_b.wav").write_bytes(b"audio-b")

        clap_dir = model_root / "audio_embedding" / "clap"
        imagebind_dir = model_root / "audio_embedding" / "imagebind"
        clap_dir.mkdir(parents=True)
        imagebind_dir.mkdir(parents=True)
        (clap_dir / "sample_a.npy").write_bytes(b"npy")
        (clap_dir / "sample_b.npy").write_bytes(b"npy")
        (imagebind_dir / "sample_a.npy").write_bytes(b"npy")
        (imagebind_dir / "sample_b.npy").write_bytes(b"npy")

    args = _build_parser().parse_args(
        [
            "batch-score",
            "--base-data-dir",
            str(base_data_dir),
            "--gen-dirs",
            "m1",
            "m2",
            "--ground-truth-embedding-dir",
            str(gt_root),
            "--output-dir",
            str(output_dir),
            "--report-name",
            "cprs_result.md",
            "--model",
            "all",
        ]
    )

    dummy_metrics = {
        "mean_cprs": 0.8,
        "std_cprs": 0.1,
        "mean_cosine": 0.7,
        "std_cosine": 0.1,
        "mean_proj_coeff": 0.9,
        "std_proj_coeff": 0.1,
        "mean_proj_gauss": 0.8,
        "std_proj_gauss": 0.1,
        "proj_deviation": 0.1,
        "n_pairs": 10.0,
    }

    with patch("phyavbench.cli.extract_clap_embeddings_multi_target") as mocked_clap, patch(
        "phyavbench.cli.extract_imagebind_embeddings_multi_target"
    ) as mocked_imagebind, patch(
        "phyavbench.cli.load_ground_truth_embeddings", return_value={"sample": np.array([1.0, 2.0])}
    ) as mocked_load_gt, patch(
        "phyavbench.cli.load_generated_embeddings", return_value={"sample": np.array([1.0, 2.0])}
    ) as mocked_load_gen, patch("phyavbench.cli.compute_cprs_score", return_value=dummy_metrics), patch(
        "phyavbench.cli._compute_raw_cprs_rows", return_value=[]
    ), patch("phyavbench.cli._render_sorted_batch_report", return_value="# report"), patch(
        "phyavbench.cli.write_markdown_report"
    ), patch("phyavbench.cli._write_raw_cprs_csv"):
        code = _handle_batch_score_command(args)

    assert code == 0
    mocked_clap.assert_not_called()
    mocked_imagebind.assert_not_called()
    assert mocked_load_gt.call_count == 2
    assert mocked_load_gen.call_count == 4


def test_score_parser_accepts_model_option() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "score",
            "/tmp/pred",
            "/tmp/gt",
            "--model",
            "clap",
        ]
    )
    assert args.model == "clap"


def test_clean_parser_accepts_required_args() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "clean",
            "--base-data-dir",
            "/tmp/data",
            "--gen-dirs",
            "a",
            "b",
        ]
    )

    assert args.base_data_dir == "/tmp/data"
    assert args.gen_dirs == ["a", "b"]


def test_render_sorted_batch_report_sorts_by_cprs(tmp_path: Path) -> None:
    content = _render_sorted_batch_report(
        rows=[
            {
                "gen_dir": "model_b",
                "section": "CLAP",
                "mean_cprs": 0.6,
                "std_cprs": 0.1,
                "n_pairs": 10.0,
            },
            {
                "gen_dir": "model_a",
                "section": "IMAGEBIND",
                "mean_cprs": 0.8,
                "std_cprs": 0.2,
                "n_pairs": 12.0,
            },
        ],
        sections=["CLAP", "IMAGEBIND"],
        base_data_dir=tmp_path / "data",
        ground_truth_embedding_dir=tmp_path / "gt",
        generated_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
    )

    assert "### CLAP" in content
    assert "### IMAGEBIND" in content

    model_a_pos = content.find("| 1 | model_a | 0.800000")
    model_b_pos = content.find("| 1 | model_b | 0.600000")
    assert model_a_pos != -1
    assert model_b_pos != -1


def test_compute_raw_cprs_rows_shape_and_fields() -> None:
    predicted = {
        "sample1": np.array([1.0, 2.0]),
        "sample2": np.array([2.0, 3.0]),
    }
    ground_truth = {
        "sample1": np.array([1.0, 2.0]),
        "sample2": np.array([2.0, 3.0]),
    }

    rows = _compute_raw_cprs_rows(
        model_name="veo3.1",
        predicted_embeddings=predicted,
        ground_truth_embeddings=ground_truth,
    )

    assert len(rows) == 2
    assert rows[0]["model"] == "veo3.1"
    assert "sample_id" in rows[0]
    assert "cprs" in rows[0]
    assert "cos" in rows[0]
    assert "proj_coeff" in rows[0]
    assert "proj_gauss" in rows[0]
    assert "|proj_coeff-1|" in rows[0]


def test_write_raw_cprs_csv_writes_expected_header(tmp_path: Path) -> None:
    csv_path = tmp_path / "out" / "clap_cprs_raw.csv"
    rows = [
        {
            "model": "sora2",
            "sample_id": "m01_c01_t01_s02_g001",
            "cprs": 0.1,
            "cos": -0.2,
            "proj_coeff": -0.1,
            "proj_gauss": 0.01,
            "|proj_coeff-1|": 1.1,
        }
    ]

    _write_raw_cprs_csv(csv_path, rows)

    content = csv_path.read_text(encoding="utf-8")
    assert "model,sample_id,cprs,cos,proj_coeff,proj_gauss,|proj_coeff-1|" in content
    assert "sora2,m01_c01_t01_s02_g001" in content


def test_handle_clean_command_deletes_targets(tmp_path: Path) -> None:
    base_data_dir = tmp_path / "test_data"
    model_root = base_data_dir / "veo3.1"
    (model_root / "audio").mkdir(parents=True)
    (model_root / "audio_embedding").mkdir(parents=True)
    (model_root / "audio" / "x.wav").write_bytes(b"x")
    (model_root / "audio_embedding" / "x.npy").write_bytes(b"x")
    (model_root / "cprs.md").write_text("old", encoding="utf-8")

    args = _build_parser().parse_args(
        [
            "clean",
            "--base-data-dir",
            str(base_data_dir),
            "--gen-dirs",
            "veo3.1",
        ]
    )

    code = _handle_clean_command(args)
    assert code == 0
    assert not (model_root / "audio").exists()
    assert not (model_root / "audio_embedding").exists()
    assert not (model_root / "cprs.md").exists()
