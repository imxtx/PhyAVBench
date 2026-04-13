import tempfile
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pytest

import sys  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phyavbench.utils.scoring import (
    cprs,
    compute_cprs_score,
    load_embedding_directories,
    render_combined_markdown_report,
    render_markdown_report,
    write_markdown_report,
)


def test_cprs_identical_vectors():
    """Test CPRS with identical vectors should give 1.0"""
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 3.0])
    cprs_val, cos_sim, proj_coeff, proj_gauss = cprs(a, b)

    # For identical vectors: cosine similarity = 1, projection coefficient = 1
    assert abs(cos_sim - 1.0) < 0.01
    assert abs(proj_coeff - 1.0) < 0.01
    assert abs(cprs_val - 1.0) < 0.01


def test_cprs_opposite_vectors():
    """Test CPRS with opposite vectors should give lower scores"""
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([-1.0, 0.0, 0.0])
    cprs_val, cos_sim, proj_coeff, proj_gauss = cprs(a, b)

    # For opposite vectors: cosine similarity = -1, projection coefficient = -1
    assert cos_sim < 0
    assert proj_coeff < 0
    assert cprs_val < 1.0


def test_compute_cprs_score():
    """Test detailed CPRS computation"""
    predicted = {
        "sample1": np.array([1.0, 2.0]),
        "sample2": np.array([2.0, 3.0]),
    }
    ground_truth = {
        "sample1": np.array([1.0, 2.0]),
        "sample2": np.array([2.0, 3.0]),
    }

    metrics = compute_cprs_score(predicted, ground_truth)
    assert isinstance(metrics, dict)
    assert "mean_cprs" in metrics
    assert "std_cprs" in metrics
    assert "mean_cosine" in metrics
    assert "mean_proj_coeff" in metrics
    assert "n_pairs" in metrics
    # For identical vectors, CPRS should be ~1.0
    assert abs(metrics["mean_cprs"] - 1.0) < 0.01


def test_compute_cprs_score_with_unmatched_samples():
    """Test CPRS computation skips unmatched samples"""
    predicted = {
        "sample1": np.array([1.0, 2.0]),
        "sample3": np.array([3.0, 4.0]),  # Not in ground truth
    }
    ground_truth = {
        "sample1": np.array([1.0, 2.0]),
        "sample2": np.array([2.0, 3.0]),  # Not in predictions
    }

    metrics = compute_cprs_score(predicted, ground_truth)
    # Should only score sample1
    assert abs(metrics["mean_cprs"] - 1.0) < 0.01
    assert metrics["n_pairs"] == 1.0


def test_load_embedding_directories(tmp_path: Path):
    """Test loading embeddings: GT has direction vectors, gen has a/b pairs"""
    # GT dir: pre-computed direction vectors
    gt_dir = tmp_path / "gt_embeddings"
    gt_dir.mkdir(parents=True)
    gt_dir_vec_1 = np.array([1.0, 0.0])  # b - a for sample1
    gt_dir_vec_2 = np.array([1.0, 1.0])  # b - a for sample2
    np.save(gt_dir / "sample1.npy", gt_dir_vec_1)
    np.save(gt_dir / "sample2.npy", gt_dir_vec_2)

    # Gen dir: a and b embeddings
    gen_dir = tmp_path / "gen_embeddings"
    gen_dir.mkdir(parents=True)
    # sample1: a=[0.0, 0.0], b=[0.9, 0.1], direction = [0.9, 0.1]
    np.save(gen_dir / "sample1_a.npy", np.array([0.0, 0.0]))
    np.save(gen_dir / "sample1_b.npy", np.array([0.9, 0.1]))
    # sample2: a=[0.0, 0.0], b=[1.1, 0.9], direction = [1.1, 0.9]
    np.save(gen_dir / "sample2_a.npy", np.array([0.0, 0.0]))
    np.save(gen_dir / "sample2_b.npy", np.array([1.1, 0.9]))

    gen_dirs, gt_dirs = load_embedding_directories(gen_dir, gt_dir)

    assert "sample1" in gen_dirs
    assert "sample1" in gt_dirs
    assert "sample2" in gen_dirs
    assert "sample2" in gt_dirs

    # Verify gen directions are computed as b - a
    assert np.allclose(gen_dirs["sample1"], np.array([0.9, 0.1]))
    assert np.allclose(gen_dirs["sample2"], np.array([1.1, 0.9]))
    # Verify GT directions are loaded as-is
    assert np.allclose(gt_dirs["sample1"], gt_dir_vec_1)
    assert np.allclose(gt_dirs["sample2"], gt_dir_vec_2)


def test_render_markdown_report(tmp_path: Path):
    """Test markdown report rendering with metrics dictionary"""
    gen_dir = tmp_path / "gen"
    gt_dir = tmp_path / "gt"

    metrics = {
        "mean_cprs": 0.85,
        "std_cprs": 0.05,
        "mean_cosine": 0.80,
        "std_cosine": 0.08,
        "mean_proj_coeff": 0.90,
        "std_proj_coeff": 0.10,
        "mean_proj_gauss": 0.88,
        "std_proj_gauss": 0.07,
        "proj_deviation": 0.10,
        "n_pairs": 100.0,
    }

    markdown = render_markdown_report(
        metrics=metrics,
        embedding_dir=gen_dir,
        ground_truth_embedding_dir=gt_dir,
        generated_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
    )

    assert "CPRS Scoring Report" in markdown
    assert "0.850000" in markdown
    assert str(gen_dir) in markdown
    assert str(gt_dir) in markdown
    assert "Detailed Statistics" in markdown


def test_render_combined_markdown_report(tmp_path: Path):
    embedding_root = tmp_path / "model"
    gt_dir = tmp_path / "gt"
    sections = [
        (
            "CLAP",
            embedding_root / "clap" / "audio_embeddings",
            {
                "mean_cprs": 0.80,
                "std_cprs": 0.05,
                "mean_cosine": 0.81,
                "std_cosine": 0.04,
                "mean_proj_coeff": 0.90,
                "std_proj_coeff": 0.03,
                "mean_proj_gauss": 0.88,
                "std_proj_gauss": 0.02,
                "proj_deviation": 0.10,
                "n_pairs": 10.0,
            },
        ),
        (
            "IMAGEBIND",
            embedding_root / "imagebind" / "audio_embeddings",
            {
                "mean_cprs": 0.70,
                "std_cprs": 0.06,
                "mean_cosine": 0.71,
                "std_cosine": 0.05,
                "mean_proj_coeff": 0.85,
                "std_proj_coeff": 0.04,
                "mean_proj_gauss": 0.82,
                "std_proj_gauss": 0.03,
                "proj_deviation": 0.15,
                "n_pairs": 20.0,
            },
        ),
    ]

    markdown = render_combined_markdown_report(
        sections=sections,
        embedding_root=embedding_root,
        ground_truth_embedding_dir=gt_dir,
        generated_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
    )

    assert "CPRS Scoring Report" in markdown
    assert "CLAP" in markdown
    assert "IMAGEBIND" in markdown
    assert "0.733333" in markdown


def test_write_markdown_report(tmp_path: Path):
    """Test markdown report writing"""
    output_path = tmp_path / "output" / "report.md"
    content = "# Test Report\n\nThis is a test."

    write_markdown_report(output_path, content)

    assert output_path.exists()
    assert output_path.read_text() == content
    assert (tmp_path / "output").exists()
