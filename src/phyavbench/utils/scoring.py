from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

EmbeddingMap = Dict[str, np.ndarray]


def _load_gt_directions(gt_dir: Path) -> EmbeddingMap:
    """Load ground-truth direction vectors from directory (pre-computed .npy files)."""
    gt_dir = Path(gt_dir)
    npy_files = sorted(gt_dir.glob("*.npy"))
    if not npy_files:
        raise FileNotFoundError(f"No GT direction vectors found in {gt_dir}")

    directions: EmbeddingMap = {}
    for file_path in tqdm(npy_files, desc="Loading GT directions", unit="file"):
        directions[file_path.stem] = np.load(file_path)
    return directions


def _load_gen_directions(gen_dir: Path) -> EmbeddingMap:
    """Load generated directions by pairing <prompt>_a.npy and <prompt>_b.npy files, computing b - a."""
    gen_dir = Path(gen_dir)
    npy_files = sorted(gen_dir.glob("*_a.npy"))
    if not npy_files:
        raise FileNotFoundError(f"No generated embedding pairs found in {gen_dir}")

    directions: EmbeddingMap = {}
    for a_file in tqdm(npy_files, desc="Loading generated directions", unit="file"):
        prompt = a_file.name[: -len("_a.npy")]
        b_file = gen_dir / f"{prompt}_b.npy"
        if not b_file.exists():
            continue
        a = np.load(a_file)
        b = np.load(b_file)
        directions[prompt] = b - a
    return directions


def _safe_mean(xs: List[float]) -> float:
    """Compute mean of a list, returning NaN for empty lists."""
    return float(np.mean(xs)) if xs else float("nan")


def _safe_std(xs: List[float]) -> float:
    """Compute standard deviation of a list, returning NaN for empty lists."""
    return float(np.std(xs)) if xs else float("nan")


def cprs(a: np.ndarray, b: np.ndarray, k: float = 5.0) -> Tuple[float, float, float, float]:
    """
    Compute CPRS (Cosine-Projection-based Relevance Score) between two vectors.

    Args:
        a: First embedding vector (generated direction)
        b: Second embedding vector (ground-truth direction)
        k: Gaussian kernel decay factor (default: 5.0)

    Returns:
        Tuple of (cprs_value, cosine_similarity, projection_coefficient, projection_gaussian)
    """
    eps = 1e-8
    a = np.asarray(a)
    b = np.asarray(b)

    # 1. Cosine similarity normalized to [0, 1]
    cos_theta = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps))
    c = (cos_theta + 1.0) / 2.0

    # 2. Normalized projection coefficient of a onto b
    p = float(np.dot(a, b) / (np.linalg.norm(b) ** 2 + eps))

    # 3. Gaussian normalization
    proj_gauss = float(np.exp(-k * (p - 1.0) ** 2))

    # 4. CPRS = average of cosine and projection terms
    cprs_val = float((c + proj_gauss) / 2.0)

    return cprs_val, cos_theta, p, proj_gauss


def load_embedding_directories(
    embedding_dir: Path,
    ground_truth_embedding_dir: Path,
) -> Tuple[EmbeddingMap, EmbeddingMap]:
    """Load generated and ground-truth embedding directions.

    GT dir contains pre-computed direction vectors (.npy files).
    Generated dir contains paired embeddings (<prompt>_a.npy, <prompt>_b.npy),
    directions are computed as b - a.

    Args:
        embedding_dir: Path to directory with generated embeddings _a/_b pairs
        ground_truth_embedding_dir: Path to directory with GT direction vectors

    Returns:
        Tuple of (generated_directions, ground_truth_directions) keyed by prompt stem
    """
    gen_directions = _load_gen_directions(embedding_dir)
    gt_directions = _load_gt_directions(ground_truth_embedding_dir)

    return gen_directions, gt_directions


def load_generated_embeddings(embedding_dir: Path) -> EmbeddingMap:
    """Load generated embedding directions from one embedding directory."""
    return _load_gen_directions(embedding_dir)


def load_ground_truth_embeddings(ground_truth_embedding_dir: Path) -> EmbeddingMap:
    """Load ground-truth embedding directions from one directory."""
    return _load_gt_directions(ground_truth_embedding_dir)


def compute_cprs_score(
    predicted_embeddings: EmbeddingMap,
    ground_truth_embeddings: EmbeddingMap,
    k: float = 5.0,
) -> Dict[str, float]:
    """Compute detailed CPRS metrics from predicted and ground-truth embeddings.

    Args:
        predicted_embeddings: Dictionary of generated embedding directions
        ground_truth_embeddings: Dictionary of GT embedding directions
        k: Gaussian kernel decay factor (default: 5.0)

    Returns:
        Dictionary with mean/std for all metrics:
        - mean_cprs, std_cprs
        - mean_cosine, std_cosine
        - mean_proj_coeff, std_proj_coeff
        - mean_proj_gauss, std_proj_gauss
        - proj_deviation
        - n_pairs
    """
    return _compute_detailed_scores(predicted_embeddings, ground_truth_embeddings, k=k)


def _compute_detailed_scores(
    predicted_embeddings: EmbeddingMap,
    ground_truth_embeddings: EmbeddingMap,
    k: float = 5.0,
) -> Dict[str, float]:
    """Compute detailed CPRS scores including statistics.

    Args:
        predicted_embeddings: Dictionary of generated embedding directions
        ground_truth_embeddings: Dictionary of GT embedding directions
        k: Gaussian kernel decay factor (default: 5.0)

    Returns:
        Dictionary with mean/std for cprs, cosine, projection coefficient, etc.
    """
    cprs_vals: List[float] = []
    cos_vals: List[float] = []
    proj_coeff_vals: List[float] = []
    proj_gauss_vals: List[float] = []
    n_matched = 0

    for pid, emb_gen in predicted_embeddings.items():
        if pid not in ground_truth_embeddings:
            continue
        n_matched += 1
        emb_gt = ground_truth_embeddings[pid]

        g = emb_gen.reshape(-1)
        r = emb_gt.reshape(-1)

        cprs_val, cos_sim, proj_coeff, proj_gauss = cprs(g, r, k=k)
        cprs_vals.append(cprs_val)
        cos_vals.append(cos_sim)
        proj_coeff_vals.append(proj_coeff)
        proj_gauss_vals.append(proj_gauss)

    return {
        "mean_cprs": _safe_mean(cprs_vals),
        "std_cprs": _safe_std(cprs_vals),
        "mean_cosine": _safe_mean(cos_vals),
        "std_cosine": _safe_std(cos_vals),
        "mean_proj_coeff": _safe_mean(proj_coeff_vals),
        "std_proj_coeff": _safe_std(proj_coeff_vals),
        "mean_proj_gauss": _safe_mean(proj_gauss_vals),
        "std_proj_gauss": _safe_std(proj_gauss_vals),
        "proj_deviation": (abs(_safe_mean(proj_coeff_vals) - 1.0) if proj_coeff_vals else float("nan")),
        "n_pairs": float(n_matched),
    }


def render_markdown_report(
    metrics: Dict[str, float],
    embedding_dir: Path,
    ground_truth_embedding_dir: Path,
    generated_at: datetime,
) -> str:
    """Render markdown report content for scoring output.

    Args:
        metrics: Dictionary of CPRS metrics from compute_cprs_score
        embedding_dir: Path to generated embeddings directory
        ground_truth_embedding_dir: Path to GT embeddings directory
        generated_at: Timestamp of report generation

    Returns:
        Markdown-formatted report string
    """
    lines: List[str] = []

    # Header
    lines.append("# CPRS Scoring Report")
    lines.append("")

    # Metadata
    lines.append(f"**Generated:** {generated_at.isoformat(timespec='seconds')}")
    lines.append("")

    lines.append("## Configuration")
    lines.append(f"- Embedding directory: `{embedding_dir}`")
    lines.append(f"- Ground-truth directory: `{ground_truth_embedding_dir}`")
    lines.append("")

    # Overall score and statistics
    lines.append("## Results")
    lines.append("")

    lines.append("### Summary")
    lines.append(f"**Overall CPRS Score:** `{metrics['mean_cprs']:.6f}`")
    lines.append(f"**Matched Pairs:** {int(metrics['n_pairs'])}")
    lines.append("")

    lines.append("### Detailed Statistics")
    lines.append("")
    lines.append("| Metric | Mean | Std |")
    lines.append("|--------|------|-----|")
    lines.append(f"| CPRS | {metrics['mean_cprs']:.6f} | {metrics['std_cprs']:.6f} |")
    lines.append(f"| Cosine Similarity | {metrics['mean_cosine']:.6f} | {metrics['std_cosine']:.6f} |")
    lines.append(f"| Projection Coefficient | {metrics['mean_proj_coeff']:.6f} | {metrics['std_proj_coeff']:.6f} |")
    lines.append(f"| Projection Gaussian | {metrics['mean_proj_gauss']:.6f} | {metrics['std_proj_gauss']:.6f} |")
    lines.append("")
    lines.append(f"**Projection Deviation:** {metrics['proj_deviation']:.6f}")
    lines.append("")

    return "\n".join(lines)


def render_combined_markdown_report(
    sections: List[Tuple[str, Path, Dict[str, float]]],
    embedding_root: Path,
    ground_truth_embedding_dir: Path,
    generated_at: datetime,
) -> str:
    """Render a combined markdown report for multiple embedding sections."""
    total_pairs = sum(section_metrics["n_pairs"] for _, _, section_metrics in sections)
    overall_cprs = (
        sum(section_metrics["mean_cprs"] * section_metrics["n_pairs"] for _, _, section_metrics in sections)
        / total_pairs
        if total_pairs
        else float("nan")
    )

    lines: List[str] = []
    lines.append("# CPRS Scoring Report")
    lines.append("")
    lines.append(f"**Generated:** {generated_at.isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("## Configuration")
    lines.append(f"- Embedding root: `{embedding_root}`")
    lines.append(f"- Ground-truth directory: `{ground_truth_embedding_dir}`")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("### Summary")
    lines.append(f"**Overall CPRS Score:** `{overall_cprs:.6f}`")
    lines.append(f"**Matched Pairs:** {int(total_pairs)}")
    lines.append("")

    for section_name, embedding_dir, metrics in sections:
        lines.append(f"### {section_name}")
        lines.append(f"- Embedding directory: `{embedding_dir}`")
        lines.append(f"- Ground-truth directory: `{ground_truth_embedding_dir}`")
        lines.append("")
        lines.append("| Metric | Mean | Std |")
        lines.append("|--------|------|-----|")
        lines.append(f"| CPRS | {metrics['mean_cprs']:.6f} | {metrics['std_cprs']:.6f} |")
        lines.append(f"| Cosine Similarity | {metrics['mean_cosine']:.6f} | {metrics['std_cosine']:.6f} |")
        lines.append(
            f"| Projection Coefficient | {metrics['mean_proj_coeff']:.6f} | {metrics['std_proj_coeff']:.6f} |"
        )
        lines.append(f"| Projection Gaussian | {metrics['mean_proj_gauss']:.6f} | {metrics['std_proj_gauss']:.6f} |")
        lines.append("")
        lines.append(f"**Projection Deviation:** {metrics['proj_deviation']:.6f}")
        lines.append(f"**Matched Pairs:** {int(metrics['n_pairs'])}")
        lines.append("")

    return "\n".join(lines)


def write_markdown_report(markdown_path: Path, content: str) -> None:
    """Write markdown report content to disk.

    Args:
        markdown_path: Path where the report should be written
        content: Markdown content to write
    """
    markdown_path = Path(markdown_path)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)

    with open(markdown_path, "w", encoding="utf-8") as f:
        f.write(content)
