from __future__ import annotations

import math
import struct
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

from tqdm import tqdm


def collect_audio_files(directory: Path) -> List[Path]:
    exts = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
    files = [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in exts]
    return sorted(files)


def _as_nested_list(value: object) -> object:
    if hasattr(value, "tolist"):
        return getattr(value, "tolist")()
    return value


def _infer_shape(value: object) -> tuple[int, ...]:
    if isinstance(value, (list, tuple)):
        if not value:
            return (0,)

        child_shape = _infer_shape(value[0])
        for item in value[1:]:
            if _infer_shape(item) != child_shape:
                raise ValueError("CLAP embeddings must have a consistent rectangular shape")
        return (len(value),) + child_shape

    return ()


def _flatten_numeric_values(value: object) -> List[float]:
    if isinstance(value, (list, tuple)):
        flattened: List[float] = []
        for item in value:
            flattened.extend(_flatten_numeric_values(item))
        return flattened

    return [float(value)]


def _write_npy_file(target: Path, embedding: object) -> None:
    nested_value = _as_nested_list(embedding)
    shape = _infer_shape(nested_value)
    if shape == ():
        nested_value = [nested_value]
        shape = (1,)

    flat_values = _flatten_numeric_values(nested_value)
    header_dict = {"descr": "<f8", "fortran_order": False, "shape": shape}
    header = repr(header_dict) + " "
    magic_and_version = 10
    padding = (-(magic_and_version + len(header) + 1)) % 16
    header = header[:-1] + (" " * padding) + "\n"

    with target.open("wb") as file_handle:
        file_handle.write(b"\x93NUMPY")
        file_handle.write(bytes([1, 0]))
        file_handle.write(struct.pack("<H", len(header)))
        file_handle.write(header.encode("latin1"))
        file_handle.write(struct.pack(f"<{len(flat_values)}d", *flat_values))


def _extract_clap_embeddings_from_audio_dir(
    audio_dir: Path,
    clap_output_dir: Path,
    batch_size: int,
) -> None:
    import laion_clap  # pyright: ignore[reportMissingImports]

    wav_files = collect_audio_files(audio_dir)
    if not wav_files:
        raise RuntimeError(f"No audio files found in directory: {audio_dir}")

    clap_output_dir.mkdir(parents=True, exist_ok=True)

    model = laion_clap.CLAP_Module(enable_fusion=False)
    model.load_ckpt()

    total = len(wav_files)
    batch_count = math.ceil(total / batch_size)
    for idx in tqdm(range(batch_count), desc="[clap]", unit="batch"):
        start = idx * batch_size
        end = min(start + batch_size, total)
        batch = wav_files[start:end]

        try:
            embeddings = model.get_audio_embedding_from_filelist(
                [str(path) for path in batch],
                use_tensor=False,
            )
        except Exception as exc:
            raise RuntimeError(f"CLAP inference failed for directory {audio_dir}: {exc}") from exc

        if len(embeddings) != len(batch):
            raise RuntimeError(
                "CLAP model returned a different number of embeddings than input files: "
                f"expected {len(batch)}, got {len(embeddings)}"
            )

        for wav_path, embedding in zip(batch, embeddings):
            target = clap_output_dir / f"{wav_path.stem}.npy"
            _write_npy_file(target, embedding)

    print(f"[clap] done: {total} files -> {clap_output_dir}")


def _extract_imagebind_embeddings_from_audio_files(
    audio_files: List[Path],
    imagebind_output_dir: Path,
    batch_size: int,
) -> None:
    import torch  # pyright: ignore[reportMissingImports]
    from imagebind import (  # pyright: ignore[reportMissingImports]
        data as imagebind_data,
    )
    from imagebind.models import (  # pyright: ignore[reportMissingImports]
        imagebind_model,
    )
    from imagebind.models.imagebind_model import (  # pyright: ignore[reportMissingImports]
        ModalityType,
    )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    imagebind_output_dir.mkdir(parents=True, exist_ok=True)

    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)

    total = len(audio_files)
    batch_count = math.ceil(total / batch_size)
    for idx in tqdm(range(batch_count), desc="[imagebind]", unit="batch"):
        start = idx * batch_size
        end = min(start + batch_size, total)
        batch = audio_files[start:end]

        try:
            with torch.no_grad():
                inputs = {
                    ModalityType.AUDIO: imagebind_data.load_and_transform_audio_data(
                        [str(path) for path in batch],
                        device,
                    )
                }
                outputs = model(inputs)
                embeddings = outputs[ModalityType.AUDIO].detach().cpu().tolist()
        except Exception as exc:
            raise RuntimeError(f"ImageBind inference failed: {exc}") from exc

        if len(embeddings) != len(batch):
            raise RuntimeError(
                "ImageBind model returned a different number of embeddings than input files: "
                f"expected {len(batch)}, got {len(embeddings)}"
            )

        for wav_path, embedding in zip(batch, embeddings):
            target = imagebind_output_dir / f"{wav_path.stem}.npy"
            _write_npy_file(target, embedding)

    print(f"[imagebind] done: {total} files -> {imagebind_output_dir}")


def _validate_audio_embedding_targets(
    audio_embedding_targets: List[Tuple[Path, Path]],
    model_name: str,
) -> List[Tuple[Path, Path]]:
    if not audio_embedding_targets:
        raise ValueError(f"No audio files provided for {model_name} embedding extraction")

    validated_targets: List[Tuple[Path, Path]] = []
    for audio_file, embedding_output_dir in audio_embedding_targets:
        resolved_audio_file = Path(audio_file)
        resolved_embedding_output_dir = Path(embedding_output_dir)

        if not resolved_audio_file.exists() or not resolved_audio_file.is_file():
            raise FileNotFoundError(f"Audio file not found or not a file: {resolved_audio_file}")

        validated_targets.append((resolved_audio_file, resolved_embedding_output_dir))

    return validated_targets


def extract_clap_embeddings_multi_target(
    audio_embedding_targets: List[Tuple[Path, Path]],
    batch_size: int = 60,
) -> None:
    """Extract CLAP embeddings for multiple audio files and output roots in one run.

    This function loads the CLAP model exactly once, then writes each embedding to
    <embedding_output_dir>/clap/<audio_stem>.npy according to the provided mapping.
    """
    if batch_size <= 0:
        raise ValueError("CLAP batch size must be a positive integer")

    validated_targets = _validate_audio_embedding_targets(
        audio_embedding_targets=audio_embedding_targets,
        model_name="CLAP",
    )

    import laion_clap  # pyright: ignore[reportMissingImports]

    for _, embedding_output_dir in validated_targets:
        (embedding_output_dir / "clap").mkdir(parents=True, exist_ok=True)

    audio_files = [audio_file for audio_file, _ in validated_targets]
    total = len(audio_files)
    batch_count = math.ceil(total / batch_size)

    model = laion_clap.CLAP_Module(enable_fusion=False)
    model.load_ckpt()

    for idx in tqdm(range(batch_count), desc="[clap]", unit="batch"):
        start = idx * batch_size
        end = min(start + batch_size, total)
        batch_targets = validated_targets[start:end]
        batch_files = [audio_file for audio_file, _ in batch_targets]

        try:
            embeddings = model.get_audio_embedding_from_filelist(
                [str(path) for path in batch_files],
                use_tensor=False,
            )
        except Exception as exc:
            raise RuntimeError(f"CLAP inference failed: {exc}") from exc

        if len(embeddings) != len(batch_targets):
            raise RuntimeError(
                "CLAP model returned a different number of embeddings than input files: "
                f"expected {len(batch_targets)}, got {len(embeddings)}"
            )

        for (wav_path, embedding_output_dir), embedding in zip(batch_targets, embeddings):
            target = embedding_output_dir / "clap" / f"{wav_path.stem}.npy"
            _write_npy_file(target, embedding)

    print(f"[clap] done: {total} files across multiple targets")


def extract_imagebind_embeddings_multi_target(
    audio_embedding_targets: List[Tuple[Path, Path]],
    batch_size: int = 60,
) -> None:
    """Extract ImageBind embeddings for multiple audio files and output roots in one run.

    This function loads the ImageBind model exactly once, then writes each embedding
    to <embedding_output_dir>/imagebind/<audio_stem>.npy according to the mapping.
    """
    if batch_size <= 0:
        raise ValueError("ImageBind batch size must be a positive integer")

    validated_targets = _validate_audio_embedding_targets(
        audio_embedding_targets=audio_embedding_targets,
        model_name="ImageBind",
    )

    import torch  # pyright: ignore[reportMissingImports]
    from imagebind import (  # pyright: ignore[reportMissingImports]
        data as imagebind_data,
    )
    from imagebind.models import (  # pyright: ignore[reportMissingImports]
        imagebind_model,
    )
    from imagebind.models.imagebind_model import (  # pyright: ignore[reportMissingImports]
        ModalityType,
    )

    for _, embedding_output_dir in validated_targets:
        (embedding_output_dir / "imagebind").mkdir(parents=True, exist_ok=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    audio_files = [audio_file for audio_file, _ in validated_targets]
    total = len(audio_files)
    batch_count = math.ceil(total / batch_size)

    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)

    for idx in tqdm(range(batch_count), desc="[imagebind]", unit="batch"):
        start = idx * batch_size
        end = min(start + batch_size, total)
        batch_targets = validated_targets[start:end]
        batch_files = [audio_file for audio_file, _ in batch_targets]

        try:
            with torch.no_grad():
                inputs = {
                    ModalityType.AUDIO: imagebind_data.load_and_transform_audio_data(
                        [str(path) for path in batch_files],
                        device,
                    )
                }
                outputs = model(inputs)
                embeddings = outputs[ModalityType.AUDIO].detach().cpu().tolist()
        except Exception as exc:
            raise RuntimeError(f"ImageBind inference failed: {exc}") from exc

        if len(embeddings) != len(batch_targets):
            raise RuntimeError(
                "ImageBind model returned a different number of embeddings than input files: "
                f"expected {len(batch_targets)}, got {len(embeddings)}"
            )

        for (wav_path, embedding_output_dir), embedding in zip(batch_targets, embeddings):
            target = embedding_output_dir / "imagebind" / f"{wav_path.stem}.npy"
            _write_npy_file(target, embedding)

    print(f"[imagebind] done: {total} files across multiple targets")


def extract_audio_with_ffmpeg(
    video_dir: Path,
    audio_output_dir: Path,
    video_files: Optional[List[Path]] = None,
    skip_existing: bool = False,
) -> List[Path]:
    """Extract WAV audio files from all mp4 files in a directory.

    The extraction follows a two-step pipeline for each video:
    1) ffmpeg resamples to 16k intermediate wav
    2) sox extracts the first channel to the final wav

    Raises:
        FileNotFoundError: Input directory does not exist or is not a directory.
        ValueError: Input directory contains no mp4 files.
        RuntimeError: ffmpeg/sox is missing or command execution fails.
    """
    if not video_dir.exists() or not video_dir.is_dir():
        raise FileNotFoundError(f"Input video directory not found or not a directory: {video_dir}")

    if video_files is None:
        video_files = sorted(path for path in video_dir.iterdir() if path.is_file() and path.suffix.lower() == ".mp4")
    else:
        video_files = [Path(path) for path in video_files]
        for video_path in video_files:
            if not video_path.exists() or not video_path.is_file():
                raise FileNotFoundError(f"Input video file not found or not a file: {video_path}")
            if video_path.suffix.lower() != ".mp4":
                raise ValueError(f"Input video file must be an mp4 file: {video_path}")

    if not video_files:
        raise ValueError(f"No mp4 files found in input directory: {video_dir}")

    audio_output_dir.mkdir(parents=True, exist_ok=True)
    output_audio_files: List[Path] = []
    for video_path in tqdm(video_files, desc="[audio]", unit="file"):
        intermediate_audio_path = audio_output_dir / f"{video_path.stem}_intermediate.wav"
        output_audio_path = audio_output_dir / f"{video_path.stem}.wav"

        if skip_existing and output_audio_path.exists() and output_audio_path.is_file():
            output_audio_files.append(output_audio_path)
            continue

        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-ar",
            "16000",
            str(intermediate_audio_path),
        ]
        sox_cmd = [
            "sox",
            str(intermediate_audio_path),
            str(output_audio_path),
            "remix",
            "1",
        ]

        try:
            ffmpeg_completed = subprocess.run(
                ffmpeg_cmd,
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("ffmpeg command not found. Please install ffmpeg and ensure it is in PATH.") from exc

        if ffmpeg_completed.returncode != 0:
            detail = (ffmpeg_completed.stderr or ffmpeg_completed.stdout or "unknown ffmpeg error").strip()
            raise RuntimeError(
                f"ffmpeg failed for {video_path} with exit code {ffmpeg_completed.returncode}: {detail}"
            )

        try:
            sox_completed = subprocess.run(
                sox_cmd,
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("sox command not found. Please install sox and ensure it is in PATH.") from exc

        if sox_completed.returncode != 0:
            detail = (sox_completed.stderr or sox_completed.stdout or "unknown sox error").strip()
            raise RuntimeError(f"sox failed for {video_path} with exit code {sox_completed.returncode}: {detail}")

        if intermediate_audio_path.exists():
            intermediate_audio_path.unlink()

        if not output_audio_path.exists():
            raise RuntimeError(
                f"Audio extraction reported success but output audio file was not created: {output_audio_path}"
            )

        output_audio_files.append(output_audio_path)

    return output_audio_files


def extract_clap_embeddings(
    audio_files: List[Path],
    embedding_output_dir: Path,
    batch_size: int = 60,
) -> None:
    """Extract and save CLAP embeddings for audio files from one directory.

    The embedding model is loaded in-process so the pipeline no longer shells out
    to a separate script.

    Raises:
        ValueError: Audio file list is empty, spans multiple directories, or batch size is invalid.
        FileNotFoundError: Audio directory does not exist.
        ModuleNotFoundError: Required module laion_clap is unavailable.
        RuntimeError: CLAP inference fails.
    """
    if not audio_files:
        raise ValueError("No audio files provided for CLAP embedding extraction")

    audio_dirs = {path.parent.resolve() for path in audio_files}
    if len(audio_dirs) != 1:
        raise ValueError("CLAP extraction expects audio files from a single directory")
    if batch_size <= 0:
        raise ValueError("CLAP batch size must be a positive integer")

    audio_dir = next(iter(audio_dirs))
    if not audio_dir.exists() or not audio_dir.is_dir():
        raise FileNotFoundError(f"Audio directory not found or not a directory: {audio_dir}")

    clap_output_dir = embedding_output_dir / "clap"
    _extract_clap_embeddings_from_audio_dir(
        audio_dir=audio_dir, clap_output_dir=clap_output_dir, batch_size=batch_size
    )


def extract_imagebind_embeddings(
    audio_files: List[Path],
    embedding_output_dir: Path,
    batch_size: int = 60,
) -> None:
    """Extract and save ImageBind embeddings for audio files from one directory.

    Raises:
        ValueError: Audio file list is empty, spans multiple directories, or batch size is invalid.
        FileNotFoundError: Audio directory does not exist.
        ModuleNotFoundError: Required modules torch/imagebind are unavailable.
        RuntimeError: ImageBind inference fails.
    """
    if not audio_files:
        raise ValueError("No audio files provided for ImageBind embedding extraction")

    audio_dirs = {path.parent.resolve() for path in audio_files}
    if len(audio_dirs) != 1:
        raise ValueError("ImageBind extraction expects audio files from a single directory")
    if batch_size <= 0:
        raise ValueError("ImageBind batch size must be a positive integer")

    audio_dir = next(iter(audio_dirs))
    if not audio_dir.exists() or not audio_dir.is_dir():
        raise FileNotFoundError(f"Audio directory not found or not a directory: {audio_dir}")

    imagebind_output_dir = embedding_output_dir / "imagebind"
    resolved_audio_files = collect_audio_files(audio_dir)
    if not resolved_audio_files:
        raise RuntimeError(f"No audio files found in directory: {audio_dir}")

    _extract_imagebind_embeddings_from_audio_files(
        audio_files=resolved_audio_files,
        imagebind_output_dir=imagebind_output_dir,
        batch_size=batch_size,
    )
