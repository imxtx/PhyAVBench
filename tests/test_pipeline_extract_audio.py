from pathlib import Path
import subprocess
import sys
from unittest.mock import patch


# Allow importing from src-layout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phyavbench.utils.pipeline import extract_audio_with_ffmpeg


def test_extract_audio_with_ffmpeg_missing_input_dir_raises() -> None:
    missing_dir = Path("/tmp/this_video_dir_should_not_exist_123")
    output_dir = Path("/tmp/phyavbench_audio_out")

    try:
        extract_audio_with_ffmpeg(missing_dir, output_dir)
    except FileNotFoundError as exc:
        assert "Input video directory not found" in str(exc)
    else:
        assert False, "Expected FileNotFoundError"


def test_extract_audio_with_ffmpeg_no_mp4_files_raises(tmp_path: Path) -> None:
    video_dir = tmp_path / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    (video_dir / "note.txt").write_text("not-a-video", encoding="utf-8")

    try:
        extract_audio_with_ffmpeg(video_dir, tmp_path / "audio")
    except ValueError as exc:
        assert "No mp4 files found" in str(exc)
    else:
        assert False, "Expected ValueError"


def test_extract_audio_with_ffmpeg_missing_ffmpeg_raises_runtime_error(tmp_path: Path) -> None:
    video_dir = tmp_path / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    (video_dir / "demo.mp4").write_bytes(b"fake-video")

    with patch("phyavbench.utils.pipeline.subprocess.run", side_effect=FileNotFoundError):
        try:
            extract_audio_with_ffmpeg(video_dir, tmp_path / "audio")
        except RuntimeError as exc:
            assert "ffmpeg command not found" in str(exc)
        else:
            assert False, "Expected RuntimeError"


def test_extract_audio_with_ffmpeg_nonzero_exit_raises_runtime_error(tmp_path: Path) -> None:
    video_dir = tmp_path / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    video = video_dir / "demo.mp4"
    video.write_bytes(b"fake-video")

    failed = subprocess.CompletedProcess(
        args=["ffmpeg"],
        returncode=1,
        stdout="",
        stderr="invalid input stream",
    )

    with patch("phyavbench.utils.pipeline.subprocess.run", return_value=failed):
        try:
            extract_audio_with_ffmpeg(video_dir, tmp_path / "audio")
        except RuntimeError as exc:
            assert "ffmpeg failed" in str(exc)
            assert "invalid input stream" in str(exc)
            assert str(video) in str(exc)
        else:
            assert False, "Expected RuntimeError"


def test_extract_audio_with_ffmpeg_sox_nonzero_exit_raises_runtime_error(tmp_path: Path) -> None:
    video_dir = tmp_path / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    video = video_dir / "demo.mp4"
    video.write_bytes(b"fake-video")
    audio_dir = tmp_path / "audio"

    ffmpeg_ok = subprocess.CompletedProcess(args=["ffmpeg"], returncode=0, stdout="", stderr="")
    sox_fail = subprocess.CompletedProcess(
        args=["sox"],
        returncode=2,
        stdout="",
        stderr="sox parse error",
    )

    with patch("phyavbench.utils.pipeline.subprocess.run", side_effect=[ffmpeg_ok, sox_fail]):
        try:
            extract_audio_with_ffmpeg(video_dir, audio_dir)
        except RuntimeError as exc:
            assert "sox failed" in str(exc)
            assert "sox parse error" in str(exc)
            assert str(video) in str(exc)
        else:
            assert False, "Expected RuntimeError"


def test_extract_audio_with_ffmpeg_missing_sox_raises_runtime_error(tmp_path: Path) -> None:
    video_dir = tmp_path / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    (video_dir / "demo.mp4").write_bytes(b"fake-video")
    audio_dir = tmp_path / "audio"

    ffmpeg_ok = subprocess.CompletedProcess(args=["ffmpeg"], returncode=0, stdout="", stderr="")

    with patch(
        "phyavbench.utils.pipeline.subprocess.run",
        side_effect=[ffmpeg_ok, FileNotFoundError()],
    ):
        try:
            extract_audio_with_ffmpeg(video_dir, audio_dir)
        except RuntimeError as exc:
            assert "sox command not found" in str(exc)
        else:
            assert False, "Expected RuntimeError"


def test_extract_audio_with_ffmpeg_success_returns_output_files_for_all_mp4(tmp_path: Path) -> None:
    video_dir = tmp_path / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    first_video = video_dir / "a.mp4"
    second_video = video_dir / "b.MP4"
    non_video = video_dir / "ignore.mov"

    first_video.write_bytes(b"fake-video-a")
    second_video.write_bytes(b"fake-video-b")
    non_video.write_bytes(b"not-processed")

    audio_dir = tmp_path / "audio"

    def fake_run(cmd, *args, **kwargs):  # type: ignore[no-untyped-def]
        tool = cmd[0]
        audio_dir.mkdir(parents=True, exist_ok=True)
        if tool == "ffmpeg":
            intermediate = Path(cmd[-1])
            intermediate.write_bytes(b"intermediate")
            return subprocess.CompletedProcess(args=["ffmpeg"], returncode=0, stdout="", stderr="")
        if tool == "sox":
            output = Path(cmd[2])
            output.write_bytes(b"fake-audio")
            return subprocess.CompletedProcess(args=["sox"], returncode=0, stdout="", stderr="")
        return subprocess.CompletedProcess(args=["ffmpeg"], returncode=0, stdout="", stderr="")

    with patch("phyavbench.utils.pipeline.subprocess.run", side_effect=fake_run):
        outputs = extract_audio_with_ffmpeg(video_dir, audio_dir)

    assert outputs == [audio_dir / "a.wav", audio_dir / "b.wav"]
    assert all(path.exists() for path in outputs)
    assert not (audio_dir / "a_intermediate.wav").exists()
    assert not (audio_dir / "b_intermediate.wav").exists()
