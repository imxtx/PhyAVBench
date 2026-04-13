from pathlib import Path
import types
import sys
from unittest.mock import patch


# Allow importing from src-layout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phyavbench.utils.pipeline import extract_clap_embeddings
from phyavbench.utils.pipeline import extract_clap_embeddings_multi_target


class _FakeClapModule:
    def __init__(self) -> None:
        self.loaded = False
        self.calls = []

    class CLAP_Module:  # noqa: N801 - match third-party API
        def __init__(self, enable_fusion: bool = False) -> None:
            self.enable_fusion = enable_fusion
            self.loaded = False
            self.calls = []

        def load_ckpt(self) -> None:
            self.loaded = True

        def get_audio_embedding_from_filelist(self, filelist, use_tensor=False):  # type: ignore[no-untyped-def]
            self.calls.append((list(filelist), use_tensor))
            return [[1.0, 2.0] for _ in filelist]


def test_extract_clap_embeddings_empty_audio_files_raises() -> None:
    try:
        extract_clap_embeddings(audio_files=[], embedding_output_dir=Path("/tmp/embed"))
    except ValueError as exc:
        assert "No audio files provided" in str(exc)
    else:
        assert False, "Expected ValueError"


def test_extract_clap_embeddings_multiple_audio_dirs_raises(tmp_path: Path) -> None:
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir(parents=True, exist_ok=True)
    dir_b.mkdir(parents=True, exist_ok=True)
    file_a = dir_a / "x.wav"
    file_b = dir_b / "y.wav"
    file_a.write_bytes(b"x")
    file_b.write_bytes(b"y")

    try:
        extract_clap_embeddings(
            audio_files=[file_a, file_b],
            embedding_output_dir=tmp_path / "embeddings",
        )
    except ValueError as exc:
        assert "single directory" in str(exc)
    else:
        assert False, "Expected ValueError"


def test_extract_clap_embeddings_missing_module_raises_module_not_found(
    tmp_path: Path,
) -> None:
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    audio_file = audio_dir / "x.wav"
    audio_file.write_bytes(b"x")

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]
        if name == "laion_clap":
            raise ModuleNotFoundError(name)
        return original_import(name, globals, locals, fromlist, level)

    original_import = __import__

    with patch("builtins.__import__", side_effect=fake_import):
        try:
            extract_clap_embeddings(
                audio_files=[audio_file],
                embedding_output_dir=tmp_path / "embeddings",
                batch_size=32,
            )
        except ModuleNotFoundError as exc:
            assert "laion_clap" in str(exc)
        else:
            assert False, "Expected ModuleNotFoundError"


def test_extract_clap_embeddings_rejects_non_positive_batch_size(
    tmp_path: Path,
) -> None:
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    audio_file = audio_dir / "x.wav"
    audio_file.write_bytes(b"x")

    try:
        extract_clap_embeddings(
            audio_files=[audio_file],
            embedding_output_dir=tmp_path / "embeddings",
            batch_size=0,
        )
    except ValueError as exc:
        assert "positive integer" in str(exc)
    else:
        assert False, "Expected ValueError"


def test_extract_clap_embeddings_success_saves_embeddings(tmp_path: Path) -> None:
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    audio_file = audio_dir / "x.wav"
    audio_file.write_bytes(b"x")
    embedding_output_dir = tmp_path / "embeddings"

    fake_module = types.SimpleNamespace(CLAP_Module=_FakeClapModule.CLAP_Module)

    with patch.dict(sys.modules, {"laion_clap": fake_module}):
        extract_clap_embeddings(
            audio_files=[audio_file],
            embedding_output_dir=embedding_output_dir,
            batch_size=24,
        )

    saved = embedding_output_dir / "clap" / "x.npy"
    assert saved.exists()
    assert saved.read_bytes().startswith(b"\x93NUMPY")


def test_extract_clap_embeddings_multi_target_saves_to_each_root(
    tmp_path: Path,
) -> None:
    audio_dir_a = tmp_path / "audio_a"
    audio_dir_b = tmp_path / "audio_b"
    audio_dir_a.mkdir(parents=True, exist_ok=True)
    audio_dir_b.mkdir(parents=True, exist_ok=True)
    file_a = audio_dir_a / "x.wav"
    file_b = audio_dir_b / "y.wav"
    file_a.write_bytes(b"x")
    file_b.write_bytes(b"y")

    fake_module = types.SimpleNamespace(CLAP_Module=_FakeClapModule.CLAP_Module)

    with patch.dict(sys.modules, {"laion_clap": fake_module}):
        extract_clap_embeddings_multi_target(
            audio_embedding_targets=[
                (file_a, tmp_path / "model_a" / "audio_embedding"),
                (file_b, tmp_path / "model_b" / "audio_embedding"),
            ],
            batch_size=2,
        )

    assert (tmp_path / "model_a" / "audio_embedding" / "clap" / "x.npy").exists()
    assert (tmp_path / "model_b" / "audio_embedding" / "clap" / "y.npy").exists()
