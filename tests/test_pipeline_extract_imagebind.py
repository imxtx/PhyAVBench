from pathlib import Path
import sys
import types
from unittest.mock import patch


# Allow importing from src-layout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phyavbench.utils.pipeline import extract_imagebind_embeddings
from phyavbench.utils.pipeline import extract_imagebind_embeddings_multi_target


class _FakeTensor:
    def __init__(self, values):  # type: ignore[no-untyped-def]
        self._values = values

    def detach(self):  # type: ignore[no-untyped-def]
        return self

    def cpu(self):  # type: ignore[no-untyped-def]
        return self

    def tolist(self):  # type: ignore[no-untyped-def]
        return self._values


class _FakeImageBindModel:
    def eval(self) -> None:
        pass

    def to(self, _device: str) -> None:
        pass

    def __call__(self, inputs):  # type: ignore[no-untyped-def]
        batch_size = len(inputs["audio"])
        values = [[float(index), float(index) + 0.5] for index in range(batch_size)]
        return {"audio": _FakeTensor(values)}


def test_extract_imagebind_embeddings_empty_audio_files_raises() -> None:
    try:
        extract_imagebind_embeddings(
            audio_files=[], embedding_output_dir=Path("/tmp/embed")
        )
    except ValueError as exc:
        assert "No audio files provided" in str(exc)
    else:
        assert False, "Expected ValueError"


def test_extract_imagebind_embeddings_multiple_audio_dirs_raises(
    tmp_path: Path,
) -> None:
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir(parents=True, exist_ok=True)
    dir_b.mkdir(parents=True, exist_ok=True)
    file_a = dir_a / "x.wav"
    file_b = dir_b / "y.wav"
    file_a.write_bytes(b"x")
    file_b.write_bytes(b"y")

    try:
        extract_imagebind_embeddings(
            audio_files=[file_a, file_b],
            embedding_output_dir=tmp_path / "embeddings",
        )
    except ValueError as exc:
        assert "single directory" in str(exc)
    else:
        assert False, "Expected ValueError"


def test_extract_imagebind_embeddings_rejects_non_positive_batch_size(
    tmp_path: Path,
) -> None:
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    audio_file = audio_dir / "x.wav"
    audio_file.write_bytes(b"x")

    try:
        extract_imagebind_embeddings(
            audio_files=[audio_file],
            embedding_output_dir=tmp_path / "embeddings",
            batch_size=0,
        )
    except ValueError as exc:
        assert "positive integer" in str(exc)
    else:
        assert False, "Expected ValueError"


def test_extract_imagebind_embeddings_missing_torch_raises_module_not_found(
    tmp_path: Path,
) -> None:
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    audio_file = audio_dir / "x.wav"
    audio_file.write_bytes(b"x")

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]
        if name == "torch":
            raise ModuleNotFoundError(name)
        return original_import(name, globals, locals, fromlist, level)

    original_import = __import__

    with patch("builtins.__import__", side_effect=fake_import):
        try:
            extract_imagebind_embeddings(
                audio_files=[audio_file],
                embedding_output_dir=tmp_path / "embeddings",
                batch_size=8,
            )
        except ModuleNotFoundError as exc:
            assert "torch" in str(exc)
        else:
            assert False, "Expected ModuleNotFoundError"


def test_extract_imagebind_embeddings_success_saves_embeddings(tmp_path: Path) -> None:
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    (audio_dir / "x.wav").write_bytes(b"x")
    (audio_dir / "y.wav").write_bytes(b"y")

    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
        no_grad=lambda: _NoGradContext(),
    )
    fake_modality_type = types.SimpleNamespace(AUDIO="audio")
    fake_data_module = types.SimpleNamespace(
        load_and_transform_audio_data=lambda filelist, _device: filelist
    )
    fake_imagebind_model_module = types.SimpleNamespace(
        imagebind_huge=lambda pretrained=True: _FakeImageBindModel()
    )

    with patch.dict(
        sys.modules,
        {
            "torch": fake_torch,
            "imagebind": types.SimpleNamespace(data=fake_data_module),
            "imagebind.data": fake_data_module,
            "imagebind.models": types.SimpleNamespace(
                imagebind_model=fake_imagebind_model_module
            ),
            "imagebind.models.imagebind_model": types.SimpleNamespace(
                ModalityType=fake_modality_type,
                imagebind_huge=fake_imagebind_model_module.imagebind_huge,
            ),
        },
    ):
        extract_imagebind_embeddings(
            audio_files=[audio_dir / "x.wav", audio_dir / "y.wav"],
            embedding_output_dir=tmp_path / "embeddings",
            batch_size=1,
        )

    saved_dir = tmp_path / "embeddings" / "imagebind"
    assert (saved_dir / "x.npy").exists()
    assert (saved_dir / "y.npy").exists()
    assert (saved_dir / "x.npy").read_bytes().startswith(b"\x93NUMPY")


def test_extract_imagebind_embeddings_multi_target_saves_to_each_root(
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

    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
        no_grad=lambda: _NoGradContext(),
    )
    fake_modality_type = types.SimpleNamespace(AUDIO="audio")
    fake_data_module = types.SimpleNamespace(
        load_and_transform_audio_data=lambda filelist, _device: filelist
    )
    fake_imagebind_model_module = types.SimpleNamespace(
        imagebind_huge=lambda pretrained=True: _FakeImageBindModel()
    )

    with patch.dict(
        sys.modules,
        {
            "torch": fake_torch,
            "imagebind": types.SimpleNamespace(data=fake_data_module),
            "imagebind.data": fake_data_module,
            "imagebind.models": types.SimpleNamespace(
                imagebind_model=fake_imagebind_model_module
            ),
            "imagebind.models.imagebind_model": types.SimpleNamespace(
                ModalityType=fake_modality_type,
                imagebind_huge=fake_imagebind_model_module.imagebind_huge,
            ),
        },
    ):
        extract_imagebind_embeddings_multi_target(
            audio_embedding_targets=[
                (file_a, tmp_path / "model_a" / "audio_embedding"),
                (file_b, tmp_path / "model_b" / "audio_embedding"),
            ],
            batch_size=2,
        )

    assert (tmp_path / "model_a" / "audio_embedding" / "imagebind" / "x.npy").exists()
    assert (tmp_path / "model_b" / "audio_embedding" / "imagebind" / "y.npy").exists()


class _NoGradContext:
    def __enter__(self):  # type: ignore[no-untyped-def]
        return None

    def __exit__(self, exc_type, exc, tb):  # type: ignore[no-untyped-def]
        return False
