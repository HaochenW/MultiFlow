from __future__ import annotations

import hashlib

import pytest

from multiflow_omics.datasets import DatasetFile, verify_dataset_file


def test_dataset_file_verification(tmp_path):
    payload = b"paired-h5mu-test"
    path = tmp_path / "file.h5mu"
    path.write_bytes(payload)
    dataset = DatasetFile(
        name="test",
        filename=path.name,
        url="https://example.invalid/file",
        size_bytes=len(payload),
        md5=hashlib.md5(payload).hexdigest(),  # noqa: S324 - integrity fixture
        doi="10.example/test",
        license="CC0",
    )
    verify_dataset_file(path, dataset)
    with pytest.raises(ValueError, match="size mismatch"):
        verify_dataset_file(
            path,
            DatasetFile(**{**dataset.__dict__, "size_bytes": len(payload) + 1}),
        )
