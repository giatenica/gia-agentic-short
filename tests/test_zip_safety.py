import io
import zipfile
from pathlib import Path

import pytest

from src.utils.zip_safety import extract_zip_bytes_safely


def _make_zip_bytes(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, content in entries.items():
            zf.writestr(name, content)
    return buf.getvalue()


@pytest.mark.unit
def test_extract_zip_bytes_safely_blocks_traversal(tmp_path: Path):
    payload = _make_zip_bytes({"../escape.txt": b"nope", "ok.txt": b"yes"})

    res = extract_zip_bytes_safely(
        content=payload,
        dest_dir=tmp_path,
        max_files=10,
        max_total_uncompressed_bytes=1024 * 1024,
    )

    extracted_names = {p.name for p in res.extracted_paths}
    assert "ok.txt" in extracted_names
    assert "escape.txt" not in extracted_names
    assert (tmp_path / "ok.txt").exists()


@pytest.mark.unit
def test_extract_zip_bytes_safely_enforces_limits(tmp_path: Path):
    payload = _make_zip_bytes({"a.txt": b"a" * 10, "b.txt": b"b" * 10})

    res = extract_zip_bytes_safely(
        content=payload,
        dest_dir=tmp_path,
        max_files=1,
        max_total_uncompressed_bytes=1024 * 1024,
    )

    assert len(res.extracted_paths) == 1
    assert res.truncated is True
