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


def _make_zip_bytes_with_zipinfo(entries: list[zipfile.ZipInfo], payloads: list[bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for info, content in zip(entries, payloads, strict=True):
            zf.writestr(info, content)
    return buf.getvalue()


def _set_encrypted_flag_for_member(zip_bytes: bytes, *, filename: str) -> bytes:
    """Patch ZIP bytes to mark a specific member as encrypted.

    stdlib zipfile cannot produce encrypted ZIPs, but we can set the encryption
    flag bit in headers. Our extractor checks this bit and should skip those
    entries without attempting to open them.
    """
    data = bytearray(zip_bytes)
    needle = filename.encode("utf-8")

    def _patch_at(name_pos: int, *, header_len: int, signature: bytes, flag_offset: int) -> None:
        header_start = name_pos - header_len
        if header_start < 0:
            return
        if data[header_start : header_start + 4] != signature:
            return
        flags_pos = header_start + flag_offset
        flags = int.from_bytes(data[flags_pos : flags_pos + 2], "little")
        flags |= 0x1
        data[flags_pos : flags_pos + 2] = flags.to_bytes(2, "little")

    # Local file header has a fixed 30-byte header before the filename.
    # Central directory header has a fixed 46-byte header before the filename.
    idx = 0
    while True:
        pos = data.find(needle, idx)
        if pos == -1:
            break
        _patch_at(pos, header_len=30, signature=b"PK\x03\x04", flag_offset=6)
        _patch_at(pos, header_len=46, signature=b"PK\x01\x02", flag_offset=8)
        idx = pos + 1

    return bytes(data)


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
def test_extract_zip_bytes_safely_blocks_absolute_paths_and_backslash_traversal(tmp_path: Path):
    payload = _make_zip_bytes(
        {
            "/etc/passwd": b"nope",
            "..\\escape.txt": b"nope",
            "ok.txt": b"yes",
        }
    )

    res = extract_zip_bytes_safely(
        content=payload,
        dest_dir=tmp_path,
        max_files=10,
        max_total_uncompressed_bytes=1024 * 1024,
    )

    extracted_names = {p.name for p in res.extracted_paths}
    assert "ok.txt" in extracted_names
    assert "passwd" not in extracted_names
    assert "escape.txt" not in extracted_names


@pytest.mark.unit
def test_extract_zip_bytes_safely_enforces_max_filename_length(tmp_path: Path):
    payload = _make_zip_bytes({"this_component_is_too_long.txt": b"x", "ok.txt": b"y"})

    res = extract_zip_bytes_safely(
        content=payload,
        dest_dir=tmp_path,
        max_files=10,
        max_total_uncompressed_bytes=1024 * 1024,
        max_filename_length=10,
    )

    extracted_names = {p.name for p in res.extracted_paths}
    assert "ok.txt" in extracted_names
    assert "this_component_is_too_long.txt" not in extracted_names
    assert res.skipped_entries >= 1


@pytest.mark.unit
def test_extract_zip_bytes_safely_skips_encrypted_entries(tmp_path: Path):
    raw = _make_zip_bytes({"secret.txt": b"secret", "ok.txt": b"ok"})
    payload = _set_encrypted_flag_for_member(raw, filename="secret.txt")

    res = extract_zip_bytes_safely(
        content=payload,
        dest_dir=tmp_path,
        max_files=10,
        max_total_uncompressed_bytes=1024 * 1024,
    )

    extracted_names = {p.name for p in res.extracted_paths}
    assert "ok.txt" in extracted_names
    assert "secret.txt" not in extracted_names
    assert res.skipped_entries >= 1


@pytest.mark.unit
def test_extract_zip_bytes_safely_skips_symlink_entries(tmp_path: Path):
    # Craft a symlink-like ZipInfo via POSIX mode bits.
    link = zipfile.ZipInfo("link")
    link.create_system = 3  # Unix
    link.external_attr = (0o120777 << 16)  # symlink

    ok = zipfile.ZipInfo("ok.txt")

    payload = _make_zip_bytes_with_zipinfo([link, ok], [b"/etc/passwd", b"ok"])

    res = extract_zip_bytes_safely(
        content=payload,
        dest_dir=tmp_path,
        max_files=10,
        max_total_uncompressed_bytes=1024 * 1024,
    )

    extracted_names = {p.name for p in res.extracted_paths}
    assert "ok.txt" in extracted_names
    assert "link" not in extracted_names
    assert res.skipped_entries >= 1


@pytest.mark.unit
def test_extract_zip_bytes_safely_counts_directory_entries_as_skipped(tmp_path: Path):
    payload = _make_zip_bytes({"dir/": b"", "dir/file.txt": b"x"})

    res = extract_zip_bytes_safely(
        content=payload,
        dest_dir=tmp_path,
        max_files=10,
        max_total_uncompressed_bytes=1024 * 1024,
    )

    extracted_names = {p.as_posix().split("/")[-1] for p in res.extracted_paths}
    assert "file.txt" in extracted_names
    assert res.skipped_entries >= 1


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


@pytest.mark.unit
def test_extract_zip_bytes_safely_truncates_on_total_size(tmp_path: Path):
    payload = _make_zip_bytes({"big.txt": b"x" * 20})

    res = extract_zip_bytes_safely(
        content=payload,
        dest_dir=tmp_path,
        max_files=10,
        max_total_uncompressed_bytes=10,
    )

    assert res.truncated is True
    assert res.extracted_paths == []
