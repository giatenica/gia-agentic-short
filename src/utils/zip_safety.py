"""ZIP safety utilities.

This module provides a small helper for extracting ZIP payloads safely.

Threat model:
- Zip-slip path traversal ("../" paths, absolute paths, backslashes)
- Zip bombs (many files, huge uncompressed totals)
- Symlinks (writing outside destination via link entries)
- Encrypted entries (cannot be inspected safely)

The goal is deterministic, defensive extraction behavior for local intake.
"""

from __future__ import annotations

import io
import shutil
import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import List, Optional


@dataclass(frozen=True)
class ZipExtractionResult:
    extracted_paths: List[Path]
    skipped_entries: int
    truncated: bool


def _is_symlink(info: zipfile.ZipInfo) -> bool:
    # On Unix-like systems, external_attr stores the POSIX mode in the top 16 bits.
    # Note: ZIPs created on Windows may not carry POSIX mode bits reliably.
    # This check is best-effort and intentionally conservative.
    mode = (int(info.external_attr) >> 16) & 0o170000
    return mode == stat.S_IFLNK


def _is_encrypted(info: zipfile.ZipInfo) -> bool:
    # General purpose bit 0 indicates encryption.
    return bool(int(info.flag_bits) & 0x1)


def _safe_member_relpath(name: str) -> Optional[PurePosixPath]:
    # Normalize separators to avoid Windows-style traversal.
    normalized = (name or "").replace("\\", "/")
    if not normalized:
        return None

    p = PurePosixPath(normalized)

    # Reject absolute paths or drive-letter-ish patterns.
    if p.is_absolute():
        return None

    # Reject traversal segments.
    parts = p.parts
    if any(part == ".." for part in parts):
        return None

    # Explicitly reject leading ./ to keep behavior stable.
    if parts and parts[0] == ".":
        return None

    return p


def extract_zip_bytes_safely(
    *,
    content: bytes,
    dest_dir: Path,
    max_files: int,
    max_total_uncompressed_bytes: int,
    max_filename_length: int = 255,
) -> ZipExtractionResult:
    """Safely extract a ZIP archive provided as bytes.

    Args:
        content: Raw ZIP file bytes.
        dest_dir: Directory into which files will be extracted.
        max_files: Hard cap on number of extracted files.
        max_total_uncompressed_bytes: Hard cap on sum of extracted file sizes.
        max_filename_length: Cap on individual path component lengths.

    Returns:
        ZipExtractionResult

    Raises:
        zipfile.BadZipFile: When content is not a valid ZIP.
        ValueError: When limits are invalid.
    """
    if max_files <= 0:
        raise ValueError("max_files must be > 0")
    if max_total_uncompressed_bytes <= 0:
        raise ValueError("max_total_uncompressed_bytes must be > 0")

    dest_dir = dest_dir.resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    extracted: List[Path] = []
    skipped = 0
    truncated = False

    total_uncompressed = 0

    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        for info in zf.infolist():
            if info.is_dir():
                skipped += 1
                continue

            if _is_encrypted(info) or _is_symlink(info):
                skipped += 1
                continue

            rel = _safe_member_relpath(info.filename)
            if rel is None:
                skipped += 1
                continue

            # Enforce per-component filename length.
            if any(len(part) > max_filename_length for part in rel.parts):
                skipped += 1
                continue

            try:
                size = int(info.file_size)
            except Exception:
                skipped += 1
                continue

            # Guard against negative/invalid sizes.
            if size < 0:
                skipped += 1
                continue

            if len(extracted) >= max_files:
                truncated = True
                break

            if total_uncompressed + size > max_total_uncompressed_bytes:
                truncated = True
                break

            # Resolve the final path and ensure it stays within dest_dir.
            out_path = (dest_dir / Path(*rel.parts)).resolve()
            if not out_path.is_relative_to(dest_dir):
                skipped += 1
                continue

            out_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                with zf.open(info) as src, open(out_path, "wb") as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)
            except OSError:
                skipped += 1
                continue

            extracted.append(out_path)
            total_uncompressed += size

    return ZipExtractionResult(extracted_paths=extracted, skipped_entries=skipped, truncated=truncated)
