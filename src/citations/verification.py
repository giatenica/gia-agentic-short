"""Citation verification utilities.

Implements a simple verification chain:
- Primary: Crossref
- Fallback: OpenAlex

Also provides stale detection via a last_checked timestamp in the CitationRecord.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from src.citations.crossref import normalize_doi, resolve_crossref_doi_to_record
from src.citations.openalex import resolve_openalex_doi_to_record


def _utc_now_iso_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_iso8601(value: str) -> Optional[datetime]:
    if not isinstance(value, str) or not value.strip():
        return None
    v = value.strip()
    try:
        if v.endswith("Z"):
            v = v.replace("Z", "+00:00")
        dt = datetime.fromisoformat(v)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


@dataclass(frozen=True)
class CitationVerificationPolicy:
    max_age_hours: int = 24


def is_verification_stale(*, last_checked: Optional[str], now: Optional[str], policy: CitationVerificationPolicy) -> bool:
    if last_checked is None:
        return True

    last_dt = _parse_iso8601(last_checked)
    if last_dt is None:
        return True

    now_dt = _parse_iso8601(now) if now is not None else datetime.now(timezone.utc)
    if now_dt is None:
        now_dt = datetime.now(timezone.utc)

    max_age = timedelta(hours=int(policy.max_age_hours))
    return (now_dt - last_dt) > max_age


def _error_dict(exc: BaseException) -> Dict[str, str]:
    return {"type": type(exc).__name__, "message": str(exc)}


def resolve_doi_to_record_with_fallback(
    *,
    doi: str,
    citation_key: str,
    created_at: Optional[str] = None,
    existing_record: Optional[Dict[str, Any]] = None,
    policy: Optional[CitationVerificationPolicy] = None,
    now: Optional[str] = None,
) -> Dict[str, Any]:
    """Resolve DOI into a CitationRecord with verification metadata.

    If an existing_record is provided and it is verified and not stale, return
    it without making network calls.
    """

    policy_val = policy or CitationVerificationPolicy()
    now_iso = now or _utc_now_iso_z()
    try:
        normalized = normalize_doi(doi)
    except ValueError as exc:
        return {
            "schema_version": "1.0",
            "citation_key": citation_key,
            "status": "unverified",
            "title": "(missing title)",
            "authors": ["(unknown)"],
            "year": 1900,
            "created_at": created_at or now_iso,
            "identifiers": {"doi": str(doi).strip(), "arxiv": None},
            "verification": {
                "status": "manual",
                "provider_used": None,
                "last_checked": now_iso,
                "attempts": [
                    {
                        "provider": "normalize_doi",
                        "ok": False,
                        "checked_at": now_iso,
                        "error": _error_dict(exc),
                    }
                ],
            },
            "manual_verification_required": True,
            "notes": "Invalid DOI; manual verification required.",
        }

    if isinstance(existing_record, dict):
        existing_status = str(existing_record.get("status") or "").strip()
        existing_doi = None
        identifiers = existing_record.get("identifiers")
        if isinstance(identifiers, dict):
            existing_doi = identifiers.get("doi")
        if isinstance(existing_doi, str) and existing_doi.strip():
            try:
                existing_doi = normalize_doi(existing_doi)
            except ValueError:
                existing_doi = existing_doi.strip()

        verification = existing_record.get("verification")
        last_checked = None
        if isinstance(verification, dict):
            last_checked = verification.get("last_checked")

        if existing_status == "verified" and existing_doi == normalized:
            if not is_verification_stale(last_checked=last_checked, now=now_iso, policy=policy_val):
                return existing_record

    attempts: list[Dict[str, Any]] = []

    try:
        rec = resolve_crossref_doi_to_record(doi=normalized, citation_key=citation_key, created_at=created_at)
        attempts.append({"provider": "crossref", "ok": True, "checked_at": now_iso})
        rec["verification"] = {
            "status": "verified",
            "provider_used": "crossref",
            "last_checked": now_iso,
            "attempts": attempts,
        }
        rec["manual_verification_required"] = False
        return rec
    except Exception as exc:
        attempts.append({"provider": "crossref", "ok": False, "checked_at": now_iso, "error": _error_dict(exc)})

    try:
        rec = resolve_openalex_doi_to_record(
            doi=normalized,
            citation_key=citation_key,
            created_at=created_at,
        )
        attempts.append({"provider": "openalex", "ok": True, "checked_at": now_iso})
        rec["verification"] = {
            "status": "verified",
            "provider_used": "openalex",
            "last_checked": now_iso,
            "attempts": attempts,
        }
        rec["manual_verification_required"] = False
        return rec
    except Exception as exc:
        attempts.append({"provider": "openalex", "ok": False, "checked_at": now_iso, "error": _error_dict(exc)})

    # Fall back to a record that indicates manual verification is needed.
    return {
        "schema_version": "1.0",
        "citation_key": citation_key,
        "status": "unverified",
        "title": "(missing title)",
        "authors": ["(unknown)"],
        "year": 1900,
        "created_at": created_at or now_iso,
        "identifiers": {"doi": normalized, "arxiv": None},
        "verification": {
            "status": "manual",
            "provider_used": None,
            "last_checked": now_iso,
            "attempts": attempts,
        },
        "manual_verification_required": True,
        "notes": "Automatic verification failed for this DOI; manual verification required.",
    }
