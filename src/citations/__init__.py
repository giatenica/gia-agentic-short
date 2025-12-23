"""Citations package.

Filesystem-first citation registry and validation helpers.
"""

from .crossref import (

    CrossrefClient,
    CrossrefClientConfig,
    CrossrefError,
    CrossrefNotFoundError,
    normalize_doi,
    resolve_crossref_doi_and_upsert,
    resolve_crossref_doi_to_record,
)

__all__ = [

    "CrossrefClient",
    "CrossrefClientConfig",
    "CrossrefError",
    "CrossrefNotFoundError",
    "normalize_doi",
    "resolve_crossref_doi_and_upsert",
    "resolve_crossref_doi_to_record",
]
