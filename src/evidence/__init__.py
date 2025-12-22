""" 
Evidence Ledger
===============
Append-only evidence storage for extracting and tracking citations, quotes, tables, and claims.

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

from .store import EvidenceStore
from .source_fetcher import SourceFetcherTool, LocalSource, discover_local_sources
from .parser import MVPLineBlockParser, DocumentParser, ParsedDocument, TextBlock, TextSpan, parse_to_blocks

__all__ = [
	"EvidenceStore",
	"SourceFetcherTool",
	"LocalSource",
	"discover_local_sources",
	"MVPLineBlockParser",
	"DocumentParser",
	"ParsedDocument",
	"TextBlock",
	"TextSpan",
	"parse_to_blocks",
]
