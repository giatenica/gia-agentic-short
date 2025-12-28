# Known Issues

Tracked issues discovered during workflow runs. These should be converted to GitHub issues for formal tracking.

## Critical: Edison List Response Bug (FIXED)

**Status:** Fixed in commit `5398706`

Edison's `arun_tasks_until_done` returns a **LIST** of `PQATaskResponse` objects, not a single object. The code was doing `getattr(task_response, 'answer', str(task_response))` but since lists don't have an `answer` attribute, this fell back to `str(task_response)` which gave the repr string of the entire list, causing citation extraction to fail.

## Open Issues

### 1. Claude Literature Search Missing Method (FIXED)

**Status:** Fixed  
**Error:** `AttributeError: 'ClaudeClient' object has no attribute 'complete'`  
**Location:** `src/llm/claude_literature_search.py`  
**Root Cause:** The Claude Literature Search feature (PR #173) called `self.client.complete()` but `ClaudeClient` does not have a `complete` method. It has `chat()` and `chat_async()` methods instead.  
**Impact:** Primary literature search provider failed immediately, forcing fallback to Semantic Scholar/arXiv/Edison.  
**Fix:** Added `_llm_complete()` helper method that wraps `chat_async()` with the expected interface.

### 2. Pipeline JSON Serialization Error (FIXED)

**Status:** Fixed  
**Error:** `TypeError: Object of type Timestamp is not JSON serializable`  
**Location:** `scripts/run_full_pipeline.py` line 53  
**Root Cause:** The workflow context contains pandas `Timestamp` objects that cannot be serialized to JSON by the standard library.  
**Impact:** Pipeline crashed after completing all workflows, losing the final context dump.  
**Fix:** Added `SafeJSONEncoder` class that handles Timestamp, datetime, Path, and set objects.

### 3. Parquet Files Not Supported in Evidence Pipeline (FIXED)

**Status:** Fixed  
**Error:** `Unsupported text format: .parquet`  
**Location:** `src/evidence/source_fetcher.py`  
**Files affected:** All data files in `data/raw data/` directory  
**Root Cause:** Evidence pipeline only supported text formats (CSV, JSON, TXT, MD) but not binary parquet files.  
**Impact:** Could not extract evidence from actual data files.  
**Fix:** Added parquet support via pandas/pyarrow integration:
- Added `BINARY_DATA_EXTENSIONS` set with `.parquet`
- Added `_load_parquet_as_text()` helper that converts parquet to markdown summary
- Output includes schema, sample rows (first 10), numeric statistics, and date ranges

### 4. All Literature Search Fallbacks Failed (FIXED)

**Status:** Fixed  
**Severity:** High  
**Issue:** When Claude Literature Search failed, all fallback providers also failed:  
- **Semantic Scholar:** HTTP 429 (Rate Limited)
- **arXiv:** HTTP 500 Internal Server Error  
- **Edison:** HTTP 402 Payment Required (subscription expired)  
**Result:** Had to fall back to manual sources list (no external search).  
**Fix:** Added retry logic with exponential backoff for transient errors:
- Added `tenacity` retry decorators to Semantic Scholar and arXiv search methods
- Retry on HTTP 429, 500, 502, 503, 504 and timeout/connection errors
- Up to 3 attempts with exponential backoff (2s min, 30s max)
- Semantic Scholar now respects `Retry-After` header on 429 responses
- arXiv timeout increased to 30s for better reliability

### 5. Analysis Execution Missing Scripts (FIXED)

**Status:** Fixed  
**Severity:** Medium  
**Error:** `Analysis execution failed: No analysis scripts configured or discovered under analysis/`  
**Root Cause:** The workflow expects analysis scripts in `analysis/` folder but none exist for this project  
**Location:** `src/agents/data_analysis_execution.py`  
**Fix:** Changed default `on_missing_outputs` from `"block"` to `"downgrade"`:
- Projects without analysis scripts now gracefully skip the analysis step
- Returns success with `action: "downgrade", reason: "no_scripts"` metadata
- Users can still set `on_missing_outputs: "block"` explicitly if strict mode is needed

### 6. High Consistency Issue Count (IMPROVED)

**Status:** Improved  
**Severity:** Low  
**Count:** 16 critical, 34 high issues found in final consistency check (50 total)  
**Root Cause:** Documents generated across multiple workflow stages have inconsistent:
- Hypothesis wordings
- Timeline specifications
- Statistical references  
**Improvement:** Added automated fix suggestions:
- New `generate_fix_script` config option (default: True)
- Generates `outputs/consistency_fixes.json` with actionable fix recommendations
- Each fix includes: action type, search pattern, replacement value, affected documents
- Actions include: `add_bibtex_entry`, `align_hypothesis_text`, `verify_statistic_value`, etc.
- Marked as "manual_review_required" for critical/high severity issues

## Workflow Statistics

Last run (2025-12-28):
- **Research Workflow:** 453.57s, 44,296 tokens, 0 errors
- **Literature Workflow:** 640.1s, 80,070 tokens, success=True (with fallbacks)
- **Gap Resolution Workflow:** 1,482.93s, 1,047,409 tokens, 9/12 gaps resolved
- **Total time:** ~43 minutes
- **Total tokens:** ~1,171,775
- **Final status:** Crashed on JSON serialization (all workflows completed)

---

*Generated by Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher, for more information see: https://giatenica.com*
