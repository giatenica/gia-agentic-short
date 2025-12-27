# Pipeline Troubleshooting Guide

This guide helps diagnose and fix common issues in the GIA Agentic Research Pipeline.

## Table of Contents

1. [Data Flow Verification](#1-data-flow-verification)
2. [Common Failure Patterns](#2-common-failure-patterns)
3. [Debug Commands](#3-debug-commands)
4. [Phase-Specific Issues](#4-phase-specific-issues)
5. [Quality Gates](#5-quality-gates)
6. [Evaluation Metrics](#6-evaluation-metrics)
7. [External Dependencies](#7-external-dependencies)
8. [Quick Diagnostic Script](#quick-diagnostic-script)

---

## Data Flow Verification

Use this checklist to verify each pipeline phase produced expected outputs.

### Phase 1: Intake

```
project_folder/
├── project.json        # Must exist and be valid JSON
└── intake_form.json    # Optional, created if web form used
```

**Verification:**
```bash
# Check project.json is valid
cat project.json | python -m json.tool > /dev/null && echo "Valid JSON"

# Required fields
cat project.json | jq '.research_question, .hypothesis'
```

### Phase 2: Literature

```
project_folder/
├── citations_data.json              # Edison search results
├── bibliography/
│   └── citations.json               # Processed citations (CitationRecord schema)
└── sources/
    └── <source_id>/
        ├── raw/                     # Downloaded PDFs
        ├── parsed.json              # Parsed content
        └── evidence.json            # Extracted evidence items
```

**Verification:**
```bash
# Check citation count
cat bibliography/citations.json | jq length

# Check sources exist
ls -la sources/

# Count evidence files
find sources -name 'evidence.json' | wc -l

# Check total evidence items
find sources -name 'evidence.json' -exec cat {} \; | jq -s 'map(length) | add'
```

### Phase 3: Gap Resolution

```
project_folder/
├── claims/
│   └── claims.json                  # Generated claims
└── outputs/
    └── metrics.json                 # Analysis metrics (if analysis ran)
```

**Verification:**
```bash
# Check claims count
cat claims/claims.json | jq length

# Check metrics
cat outputs/metrics.json | jq '.[] | .metric_key'
```

### Phase 4: Writing

```
project_folder/
└── outputs/
    ├── sections/
    │   ├── introduction.tex
    │   ├── related_work.tex
    │   ├── methods.tex
    │   ├── results.tex
    │   └── discussion.tex
    ├── degradation_summary.json     # Pipeline degradations
    └── evaluation_results.json      # Quality metrics
```

**Verification:**
```bash
# Check all sections exist
ls -la outputs/sections/*.tex

# Check for placeholder text
grep -l 'Evidence is not yet available' outputs/sections/*.tex

# Check degradation count
cat outputs/degradation_summary.json | jq '.degradations | length'
```

---


## 2. Common Failure Patterns

| Symptom | Cause | Fix | Reference |
|---------|-------|-----|-----------|
| Empty citations.json | Citation population not called | Enable citation flow in literature workflow | [#145](https://github.com/giatenica/gia-agentic-short/issues/145) |
| No sources/ directory | Source acquisition not run | Run source acquisition after literature search | [#146](https://github.com/giatenica/gia-agentic-short/issues/146), [#150](https://github.com/giatenica/gia-agentic-short/issues/150) |
| Placeholder text in sections | Evidence not available | Verify sources/*/evidence.json exist and are populated | |
| "Evidence is not yet available" | Evidence pipeline not enabled | Set `evidence_pipeline: true` in workflow_context | |
| Gates all pass but output poor | Gates disabled or thresholds too low | Check gate configuration and enable strict mode | [#149](https://github.com/giatenica/gia-agentic-short/issues/149) |
| Analysis metrics missing | No analysis scripts executed | Run analysis execution (A24) before writing | [#148](https://github.com/giatenica/gia-agentic-short/issues/148) |
| "Results are pending metric computation" | outputs/metrics.json missing | Execute data analysis and generate metrics | |
| No PDF output | LaTeX compilation failed | Check temp/compile.log for errors | |
| Missing bibliography entries | references.bib not generated | Verify citations_data.json populated | |
| Section writers downgrade | Required artifacts missing | Follow data flow verification checklist | |

---


## 3. Debug Commands

### Citation Flow

Check if citations are being captured:

```bash
# Count citations in primary registry
cat bibliography/citations.json | jq length

# Count citations in data file
cat citations_data.json | jq length

# List all citation keys
cat bibliography/citations.json | jq -r '.[].key'

# Verify citation in literature review
grep -c "\\cite{" LITERATURE_REVIEW.md
```

### Evidence Availability

Check if evidence extraction is working:

```bash
# Count evidence files
find sources -name 'evidence.json' | wc -l

# List all evidence files
find sources -name 'evidence.json'

# Check evidence items per source
for f in sources/*/evidence.json; do
  echo "$f: $(jq length $f) items"
done

# Verify evidence types
jq -r '.[].type' sources/*/evidence.json | sort | uniq -c
```

### Section Quality

Check for degraded output:

```bash
# Find all placeholder patterns
grep -r "Evidence is not yet available" outputs/
grep -r "No quote evidence items" outputs/
grep -r "Results are pending" outputs/
grep -r "statements are non-definitive" outputs/

# Count citations in sections
grep -c "\\cite{" outputs/sections/*.tex

# Check section file sizes
ls -lh outputs/sections/*.tex
```

### Metrics and Analysis

Check if computational analysis ran:

```bash
# Check metrics file
cat outputs/metrics.json | jq length

# List metric types
jq -r '.[].metric_type' outputs/metrics.json | sort | uniq -c

# Check artifacts
ls -la outputs/artifacts.json

# Verify claims referencing metrics
jq -r '.[] | select(.metric_keys | length > 0) | .claim_id' claims/claims.json
```

### Workflow State

Check overall pipeline state:

```bash
# Check workflow context (if available)
cat full_pipeline_context.json | jq '.success, .errors'

# List all generated artifacts
find . -name '*.json' -o -name '*.md' -o -name '*.tex' | sort

# Check file timestamps to see pipeline progression
ls -lt *.md outputs/sections/*.tex | head -20
```

---


## 4. Phase-Specific Issues

### Phase 1: Intake Issues

**Problem:** Data analysis fails or produces incomplete results

**Diagnosis:**
```bash
# Check project.json structure
cat project.json | jq .

# Verify data files are accessible
ls -la data/
```

**Solutions:**
- Ensure project.json has required fields: `project_title`, `research_question`
- Verify data files are in supported formats (CSV, JSON, Excel)
- Check data file permissions

**Problem:** RESEARCH_OVERVIEW.md is too generic

**Solutions:**
- Provide more detailed project.json with background and goals
- Add notes.md with additional context
- Include preliminary_findings in project.json

### Phase 2: Literature Issues

**Problem:** Literature search returns no results

**Diagnosis:**
```bash
# Check Edison API configuration
echo $EDISON_API_KEY

# Check citations_data.json
cat citations_data.json
```

**Solutions:**
- Verify EDISON_API_KEY environment variable is set
- Check network connectivity
- Review search terms in PROJECT_PLAN.md
- Enable graceful degradation if API unavailable

**Problem:** Sources not acquired after literature search

**Solutions:**
- Run source acquisition explicitly: `python scripts/run_source_acquisition.py <project>`
- Check PDF download configuration
- Verify DOI/URL availability in citations_data.json

### Phase 3: Gap Resolution Issues

**Problem:** Claims not linking to evidence

**Diagnosis:**
```bash
# Check claims structure
cat claims/claims.json | jq '.[0]'

# Verify evidence_keys in claims
jq -r '.[].evidence_keys[]' claims/claims.json
```

**Solutions:**
- Ensure evidence pipeline ran (Phase 2)
- Verify evidence.json files exist in sources/
- Check that evidence items have unique IDs

### Phase 4: Writing Issues

**Problem:** All sections have placeholder text

**Root cause:** Missing prerequisite artifacts

**Solutions:**
1. Verify evidence availability (see Debug Commands)
2. Check citation registry population
3. Ensure metrics.json exists for Results/Methods sections
4. Enable evidence_pipeline in workflow_context

**Problem:** LaTeX compilation fails

**Diagnosis:**
```bash
# Check LaTeX logs
cat temp/compile.log

# Verify main.tex structure
head -50 temp/main.tex
```

**Solutions:**
- Install required LaTeX packages
- Check for invalid LaTeX commands in sections
- Verify bibliography file exists
- Review compile.log for specific errors

---


## Quality Gates

The pipeline has 7 quality gates that can operate in `block` or `downgrade` mode.

| Gate | Purpose | Key Files |
|------|---------|-----------|
| `evidence_gate` | Verify evidence exists for claims | `sources/*/evidence.json` |
| `citation_gate` | Verify citations are in bibliography | `bibliography/citations.json` |
| `computation_gate` | Verify computed metrics are valid | `outputs/metrics.json`, `claims/claims.json` |
| `claim_evidence_gate` | Verify claims have supporting evidence | `claims/claims.json` |
| `literature_gate` | Verify literature coverage | `bibliography/citations.json` |
| `analysis_gate` | Verify analysis was executed | `outputs/metrics.json`, `outputs/artifacts.json` |
| `citation_accuracy_gate` | Verify citation formatting | `outputs/sections/*.tex` |

### Checking Gate Status

```bash
# Check which gates triggered
cat outputs/degradation_summary.json | jq '.degradations[] | select(.stage | test("gate|citation|evidence|computation|analysis"))'
```

### Configuring Gates

Gates are configured via `workflow_overrides` in the pipeline call:

```python
from src.pipeline.runner import run_full_pipeline

result = await run_full_pipeline(
    project_folder,
    workflow_overrides={
        "citation_gate": {"enabled": True, "on_missing": "block"},  # Strict
        "evidence_gate": {"enabled": True, "on_missing": "downgrade"},  # Warn only
    }
)
```

---


## Evaluation Metrics

The pipeline runs post-completion evaluation that writes to `outputs/evaluation_results.json`.

### Available Metrics

| Metric | What It Checks | Ideal Score |
|--------|---------------|-------------|
| `completeness` | All 5 section .tex files exist | 1.0 (5/5) |
| `evidence_coverage` | Sources have evidence.json | 1.0 (100%) |
| `citation_coverage` | Citations are verified | 1.0 (100%) |
| `claims_coverage` | Claims generated from metrics | 1.0 |

### Checking Evaluation Results

```bash
# Overall score
cat outputs/evaluation_results.json | jq '.overall_score'

# Per-metric breakdown
cat outputs/evaluation_results.json | jq '.metrics[] | {name, score: .score, max: .max_score}'

# Specific issues
cat outputs/evaluation_results.json | jq '.metrics[] | select(.normalized_score < 1.0) | {name, details}'
```

### Configuring Evaluation

```python
result = await run_full_pipeline(
    project_folder,
    workflow_overrides={
        "evaluation": {
            "enabled": True,
            "min_quality_score": 0.7,  # Require 70% quality
            "metrics": ["completeness", "evidence_coverage"],  # Only run these
        }
    }
)
```

---


## 5. External Dependencies

### Required API Keys

| Service | Environment Variable | Used For | Required? |
|---------|---------------------|----------|-----------|
| Anthropic | `ANTHROPIC_API_KEY` | All agent operations | Yes |
| Edison | `EDISON_API_KEY` | Literature search | No (degraded mode) |
| Crossref | N/A | Citation verification | No |

**Check API key configuration:**
```bash
# Verify keys are set (without exposing values)
env | grep -E "(ANTHROPIC|EDISON)_API_KEY" | sed 's/=.*/=***/'
```

### LaTeX Installation

**Required for PDF generation:**
- pdflatex
- bibtex
- Standard packages: amsmath, graphicx, hyperref, natbib

**Verify installation:**
```bash
pdflatex --version
bibtex --version
```

### Python Dependencies

**Check installed packages:**
```bash
pip list | grep -E "(anthropic|pydantic|jsonschema)"
```

**Install missing dependencies:**
```bash
pip install -r requirements.txt
```

---


## Quick Diagnostic Script

Save this as `diagnose_pipeline.sh` for quick health checks:

```bash
#!/bin/bash
# Pipeline health check script

PROJECT=$1
if [ -z "$PROJECT" ]; then
  echo "Usage: $0 <project_folder>"
  exit 1
fi

cd "$PROJECT" || exit 1

echo "=== Pipeline Health Check ==="
echo

echo "Phase 1 (Intake):"
[ -f "project.json" ] && echo "✓ project.json" || echo "✗ project.json"
[ -f "RESEARCH_OVERVIEW.md" ] && echo "✓ RESEARCH_OVERVIEW.md" || echo "✗ RESEARCH_OVERVIEW.md"
echo

echo "Phase 2 (Literature):"
[ -f "citations_data.json" ] && echo "✓ citations_data.json ($(jq length citations_data.json 2>/dev/null || echo 0) entries)" || echo "✗ citations_data.json"
[ -f "bibliography/citations.json" ] && echo "✓ bibliography/citations.json ($(jq length bibliography/citations.json 2>/dev/null || echo 0) entries)" || echo "✗ bibliography/citations.json"
[ -d "sources" ] && echo "✓ sources/ ($(find sources -name 'evidence.json' 2>/dev/null | wc -l) evidence files)" || echo "✗ sources/"
echo

echo "Phase 3 (Gap Resolution):"
[ -f "claims/claims.json" ] && echo "✓ claims/claims.json ($(jq length claims/claims.json 2>/dev/null || echo 0) claims)" || echo "✗ claims/claims.json"
echo

echo "Phase 4 (Writing):"
[ -d "outputs/sections" ] && echo "✓ outputs/sections/ ($(ls outputs/sections/*.tex 2>/dev/null | wc -l) files)" || echo "✗ outputs/sections/"
[ -f "temp/main.tex" ] && echo "✓ temp/main.tex" || echo "✗ temp/main.tex"

echo
echo "Checking for placeholder text..."
if grep -rq "Evidence is not yet available" outputs/ 2>/dev/null; then
  echo "⚠ Placeholder text found in outputs/"
else
  echo "✓ No placeholder text"
fi

echo
echo "=== End Health Check ==="
```

---


## Getting Help

If issues persist after following this guide:

1. Check the GitHub issues for similar problems
2. Review logs in `.evidence/` and `temp/` directories
3. Run the pipeline with increased verbosity/tracing
4. Examine `full_pipeline_context.json` for error details
5. Open a new issue with:
   - Pipeline phase where failure occurred
   - Error messages or symptoms
   - Output of diagnostic script above
   - Relevant log snippets

---


---

**Author:** Gia Tenica*

*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher, for more information see: https://giatenica.com
