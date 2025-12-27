# Troubleshooting Guide

This guide helps diagnose and resolve common issues with the GIA research pipeline.

## Table of Contents

1. [Data Flow Verification](#data-flow-verification)
2. [Common Failure Patterns](#common-failure-patterns)
3. [Debug Commands](#debug-commands)
4. [Quality Gates](#quality-gates)
5. [Evaluation Metrics](#evaluation-metrics)

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

## Common Failure Patterns

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| Empty `citations.json` | Citation population skipped | Verify Edison API key is set; check `citations_data.json` exists |
| No `sources/` directory | Source acquisition not run | Verify literature workflow completed; check logs for errors |
| Placeholder text in sections | Evidence not available | Check `sources/*/evidence.json` files exist and have content |
| All gates pass but output is poor | Gates in warn mode | Review `degradation_summary.json` for warnings |
| Missing `metrics.json` | Analysis scripts failed | Check `analysis/` folder has executable scripts; review logs |
| Low evaluation score | Output quality issues | Review `evaluation_results.json` for specific metric failures |
| Pipeline halts early | Phase failure | Check `workflow_context.json` for error details |

### Empty Citations

**Symptoms:** `bibliography/citations.json` is empty or missing.

**Diagnosis:**
```bash
# Check raw Edison results
cat citations_data.json | jq length

# If empty, Edison search failed
# If has data, citation population failed
```

**Solutions:**
1. Verify `EDISON_API_KEY` environment variable is set
2. Check `citations_data.json` for error messages
3. Verify research question in `project.json` is searchable

### Missing Evidence

**Symptoms:** Section writers produce placeholder text.

**Diagnosis:**
```bash
# Count sources with evidence
for d in sources/*/; do
  if [ -f "${d}evidence.json" ]; then
    count=$(cat "${d}evidence.json" | jq length)
    echo "${d}: ${count} items"
  else
    echo "${d}: NO EVIDENCE FILE"
  fi
done
```

**Solutions:**
1. Verify sources were downloaded (check `sources/*/raw/`)
2. Run evidence extraction manually if needed
3. Check source documents are parseable (not scanned images)

### Quality Gate Failures

**Symptoms:** Pipeline produces warnings or blocks on quality gates.

**Diagnosis:**
```bash
# Check degradation summary
cat outputs/degradation_summary.json | jq '.degradations[] | {stage, reason_code, severity}'
```

**Solutions:**
1. Review specific gate that triggered
2. Add missing evidence or citations
3. Switch gate to warn mode if blocking is not desired

---

## Debug Commands

### Pipeline Status

```bash
# Full pipeline status
python -c "
from src.pipeline.context import WorkflowContext
ctx = WorkflowContext.read_json('workflow_context.json')
if ctx:
    print(f'Success: {ctx.success}')
    print(f'Phases: {list(ctx.phase_results.keys())}')
    print(f'Errors: {ctx.errors}')
"
```

### Citation Flow

```bash
# Count citations at each stage
echo "Edison results: $(cat citations_data.json 2>/dev/null | jq length || echo 0)"
echo "Bibliography: $(cat bibliography/citations.json 2>/dev/null | jq length || echo 0)"
echo "Verified: $(cat bibliography/citations.json 2>/dev/null | jq '[.[] | select(.status=="verified")] | length' || echo 0)"
```

### Evidence Coverage

```bash
# Check evidence distribution
for d in sources/*/; do
  name=$(basename "$d")
  count=$(cat "${d}evidence.json" 2>/dev/null | jq length || echo "0")
  echo "$name: $count items"
done | sort -t: -k2 -n -r
```

### Section Quality

```bash
# Check for common issues
echo "=== Placeholder Text ==="
grep -l 'Evidence is not yet available' outputs/sections/*.tex 2>/dev/null

echo "=== Empty Sections ==="
for f in outputs/sections/*.tex; do
  size=$(wc -c < "$f")
  if [ "$size" -lt 500 ]; then
    echo "$f: $size bytes (possibly empty)"
  fi
done

echo "=== Missing Citations ==="
grep -l '\[CITATION NEEDED\]' outputs/sections/*.tex 2>/dev/null
```

### Analysis Output

```bash
# Check analysis artifacts
echo "Metrics: $(cat outputs/metrics.json 2>/dev/null | jq length || echo 0)"
echo "Claims: $(cat claims/claims.json 2>/dev/null | jq length || echo 0)"
echo "Artifacts: $(cat outputs/artifacts.json 2>/dev/null | jq '.outputs | length' || echo 0)"
```

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

## Getting Help

If you encounter issues not covered here:

1. Check the logs in your terminal for error messages
2. Review `outputs/degradation_summary.json` for pipeline warnings
3. Check `workflow_context.json` for phase completion status
4. Open an issue on the repository with:
   - Error message or symptom
   - Relevant file contents (sanitized)
   - Steps to reproduce
