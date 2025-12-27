# Pipeline Troubleshooting Guide

This guide helps diagnose and fix common issues in the GIA Agentic Research Pipeline.

## Table of Contents

1. [Data Flow Verification](#1-data-flow-verification)
2. [Common Failure Patterns](#2-common-failure-patterns)
3. [Debug Commands](#3-debug-commands)
4. [Phase-Specific Issues](#4-phase-specific-issues)
5. [External Dependencies](#5-external-dependencies)

---

## 1. Data Flow Verification

Use this checklist to verify each pipeline phase produced expected outputs:

### Phase 1: Intake

```
Phase 1 (Intake)
├── project.json exists and valid
├── RESEARCH_OVERVIEW.md generated
├── DATA_ANALYSIS.md generated (if data provided)
└── workflow_context initialized
```

**Verification commands:**
```bash
# Check project configuration
cat project.json | jq .

# Verify intake outputs
ls -la RESEARCH_OVERVIEW.md DATA_ANALYSIS.md
```

### Phase 2: Literature

```
Phase 2 (Literature)
├── LITERATURE_REVIEW.md generated
├── PROJECT_PLAN.md created
├── citations_data.json populated
├── bibliography/citations.json exists
├── references.bib created
└── sources/ directory has subdirectories (if evidence pipeline enabled)
```

**Verification commands:**
```bash
# Check citation data
cat citations_data.json | jq length
cat bibliography/citations.json | jq length

# Verify literature outputs
ls -la LITERATURE_REVIEW.md PROJECT_PLAN.md references.bib

# Check sources directory
ls -d sources/*/
```

### Phase 3: Gap Resolution (Optional)

```
Phase 3 (Gap Resolution)
├── claims/claims.json populated
├── evidence items linked to claims
└── RESEARCH_OVERVIEW.md updated
```

**Verification commands:**
```bash
# Check claims
cat claims/claims.json | jq length

# Verify evidence linkage
grep -r "claim_id" sources/*/evidence.json
```

### Phase 4: Writing

```
Phase 4 (Writing)
├── outputs/sections/*.tex files exist
├── temp/main.tex assembled
├── outputs/draft.pdf generated (if LaTeX compiled)
└── No placeholder text in outputs
```

**Verification commands:**
```bash
# Check section files
ls -la outputs/sections/*.tex

# Check for placeholder text
grep -r "Evidence is not yet available" outputs/
grep -r "No quote evidence items" outputs/
grep -r "Results are pending" outputs/

# Verify main document
ls -la temp/main.tex outputs/draft.pdf
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

**Author:** Gia Tenica*

*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher, for more information see: https://giatenica.com
