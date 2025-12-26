import pytest

from src.evidence.extraction import extract_evidence_items
from src.utils.schema_validation import validate_evidence_item


@pytest.mark.unit
def test_extract_evidence_items_schema_and_kind_selection():
    parsed = {
        "blocks": [
            {
                "kind": "paragraph",
                "span": {"start_line": 1, "end_line": 2},
                "text": '"Quoted statement" with a 42% increase.',
            },
            {
                "kind": "heading",
                "span": {"start_line": 3, "end_line": 3},
                "text": "# Heading",
            },
        ]
    }

    items = extract_evidence_items(
        parsed=parsed,
        source_id="src1",
        created_at="2025-01-01T00:00:00+00:00",
        max_items=10,
    )

    assert len(items) == 1
    assert items[0]["kind"] == "quote"
    assert items[0]["locator"]["span"]["start_line"] == 1

    for item in items:
        validate_evidence_item(item)


@pytest.mark.unit
def test_extract_evidence_items_deterministic_ids_with_fixed_timestamp():
    parsed = {
        "blocks": [
            {
                "kind": "paragraph",
                "span": {"start_line": 10, "end_line": 12},
                "text": "This block contains a metric: 3.14%.",
            }
        ]
    }

    items_a = extract_evidence_items(
        parsed=parsed,
        source_id="srcX",
        created_at="2025-01-01T00:00:00+00:00",
        max_items=5,
    )
    items_b = extract_evidence_items(
        parsed=parsed,
        source_id="srcX",
        created_at="2025-01-01T00:00:00+00:00",
        max_items=5,
    )

    assert items_a == items_b
    assert items_a[0]["evidence_id"] == items_b[0]["evidence_id"]


@pytest.mark.unit
def test_extract_evidence_items_table_and_figure_caption_detection():
    parsed = {
        "blocks": [
            {
                "kind": "paragraph",
                "span": {"start_line": 1, "end_line": 1},
                "text": "Table 1: Summary statistics for the sample.",
            },
            {
                "kind": "paragraph",
                "span": {"start_line": 2, "end_line": 2},
                "text": "Figure 2. Time-series evolution of the spread.",
            },
            {
                "kind": "paragraph",
                "span": {"start_line": 3, "end_line": 3},
                "text": "As shown in Table 3 we report results.",
            },
            {
                "kind": "paragraph",
                "span": {"start_line": 4, "end_line": 4},
                "text": "TABLE IV: Robustness checks.",
            },
            {
                "kind": "paragraph",
                "span": {"start_line": 5, "end_line": 5},
                "text": "Fig. 5 — Event window timeline.",
            },
            {
                "kind": "paragraph",
                "span": {"start_line": 6, "end_line": 6},
                "text": "Table 7- \"Quoted caption\" with a 10% change.",
            },
            {
                "kind": "paragraph",
                "span": {"start_line": 7, "end_line": 7},
                "text": "Table 1 Summary statistics without punctuation.",
            },
        ]
    }

    items = extract_evidence_items(
        parsed=parsed,
        source_id="src1",
        created_at="2025-01-01T00:00:00+00:00",
        max_items=10,
        min_excerpt_chars=5,
    )

    kinds = [i["kind"] for i in items]
    assert "table" in kinds
    assert "figure" in kinds

    # Non-detection case: mention of Table 3 mid-sentence should not be treated as a caption.
    mention_items = [i for i in items if i["excerpt"].startswith("As shown in Table 3")]
    assert len(mention_items) == 1
    assert mention_items[0]["kind"] not in {"table", "figure"}

    # Edge cases: roman numerals, Fig. abbreviation, dash delimiter, case variations.
    assert any(i["excerpt"].startswith("TABLE IV") and i["kind"] == "table" for i in items)
    assert any(i["excerpt"].startswith("Fig. 5") and i["kind"] == "figure" for i in items)

    # Caption precedence: table/figure must win over quote/metric classification.
    precedence_items = [i for i in items if i["excerpt"].startswith("Table 7-")]
    assert len(precedence_items) == 1
    assert precedence_items[0]["kind"] == "table"

    # Regex should not be overly permissive; without punctuation after the number, this should not count as a caption.
    no_punct_items = [i for i in items if i["excerpt"].startswith("Table 1 Summary")]
    assert len(no_punct_items) == 1
    assert no_punct_items[0]["kind"] != "table"

    for item in items:
        validate_evidence_item(item)


@pytest.mark.unit
def test_extract_evidence_items_propagates_page_span_into_locator():
    parsed = {
        "blocks": [
            {
                "kind": "paragraph",
                "span": {"start_page": 2, "end_page": 2, "start_line": 10, "end_line": 10},
                "text": "A sufficiently long statement that should become evidence.",
            }
        ]
    }

    items = extract_evidence_items(
        parsed=parsed,
        source_id="src_page",
        created_at="2025-01-01T00:00:00+00:00",
        max_items=5,
        min_excerpt_chars=5,
    )

    assert len(items) == 1
    span = items[0]["locator"].get("span")
    assert isinstance(span, dict)
    assert span.get("start_page") == 2
    assert span.get("end_page") == 2
    assert span.get("start_line") == 10
    assert span.get("end_line") == 10

    for item in items:
        validate_evidence_item(item)
