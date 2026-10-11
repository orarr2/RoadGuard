"""query_repair_standards: READ-ONLY RAG tool for the Research agent.

Returns a structured JSON payload with tool_result_ids that the Decision
agent must cite from. In class this hits ChromaDB; the offline stub holds
a tiny fixed index so tests are reproducible.
"""

from __future__ import annotations

SCHEMA = {
    "type": "function",
    "function": {
        "name": "query_repair_standards",
        "description": (
            "Retrieve road maintenance procedure passages from the local "
            "corpus. Returns up to k passages plus their article ids. "
            "Every id returned here is the ONLY thing the Decision agent "
            "may cite as a source. Do not invent urls or article numbers."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "hazard_class": {"type": "string", "enum": ["D00", "D10", "D20", "D40"]},
                "road_type": {
                    "type": "string",
                    "enum": ["main", "secondary", "bike_path", "parking"],
                },
                "k": {"type": "integer", "minimum": 1, "maximum": 5, "default": 3},
            },
            "required": ["hazard_class", "road_type"],
        },
    },
}


def _entry(
    passage: str,
    url_slug: str,
    body: str,
    date: str,
    types: list[str],
    ids: list[str],
) -> dict:
    return {
        "passage": passage,
        "source_url": f"https://iroads.example/standards/{url_slug}",
        "issuing_body": body,
        "date": date,
        "hazard_types": types,
        "article_ids": ids,
    }


_STUB_CORPUS = {
    ("D40", "main"): [
        _entry(
            "Potholes exceeding 50mm depth on arterial roads shall be patched within 24 hours.",
            "pavement-2024-sec-4.3",
            "Israel National Roads Company",
            "2024-03-15",
            ["D40"],
            ["sec-4.3"],
        ),
        _entry(
            "Lane obstruction classified as hazard requires immediate signage.",
            "traffic-safety-sec-2.1",
            "Israel Traffic Safety Authority",
            "2023-11-02",
            ["D40", "D20"],
            ["sec-2.1"],
        ),
    ],
    ("D40", "secondary"): [
        _entry(
            "Potholes on secondary roads enter the standard queue when no lane is blocked.",
            "pavement-2024-sec-4.4",
            "Israel National Roads Company",
            "2024-03-15",
            ["D40"],
            ["sec-4.4"],
        ),
    ],
    ("D40", "parking"): [
        _entry(
            "Potholes in parking facilities are routed to the owner's maintenance queue.",
            "municipal-parking-sec-3.1",
            "Municipality Public Works",
            "2024-01-10",
            ["D40"],
            ["parking-sec-3.1"],
        ),
    ],
    ("D40", "bike_path"): [
        _entry(
            "Potholes on bike paths require patching within 7 days to prevent injury.",
            "bike-safety-sec-2.3",
            "Transport Ministry",
            "2024-02-20",
            ["D40"],
            ["bike-sec-2.3"],
        ),
    ],
    ("D20", "main"): [
        _entry(
            "Alligator cracking on arterial roads is scheduled in the standard queue.",
            "pavement-2024-sec-5.1",
            "Israel National Roads Company",
            "2024-03-15",
            ["D20"],
            ["sec-5.1"],
        ),
    ],
    ("D20", "secondary"): [
        _entry(
            "Alligator cracking on secondary roads enters routine queue.",
            "pavement-2024-sec-5.2",
            "Israel National Roads Company",
            "2024-03-15",
            ["D20"],
            ["sec-5.2"],
        ),
    ],
    ("D20", "bike_path"): [
        _entry(
            "Alligator cracking on bike paths requires repair within 14 days.",
            "bike-safety-sec-2.4",
            "Transport Ministry",
            "2024-02-20",
            ["D20"],
            ["bike-sec-2.4"],
        ),
    ],
    ("D10", "secondary"): [
        _entry(
            "Transverse cracks on secondary roads are routine-queue items.",
            "pavement-2024-sec-5.3",
            "Israel National Roads Company",
            "2024-03-15",
            ["D10"],
            ["sec-5.3"],
        ),
    ],
    ("D10", "parking"): [
        _entry(
            "Transverse cracks in parking facilities enter owner maintenance.",
            "municipal-parking-sec-3.2",
            "Municipality Public Works",
            "2024-01-10",
            ["D10"],
            ["parking-sec-3.2"],
        ),
    ],
    ("D00", "secondary"): [
        _entry(
            "Longitudinal cracks on secondary roads are tracked in the pavement log.",
            "pavement-2024-sec-5.4",
            "Israel National Roads Company",
            "2024-03-15",
            ["D00"],
            ["sec-5.4"],
        ),
    ],
    ("D00", "bike_path"): [
        _entry(
            "Longitudinal cracks on bike paths are logged for annual maintenance.",
            "bike-safety-sec-2.5",
            "Transport Ministry",
            "2024-02-20",
            ["D00"],
            ["bike-sec-2.5"],
        ),
    ],
}


def query_repair_standards(
    hazard_class: str,
    road_type: str,
    k: int = 3,
    rag_client=None,
) -> dict:
    """Returns passages + tool_result_ids. Shape is identical for ChromaDB and stub."""
    if rag_client is not None:
        passages = rag_client.query(hazard_class=hazard_class, road_type=road_type, k=k)
    else:
        passages = _STUB_CORPUS.get((hazard_class, road_type), [])[:k]

    coverage = (
        "direct_match"
        if any(hazard_class in p.get("hazard_types", []) for p in passages)
        else ("partial_match" if passages else "no_match")
    )

    tool_result_ids = [aid for p in passages for aid in p.get("article_ids", [])]

    return {
        "procedures": passages,
        "tool_result_ids": tool_result_ids,
        "coverage": coverage,
    }
