"""The five canonical tools (per Charter and DS23 Agent Contract).

Each tool declares a JSON Schema that Groq function-calling consumes
verbatim, and a plain Python callable that executes offline. Tools are
grouped by which agent is allowed to use them (least privilege):

  Perception: lookup_similar_cases
  Research:   query_repair_standards (read-only)
  Decision:   check_hazard_severity, request_additional_evidence,
              create_draft_work_order, flag_for_urgent_review
"""

from .check_hazard_severity import SCHEMA as SEVERITY_SCHEMA
from .check_hazard_severity import check_hazard_severity
from .create_draft_work_order import SCHEMA as DRAFT_SCHEMA
from .create_draft_work_order import create_draft_work_order
from .flag_for_urgent_review import SCHEMA as FLAG_SCHEMA
from .flag_for_urgent_review import flag_for_urgent_review
from .lookup_similar_cases import SCHEMA as SIMILAR_SCHEMA
from .lookup_similar_cases import lookup_similar_cases
from .query_repair_standards import SCHEMA as RAG_SCHEMA
from .query_repair_standards import query_repair_standards
from .request_additional_evidence import SCHEMA as EVIDENCE_SCHEMA
from .request_additional_evidence import request_additional_evidence

PERCEPTION_TOOLS = {
    "lookup_similar_cases": (lookup_similar_cases, SIMILAR_SCHEMA),
}

RESEARCH_TOOLS = {
    "query_repair_standards": (query_repair_standards, RAG_SCHEMA),
}

DECISION_TOOLS = {
    "check_hazard_severity": (check_hazard_severity, SEVERITY_SCHEMA),
    "request_additional_evidence": (request_additional_evidence, EVIDENCE_SCHEMA),
    "create_draft_work_order": (create_draft_work_order, DRAFT_SCHEMA),
    "flag_for_urgent_review": (flag_for_urgent_review, FLAG_SCHEMA),
}

__all__ = [
    "DECISION_TOOLS",
    "PERCEPTION_TOOLS",
    "RESEARCH_TOOLS",
    "check_hazard_severity",
    "create_draft_work_order",
    "flag_for_urgent_review",
    "lookup_similar_cases",
    "query_repair_standards",
    "request_additional_evidence",
]
