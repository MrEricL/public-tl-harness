"""Versioned, focused source/draft judgments. IDs are local routing keys."""

from __future__ import annotations

from typing import Any

from .models import QUESTION_SCHEMA_VERSION, DecisionRequest, digest


QUESTIONS: dict[str, dict[str, Any]] = {
    "actor_relation_error": {"type": "noul", "instructions": "Does `draft` materially change who acts, receives an action, or relates to whom in `source`?"},
    "negation_condition_error": {"type": "noul", "instructions": "Does `draft` reverse or lose a material negation, condition, permission, necessity, or causal dependency from `source`?"},
    "quantity_time_error": {"type": "noul", "instructions": "Does `draft` change a material number, duration, time, sequence, or comparison in `source`?"},
    "omitted_material": {"type": "noul", "instructions": "Does `draft` omit a meaningful event, description, qualification, or relationship stated in `source`?"},
    "unsupported_addition": {"type": "noul", "instructions": "Does `draft` assert a material claim that is unsupported by `source` and `context`?"},
    "terminology_conflict": {"type": "noul", "instructions": "Does `draft` misuse a relevant term or create a real story inconsistency given `source` and `context`, beyond a harmless synonym?"},
    "source_sufficiency": {"type": "choice", "instructions": "How sufficient is `source` to assess the meaning expressed by `draft`?", "criteria": {"sufficient": "The source supplies the relevant meaning clearly.", "ambiguous": "The source is present but permits materially different readings.", "incomplete": "The source is physically missing or cut off at the point needed to judge."}},
    "readability_problem": {"type": "score", "instructions": "How difficult is `draft` to read as natural English fiction? Judge readability, not fidelity.", "criteria": ["Natural and clear English prose", "Somewhat awkward but readily understandable", "Seriously difficult to follow", "Incoherent or unreadable"]},
}

PATCH_QUESTION: dict[str, Any] = {"type": "noul", "instructions": "Compared with `draft`, does `candidate` introduce a material claim unsupported by `source` and `context`?"}


def question_schema(request: DecisionRequest) -> dict[str, dict[str, Any]]:
    questions = dict(QUESTIONS)
    if request.candidate_text is not None:
        questions["patch_unsupported_claim"] = PATCH_QUESTION
    return questions


def question_schema_digest(request: DecisionRequest) -> str:
    return digest({"version": QUESTION_SCHEMA_VERSION, "questions": question_schema(request)})


def request_payload(request: DecisionRequest) -> dict[str, Any]:
    state = {"source": request.source_text, "draft": request.draft_text, "context": request.context_text}
    if request.candidate_text is not None:
        state["candidate"] = request.candidate_text
    return {"state": state, "model": request.requested_model, "questions": question_schema(request)}


def request_id(request: DecisionRequest) -> str:
    return digest({"segment_id": request.segment_id, "payload": request_payload(request), "question_schema_version": QUESTION_SCHEMA_VERSION, "routing_policy_version": request.routing_policy_version, "routing_thresholds": request.routing_thresholds})
