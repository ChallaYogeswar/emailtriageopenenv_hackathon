"""
Deterministic graders for all 3 tasks.
Each grader returns a score strictly in (0.0, 1.0) — never exactly 0 or 1.
Graders are called by the environment and also by the /tasks endpoint.
"""

from __future__ import annotations

from typing import Dict, List, Any
from tasks.email_data import GROUND_TRUTH


def _clamp(score: float) -> float:
    """Clamp score to strictly open interval (0.0, 1.0)."""
    return round(max(0.01, min(score, 0.99)), 4)


def _classification_score(email_id: str, priority: str, category: str) -> float:
    """Score a single classification action."""
    gt = GROUND_TRUTH.get(email_id, {})
    score = 0.0
    if priority and gt.get("priority") == priority:
        score += 0.5
    if category and gt.get("category") == category:
        score += 0.5
    return score


def _reply_quality_score(email_id: str, tone: str, body: str) -> float:
    """Score reply quality: tone correctness + body non-empty."""
    gt = GROUND_TRUTH.get(email_id, {})
    expected_actions = gt.get("gt_actions", [])
    expected_reply = next((a for a in expected_actions if a.startswith("reply:")), None)
    score = 0.0
    if expected_reply:
        expected_tone = expected_reply.split(":")[1]
        if tone == expected_tone:
            score += 0.6
        elif tone in ("apologetic", "formal") and expected_tone in ("apologetic", "formal"):
            score += 0.3  # partial credit for close tones
    if body and len(body.strip()) > 20:
        score += 0.4  # non-trivial reply body
    return min(score, 1.0)


def _escalation_score(email_id: str, escalate_to: str) -> float:
    """Score escalation target correctness."""
    gt = GROUND_TRUTH.get(email_id, {})
    expected_actions = gt.get("gt_actions", [])
    expected_escalations = [a.split(":")[1] for a in expected_actions if a.startswith("escalate:")]
    if not expected_escalations:
        return -0.3  # penalize unnecessary escalation
    if escalate_to in expected_escalations:
        return 1.0
    return 0.0


def _spam_detection_score(email_id: str, flagged: bool) -> float:
    """Score spam detection: true positives rewarded, false positives heavily penalized."""
    gt = GROUND_TRUTH.get(email_id, {})
    is_spam = gt.get("category") == "spam"
    if flagged and is_spam:
        return 1.0   # true positive
    if flagged and not is_spam:
        return -0.5  # false positive — penalize hard
    if not flagged and is_spam:
        return 0.0   # missed — no score
    return 0.0       # true negative — no score (expected)


# ──────────────────────────────────────────────────────────────
# TASK 1 GRADER
# ──────────────────────────────────────────────────────────────

def grade_task1(email_states: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Task 1: Basic Inbox Triage.
    Score = average classification accuracy across all 10 emails.
    """
    total_score = 0.0
    breakdown = {}
    expected_ids = {f"t1_e{i}" for i in range(1, 11)}

    for email in email_states:
        eid = email["id"]
        if eid not in expected_ids:
            continue
        priority = email.get("priority_label") or ""
        category = email.get("category_label") or ""
        score = _classification_score(eid, priority, category)
        breakdown[eid] = {
            "score": score,
            "got_priority": priority,
            "got_category": category,
            "expected_priority": GROUND_TRUTH.get(eid, {}).get("priority"),
            "expected_category": GROUND_TRUTH.get(eid, {}).get("category"),
        }
        total_score += score

    final = total_score / len(expected_ids) if expected_ids else 0.0
    return {
        "score": _clamp(final),
        "breakdown": breakdown,
        "task": "task_1_basic_triage",
    }


# ──────────────────────────────────────────────────────────────
# TASK 2 GRADER
# ──────────────────────────────────────────────────────────────

def grade_task2(email_states: List[Dict[str, Any]], action_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Task 2: Reply and Escalate.
    Weighted score:
      - Classification (40%): accuracy on all 15 emails
      - Reply quality (30%): tone + non-empty body on emails needing replies
      - Escalation routing (20%): correct team for urgent escalations
      - Spam flagging (10%): true positives, penalize false positives
    """
    classification_score = 0.0
    reply_score = 0.0
    escalation_score = 0.0
    spam_score = 0.0
    expected_ids = {f"t1_e{i}" for i in range(1, 11)} | {f"t2_e{i}" for i in range(11, 16)}

    # Collect per-email action results
    reply_by_email: Dict[str, Dict] = {}
    escalation_by_email: Dict[str, str] = {}

    for action in action_history:
        eid = action.get("email_id")
        if not eid:
            continue
        atype = action.get("action_type")
        if atype == "reply":
            reply_by_email[eid] = {"tone": action.get("tone", ""), "body": action.get("body", "")}
        elif atype == "escalate":
            escalation_by_email[eid] = action.get("escalate_to", "")

    n_classified = 0
    n_replies_expected = 0
    n_escalations_expected = 0
    n_spam_expected = 0

    for email in email_states:
        eid = email["id"]
        if eid not in expected_ids:
            continue

        gt = GROUND_TRUTH.get(eid, {})
        gt_actions = gt.get("gt_actions", [])

        # Classification
        priority = email.get("priority_label") or ""
        category = email.get("category_label") or ""
        classification_score += _classification_score(eid, priority, category)
        n_classified += 1

        # Replies
        needs_reply = any(a.startswith("reply:") for a in gt_actions)
        if needs_reply:
            n_replies_expected += 1
            if eid in reply_by_email:
                r = reply_by_email[eid]
                reply_score += _reply_quality_score(eid, r["tone"], r["body"])

        # Escalations
        needs_escalation = any(a.startswith("escalate:") for a in gt_actions)
        if needs_escalation:
            n_escalations_expected += 1
            if eid in escalation_by_email:
                escalation_score += max(0, _escalation_score(eid, escalation_by_email[eid]))
            # Also penalize escalations on emails that don't need them
        elif eid in escalation_by_email:
            escalation_score -= 0.2  # unnecessary escalation penalty

        # Spam
        is_gt_spam = gt.get("category") == "spam"
        if is_gt_spam:
            n_spam_expected += 1
            flagged = email.get("is_spam_flagged", False)
            spam_score += _spam_detection_score(eid, flagged)
        elif email.get("is_spam_flagged", False):
            spam_score -= 0.5  # false positive

    classification_final = (classification_score / n_classified) if n_classified else 0.0
    reply_final = (reply_score / n_replies_expected) if n_replies_expected else 1.0
    escalation_final = (escalation_score / n_escalations_expected) if n_escalations_expected else 1.0
    spam_final = (spam_score / n_spam_expected) if n_spam_expected else 1.0

    weighted = (
        0.40 * classification_final
        + 0.30 * reply_final
        + 0.20 * max(0.0, escalation_final)
        + 0.10 * max(0.0, spam_final)
    )

    return {
        "score": _clamp(weighted),
        "breakdown": {
            "classification": round(classification_final, 4),
            "reply_quality": round(reply_final, 4),
            "escalation_routing": round(escalation_final, 4),
            "spam_detection": round(spam_final, 4),
        },
        "task": "task_2_reply_and_escalate",
    }


# ──────────────────────────────────────────────────────────────
# TASK 3 GRADER
# ──────────────────────────────────────────────────────────────

def grade_task3(email_states: List[Dict[str, Any]], action_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Task 3: Full Workflow.
    Additional scoring on top of Task 2:
      - Trap emails: false-positive spam flag on trap emails (t3_e16) → heavy penalty
      - Phishing detection: correctly flagging t3_e17 as spam
      - Thread continuity: follow-up emails (t3_e13, t3_e18) escalated/replied
      - Multi-action completeness: emails needing 3+ actions graded for completeness
    Weights: classification 30%, reply 20%, escalation 20%, spam/traps 20%, completeness 10%
    """
    # Re-use task2 logic for base scoring, then add trap scoring
    base = grade_task2(email_states, action_history)
    base_score = base["score"]

    trap_score = 0.0
    # t3_e16 = legitimate security alert — should NOT be flagged as spam
    # t3_e17 = phishing — SHOULD be flagged as spam
    trap_email = next((e for e in email_states if e["id"] == "t3_e16"), None)
    phish_email = next((e for e in email_states if e["id"] == "t3_e17"), None)

    trap_penalty = 0.0
    if trap_email and trap_email.get("is_spam_flagged"):
        trap_penalty = -0.3  # heavy false-positive penalty
    if phish_email and phish_email.get("is_spam_flagged"):
        trap_score = 0.15  # correctly identified phishing

    # Thread continuity: t3_e18 is follow-up to t1_e2 — check escalated + replied
    thread_score = 0.0
    escalations_on_e18 = [a for a in action_history if a.get("email_id") == "t3_e18" and a.get("action_type") == "escalate"]
    replies_on_e18 = [a for a in action_history if a.get("email_id") == "t3_e18" and a.get("action_type") == "reply"]
    if escalations_on_e18:
        thread_score += 0.05
    if replies_on_e18:
        thread_score += 0.05

    # Multi-action completeness on t3_e20 (server outage — needs classify + 2x escalate)
    completeness_score = 0.0
    e20_actions = [a for a in action_history if a.get("email_id") == "t3_e20"]
    e20_types = {a.get("action_type") for a in e20_actions}
    e20_escalation_targets = {a.get("escalate_to") for a in e20_actions if a.get("action_type") == "escalate"}
    if "classify" in e20_types:
        completeness_score += 0.03
    if "technical_team" in e20_escalation_targets:
        completeness_score += 0.04
    if "manager" in e20_escalation_targets:
        completeness_score += 0.03

    final = min(max(base_score * 0.7 + trap_score + trap_penalty + thread_score + completeness_score, 0.0), 1.0)

    return {
        "score": _clamp(final),
        "breakdown": {
            **base["breakdown"],
            "trap_handling": round(trap_score + trap_penalty, 4),
            "thread_continuity": round(thread_score, 4),
            "multi_action_completeness": round(completeness_score, 4),
        },
        "task": "task_3_full_workflow",
    }


GRADERS = {
    "task_1_basic_triage": lambda states, history: grade_task1(states),
    "task_2_reply_and_escalate": grade_task2,
    "task_3_full_workflow": grade_task3,
}