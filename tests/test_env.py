"""
Test suite for Email Triage OpenEnv.
Verifies OpenEnv spec compliance, grader determinism, and reward shaping.
Run with: python -m pytest tests/ -v
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from env.environment import EmailTriageEnv
from env.models import Action, ActionType, Priority, Category, Tone, EscalationTarget
from graders.graders import grade_task1, grade_task2, grade_task3
from tasks.email_data import TASK_EMAILS, GROUND_TRUTH


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────

@pytest.fixture
def env():
    return EmailTriageEnv()


# ──────────────────────────────────────────────────────────────
# Spec compliance tests
# ──────────────────────────────────────────────────────────────

def test_reset_returns_observation(env):
    obs = env.reset("task_1_basic_triage")
    assert obs is not None
    assert obs.inbox_summary is not None
    assert len(obs.inbox_summary) == 10
    assert obs.step_number == 0
    assert obs.task_objective != ""


def test_step_returns_full_result(env):
    env.reset("task_1_basic_triage")
    action = Action(action_type=ActionType.NOOP)
    result = env.step(action)
    assert result.observation is not None
    assert result.reward is not None
    assert isinstance(result.done, bool)
    assert isinstance(result.info, dict)


def test_state_returns_env_state(env):
    env.reset("task_1_basic_triage")
    state = env.state()
    assert state.task_id == "task_1_basic_triage"
    assert state.step_number == 0
    assert not state.done
    assert len(state.emails) == 10


def test_reward_is_bounded(env):
    """Reward must always be in [-1.0, 1.0]."""
    env.reset("task_1_basic_triage")
    for _ in range(5):
        action = Action(action_type=ActionType.NOOP)
        result = env.step(action)
        assert -1.0 <= result.reward.value <= 1.0


def test_all_3_tasks_exist():
    assert "task_1_basic_triage" in TASK_EMAILS
    assert "task_2_reply_and_escalate" in TASK_EMAILS
    assert "task_3_full_workflow" in TASK_EMAILS


def test_task1_has_10_emails():
    assert len(TASK_EMAILS["task_1_basic_triage"]) == 10


def test_task2_has_15_emails():
    assert len(TASK_EMAILS["task_2_reply_and_escalate"]) == 15


def test_task3_has_20_emails():
    assert len(TASK_EMAILS["task_3_full_workflow"]) == 20


# ──────────────────────────────────────────────────────────────
# Focus action tests
# ──────────────────────────────────────────────────────────────

def test_focus_exposes_email_content(env):
    env.reset("task_1_basic_triage")
    action = Action(action_type=ActionType.FOCUS, email_id="t1_e1")
    result = env.step(action)
    assert result.observation.current_email is not None
    assert result.observation.current_email.id == "t1_e1"


def test_focus_invalid_id_penalty(env):
    env.reset("task_1_basic_triage")
    action = Action(action_type=ActionType.FOCUS, email_id="nonexistent_id")
    result = env.step(action)
    assert result.reward.value < 0


# ──────────────────────────────────────────────────────────────
# Classify action tests
# ──────────────────────────────────────────────────────────────

def test_correct_classification_positive_reward(env):
    env.reset("task_1_basic_triage")
    # t1_e1 is urgent/technical
    env.step(Action(action_type=ActionType.FOCUS, email_id="t1_e1"))
    result = env.step(Action(
        action_type=ActionType.CLASSIFY,
        priority=Priority.URGENT,
        category=Category.TECHNICAL,
    ))
    assert result.reward.value > 0


def test_wrong_classification_small_penalty(env):
    env.reset("task_1_basic_triage")
    env.step(Action(action_type=ActionType.FOCUS, email_id="t1_e1"))
    result = env.step(Action(
        action_type=ActionType.CLASSIFY,
        priority=Priority.LOW,
        category=Category.GENERAL,
    ))
    assert result.reward.value < 0


# ──────────────────────────────────────────────────────────────
# Spam detection tests
# ──────────────────────────────────────────────────────────────

def test_true_positive_spam_reward(env):
    env.reset("task_1_basic_triage")
    # t1_e4 is spam
    env.step(Action(action_type=ActionType.FOCUS, email_id="t1_e4"))
    result = env.step(Action(action_type=ActionType.FLAG_SPAM, confidence=0.95))
    assert result.reward.value > 0


def test_false_positive_spam_penalty(env):
    env.reset("task_1_basic_triage")
    # t1_e1 is NOT spam — flagging it should penalize
    env.step(Action(action_type=ActionType.FOCUS, email_id="t1_e1"))
    result = env.step(Action(action_type=ActionType.FLAG_SPAM, confidence=0.9))
    assert result.reward.value < 0


# ──────────────────────────────────────────────────────────────
# Loop detection tests
# ──────────────────────────────────────────────────────────────

def test_loop_detection_applies_penalty(env):
    env.reset("task_1_basic_triage")
    rewards = []
    for _ in range(5):
        result = env.step(Action(action_type=ActionType.NOOP))
        rewards.append(result.reward.value)
    # After 3 noops, loop penalty should kick in
    assert rewards[4] < rewards[0]


# ──────────────────────────────────────────────────────────────
# Episode termination tests
# ──────────────────────────────────────────────────────────────

def test_episode_ends_at_max_steps(env):
    env.reset("task_1_basic_triage")
    result = None
    for _ in range(35):  # task1 max is 30
        result = env.step(Action(action_type=ActionType.NOOP))
        if result.done:
            break
    assert result.done


def test_step_after_done_safe(env):
    env.reset("task_1_basic_triage")
    for _ in range(35):
        result = env.step(Action(action_type=ActionType.NOOP))
        if result.done:
            break
    # Extra step after done should not crash
    result2 = env.step(Action(action_type=ActionType.NOOP))
    assert result2.done


# ──────────────────────────────────────────────────────────────
# Grader tests
# ──────────────────────────────────────────────────────────────

def test_grader_task1_zero_score_no_labels():
    """Unclassified inbox should score 0."""
    email_states = [
        {"id": f"t1_e{i}", "priority_label": None, "category_label": None}
        for i in range(1, 11)
    ]
    result = grade_task1(email_states)
    assert result["score"] == 0.0


def test_grader_task1_perfect_score():
    """Perfectly classified inbox should score 1.0."""
    from tasks.email_data import GROUND_TRUTH
    email_states = []
    for i in range(1, 11):
        eid = f"t1_e{i}"
        gt = GROUND_TRUTH.get(eid, {})
        email_states.append({
            "id": eid,
            "priority_label": gt.get("priority"),
            "category_label": gt.get("category"),
        })
    result = grade_task1(email_states)
    assert result["score"] == 1.0


def test_grader_task1_score_in_range():
    email_states = [
        {"id": f"t1_e{i}", "priority_label": "normal", "category_label": "general"}
        for i in range(1, 11)
    ]
    result = grade_task1(email_states)
    assert 0.0 <= result["score"] <= 1.0


def test_grader_task2_score_in_range():
    email_states = [
        {"id": f"t1_e{i}", "priority_label": "normal", "category_label": "general",
         "is_spam_flagged": False, "is_escalated": False, "has_reply": False}
        for i in range(1, 11)
    ] + [
        {"id": f"t2_e{i}", "priority_label": "normal", "category_label": "general",
         "is_spam_flagged": False, "is_escalated": False, "has_reply": False}
        for i in range(11, 16)
    ]
    result = grade_task2(email_states, [])
    assert 0.0 <= result["score"] <= 1.0


def test_grader_deterministic():
    """Same inputs must always produce same score."""
    email_states = [
        {"id": f"t1_e{i}", "priority_label": "urgent", "category_label": "technical"}
        for i in range(1, 11)
    ]
    score1 = grade_task1(email_states)["score"]
    score2 = grade_task1(email_states)["score"]
    assert score1 == score2


def test_grader_task3_trap_penalty():
    """Flagging legitimate security alert (t3_e16) as spam should hurt score."""
    from tasks.email_data import GROUND_TRUTH, TASK_EMAILS
    task3 = TASK_EMAILS["task_3_full_workflow"]
    email_states = []
    for e in task3:
        eid = e["id"]
        gt = GROUND_TRUTH.get(eid, {})
        email_states.append({
            "id": eid,
            "priority_label": gt.get("priority"),
            "category_label": gt.get("category"),
            "is_spam_flagged": eid == "t3_e16",  # FALSE positive trap
            "is_escalated": False,
            "has_reply": False,
        })
    result_with_false_positive = grade_task3(email_states, [])

    # Now without the false positive
    for s in email_states:
        s["is_spam_flagged"] = False
    result_clean = grade_task3(email_states, [])

    assert result_clean["score"] > result_with_false_positive["score"]
