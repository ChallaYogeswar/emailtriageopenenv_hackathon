"""
Synthetic email datasets for all 3 tasks.
Each email has a ground-truth label used by graders.
"""

from typing import Dict, List

# Ground-truth answers (used only by graders, never exposed to agent)
GROUND_TRUTH: Dict[str, Dict] = {}


def _e(eid, subject, sender, body, ts, priority, category, thread_id=None, attachments=None, gt_actions=None):
    """Helper to build email dict + register ground truth."""
    GROUND_TRUTH[eid] = {
        "priority": priority,
        "category": category,
        "gt_actions": gt_actions or [],
    }
    return {
        "id": eid,
        "subject": subject,
        "sender": sender,
        "timestamp": ts,
        "body": body,
        "thread_id": thread_id or eid,
        "attachments": attachments or [],
        "labels": [],
    }


# ──────────────────────────────────────────────────────────────
# TASK 1 — Basic Inbox Triage (10 emails, easy)
# Objective: Correctly classify priority + category for each email.
# ──────────────────────────────────────────────────────────────

TASK1_EMAILS = [
    _e("t1_e1", "Cannot login to my account",
       "alice@example.com",
       "Hi, I've been trying to log in for the past hour and keep getting 'invalid credentials'. "
       "I need access urgently for a client presentation in 2 hours. Please help ASAP.",
       "2024-03-15T09:00:00Z", "urgent", "technical",
       gt_actions=["classify:urgent:technical"]),

    _e("t1_e2", "Invoice #4521 – Wrong amount charged",
       "bob@company.org",
       "Hello, I received my invoice this month and was charged $450 instead of the agreed $200. "
       "Please correct this and send a revised invoice. Thank you.",
       "2024-03-15T09:15:00Z", "high", "billing",
       gt_actions=["classify:high:billing"]),

    _e("t1_e3", "Question about your premium plan",
       "carol@gmail.com",
       "Hi there! I was wondering what features are included in the premium plan and "
       "how it differs from the basic plan. No rush, just curious.",
       "2024-03-15T09:30:00Z", "low", "general",
       gt_actions=["classify:low:general"]),

    _e("t1_e4", "🎉 You've WON a $1000 Amazon gift card! Click HERE",
       "noreply@prize-winner-2024.ru",
       "Congratulations! You have been selected as our lucky winner! "
       "Click the link below to claim your prize NOW before it expires! "
       "http://totally-legit-prizes.ru/claim?user=target",
       "2024-03-15T09:45:00Z", "low", "spam",
       gt_actions=["classify:low:spam", "flag_spam"]),

    _e("t1_e5", "Weekly team standup notes",
       "manager@ourcompany.com",
       "Hi team, here are the notes from today's standup:\n"
       "- Project X on track\n- Deploy scheduled for Friday\n- Please review PRs by EOD.\nThanks!",
       "2024-03-15T10:00:00Z", "normal", "internal",
       gt_actions=["classify:normal:internal"]),

    _e("t1_e6", "API integration keeps timing out",
       "dev@startup.io",
       "We're integrating your REST API and getting consistent 504 timeouts on the /data/export endpoint. "
       "This is blocking our production launch. We've tried increasing timeout to 60s with no luck. "
       "Reproduction steps attached.",
       "2024-03-15T10:15:00Z", "urgent", "technical",
       gt_actions=["classify:urgent:technical"]),

    _e("t1_e7", "Subscription renewal confirmation",
       "billing@ourcompany.com",
       "Your annual subscription has been successfully renewed. "
       "Amount: $299.00. Next renewal: March 15, 2025. Receipt attached.",
       "2024-03-15T10:30:00Z", "low", "billing",
       gt_actions=["classify:low:billing"]),

    _e("t1_e8", "How do I export my data?",
       "user123@yahoo.com",
       "Hello, I'd like to export all my data from the platform before I cancel my subscription. "
       "Can you walk me through the steps? Thanks.",
       "2024-03-15T10:45:00Z", "normal", "general",
       gt_actions=["classify:normal:general"]),

    _e("t1_e9", "URGENT: Data breach notification",
       "cto@bigclient.com",
       "We have detected suspicious activity in our account that may indicate unauthorized access. "
       "Multiple login attempts from IP 185.220.x.x at 3AM. This needs immediate attention. "
       "We may need to involve our legal team.",
       "2024-03-15T11:00:00Z", "urgent", "technical",
       gt_actions=["classify:urgent:technical", "escalate:technical_team"]),

    _e("t1_e10", "Feedback on your product",
       "happy.customer@gmail.com",
       "Just wanted to say your product has been fantastic for our team! "
       "We've cut our reporting time by 40%. Keep up the great work!",
       "2024-03-15T11:15:00Z", "low", "general",
       gt_actions=["classify:low:general"]),
]


# ──────────────────────────────────────────────────────────────
# TASK 2 — Reply and Escalate (15 emails, medium)
# Objective: Correct classification + appropriate reply/escalation.
# ──────────────────────────────────────────────────────────────

TASK2_EMAILS = TASK1_EMAILS + [
    _e("t2_e11", "Payment failed – account suspended",
       "frustrated@client.com",
       "My payment failed AGAIN and now my account is suspended! I have a demo tomorrow "
       "and I cannot afford downtime. This is the third time this has happened. "
       "I need this resolved in the next hour or I'm cancelling and filing a dispute.",
       "2024-03-15T11:30:00Z", "urgent", "billing",
       gt_actions=["classify:urgent:billing", "reply:apologetic", "escalate:billing_team"]),

    _e("t2_e12", "Legal: Unauthorized use of our trademark",
       "legal@bigcorp.com",
       "Dear Sir/Madam, We have identified use of our registered trademark 'CloudSync Pro' "
       "on your platform without authorization. This constitutes infringement under 15 U.S.C. § 1114. "
       "We demand immediate removal or we will pursue legal remedies. Please respond within 48 hours.",
       "2024-03-15T11:45:00Z", "urgent", "general",
       gt_actions=["classify:urgent:general", "escalate:legal"]),

    _e("t2_e13", "Re: API timeout issue – still not fixed",
       "dev@startup.io",
       "Hi, following up on my earlier email about the API timeouts. "
       "It's been 6 hours and we still have no resolution. Our launch is in 2 hours. "
       "Please escalate this immediately.",
       "2024-03-15T12:00:00Z", "urgent", "technical",
       thread_id="t1_e6",
       gt_actions=["classify:urgent:technical", "escalate:technical_team", "reply:apologetic"]),

    _e("t2_e14", "Earn $5000/week from home! No experience needed!",
       "money@quick-rich-2024.biz",
       "Dear Friend, Are you tired of your 9-5? Our PROVEN system lets you earn $5000 per week "
       "from the comfort of your home. No experience needed! Just click here: http://scam.biz/join",
       "2024-03-15T12:15:00Z", "low", "spam",
       gt_actions=["classify:low:spam", "flag_spam", "archive:spam"]),

    _e("t2_e15", "How to reset 2FA on our account?",
       "ops@medium-business.com",
       "We recently onboarded two new team members and need to reset 2FA for their accounts. "
       "Our IT admin is unavailable this week. Can you guide us through the process?",
       "2024-03-15T12:30:00Z", "normal", "technical",
       gt_actions=["classify:normal:technical", "reply:friendly"]),
]


# ──────────────────────────────────────────────────────────────
# TASK 3 — Full Workflow (20 emails, hard)
# Traps: legitimate emails that look spammy, thread dependencies
# ──────────────────────────────────────────────────────────────

TASK3_EMAILS = TASK2_EMAILS + [
    # Trap: Looks spammy but is a legitimate security alert
    _e("t3_e16", "⚠️ URGENT: Your account password was changed",
       "security-noreply@ourcompany-platform.com",
       "A password change was made to your account from IP 203.0.113.42 (Singapore). "
       "If this was NOT you, click here immediately to secure your account and contact support. "
       "Reference: SEC-2024-8821.",
       "2024-03-15T12:45:00Z", "urgent", "technical",
       gt_actions=["classify:urgent:technical"]),  # NOT spam - trap!

    # Trap: Looks internal but is external phishing
    _e("t3_e17", "IT Department: Please verify your credentials",
       "it-helpdesk@ourcompany-support-desk.ru",
       "Dear Employee, Our systems require you to re-verify your login credentials. "
       "Please visit http://ourcompany-support-desk.ru/verify and enter your username and password. "
       "Failure to comply within 24h will result in account lockout. - IT Dept",
       "2024-03-15T13:00:00Z", "low", "spam",
       gt_actions=["classify:low:spam", "flag_spam"]),  # Phishing - IS spam

    _e("t3_e18", "Re: Invoice dispute – escalation required",
       "bob@company.org",
       "Hi, I still haven't heard back about my invoice dispute from this morning. "
       "I've now reviewed our contract and the overcharge is clearly in violation of clause 4.2. "
       "I need a written confirmation of refund by 5PM today or I'll involve my legal counsel.",
       "2024-03-15T13:15:00Z", "urgent", "billing",
       thread_id="t1_e2",
       gt_actions=["classify:urgent:billing", "escalate:billing_team", "reply:apologetic"]),

    _e("t3_e19", "New feature request: Bulk CSV import",
       "product.feedback@enterprise.com",
       "Hi team, we've been using your platform for 6 months and love it. "
       "Our main pain point is manually uploading data. A bulk CSV import feature "
       "would save us ~3 hours/week. We'd be happy to be beta testers. "
       "Priority for us: medium-high. Happy to jump on a call.",
       "2024-03-15T13:30:00Z", "normal", "general",
       gt_actions=["classify:normal:general", "reply:friendly"]),

    _e("t3_e20", "Server outage – ALL customers affected",
       "monitoring@ourcompany.com",
       "CRITICAL ALERT: Production server cluster US-EAST-1 is down. "
       "Error rate: 100%. Affected services: API, Dashboard, Auth. "
       "Duration: 8 minutes and counting. Incident ID: INC-2024-0315-001. "
       "On-call engineer notified. All hands required.",
       "2024-03-15T13:45:00Z", "urgent", "technical",
       gt_actions=["classify:urgent:technical", "escalate:technical_team", "escalate:manager"]),
]


TASK_EMAILS = {
    "task_1_basic_triage": TASK1_EMAILS,
    "task_2_reply_and_escalate": TASK2_EMAILS,
    "task_3_full_workflow": TASK3_EMAILS,
}

TASK_OBJECTIVES = {
    "task_1_basic_triage": (
        "Process the inbox and correctly classify each email by priority "
        "(urgent/high/normal/low) and category (billing/technical/general/spam/internal). "
        "Use 'focus' to read an email, then 'classify' to label it."
    ),
    "task_2_reply_and_escalate": (
        "Process the inbox: classify all emails, send appropriate replies to customer queries, "
        "escalate critical issues to the correct team, and flag/archive spam. "
        "Prioritize urgent emails first."
    ),
    "task_3_full_workflow": (
        "Handle a complex inbox with interdependencies and traps. "
        "Correctly identify phishing emails (do NOT flag legitimate security alerts as spam). "
        "Handle thread follow-ups, multi-step actions, and ensure zero false-positive spam flags "
        "on legitimate emails. End state will be fully graded."
    ),
}

TASK_MAX_STEPS = {
    "task_1_basic_triage": 30,
    "task_2_reply_and_escalate": 50,
    "task_3_full_workflow": 80,
}
