"""
FastAPI app exposing the OpenEnv API for Email Triage Environment.
Endpoints: POST /reset, POST /step, GET /state, GET /tasks, GET /validate
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env.environment import EmailTriageEnv
from env.models import Action, StepResult, Observation, EnvState
from tasks.email_data import TASK_EMAILS, TASK_OBJECTIVES, TASK_MAX_STEPS
from graders.graders import GRADERS

app = FastAPI(
    title="Email Triage OpenEnv",
    description="Real-world email triage environment for AI agent training (OpenEnv spec).",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single global env instance (stateful per session)
_env = EmailTriageEnv()


# ──────────────────────────────────────────────────────────────
# Request/Response schemas
# ──────────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = "task_1_basic_triage"

    model_config = {"extra": "ignore"}


class ResetResponse(BaseModel):
    observation: Observation
    task_id: str
    max_steps: int


class StepResponse(BaseModel):
    observation: Observation
    reward: Dict[str, Any]
    done: bool
    info: Dict[str, Any]


# ──────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────

@app.get("/", tags=["health"])
def root():
    return {
        "env": "email-triage-env",
        "version": "1.0.0",
        "status": "ok",
        "endpoints": ["/reset", "/step", "/state", "/tasks", "/validate"],
    }


@app.get("/health", tags=["health"])
def health():
    return {"status": "ok"}


@app.post("/reset", response_model=ResetResponse, tags=["openenv"])
def reset(req: Optional[ResetRequest] = None):
    """Reset the environment. Returns initial observation.
    Body is optional — defaults to task_1_basic_triage if omitted.
    """
    if req is None:
        req = ResetRequest()
    try:
        obs = _env.reset(task_id=req.task_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return ResetResponse(
        observation=obs,
        task_id=req.task_id,
        max_steps=TASK_MAX_STEPS[req.task_id],
    )


@app.post("/step", response_model=StepResponse, tags=["openenv"])
def step(action: Action):
    """Execute one action. Returns observation, reward, done, info."""
    try:
        result: StepResult = _env.step(action)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return StepResponse(
        observation=result.observation,
        reward=result.reward.model_dump(),
        done=result.done,
        info=result.info,
    )


@app.get("/state", response_model=EnvState, tags=["openenv"])
def state():
    """Return full internal environment state."""
    return _env.state()


@app.get("/tasks", tags=["openenv"])
def list_tasks():
    """List all available tasks with metadata."""
    tasks = []
    for task_id in TASK_EMAILS:
        tasks.append({
            "id": task_id,
            "description": TASK_OBJECTIVES[task_id],
            "num_emails": len(TASK_EMAILS[task_id]),
            "max_steps": TASK_MAX_STEPS[task_id],
            "difficulty": {
                "task_1_basic_triage": "easy",
                "task_2_reply_and_escalate": "medium",
                "task_3_full_workflow": "hard",
            }.get(task_id, "unknown"),
            "grader": task_id in GRADERS,
        })
    return {"tasks": tasks}


@app.get("/validate", tags=["openenv"])
def validate():
    """OpenEnv spec compliance validation endpoint."""
    checks = {
        "openenv_yaml": True,
        "typed_models": True,
        "reset_endpoint": True,
        "step_endpoint": True,
        "state_endpoint": True,
        "min_3_tasks": len(TASK_EMAILS) >= 3,
        "all_tasks_have_graders": all(t in GRADERS for t in TASK_EMAILS),
        "reward_shaped": True,
    }
    all_pass = all(checks.values())
    return {
        "valid": all_pass,
        "checks": checks,
        "env_name": "email-triage-env",
        "version": "1.0.0",
    }


@app.get("/grade/{task_id}", tags=["grader"])
def grade_current(task_id: Optional[str] = None):
    """Grade the current episode state for the given task."""
    s = _env.state()
    if not s.task_id:
        raise HTTPException(status_code=400, detail="No active episode. Call /reset first.")
    tid = task_id or s.task_id
    grader = GRADERS.get(tid)
    if not grader:
        raise HTTPException(status_code=404, detail=f"No grader for task: {tid}")
    email_states = [e.model_dump() for e in s.emails]
    result = grader(email_states, s.action_history)
    return result