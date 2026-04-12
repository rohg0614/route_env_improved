# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Route Env Environment.

Usage:
    uvicorn server.app:app --host 0.0.0.0 --port 7860
"""

import math
import os
import time
import yaml
import pathlib
from threading import Lock
from typing import Optional

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install with: uv sync"
    ) from e

from fastapi import HTTPException
from pydantic import BaseModel

os.environ.setdefault("ENABLE_WEB_INTERFACE", "true")

# Flat structure — models and server files are all at repo root
from models import RouteAction, RouteObservation
from server.route_env_environment import RouteEnvironment
from tasks import TASKS, TASK_ORDER
from grader import score_episode

app = create_app(
    RouteEnvironment,
    RouteAction,
    RouteObservation,
    env_name="route_env_improved",
    max_concurrent_envs=1,
)


class GraderRequest(BaseModel):
    task_name: str
    step_count: int = 0
    completed_rides: int = 0
    late_rides: int = 0
    total_reward: float = 0.0


class ScoreRecord(BaseModel):
    task_name: str
    score: float
    model: Optional[str] = None
    timestamp: Optional[float] = None


# In-memory leaderboard — stores up to 200 most recent scored runs.
_leaderboard: list[dict] = []
_leaderboard_lock = Lock()


def _tasks_response():
    return {
        "tasks": [
            {
                "name": name,
                "horizon_steps": cfg.horizon_steps,
                "node_count": cfg.node_count,
                "max_shift_hours": cfg.max_shift_hours,
                "has_grader": True,
            }
            for name, cfg in TASKS.items()
        ]
    }


def _grade_response(request: GraderRequest):
    if request.task_name not in TASKS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task '{request.task_name}'. Valid tasks: {TASK_ORDER}",
        )
    score = score_episode(
        step_count=request.step_count,
        completed_rides=request.completed_rides,
        late_rides=request.late_rides,
        total_reward=request.total_reward,
        task_name=request.task_name,
    )
    # Clamp as a final safety net before returning to the validator
    score = float(max(0.02, min(0.98, score)))
    if not (0.0 < score < 1.0):
        raise HTTPException(
            status_code=500,
            detail=f"Grader returned out-of-range score: {score}",
        )
    return {"task_name": request.task_name, "score": score}


# Register at both prefixes — HF Space proxies /web/, validator may hit either
@app.get("/tasks")
@app.get("/web/tasks")
def list_tasks():
    return _tasks_response()


@app.post("/grader")
@app.post("/web/grader")
def grade(request: GraderRequest):
    return _grade_response(request)


@app.post("/leaderboard")
@app.post("/web/leaderboard")
def record_score(record: ScoreRecord):
    """Record a scored run into the in-memory leaderboard."""
    entry = {
        "task_name": record.task_name,
        "score": round(float(record.score), 4),
        "model": record.model or "unknown",
        "timestamp": record.timestamp or time.time(),
    }
    with _leaderboard_lock:
        _leaderboard.append(entry)
        # Keep only the 200 most recent entries to cap memory use.
        if len(_leaderboard) > 200:
            _leaderboard.pop(0)
    return {"status": "recorded", "entry": entry}


@app.get("/leaderboard")
@app.get("/web/leaderboard")
def get_leaderboard():
    """Return the top 20 scores per task, sorted descending."""
    with _leaderboard_lock:
        snapshot = list(_leaderboard)
    by_task: dict[str, list[dict]] = {}
    for entry in snapshot:
        by_task.setdefault(entry["task_name"], []).append(entry)
    top: dict[str, list[dict]] = {
        task: sorted(entries, key=lambda e: -e["score"])[:20]
        for task, entries in by_task.items()
    }
    overall = sorted(snapshot, key=lambda e: -e["score"])[:20]
    return {"by_task": top, "overall_top20": overall, "total_recorded": len(snapshot)}


@app.get("/spec")
@app.get("/web/spec")
def get_spec():
    """Return openenv.yaml metadata as JSON — required by openenv validate."""
    yaml_path = pathlib.Path(__file__).resolve().parent.parent / "openenv.yaml"
    try:
        with open(yaml_path, "r") as f:
            spec_data = yaml.safe_load(f)
        return spec_data
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="openenv.yaml not found")


def main() -> None:
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()