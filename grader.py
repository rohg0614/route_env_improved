"""Deterministic per-task graders returning scores strictly in (0.0, 1.0).

Each task has a distinct grader calibrated to its difficulty:

  Easy   (6 nodes,  60 steps, lateness_budget=0.45):
    - Ride efficiency saturates at 0.68; late penalty is light (0.18).
    - A greedy always-accept agent scores ~0.76. Idle agent ~0.08.
    - Designed so any reasonable policy passes the 0.5 threshold.

  Medium (8 nodes,  84 steps, lateness_budget=0.32):
    - Ride efficiency saturates at 0.65; late penalty is moderate (0.28).
    - A greedy agent scores ~0.68. Idle agent ~0.05.
    - Requires consistent ride acceptance AND some SLA awareness to score well.

  Hard   (12 nodes, 120 steps, lateness_budget=0.15):
    - Ride efficiency saturates at 0.60; late penalty is severe (0.45).
    - A greedy always-accept agent scores ~0.36 (67% late rate in practice).
    - Only an agent that actively avoids high-wait rides passes (~0.64+).
    - This is the key differentiator: pure greedy fails, SLA-aware agent passes.

All three graders:
  - Use ride RATE (completed / step_count) not raw count, so a 120-step hard
    episode doesn't get a free time_comp bonus over a 60-step easy episode.
  - Use late RATE (late / completed) not raw count, so the penalty scales with
    how badly the agent manages SLA, not absolute episode length.
  - Are strictly in (0.02, 0.98) by hard clamp — the validator's (0, 1) check
    always passes regardless of input extremes.
"""

import math
from typing import Callable


# ── Per-task scoring functions ────────────────────────────────────────────────

def _score_easy(
    step_count: int,
    completed_rides: int,
    late_rides: int,
    total_reward: float,
) -> float:
    """Grader for the easy task (6 nodes, 60 steps, relaxed SLA).

    Calibration targets:
      idle agent  → ~0.08
      greedy      → ~0.76
      good LLM    → ~0.78
      optimal     → ~0.79
    """
    if step_count == 0:
        return 0.06

    # Primary: ride efficiency (rides per step), saturates at 0.68
    ride_rate = completed_rides / step_count
    ride_comp = min(0.68, ride_rate * 2.7)

    # Secondary: fare quality (reward per step), log-damped
    fare_rate = max(0.0, total_reward) / step_count
    fare_comp = min(0.16, math.log1p(fare_rate) * 0.06)

    # Penalty: late ride rate (fraction of rides that breached SLA)
    late_rate = late_rides / max(1, completed_rides)
    late_penalty = late_rate * 0.18  # relaxed — easy task has lateness_budget=0.45

    raw = 0.08 + ride_comp + fare_comp - late_penalty
    clamped = max(0.02, min(0.98, raw))
    floored = math.floor(clamped * 10000) / 10000
    final = max(0.02, min(0.98, floored))
    return float(final)


def _score_medium(
    step_count: int,
    completed_rides: int,
    late_rides: int,
    total_reward: float,
) -> float:
    """Grader for the medium task (8 nodes, 84 steps, moderate SLA).

    Calibration targets:
      idle agent  → ~0.05
      greedy      → ~0.68
      good LLM    → ~0.71
      optimal     → ~0.73
    """
    if step_count == 0:
        return 0.05

    ride_rate = completed_rides / step_count
    ride_comp = min(0.65, ride_rate * 2.6)

    fare_rate = max(0.0, total_reward) / step_count
    fare_comp = min(0.18, math.log1p(fare_rate) * 0.07)

    late_rate = late_rides / max(1, completed_rides)
    late_penalty = late_rate * 0.28  # moderate — lateness_budget=0.32

    raw = 0.05 + ride_comp + fare_comp - late_penalty
    clamped = max(0.02, min(0.98, raw))
    floored = math.floor(clamped * 10000) / 10000
    final = max(0.02, min(0.98, floored))
    return float(final)


def _score_hard(
    step_count: int,
    completed_rides: int,
    late_rides: int,
    total_reward: float,
) -> float:
    """Grader for the hard task (12 nodes, 120 steps, tight SLA).

    Calibration targets:
      idle agent              → ~0.04
      greedy (37/55 late)     → ~0.36  [well BELOW 0.5 threshold — greedy FAILS]
      good LLM (2/38 late)    → ~0.64  [above threshold — SLA-aware agent PASSES]
      optimal (0/45 late)     → ~0.67

    The key design property: a pure greedy always-accept agent scores ~0.36
    due to a 67% late ride rate on the hard task (lateness_budget=0.15 means
    ~75% of randomly generated rides are unsafe). An agent that reads wait_time
    and skips high-latency rides scores 0.64+, clearing the 0.5 threshold.
    This forces evaluators to see differentiated LLM reasoning.
    """
    if step_count == 0:
        return 0.04

    ride_rate = completed_rides / step_count
    ride_comp = min(0.60, ride_rate * 2.4)

    fare_rate = max(0.0, total_reward) / step_count
    fare_comp = min(0.20, math.log1p(fare_rate) * 0.08)

    late_rate = late_rides / max(1, completed_rides)
    late_penalty = late_rate * 0.45  # severe — lateness_budget=0.15, SLA is the challenge

    raw = 0.04 + ride_comp + fare_comp - late_penalty
    clamped = max(0.02, min(0.98, raw))
    floored = math.floor(clamped * 10000) / 10000
    final = max(0.02, min(0.98, floored))
    return float(final)


# ── Registry ──────────────────────────────────────────────────────────────────

_GRADER_FNS: dict[str, Callable] = {
    "easy":   _score_easy,
    "medium": _score_medium,
    "hard":   _score_hard,
}


def _make_task_grader(task_name: str) -> Callable:
    fn = _GRADER_FNS[task_name]

    def grader(
        step_count: int,
        completed_rides: int,
        late_rides: int,
        total_reward: float,
    ) -> float:
        result = fn(step_count, completed_rides, late_rides, total_reward)
        return float(max(0.02, min(0.98, result)))

    grader.__name__ = f"grader_{task_name}"
    grader.__doc__ = (
        f"Grader for task '{task_name}'. "
        f"Returns score in (0.0, 1.0) based on ride efficiency, "
        f"fare quality, and SLA adherence calibrated to {task_name} difficulty."
    )
    return grader


GRADERS: dict[str, Callable] = {
    name: _make_task_grader(name) for name in ("easy", "medium", "hard")
}


def get_grader(task_name: str) -> Callable:
    """Return the grader callable for the given task name.

    Args:
        task_name: One of 'easy', 'medium', 'hard'.

    Returns:
        A callable(step_count, completed_rides, late_rides, total_reward) -> float
        returning a score strictly in (0.0, 1.0).

    Raises:
        KeyError: If task_name is not registered.
    """
    if task_name not in GRADERS:
        raise KeyError(
            f"No grader registered for task '{task_name}'. "
            f"Available: {list(GRADERS.keys())}"
        )
    return GRADERS[task_name]


# Keep score_episode for backward compatibility (used by the environment's
# normalized_progress_score field). Routes to the task-specific grader if
# task context is available, otherwise falls back to easy grader as default.
def score_episode(
    step_count: int,
    completed_rides: int,
    late_rides: int,
    total_reward: float,
    task_name: str = "easy",
) -> float:
    """Score an episode using the task-specific grader.

    The environment calls this via _grader_score() to populate the
    normalized_progress_score field in every observation.
    """
    fn = _GRADER_FNS.get(task_name, _score_easy)
    result = fn(step_count, completed_rides, late_rides, total_reward)
    # Hard safety net — validator requires strictly (0.0, 1.0) exclusive
    return float(max(0.02, min(0.98, result)))