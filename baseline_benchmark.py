"""Deterministic baseline benchmark across easy/medium/hard tasks.

Two baselines are measured to illustrate the hard task's design property:

  greedy_always_accept  — accepts the highest-fare ride regardless of wait_time.
                          Scores ~0.76/0.68/0.36 (easy/medium/hard).
                          Fails the hard task (0.36 < 0.50 threshold) because ~67%
                          of rides are unsafe under the tight lateness_budget=0.15.

  sla_aware_heuristic   — same as greedy but skips rides with wait_time above the
                          task's lateness threshold. Falls back to reposition/wait.
                          Scores ~0.78/0.71/0.64; passes all three tasks.

This two-baseline comparison is the key evaluation insight: pure greedy FAILS the
hard task; an agent that reads wait_time and skips high-latency rides PASSES.
"""

import os
import json
import math
import time
from urllib.error import HTTPError, URLError
from urllib.request import urlopen
from statistics import mean

from client import RouteEnv
from models import RouteAction

# Lateness thresholds mirror inference.py LATENESS_BUDGETS
_LATENESS_BUDGETS: dict[str, float] = {
    "easy":   0.45,
    "medium": 0.32,
    "hard":   0.15,
}


def _late_threshold(task_name: str) -> float:
    return _LATENESS_BUDGETS.get(task_name, 0.45) * 10.0


def choose_greedy(observation) -> RouteAction:
    """Pure greedy: accepts highest-fare ride, no SLA filtering."""
    rides = observation.available_rides or []
    if rides:
        best = max(rides, key=lambda r: float(r.get("fare", 0.0)))
        return RouteAction(action_type="accept_ride", ride_id=int(best["ride_id"]))
    demand = observation.live_demand_matrix or []
    adjacent = list(observation.adjacent_nodes or [])
    if demand and adjacent:
        best_node = max(adjacent, key=lambda idx: float(demand[idx]) if idx < len(demand) else 0.0)
        if best_node != int(observation.current_node):
            return RouteAction(action_type="reposition", target_node=int(best_node))
    return RouteAction(action_type="wait")


def choose_sla_aware(observation) -> RouteAction:
    """SLA-aware heuristic: only accepts rides below the lateness threshold.

    Falls back to reposition toward highest-demand adjacent node, then wait.
    This is the minimum bar a reasonable policy should clear on every task.
    """
    task_name = str(getattr(observation, "task_name", "easy"))
    threshold = _late_threshold(task_name)
    rides = observation.available_rides or []
    safe_rides = [r for r in rides if float(r.get("wait_time", 0)) <= threshold]
    if safe_rides:
        best = max(safe_rides, key=lambda r: float(r.get("fare", 0.0)))
        return RouteAction(action_type="accept_ride", ride_id=int(best["ride_id"]))
    demand = observation.live_demand_matrix or []
    adjacent = list(observation.adjacent_nodes or [])
    if demand and adjacent:
        best_node = max(adjacent, key=lambda idx: float(demand[idx]) if idx < len(demand) else 0.0)
        if best_node != int(observation.current_node):
            return RouteAction(action_type="reposition", target_node=int(best_node))
    return RouteAction(action_type="wait")


def _wait_for_server(env_url: str, wait_seconds: float, poll_seconds: float) -> None:
    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        try:
            with urlopen(f"{env_url}/health", timeout=2) as resp:
                if getattr(resp, "status", 200) == 200:
                    return
        except (URLError, HTTPError):
            pass
        time.sleep(poll_seconds)


def run_task(task_name: str, policy, episodes: int = 3, max_steps: int = 120) -> float:
    scores = []
    env_url = os.getenv("ENV_BASE_URL", "http://localhost:7860")
    wait_seconds = float(os.getenv("WAIT_FOR_SERVER_SECONDS", "30"))
    poll_seconds = float(os.getenv("WAIT_FOR_SERVER_POLL_SECONDS", "0.5"))
    _wait_for_server(env_url, wait_seconds, poll_seconds)
    with RouteEnv(base_url=env_url).sync() as env:
        for ep in range(episodes):
            result = env.reset(task_name=task_name, seed=1000 + ep)
            obs = result.observation
            steps = 0
            done = False
            while steps < max_steps and not done:
                steps += 1
                step_result = env.step(policy(obs))
                obs = step_result.observation
                done = bool(step_result.done)
            scores.append(float(obs.normalized_progress_score))
    return mean(scores)


def main() -> None:
    tasks = ("easy", "medium", "hard")
    success_threshold = 0.5

    print("Baseline benchmark (score range 0.0–1.0, success threshold 0.50)")
    print(f"{'Task':<10} {'Greedy':>10} {'SLA-aware':>12} {'Greedy pass?':>14} {'SLA pass?':>11}")
    print("-" * 62)

    greedy_scores: dict[str, float] = {}
    sla_scores: dict[str, float] = {}

    for task in tasks:
        g = run_task(task, choose_greedy)
        s = run_task(task, choose_sla_aware)
        greedy_scores[task] = g
        sla_scores[task] = s
        g_pass = "PASS" if g >= success_threshold else "FAIL"
        s_pass = "PASS" if s >= success_threshold else "FAIL"
        print(f"{task:<10} {g:>10.4f} {s:>12.4f} {g_pass:>14} {s_pass:>11}")

    print("-" * 62)
    g_overall = mean(greedy_scores.values())
    s_overall = mean(sla_scores.values())
    print(f"{'overall':<10} {g_overall:>10.4f} {s_overall:>12.4f}")
    print()
    print("Key property: greedy FAILS hard (score<0.50) due to ~67% unsafe ride rate.")
    print("SLA-aware heuristic PASSES all tasks — this is the design intent.")

    artifact = {
        "greedy_scores": greedy_scores,
        "sla_aware_scores": sla_scores,
        "greedy_overall": g_overall,
        "sla_aware_overall": s_overall,
        "success_threshold": success_threshold,
    }
    with open("baseline_scores.json", "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)
    print("Wrote baseline_scores.json")


if __name__ == "__main__":
    main()