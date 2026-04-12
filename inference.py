import sys
import pathlib

# Ensure the project root is on sys.path so local modules resolve correctly
# regardless of the working directory the evaluator uses.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import os
import json
import math
import time
from urllib.error import HTTPError, URLError
from urllib.request import urlopen
from typing import Any

from openai import OpenAI
from dotenv import load_dotenv

from tasks import TASKS

from client import RouteEnv
from models import RouteAction

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")

API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN")
if API_KEY is None:
    raise ValueError("API_KEY or HF_TOKEN environment variable is required")

LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
MAX_STEPS_PER_TRAJECTORY = int(os.getenv("MAX_STEPS_PER_TRAJECTORY", "125"))  # hard task = 120 steps; 5-step buffer so loop never cuts off a done=True step

USE_LLM_AGENT = os.getenv("USE_LLM_AGENT", "true").lower() in ("1", "true", "yes")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")


WAIT_FOR_SERVER_SECONDS = float(os.getenv("WAIT_FOR_SERVER_SECONDS", "90"))  # HF cold boot can take 60-90s
WAIT_FOR_SERVER_POLL_SECONDS = float(os.getenv("WAIT_FOR_SERVER_POLL_SECONDS", "0.5"))

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

# ── Task constants ─────────────────────────────────────────────────────────────
# late_penalty fires when: ride.wait_time > lateness_budget * 10
# Easy:   budget=0.45 → threshold 4.5  (wait 0-4 safe, 5-8 unsafe)
# Medium: budget=0.32 → threshold 3.2  (wait 0-3 safe, 4-8 unsafe)
# Hard:   budget=0.15 → threshold 1.5  (wait 0-1 safe, 2-8 unsafe ~75%)
LATENESS_BUDGETS: dict[str, float] = {
    "easy":   0.45,
    "medium": 0.32,
    "hard":   0.15,
}






def _zone_name(task_name: str, node_idx: int) -> str:
    """Return the human-readable zone name for a node index."""
    task = TASKS.get(task_name)
    if task and task.zone_names and node_idx < len(task.zone_names):
        return task.zone_names[node_idx]
    return f"zone_{node_idx}"


# ── Ride helpers ───────────────────────────────────────────────────────────────

def _late_threshold(task_name: str) -> float:
    return LATENESS_BUDGETS.get(task_name, 0.45) * 10.0


def _ride_is_safe(ride: dict, threshold: float) -> bool:
    return float(ride.get("wait_time", 0)) <= threshold


def _expected_ride_reward(ride: dict) -> float:
    """Reward for accepting this ride: fare/60 + 2*exp(-0.1*wait_time)."""
    fare = float(ride.get("fare", 0.0))
    wait = float(ride.get("wait_time", 0.0))
    return (fare / 60.0) + 2.0 * math.exp(-0.1 * wait)


# ── Heuristic fallback (used ONLY when LLM call fails) ────────────────────────

def choose_action_heuristic(observation: Any) -> tuple[RouteAction, str]:
    """SLA-aware greedy: best safe ride > reposition to highest demand > wait."""
    task_name = str(getattr(observation, "task_name", "easy"))
    threshold = _late_threshold(task_name)
    current_node = int(observation.current_node)
    all_rides = observation.available_rides or []
    demand = observation.live_demand_matrix or []

    adjacent = list(observation.adjacent_nodes or [])

    safe_rides = [r for r in all_rides if _ride_is_safe(r, threshold)]
    if safe_rides:
        best = max(safe_rides, key=_expected_ride_reward)
        return (
            RouteAction(action_type="accept_ride", ride_id=int(best["ride_id"])),
            f"accept_ride({best['ride_id']})",
        )

    supply = list(getattr(observation, "supply_pressure", []) or [])
    if demand and adjacent:
        # Score each adjacent node: demand minus supply pressure
        def _node_score(n: int) -> float:
            d = float(demand[n]) if n < len(demand) else 0.0
            s = float(supply[n]) if n < len(supply) else 0.0
            return d - s  # high demand, low competition = good
        best_node = max(adjacent, key=_node_score)
        curr_score = _node_score(current_node)
        if _node_score(best_node) > curr_score:
            return (
                RouteAction(action_type="reposition", target_node=int(best_node)),
                f"reposition({best_node})",
            )

    return RouteAction(action_type="wait"), "wait()"


# ── LLM agent ──────────────────────────────────────────────────────────────────

def choose_action_with_openllm(
    observation: Any,
    step_idx: int,
    prev_progress: float,
    cumulative_reward: float,
) -> tuple[RouteAction, str]:
    """LLM-driven policy with reconstructed adjacency.

    The LLM sees all rides (labelled safe/unsafe), all valid reposition options
    with demand context, and a recommended action. It must reason about the
    right tradeoff every step — not just rubber-stamp accept_ride.

    Post-parse safety: unsafe rides are never allowed through.
    Invalid reposition targets fall back to heuristic.
    """
    task_name = str(getattr(observation, "task_name", "easy"))
    threshold = _late_threshold(task_name)
    current_node = int(observation.current_node)
    all_rides = observation.available_rides or []
    demand = observation.live_demand_matrix or []
    supply = list(getattr(observation, "supply_pressure", []) or [])
    shift_remaining = float(observation.shift_hours_remaining)
    steps_left = MAX_STEPS_PER_TRAJECTORY - step_idx

    adjacent = list(observation.adjacent_nodes or [])

    # ── Annotate rides ──────────────────────────────────────────────────────
    annotated_rides = []
    for r in all_rides:
        safe = _ride_is_safe(r, threshold)
        exp_r = _expected_ride_reward(r) if safe else _expected_ride_reward(r) - 1.5
        annotated_rides.append({
            "ride_id":          r.get("ride_id"),
            "fare":             r.get("fare"),
            "wait_time":        r.get("wait_time"),
            "destination":      r.get("destination"),
            "origin_zone":      r.get("origin_zone", f"zone_{r.get('origin', '?')}"),
            "destination_zone": r.get("destination_zone", f"zone_{r.get('destination', '?')}"),
            "safe":             safe,
        })
    annotated_rides.sort(key=lambda x: (not x["safe"], -float(x.get("fare", 0))))

    safe_rides   = [r for r in annotated_rides if r["safe"]]
    unsafe_rides = [r for r in annotated_rides if not r["safe"]]

    # ── Reposition options ──────────────────────────────────────────────────
    curr_demand = float(demand[current_node]) if current_node < len(demand) else 0.0
    curr_demand_pct = round(curr_demand * 100, 1)

    reposition_options = []
    for node_id in adjacent:
        node_demand = float(demand[node_id]) if node_id < len(demand) else 0.0
        node_demand_pct = round(node_demand * 100, 1)
        reposition_options.append({
            "target_node":             node_id,
            "demand_pct":              node_demand_pct,
            "current_node_demand_pct": curr_demand_pct,
            "demand_higher_than_here": node_demand_pct > curr_demand_pct,
        })
    reposition_options.sort(key=lambda x: -x["demand_pct"])

    payload = {
        "step":                   step_idx,
        "steps_remaining":        steps_left,
        "task_name":              task_name,
        "late_penalty_threshold": threshold,
        "current_node":           current_node,
        "current_node_demand_pct": curr_demand_pct,
        "current_zone": str(getattr(observation, "current_zone", "Unknown")),
        "adjacent_nodes":         adjacent,
        "driver_status":          str(observation.driver_status),
        "shift_hours_remaining":  round(shift_remaining, 2),
        "shift_pressure":         shift_remaining < 1.5,
        "normalized_progress":    round(prev_progress, 3),
        "cumulative_reward":      round(cumulative_reward, 3),
        "last_action_error":      observation.last_action_error,
        "safe_rides":             safe_rides,
        "unsafe_rides_count":     len(unsafe_rides),
        "reposition_options":     reposition_options,
        "supply_pressure_by_zone": [
            {
                "zone_index": i,
                "zone_name": _zone_name(task_name, i),
                "supply_pressure": round(float(supply[i]), 3) if i < len(supply) else 0.0,
                "demand": round(float(demand[i]), 3) if i < len(demand) else 0.0,
                "net_opportunity": round(
                    (float(demand[i]) if i < len(demand) else 0.0) -
                    (float(supply[i]) if i < len(supply) else 0.0),
                    3
                ),
            }
            for i in range(len(demand))
        ],
    }

    system_prompt = """You are an expert ride-dispatch RL agent. Every step choose ONE action to maximise total episode score.

REWARD FUNCTION:
  accept_ride(safe):   +fare/60 + 2*exp(-0.1*wait_time)    → always positive (~0.3 to 2.8)
  accept_ride(unsafe): same formula MINUS 1.5 penalty       → almost always negative — NEVER do this
  reposition(node):    typically costs ~0.25–0.37 total      → -0.9*(edge_dist/20) where edge_dist≈2-5, plus -0.15 flat tax; worthwhile to reach safe rides
  wait:                0 reward                              → free but earns nothing

DECISION RULES — strict priority order:

1. safe_rides non-empty → accept_ride from safe_rides. Choose the ride with the
   highest value: prefer high fare AND low wait_time. Use the formula fare/60 + 2*exp(-0.1*wait_time)
   mentally to rank options. NEVER pick from unsafe_rides — the -1.5 penalty makes them net negative.

1b. When choosing which safe ride to accept, also consider spatial value: prefer rides where
    origin_zone and destination_zone form a high-value corridor
    (e.g. Airport->Downtown, Stadium->Downtown). Reason from zone names, fare, and wait_time.

2. safe_rides EMPTY + any reposition_option has demand_higher_than_here=true
   → reposition to the option with highest demand_pct.
   Paying ~0.30 now to reach a higher-demand node (more safe rides spawn there) is worth it.
   target_node MUST be from adjacent_nodes list — any other value gives a -0.6 error.

2b. When choosing where to reposition, prefer zones with HIGH net_opportunity
    (demand minus supply_pressure). A zone with high demand but also many
    competitor drivers will yield fewer rides — factor in both signals.
    current_zone tells you where you are by name (e.g. "Airport") so you
    can reason spatially about city geography.

3. safe_rides EMPTY + no demand_higher_than_here=true → wait. Rides will spawn next step.

4. shift_pressure=true → accept any safe ride immediately; skip repositioning.

5. last_action_error not null → do NOT repeat that action_type.

Output ONLY a JSON object — no markdown, no text outside the JSON.
Keys: action_type (wait|accept_ride|reposition), ride_id (int or null), target_node (int or null), reasoning (one sentence).
Example: {"action_type": "reposition", "ride_id": null, "target_node": 4, "reasoning": "No safe rides at node 2; node 4 has higher demand"}"""

    user_prompt = f"State:\n{json.dumps(payload, indent=2)}\n\nWhat action do you take?"

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=256,
        )
        raw = (resp.choices[0].message.content or "").strip()
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            raw = raw[start: end + 1]
        data = json.loads(raw)

        action_type = str(data.get("action_type", "wait"))
        ride_id     = data.get("ride_id")
        target_node = data.get("target_node")

        # ── accept_ride ──────────────────────────────────────────────────────
        if action_type == "accept_ride" and ride_id is not None:
            safe_ids = {int(r["ride_id"]) for r in safe_rides}
            if int(ride_id) in safe_ids:
                return (
                    RouteAction(action_type="accept_ride", ride_id=int(ride_id)),
                    f"accept_ride({int(ride_id)})",
                )
            # LLM chose unsafe ride — correct to best safe ride
            if safe_rides:
                best_safe = safe_rides[0]
                return (
                    RouteAction(action_type="accept_ride", ride_id=int(best_safe["ride_id"])),
                    f"accept_ride({int(best_safe['ride_id'])})",
                )
            # No safe rides at all — let heuristic decide reposition/wait
            return choose_action_heuristic(observation)

        # ── reposition ───────────────────────────────────────────────────────
        if action_type == "reposition" and target_node is not None:
            if int(target_node) in adjacent:
                return (
                    RouteAction(action_type="reposition", target_node=int(target_node)),
                    f"reposition({int(target_node)})",
                )
            # LLM chose non-adjacent node — fall back to heuristic
            return choose_action_heuristic(observation)

        # ── wait ─────────────────────────────────────────────────────────────
        return RouteAction(action_type="wait"), "wait()"

    except Exception:
        return choose_action_heuristic(observation)


# ── Trajectory runner ──────────────────────────────────────────────────────────

def run_trajectory(
    env: RouteEnv,
    trajectory_idx: int,
    task_name: str = "easy",
) -> tuple[bool, int, float]:
    model_name = MODEL_NAME
    raw_rewards: list[str] = []
    prev_progress = 0.0
    final_score_set = False
    final_score = 0.0
    success = False
    step_idx = 0
    cumulative_reward = 0.0
    actual_task_name = task_name
    start_printed = False

    try:
        with env.sync() as env_client:
            reset_result = env_client.reset(task_name=task_name)
            observation = reset_result.observation
            actual_task_name = getattr(observation, "task_name", task_name)
            print(
                f"[START] task={actual_task_name} env=route_env_improved model={model_name}",
                flush=True,
            )
            start_printed = True

            while step_idx < MAX_STEPS_PER_TRAJECTORY:
                step_idx += 1

                if USE_LLM_AGENT:
                    action, action_str = choose_action_with_openllm(
                        observation, step_idx, prev_progress, cumulative_reward
                    )
                else:
                    action, action_str = choose_action_heuristic(observation)

                result = env_client.step(action)
                observation = result.observation

                raw_reward = 0.0 if result.reward is None else float(result.reward)
                progress   = float(observation.normalized_progress_score or 0.0)

                prev_progress     = progress
                cumulative_reward += raw_reward
                raw_rewards.append(f"{raw_reward:.2f}")
                done  = bool(result.done)
                error = observation.last_action_error if observation.last_action_error else "null"

                print(
                    f"[STEP] step={step_idx} action={action_str} reward={raw_reward:.2f} "
                    f"done={'true' if done else 'false'} error={error}",
                    flush=True,
                )

                if done:
                    success = progress >= 0.5
                    final_score = progress
                    final_score_set = True
                    break

            # If done never fired, use last observed progress score
            if not final_score_set and prev_progress > 0.0:
                final_score = prev_progress

        if not success and step_idx >= MAX_STEPS_PER_TRAJECTORY:
            success = False

    except BaseException as e:
        # Catches Exception AND KeyboardInterrupt/SystemExit so finally always
        # has an up-to-date final_score. Re-raise so Ctrl+C still exits cleanly.
        if not isinstance(e, (KeyboardInterrupt, SystemExit)):
            print(f"Error during execution: {e}", file=sys.stderr)
        success = False
        raise

    finally:
        # Always promote prev_progress → final_score; covers mid-episode interrupts.
        # [END] must always be emitted — even on exception — because it is in finally.
        if prev_progress > final_score:
            final_score = prev_progress
        if not start_printed:
            print(
                f"[START] task={actual_task_name} env=route_env_improved model={model_name}",
                flush=True,
            )
        print(
            f"[END] success={'true' if success else 'false'} steps={step_idx} "
            f"rewards={','.join(raw_rewards)}",
            flush=True,
        )

    return success, step_idx, cumulative_reward


# ── Episode runner ─────────────────────────────────────────────────────────────

def run_episode() -> None:
    task_names = ["easy", "medium", "hard"]

    deadline = time.time() + WAIT_FOR_SERVER_SECONDS
    while time.time() < deadline:
        try:
            with urlopen(f"{ENV_BASE_URL}/health", timeout=2) as resp:
                if getattr(resp, "status", 200) == 200:
                    break
        except (URLError, HTTPError):
            pass
        time.sleep(WAIT_FOR_SERVER_POLL_SECONDS)

    for trajectory_idx, task_name in enumerate(task_names, 1):
        env = (
            RouteEnv.from_docker_image(LOCAL_IMAGE_NAME)
            if LOCAL_IMAGE_NAME
            else RouteEnv(base_url=ENV_BASE_URL)
        )
        try:
            run_trajectory(env, trajectory_idx, task_name=task_name)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            print(f"Task '{task_name}' failed with unhandled exception: {e}", file=sys.stderr)


if __name__ == "__main__":
    run_episode()