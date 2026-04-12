#!/usr/bin/env python3
"""
validate.py — Pre-submission validation script for route_env.
============================================================
Run with the server already started on localhost:7860.

Usage:
    # Terminal 1:
    uvicorn server.app:app --host 0.0.0.0 --port 7860

    # Terminal 2:
    python validate.py
    python validate.py --server http://localhost:7860
    python validate.py --server https://your-space.hf.space
"""
import sys
import json
import argparse
import requests

TASKS = ["easy", "medium", "hard"]
PASS = "✅ PASS"
FAIL = "❌ FAIL"


def check(label: str, result: bool, detail: str = "") -> bool:
    status = PASS if result else FAIL
    line = f"  {status}  {label}"
    if detail:
        line += f"  ({detail})"
    print(line)
    return result


def get(server: str, path: str, timeout: int = 15) -> requests.Response:
    return requests.get(f"{server}{path}", timeout=timeout)


def post(server: str, path: str, payload: dict, timeout: int = 15) -> requests.Response:
    return requests.post(f"{server}{path}", json=payload, timeout=timeout)


def run_validation(server: str) -> bool:
    results = []

    print(f"\n{'='*60}")
    print(f"  route_env Pre-Submission Validator")
    print(f"  Target: {server}")
    print(f"{'='*60}\n")

    # ── 1. Health Check ────────────────────────────────────────────────────────
    print("📡  [1/6] Health Check")
    try:
        r = get(server, "/health")
        ok = r.status_code == 200
        data = r.json() if ok else {}
        results.append(check("Server responds with HTTP 200", ok))
        results.append(check("Health status is ok", data.get("status") == "ok",
                       f"status={data.get('status')}"))
    except requests.ConnectionError:
        print(f"  {FAIL}  Cannot connect to {server} — is the server running?")
        sys.exit(1)

    # ── 2. Spec Endpoint ──────────────────────────────────────────────────────
    print("\n📋  [2/6] Spec Endpoint (/spec)")
    try:
        r = get(server, "/spec")
        spec = r.json() if r.status_code == 200 else {}
        results.append(check("/spec returns HTTP 200", r.status_code == 200))
        results.append(check("spec has tasks", len(spec.get("tasks", [])) >= 3,
                       f"found={len(spec.get('tasks', []))}"))
        results.append(check("spec has name", bool(spec.get("name"))))
        results.append(check("spec has version", bool(spec.get("version"))))
        results.append(check("spec has observation_space", bool(spec.get("observation_space"))))
        results.append(check("spec has action_space", bool(spec.get("action_space"))))
    except Exception as e:
        results.append(check("/spec returns HTTP 200", False, str(e)))

    # ── 3. Reset ──────────────────────────────────────────────────────────────
    print("\n🔄  [3/6] Reset Endpoint")
    for task in TASKS:
        try:
            r = post(server, "/reset", {"task_name": task})
            ok = r.status_code == 200
            data = r.json() if ok else {}
            obs = data.get("observation", {})
            results.append(check(f"reset({task}) returns 200", ok))
            results.append(check(f"reset({task}) has observation", bool(obs)))
        except Exception as e:
            results.append(check(f"reset({task}) returns 200", False, str(e)))

    # ── 4. Step ───────────────────────────────────────────────────────────────
    print("\n👟  [4/6] Step Endpoint")
    for task in TASKS:
        try:
            post(server, "/reset", {"task_name": task})
            wait_action = {"action_type": "wait", "ride_id": None, "target_node": None}
            r = post(server, "/step", {"action": wait_action})
            ok = r.status_code == 200
            data = r.json() if ok else {}
            results.append(check(f"step({task}, wait) returns 200", ok))
            reward = data.get("reward")
            results.append(check(f"step({task}) reward in (-10, 10)",
                           reward is not None and -10 < float(reward) < 10,
                           f"reward={reward}"))
            done = data.get("done")
            results.append(check(f"step({task}) done is bool",
                           isinstance(done, bool), f"done={done}"))
        except Exception as e:
            results.append(check(f"step({task}) returns 200", False, str(e)))

    # ── 5. State ──────────────────────────────────────────────────────────────
    print("\n🗂   [5/6] State Endpoint")
    for task in TASKS:
        try:
            post(server, "/reset", {"task_name": task})
            r = get(server, "/state")
            ok = r.status_code == 200
            results.append(check(f"state({task}) returns 200", ok))
        except Exception as e:
            results.append(check(f"state({task}) returns 200", False, str(e)))

    # ── 6. Grader ─────────────────────────────────────────────────────────────
    print("\n🏆  [6/6] Grader Endpoint")
    for task in TASKS:
        try:
            r = post(server, "/grader", {
                "task_name": task,
                "step_count": 30,
                "completed_rides": 5,
                "late_rides": 1,
                "total_reward": 3.5,
            })
            ok = r.status_code == 200
            data = r.json() if ok else {}
            score = data.get("score")
            results.append(check(f"grader({task}) returns 200", ok))
            results.append(check(f"grader({task}) score in (0.0, 1.0)",
                           score is not None and 0.0 < float(score) < 1.0,
                           f"score={score}"))
        except Exception as e:
            results.append(check(f"grader({task}) returns 200", False, str(e)))

    # ── Summary ───────────────────────────────────────────────────────────────
    passed = sum(results)
    total = len(results)
    print(f"\n{'='*60}")
    print(f"  Result: {passed}/{total} checks passed")
    if passed == total:
        print(f"  🎉 ALL CHECKS PASSED — ready to submit!")
    else:
        print(f"  ⚠️  {total - passed} check(s) failed — fix before submitting")
    print(f"{'='*60}\n")
    return passed == total


def main():
    parser = argparse.ArgumentParser(description="route_env Pre-Submission Validator")
    parser.add_argument(
        "--server",
        default="http://localhost:7860",
        help="OpenEnv server URL (default: http://localhost:7860)",
    )
    args = parser.parse_args()
    success = run_validation(args.server)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
