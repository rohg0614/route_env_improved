---
title: Route Env Environment Server
emoji: 🚕
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
app_port: 7860
base_path: /web
tags:
  - openenv
---

# Route Dispatch RL Environment

Ride-hailing platforms make tens of millions of dispatch decisions every day. Each one is a multi-objective tradeoff: take the nearby low-fare ride, or reposition to a busier zone? Accept a fast pickup, or skip the late one and take the SLA penalty? No open RL benchmark has captured this core operations problem — until now.

`route_env` is a stochastic graph environment where an agent controls a single driver across a shift. Every step, the agent must balance immediate fare revenue against positional value, SLA compliance, and competitor supply pressure. A pure greedy policy fails the hard task by design. An agent that reasons about wait times, demand patterns, and competitor clustering passes.

## What makes this environment challenging

The hard task is the key differentiator. A greedy always-accept agent scores approximately 0.36 — well below the 0.5 success threshold — because roughly 67% of randomly generated rides breach the tight SLA budget (`lateness_budget=0.15`). Only an agent that actively reads `wait_time` and skips high-latency rides passes. This forces genuine reasoning, not pattern matching.

Beyond SLA compliance, the agent must reason about:

- non-stationary demand (Negative Binomial arrivals with time-of-day peaks),
- surge pricing (fares scale with local demand density),
- competitor supply pressure (other drivers chase high-demand zones, creating real scarcity),
- ride TTL (rides expire, creating genuine urgency),
- reposition cost vs. expected value tradeoff (adjacent-node moves cost ~0.30 reward).

## Why this matters for evaluating agents

This environment tests a class of reasoning that LLMs frequently fail at: sequential decisions under uncertainty where the greedy action is the wrong one. The gap between a naive policy (0.36 on hard) and an SLA-aware policy (0.64+ on hard) is large, consistent, and directly attributable to one cognitive capability — reading a single field (`wait_time`) and conditionally skipping an action. This makes `route_env` a meaningful signal for agent evaluation, not just a benchmark that separates random from non-random.

## Action and Observation Spaces

### Action (`RouteAction`)
- `action_type`: `wait | accept_ride | reposition`
- `ride_id`: required when accepting a ride
- `target_node`: required when repositioning (must be adjacent)

### Observation (`RouteObservation`)
- `task_name`: `easy | medium | hard`
- `current_node`: integer node ID of driver's current position
- `current_zone`: human-readable zone name (e.g. `"Airport"`, `"Downtown"`)
- `time_of_day_sin`, `time_of_day_cos`: cyclical hour-of-day encoding
- `driver_status`: `idle | busy | en_route`
- `shift_hours_remaining`: float hours left in the shift
- `live_demand_matrix`: per-node ride demand intensity (sums to 1.0)
- `supply_pressure`: per-node competitor driver density (sums to 1.0)
- `adjacent_nodes`: list of node IDs valid for `reposition` actions
- `available_rides`: list of ride dicts with `fare`, `wait_time`, `destination`, `origin_zone`, `destination_zone`
- `last_action_error`: error string if the previous action was invalid, else `null`
- `normalized_progress_score`: grader output in `(0.0, 1.0)` updated every step

## City Zone Model

Each task models a real city topology with named zones connected by a
fixed road network. Zone demand weights make high-traffic zones
realistically busier than residential zones.

### Easy — 6-Zone Downtown Core

Zones: Airport, Downtown, University, Hospital, Stadium, Suburbs

Connections: Airport-Downtown, Downtown-University, University-Hospital,
Hospital-Stadium, Stadium-Suburbs, Suburbs-Airport, Airport-University,
Downtown-Stadium

Demand weights: Airport (2.0x), Downtown (1.8x), Stadium (1.5x),
University (1.2x), Hospital (1.0x), Suburbs (0.8x)

### Medium — 8-Zone Mid-Size City

Zones: Airport, Downtown, Financial, University, Hospital, Stadium, Suburbs, Mall

Demand weights: Airport (2.0x), Downtown (1.8x), Financial (1.6x),
Stadium (1.5x), University (1.2x), Mall (1.1x), Hospital (1.0x), Suburbs (0.7x)

### Hard — 12-Zone Metro Area

Zones: Airport, Terminal2, Downtown, Financial, Midtown, University,
Hospital, Stadium, Convention, Suburbs_N, Suburbs_S, Mall

Demand weights: Airport (2.2x), Downtown (2.0x), Terminal2 (1.9x),
Financial (1.7x), Stadium (1.8x), Midtown (1.4x), Convention (1.3x),
University (1.2x), Mall (1.1x), Hospital (1.0x), Suburbs_N (0.7x), Suburbs_S (0.6x)

## Competitor Supply Pressure

Each episode spawns one competitor driver per zone. Competitors
independently chase high-demand zones each step with 40% probability,
creating realistic supply clustering that varies across the shift.

The `supply_pressure` observation field shows the fraction of competitor
drivers at each zone (sums to 1.0). A zone with high demand but also
many competitors yields fewer available rides.

Optimal strategy: reposition to zones with high `live_demand_matrix`
AND low `supply_pressure`. Computing `live_demand_matrix[node] - supply_pressure[node]` gives the net opportunity for each zone — the agent should prefer zones where demand exceeds competitor density.

## Ride Zone Information

Each ride in `available_rides` includes:
- `origin_zone`: human-readable pickup zone (e.g. "Airport")
- `destination_zone`: human-readable dropoff zone (e.g. "Downtown")
- `wait_time`: passenger wait time — exceeding the task lateness
  threshold triggers a -1.5 reward penalty
- `fare`: base fare modified by real-time surge pricing based on
  local demand density

The LLM agent can reason about zone corridors spatially:
"Airport to Downtown is a high-value corridor" rather than "node 0 to node 1".

## Reward Function

Per-step reward formula:

R_t = (fare/60) - 0.9*(empty_distance/20) - 0.15*did_reposition
      + 2.0*exp(-0.1*wait_time)*completed_ride - 1.5*late_penalty

Components:
- Profit engine: fare normalized to [0, 1]
- Operational burn: penalty for empty miles driven
- Action tax: small friction per reposition to discourage jitter
- Urgency bonus: exponentially decaying with wait_time — fast pickups pay more
- SLA enforcer: -1.5 for any ride exceeding lateness budget

## Tasks and Difficulty Range

Task presets are explicit in `tasks.py`:
- `easy`
- `medium`
- `hard`

Each task changes graph size, horizon, demand intensity, and constraints.

The environment supports:
- default cyclic task selection on reset, and
- explicit task selection: `reset(task_name="hard", seed=123)`.

## Programmatic Grader (`[0.0, 1.0]`)

The grader is explicit and deterministic in `grader.py`:
- input: `step_count`, `completed_rides`, `late_rides`, `total_reward`
- output: bounded scalar score in `[0.0, 1.0]`

The grader score is computed every step internally and exposed as
normalized_progress_score in the observation for logging purposes.
The inference agent does not use this field for decision-making.

## Determinism and Reproducibility

- Global deterministic seed via env var: `SEED` (default `42`)
- Optional per-episode seed override: `reset(seed=...)`
- Baselines can produce reproducible scores by fixing seeds.

## Baseline Scripts

### 1) Hackathon inference script (OpenAI client + required output format)
`inference.py`

Supports:
- `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` / `API_KEY`
- optional heuristic-only mode: `USE_LLM_AGENT=false`
- configurable server wait: `WAIT_FOR_SERVER_SECONDS` (default 90)
- configurable max steps: `MAX_STEPS_PER_TRAJECTORY` (default 125)

### 2) Deterministic benchmark script
`baseline_benchmark.py`

Runs two policies — a naive greedy and an SLA-aware heuristic — across all three tasks to illustrate the environment's core design property.

```bash
# Reproduce baseline scores locally (requires running server on :7860)
ENV_BASE_URL=http://localhost:7860 python baseline_benchmark.py
```

Expected output:
```
Baseline benchmark (score range 0.0–1.0, success threshold 0.50)
Task          Greedy    SLA-aware   Greedy pass?   SLA pass?
--------------------------------------------------------------
easy          0.7600       0.7800           PASS        PASS
medium        0.6800       0.7100           PASS        PASS
hard          0.3600       0.6400           FAIL        PASS
--------------------------------------------------------------
overall       0.6000       0.7100

Key property: greedy FAILS hard (score<0.50) due to ~67% unsafe ride rate.
SLA-aware heuristic PASSES all tasks — this is the design intent.
```

The hard task baseline of **0.36** for greedy is intentional. A pure greedy always-accept agent fails because ~67% of rides breach the SLA on the hard task (`lateness_budget=0.15`, threshold=1.5 wait steps). The SLA-aware heuristic skips those rides and scores **0.64+**, clearing the 0.5 threshold. This score gap is the environment's primary evaluation signal.

### 3) `openenv validate` output

```
openenv validate --verbose
Supported deployment modes:
  [YES] docker
  [YES] openenv_serve
  [YES] uv_run
  [YES] python_module

Usage examples:
  cd route_env-improved && uv run server
  cd route_env-improved && openenv build
  cd route_env-improved && openenv push
```

## Judge Runbook (Strict)

Use these exact commands for strict validation:

```bash
# 1) Validate spec
openenv validate --verbose

# 2) Start docker env
docker build -t route_env-env:latest .
docker rm -f route_env_local || true
docker run -d --name route_env_local -p 7860:7860 route_env-env:latest

# 3) Reproducible benchmark artifact
SEED=42 ENV_BASE_URL=http://localhost:7860 python baseline_benchmark.py

# 4) Strict baseline inference (single episode output contract)
ENV_BASE_URL=http://localhost:7860 python inference.py
```

## Setup

```bash
uv sync
openenv validate --verbose
```

## Local run (Docker)

```bash
docker build -t route_env-env:latest .
docker run -d --name route_env_local -p 7860:7860 route_env-env:latest
```

Endpoints:
- Web UI: `http://localhost:7860/web/`
- Docs: `http://localhost:7860/docs`
- Health: `http://localhost:7860/health`

## Hugging Face deployment

```bash
openenv push --repo-id <username/space-name>
```

Space metadata:
- docker sdk
- app port `7860`
- web base path `/web`

## Project Structure

```text
route_env/ (Root)
├── Dockerfile            # Root-level build config
├── pyproject.toml        # Dependency & package config
├── openenv.yaml          # Validator spec
├── inference.py          # LLM-driven baseline agent
├── client.py             # Environment client logic
├── models.py             # Shared data schemas
├── tasks.py              # Difficulty presets
├── grader.py             # Deterministic scoring logic
└── server/               # Environment server implementation
    ├── app.py            # FastAPI entry point
    └── route_env_environment.py