# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Route dispatch optimization environment implementation."""

import math
import os
import random
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from models import RouteAction, RouteObservation
from tasks import TASKS, TASK_ORDER, TaskConfig
from grader import score_episode


class RouteEnvironment(Environment):
    """Graph-based stochastic dispatch environment with dense rewards.

    Demand model improvements over a naive Poisson ring:
    - Negative Binomial arrivals (Gamma-Poisson mixture, r=3): overdispersed
      demand produces realistic boom/bust cycles that a greedy always-accept
      policy cannot exploit uniformly.
    - Surge pricing: fare is multiplied by a demand-density factor derived
      from the node's share of total active rides. High-demand nodes pay more,
      making the reposition-vs-accept tradeoff economically meaningful.
    - Ride TTL: rides expire after (lateness_budget * 20) steps, creating
      genuine scarcity and forcing the agent to act under time pressure rather
      than always finding a queue of stale rides.

    Graph topology: ring augmented with chord shortcuts (floor(n/3) and
    2*floor(n/4) hops) giving each node up to 4 neighbours.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    # Negative Binomial dispersion parameter.
    # r=3 → variance ≈ 2× mean; lower r = burstier demand.
    _NB_DISPERSION: float = 3.0

    # Surge pricing sensitivity. At 1.0, a node holding 100% of demand
    # pays double the base fare. Calibrated so surge never exceeds max_fare.
    _SURGE_SENSITIVITY: float = 0.8

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._rng = random.Random()
        self._task_idx = -1
        self._tasks = [TASKS[name] for name in TASK_ORDER]
        self._base_seed = int(os.getenv("SEED", "42"))
        self._max_fare = 60.0
        self._max_distance = 20.0
        self._reset_internal()

    def _reset_internal(self) -> None:
        self._task = self._tasks[max(self._task_idx, 0)]
        self._driver_node = 0
        self._driver_status = "idle"
        self._sim_hour = 6.0
        self._shift_hours_remaining = self._task.max_shift_hours
        self._rides: list[dict] = []
        self._next_ride_id = 1
        self._idle_wait_steps = 0
        self._total_reward = 0.0
        self._completed_rides = 0
        self._late_rides = 0
        self._total_fare = 0.0
        self._empty_distance = 0.0
        # TTL: rides expire after this many steps (scales with task difficulty).
        # Tighter lateness_budget → shorter TTL → higher scarcity.
        self._ride_ttl = max(3, int(self._task.lateness_budget * 20))
        # Competitor drivers: one per zone on average, randomly placed.
        # They chase demand each step, creating realistic supply pressure.
        n_competitors = self._task.node_count
        self._competitor_positions: list[int] = [
            self._rng.randrange(self._task.node_count)
            for _ in range(n_competitors)
        ]
        self._build_graph()
        self._spawn_rides()

    # ── Graph ─────────────────────────────────────────────────────────────────

    def _build_graph(self) -> None:
        """Fixed city-zone topology defined in TaskConfig.zone_edges.

        Each edge in zone_edges becomes a bidirectional connection.
        Edge distances are randomised once per episode reset.
        """
        n = self._task.node_count
        self._adjacency: dict[int, list[int]] = {i: [] for i in range(n)}
        self._edge_distance: dict[tuple[int, int], float] = {}

        def _add_edge(i: int, j: int) -> None:
            if i == j:
                return
            if j not in self._adjacency[i]:
                self._adjacency[i].append(j)
            if i not in self._adjacency[j]:
                self._adjacency[j].append(i)
            if (i, j) not in self._edge_distance:
                d = round(2.0 + abs(i - j) * 0.6 + self._rng.random() * 1.5, 3)
                self._edge_distance[(i, j)] = d
                self._edge_distance[(j, i)] = d

        for i, j in self._task.zone_edges:
            _add_edge(i, j)

    # ── Demand model ──────────────────────────────────────────────────────────

    def _negbin(self, mu: float) -> int:
        """Negative Binomial sample via Gamma-Poisson mixture.

        Variance = mu + mu²/r ≈ 2*mu for r=3, producing overdispersed
        arrivals that mimic real-world ride demand clustering.
        """
        if mu <= 0:
            return 0
        r = self._NB_DISPERSION
        # Draw the Poisson rate from a Gamma prior (the NB mixing distribution).
        lam = self._rng.gammavariate(r, mu / r)
        # Draw Poisson arrivals given that rate (Knuth algorithm).
        k, p, threshold = 0, 1.0, math.exp(-lam)
        while p > threshold and k < 200:
            k += 1
            p *= self._rng.random()
        return max(0, k - 1)

    def _node_intensity(self, node: int) -> float:
        """Expected arrivals per step at this node given time-of-day."""
        peak = 1.0 + 0.6 * math.sin((self._sim_hour - 8.0) * math.pi / 12.0)
        if self._task.zone_demand_weights and node < len(self._task.zone_demand_weights):
            locality = self._task.zone_demand_weights[node]
        else:
            locality = 0.8 + (node / max(1, self._task.node_count - 1)) * 0.5
        return max(0.1, self._task.base_lambda * peak * locality)

    def _surge_multiplier(self, origin_node: int) -> float:
        """Demand-density surge multiplier for a ride originating at origin_node.

        Computes the node's share of total active rides and scales fare up
        proportionally. A node holding 30% of all active rides at a
        sensitivity of 0.8 pays 1 + 0.8*2.0 = 2.6× base fare (capped by
        max_fare downstream).
        """
        total = max(1, len(self._rides))
        node_count = sum(1 for r in self._rides if r["origin"] == origin_node)
        demand_share = node_count / total
        # Normalise by the uniform expectation (1/n_nodes) so that a node
        # at exactly average demand gets multiplier = 1.0.
        n = self._task.node_count
        relative_demand = demand_share * n  # 1.0 = average, 2.0 = double average
        return 1.0 + self._SURGE_SENSITIVITY * (relative_demand - 1.0)

    def _spawn_rides(self) -> None:
        """Spawn new rides and expire stale ones (TTL-based)."""
        current_step = self._state.step_count

        # Expire rides older than TTL before computing surge (surge based on
        # active rides only, not stale ones).
        self._rides = [r for r in self._rides if current_step - r["birth_step"] < self._ride_ttl]

        for node in range(self._task.node_count):
            arrivals = self._negbin(self._node_intensity(node))
            for _ in range(arrivals):
                destination = self._rng.randrange(0, self._task.node_count)
                while destination == node:
                    destination = self._rng.randrange(0, self._task.node_count)

                estimated_distance = self._task.distance_scale * (
                    0.25 + abs(destination - node) / self._task.node_count
                )
                # Base fare + surge multiplier capped at max_fare.
                base_fare = 7.0 + estimated_distance * (1.3 + self._rng.random())
                surge = max(0.5, self._surge_multiplier(node))  # floor at 0.5
                fare = min(self._max_fare, base_fare * surge)
                wait_time = self._rng.randint(0, 8)

                self._rides.append({
                    "ride_id": self._next_ride_id,
                    "origin": node,
                    "destination": destination,
                    "fare": round(fare, 2),
                    "distance": round(estimated_distance, 2),
                    "wait_time": wait_time,
                    "birth_step": current_step,
                    "origin_zone": self._task.zone_names[node] if self._task.zone_names else f"zone_{node}",
                    "destination_zone": self._task.zone_names[destination] if self._task.zone_names else f"zone_{destination}",
                })
                self._next_ride_id += 1

    # ── Observation helpers ───────────────────────────────────────────────────

    def _live_demand(self) -> list[float]:
        counts = [0] * self._task.node_count
        for ride in self._rides:
            counts[ride["origin"]] += 1
        total = max(1, sum(counts))
        return [round(c / total, 4) for c in counts]

    def _supply_pressure(self) -> list[float]:
        """Fraction of competitor drivers at each zone.

        Sums to 1.0. A zone with high supply_pressure relative to its
        live_demand_matrix value is oversupplied — the agent should avoid it.
        """
        counts = [0] * self._task.node_count
        for pos in self._competitor_positions:
            counts[pos] += 1
        total = max(1, len(self._competitor_positions))
        return [round(c / total, 4) for c in counts]

    def _bfs_distance(self, source: int, target: int) -> int:
        """Return the shortest hop count from source to target in the current graph."""
        if source == target:
            return 0
        visited = {source}
        queue = [(source, 0)]
        while queue:
            node, dist = queue.pop(0)
            for neighbor in self._adjacency.get(node, []):
                if neighbor == target:
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        return 999  # unreachable (graph should always be connected)

    def _move_competitors(self) -> None:
        """Move competitor drivers toward high-demand zones each step.

        Each competitor independently moves to an adjacent node that is
        closer to the current highest-demand zone with 40% probability,
        or stays put with 60% probability. This creates realistic supply
        clustering that the agent must reason about.
        """
        demand = self._live_demand()
        best_node = max(range(self._task.node_count), key=lambda n: demand[n])
        new_positions = []
        for pos in self._competitor_positions:
            if self._rng.random() < 0.4:
                neighbors = self._adjacency.get(pos, [])
                if neighbors:
                    next_node = min(neighbors, key=lambda n: self._bfs_distance(n, best_node))
                    new_positions.append(next_node)
                    continue
            new_positions.append(pos)
        self._competitor_positions = new_positions

    def _rides_at_current_node(self) -> list[dict]:
        return [r for r in self._rides if r["origin"] == self._driver_node][:8]

    def _advance_time(self, step_hours: float = 5 / 60) -> None:
        self._sim_hour = (self._sim_hour + step_hours) % 24.0
        self._shift_hours_remaining = max(0.0, self._shift_hours_remaining - step_hours)
        self._state.step_count += 1
        self._spawn_rides()
        self._move_competitors()

    def _grader_score(self) -> float:
        return score_episode(
            step_count=self._state.step_count,
            completed_rides=self._completed_rides,
            late_rides=self._late_rides,
            total_reward=self._total_reward,
            task_name=self._task.name,
        )

    def _build_observation(
        self, reward: float, done: bool, last_action_error: str | None
    ) -> RouteObservation:
        theta = 2 * math.pi * (self._sim_hour / 24.0)
        adjacent_nodes = sorted(self._adjacency.get(self._driver_node, []))
        return RouteObservation(
            task_name=self._task.name,
            current_node=self._driver_node,
            time_of_day_sin=round(math.sin(theta), 6),
            time_of_day_cos=round(math.cos(theta), 6),
            driver_status=self._driver_status,
            shift_hours_remaining=round(self._shift_hours_remaining, 3),
            live_demand_matrix=self._live_demand(),
            available_rides=self._rides_at_current_node(),
            current_zone=self._task.zone_names[self._driver_node] if self._task.zone_names else "Unknown",
            supply_pressure=self._supply_pressure(),
            adjacent_nodes=adjacent_nodes,
            last_action_error=last_action_error,
            normalized_progress_score=round(_grader_score_now := self._grader_score(), 4),
            done=done,
            reward=round(reward, 4),
            metadata={
                "task_horizon": self._task.horizon_steps,
                "completed_rides": self._completed_rides,
                "late_rides": self._late_rides,
                "total_fare": round(self._total_fare, 2),
                "empty_distance": round(self._empty_distance, 2),
                "grader_score_0_to_1": round(_grader_score_now, 4),
                "adjacent_nodes": adjacent_nodes,
                "ride_ttl_steps": self._ride_ttl,
                "active_rides": len(self._rides),
                "current_zone": self._task.zone_names[self._driver_node] if self._task.zone_names else "Unknown",
                "competitor_count": len(self._competitor_positions),
                "supply_pressure": self._supply_pressure(),
            },
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def reset(
        self,
        task_name: str | None = None,
        seed: int | None = None,
    ) -> RouteObservation:
        if task_name is not None and task_name in TASKS:
            self._task_idx = TASK_ORDER.index(task_name)
        else:
            self._task_idx = (self._task_idx + 1) % len(self._tasks)
        self._state = State(episode_id=str(uuid4()), step_count=0)
        if seed is not None:
            # Explicit seed → deterministic (used by baseline_benchmark.py)
            effective_seed = seed
        else:
            # Add microsecond-level entropy so every evaluation run sees a
            # different episode, preventing trajectory memorisation.
            import time as _time
            entropy = int(_time.time() * 1_000_000) % 999_983  # large prime mod
            effective_seed = self._base_seed + self._task_idx + entropy
        self._rng.seed(effective_seed)
        self._reset_internal()
        return self._build_observation(reward=0.0, done=False, last_action_error=None)

    def step(self, action: RouteAction) -> RouteObservation:
        reward = 0.0
        last_action_error: str | None = None
        did_reposition = 0.0
        empty_distance = 0.0
        fare = 0.0
        waiting_time = float(self._idle_wait_steps)
        late_penalty = 0.0
        completed_ride = 0.0

        if self._shift_hours_remaining <= 0:
            self._total_reward += -1.0
            return self._build_observation(
                reward=-1.0, done=True, last_action_error="shift_exhausted"
            )

        if action.action_type == "wait":
            self._driver_status = "idle"
            self._idle_wait_steps += 1

        elif action.action_type == "reposition":
            if action.target_node is None:
                last_action_error = "missing_target_node"
                reward -= 0.5
            elif action.target_node not in self._adjacency.get(self._driver_node, []):
                last_action_error = "target_not_adjacent"
                reward -= 0.6
            else:
                did_reposition = 1.0
                self._driver_status = "en_route"
                empty_distance = self._edge_distance[(self._driver_node, action.target_node)]
                self._empty_distance += empty_distance
                self._driver_node = action.target_node
                self._idle_wait_steps += 1

        elif action.action_type == "accept_ride":
            if action.ride_id is None:
                last_action_error = "missing_ride_id"
                reward -= 0.5
            else:
                selected = None
                for ride in self._rides:
                    if ride["ride_id"] == action.ride_id and ride["origin"] == self._driver_node:
                        selected = ride
                        break
                if selected is None:
                    last_action_error = "ride_not_available_at_node"
                    reward -= 0.6
                else:
                    self._rides.remove(selected)
                    completed_ride = 1.0
                    self._driver_status = "busy"
                    fare = float(selected["fare"])
                    waiting_time = float(selected["wait_time"])
                    if waiting_time > (self._task.lateness_budget * 10.0):
                        late_penalty = 1.0
                        self._late_rides += 1
                    self._driver_node = int(selected["destination"])
                    self._completed_rides += 1
                    self._total_fare += fare
                    self._idle_wait_steps = 0
        else:
            last_action_error = "unsupported_action_type"
            reward -= 0.7

        # ── Master Reward Function ────────────────────────────────────────────
        # R_t = Profit_Engine - Operational_Burn - Action_Tax
        #       + Urgency_Bonus * Indicator(ride) - SLA_Enforcer
        reward += 1.0 * (fare / self._max_fare)                     # Profit Engine
        reward -= 0.9 * (empty_distance / self._max_distance)       # Operational Burn
        reward -= 0.15 * did_reposition                             # Action Tax
        if completed_ride == 1.0:
            reward += 2.0 * math.exp(-0.1 * waiting_time)          # Urgency Bonus
        reward -= 1.5 * late_penalty                                # SLA Enforcer

        self._advance_time()
        done = (
            self._state.step_count >= self._task.horizon_steps
            or self._shift_hours_remaining <= 0.0
        )
        self._total_reward += reward
        return self._build_observation(
            reward=reward, done=done, last_action_error=last_action_error
        )

    @property
    def state(self) -> State:
        return self._state
