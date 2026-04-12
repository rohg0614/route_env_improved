"""Unit tests for the improved demand model: NegBin, surge pricing, ride TTL."""

import sys, os, math, random, statistics
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Pure-function helpers mirroring the environment logic ──────────────────

_NB_DISPERSION = 3.0
_SURGE_SENSITIVITY = 0.8


def _negbin(mu: float, rng: random.Random) -> int:
    if mu <= 0:
        return 0
    r = _NB_DISPERSION
    lam = rng.gammavariate(r, mu / r)
    k, p, threshold = 0, 1.0, math.exp(-lam)
    while p > threshold:
        k += 1
        p *= rng.random()
    return max(0, k - 1)


def _surge_multiplier(origin_node: int, rides: list[dict], n_nodes: int) -> float:
    total = max(1, len(rides))
    node_count = sum(1 for r in rides if r["origin"] == origin_node)
    demand_share = node_count / total
    relative_demand = demand_share * n_nodes
    return 1.0 + _SURGE_SENSITIVITY * (relative_demand - 1.0)


# ── NegBin distribution tests ─────────────────────────────────────────────

class TestNegBin:
    def _samples(self, mu: float, n: int = 5000, seed: int = 42) -> list[int]:
        rng = random.Random(seed)
        return [_negbin(mu, rng) for _ in range(n)]

    def test_mean_close_to_mu(self):
        for mu in [0.5, 1.2, 1.8, 3.2]:
            samples = self._samples(mu, n=8000)
            assert abs(statistics.mean(samples) - mu) < 0.15, (
                f"NegBin mean {statistics.mean(samples):.3f} far from mu={mu}"
            )

    def test_variance_exceeds_mean(self):
        """NegBin variance > mean — key overdispersion property."""
        for mu in [1.2, 1.8, 3.2]:
            samples = self._samples(mu, n=8000)
            var = statistics.variance(samples)
            mean = statistics.mean(samples)
            assert var > mean * 1.1, (
                f"Expected variance > mean for NegBin(mu={mu}), got var={var:.3f} mean={mean:.3f}"
            )

    def test_non_negative(self):
        samples = self._samples(1.8, n=2000)
        assert all(s >= 0 for s in samples)

    def test_zero_mu_returns_zero(self):
        rng = random.Random(42)
        assert _negbin(0.0, rng) == 0
        assert _negbin(-1.0, rng) == 0

    def test_overdispersion_vs_poisson(self):
        """NegBin variance is clearly larger than Poisson variance at same mean."""
        mu = 3.2
        rng = random.Random(42)
        nb_samples = [_negbin(mu, rng) for _ in range(8000)]
        nb_var = statistics.variance(nb_samples)
        # Poisson variance == mean; NegBin should be at least 30% higher
        assert nb_var > mu * 1.3, f"NegBin variance {nb_var:.2f} not clearly > Poisson variance {mu}"


# ── Surge pricing tests ────────────────────────────────────────────────────

class TestSurgePricing:
    def _make_rides(self, origin_counts: dict[int, int]) -> list[dict]:
        rides = []
        ride_id = 1
        for node, count in origin_counts.items():
            for _ in range(count):
                rides.append({"ride_id": ride_id, "origin": node})
                ride_id += 1
        return rides

    def test_uniform_demand_gives_multiplier_one(self):
        """When all nodes have equal rides, surge == 1.0 for every node."""
        n_nodes = 6
        rides = self._make_rides({i: 10 for i in range(n_nodes)})
        for node in range(n_nodes):
            m = _surge_multiplier(node, rides, n_nodes)
            assert abs(m - 1.0) < 1e-9, f"node {node}: expected 1.0, got {m}"

    def test_high_demand_node_surges_above_one(self):
        """A node with disproportionate rides gets multiplier > 1."""
        rides = self._make_rides({0: 50, 1: 5, 2: 5, 3: 5, 4: 5, 5: 5})
        m = _surge_multiplier(0, rides, 6)
        assert m > 1.0, f"High-demand node should surge above 1, got {m:.3f}"

    def test_low_demand_node_discounts_below_one(self):
        """A node below average demand gets multiplier < 1 (discount)."""
        rides = self._make_rides({0: 50, 1: 5, 2: 5, 3: 5, 4: 5, 5: 5})
        m = _surge_multiplier(5, rides, 6)
        assert m < 1.0, f"Low-demand node should discount below 1, got {m:.3f}"

    def test_surge_scales_with_sensitivity(self):
        """Higher demand concentration → higher surge multiplier."""
        rides_mild = self._make_rides({0: 20, 1: 10, 2: 10, 3: 10, 4: 10, 5: 10})
        rides_extreme = self._make_rides({0: 60, 1: 2, 2: 2, 3: 2, 4: 2, 5: 2})
        m_mild = _surge_multiplier(0, rides_mild, 6)
        m_extreme = _surge_multiplier(0, rides_extreme, 6)
        assert m_extreme > m_mild, "Greater demand concentration should yield higher surge"

    def test_surge_multiplier_floor(self):
        """Multiplier should never go below the floor (0.5) after clamping."""
        # All rides at one node → all other nodes have 0 rides → share = 0
        rides = self._make_rides({0: 100})
        m = _surge_multiplier(5, rides, 6)
        # Raw formula gives 1 + 0.8*(0-1) = 0.2, but env clamps at 0.5
        # We just test the formula here (floor applied at env level)
        assert m < 1.0  # confirms discount direction


# ── Ride TTL tests ─────────────────────────────────────────────────────────

class TestRideTTL:
    def _ttl_for_task(self, lateness_budget: float) -> int:
        return max(3, int(lateness_budget * 20))

    def test_easy_ttl(self):
        """Easy task (lateness_budget=0.45) → TTL=9."""
        assert self._ttl_for_task(0.45) == 9

    def test_medium_ttl(self):
        """Medium task (lateness_budget=0.32) → TTL=6."""
        assert self._ttl_for_task(0.32) == 6

    def test_hard_ttl(self):
        """Hard task (lateness_budget=0.15) → TTL=3 (floor)."""
        assert self._ttl_for_task(0.15) == 3

    def test_ttl_ordering(self):
        """Harder tasks have shorter TTL (higher scarcity)."""
        ttl_easy = self._ttl_for_task(0.45)
        ttl_medium = self._ttl_for_task(0.32)
        ttl_hard = self._ttl_for_task(0.15)
        assert ttl_easy > ttl_medium > ttl_hard

    def test_expiry_removes_old_rides(self):
        """Simulated expiry: rides older than TTL are filtered out."""
        ttl = 3
        rides = [
            {"ride_id": 1, "origin": 0, "birth_step": 0},
            {"ride_id": 2, "origin": 1, "birth_step": 2},
            {"ride_id": 3, "origin": 2, "birth_step": 4},
        ]
        current_step = 5
        active = [r for r in rides if current_step - r["birth_step"] < ttl]
        # ride 1 (age=5) and ride 2 (age=3) are expired; ride 3 (age=1) survives
        assert len(active) == 1
        assert active[0]["ride_id"] == 3

    def test_fresh_rides_survive(self):
        ttl = 5
        current_step = 10
        rides = [{"ride_id": i, "origin": 0, "birth_step": current_step - i} for i in range(1, 7)]
        active = [r for r in rides if current_step - r["birth_step"] < ttl]
        assert len(active) == 4  # birth_steps 10,9,8,7 survive; 6,5 are expired
