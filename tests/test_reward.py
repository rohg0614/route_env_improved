"""Unit tests for the master reward function arithmetic.

Tests are pure-Python: they don't import the environment class (which
requires openenv to be installed) but directly verify the reward formula
components defined in the original plan.
"""

import math
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

MAX_FARE = 60.0
MAX_DISTANCE = 20.0


def reward(
    fare: float = 0.0,
    empty_distance: float = 0.0,
    did_reposition: float = 0.0,
    completed_ride: float = 0.0,
    waiting_time: float = 0.0,
    late_penalty: float = 0.0,
) -> float:
    r = 0.0
    r += 1.0 * (fare / MAX_FARE)
    r -= 0.9 * (empty_distance / MAX_DISTANCE)
    r -= 0.15 * did_reposition
    if completed_ride == 1.0:
        r += 2.0 * math.exp(-0.1 * waiting_time)
    r -= 1.5 * late_penalty
    return r


class TestProfitEngine:
    def test_max_fare_contributes_one(self):
        r = reward(fare=60.0)
        assert abs(r - 1.0) < 1e-9

    def test_zero_fare_contributes_zero(self):
        r = reward(fare=0.0)
        assert abs(r) < 1e-9

    def test_partial_fare(self):
        r = reward(fare=30.0)
        assert abs(r - 0.5) < 1e-9


class TestOperationalBurn:
    def test_max_distance_costs_09(self):
        r = reward(empty_distance=20.0)
        assert abs(r - (-0.9)) < 1e-9

    def test_zero_distance_no_cost(self):
        r = reward(empty_distance=0.0)
        assert abs(r) < 1e-9


class TestActionTax:
    def test_reposition_costs_015(self):
        r = reward(did_reposition=1.0)
        assert abs(r - (-0.15)) < 1e-9

    def test_no_reposition_no_tax(self):
        r = reward(did_reposition=0.0)
        assert abs(r) < 1e-9


class TestUrgencyBonus:
    def test_zero_wait_max_bonus(self):
        # 2.0 * exp(0) = 2.0
        r = reward(completed_ride=1.0, waiting_time=0.0)
        assert abs(r - 2.0) < 1e-9

    def test_bonus_decays_with_wait(self):
        r_low_wait = reward(completed_ride=1.0, waiting_time=1.0)
        r_high_wait = reward(completed_ride=1.0, waiting_time=8.0)
        assert r_low_wait > r_high_wait

    def test_indicator_function_blocks_idle_exploit(self):
        # No completed_ride means no urgency bonus regardless of waiting_time
        r = reward(completed_ride=0.0, waiting_time=0.0)
        assert abs(r) < 1e-9


class TestSLAEnforcer:
    def test_late_ride_deducts_15(self):
        r = reward(late_penalty=1.0)
        assert abs(r - (-1.5)) < 1e-9

    def test_no_late_ride_no_penalty(self):
        r = reward(late_penalty=0.0)
        assert abs(r) < 1e-9


class TestCompositeReward:
    def test_high_fare_ride_positive(self):
        """Accepting a max-fare ride with zero wait should be strongly positive."""
        r = reward(fare=60.0, completed_ride=1.0, waiting_time=0.0)
        assert r > 2.5

    def test_repositioning_to_empty_area_net_negative(self):
        """Moving empty to a far node with no following ride should be negative."""
        r = reward(empty_distance=15.0, did_reposition=1.0)
        assert r < 0.0

    def test_late_accepted_ride_can_be_negative(self):
        """A low-fare late ride can have negative total reward."""
        r = reward(fare=8.0, completed_ride=1.0, waiting_time=9.0, late_penalty=1.0)
        # fare/60 ≈ 0.133, urgency ≈ 0.82, late = -1.5 → net ≈ -0.55
        assert r < 0.0
