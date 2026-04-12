"""Tests for zone names, edges, demand weights, and task config."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
from tasks import TASKS, TASK_ORDER


class TestZoneNames:
    def test_all_tasks_have_zone_names(self):
        for name, task in TASKS.items():
            assert len(task.zone_names) > 0, f"{name} has no zone_names"

    def test_zone_names_match_node_count(self):
        for name, task in TASKS.items():
            assert len(task.zone_names) == task.node_count, (
                f"{name}: zone_names length {len(task.zone_names)} "
                f"!= node_count {task.node_count}"
            )

    def test_easy_zone_names_correct(self):
        assert TASKS["easy"].zone_names[0] == "Airport"
        assert TASKS["easy"].zone_names[1] == "Downtown"
        assert TASKS["easy"].zone_names[4] == "Stadium"
        assert TASKS["easy"].zone_names[5] == "Suburbs"

    def test_medium_zone_names_correct(self):
        assert TASKS["medium"].zone_names[0] == "Airport"
        assert TASKS["medium"].zone_names[2] == "Financial"
        assert TASKS["medium"].zone_names[7] == "Mall"

    def test_hard_zone_names_correct(self):
        assert TASKS["hard"].zone_names[0] == "Airport"
        assert TASKS["hard"].zone_names[2] == "Downtown"
        assert TASKS["hard"].zone_names[6] == "Hospital"
        assert TASKS["hard"].zone_names[9] == "Suburbs_N"

    def test_no_duplicate_zone_names_per_task(self):
        for name, task in TASKS.items():
            assert len(set(task.zone_names)) == len(task.zone_names), (
                f"{name} has duplicate zone names"
            )


class TestZoneEdges:
    def test_all_tasks_have_zone_edges(self):
        for name, task in TASKS.items():
            assert len(task.zone_edges) > 0, f"{name} has no zone_edges"

    def test_all_edges_reference_valid_nodes(self):
        for name, task in TASKS.items():
            for i, j in task.zone_edges:
                assert 0 <= i < task.node_count, (
                    f"{name}: edge ({i},{j}) references invalid node {i}"
                )
                assert 0 <= j < task.node_count, (
                    f"{name}: edge ({i},{j}) references invalid node {j}"
                )

    def test_no_self_loops(self):
        for name, task in TASKS.items():
            for i, j in task.zone_edges:
                assert i != j, f"{name}: self-loop at node {i}"

    def test_easy_airport_connects_to_downtown(self):
        edges = TASKS["easy"].zone_edges
        assert (0, 1) in edges or (1, 0) in edges

    def test_easy_airport_connects_to_university(self):
        edges = TASKS["easy"].zone_edges
        assert (0, 2) in edges or (2, 0) in edges


class TestZoneDemandWeights:
    def test_all_tasks_have_demand_weights(self):
        for name, task in TASKS.items():
            assert len(task.zone_demand_weights) > 0, (
                f"{name} has no zone_demand_weights"
            )

    def test_demand_weights_match_node_count(self):
        for name, task in TASKS.items():
            assert len(task.zone_demand_weights) == task.node_count, (
                f"{name}: demand_weights length {len(task.zone_demand_weights)} "
                f"!= node_count {task.node_count}"
            )

    def test_all_weights_positive(self):
        for name, task in TASKS.items():
            for i, w in enumerate(task.zone_demand_weights):
                assert w > 0, f"{name} node {i} has non-positive weight {w}"

    def test_airport_highest_demand_easy(self):
        weights = TASKS["easy"].zone_demand_weights
        assert weights[0] == max(weights), (
            "Airport (node 0) should have highest demand on easy"
        )

    def test_suburbs_lowest_demand_easy(self):
        weights = TASKS["easy"].zone_demand_weights
        assert weights[5] == min(weights), (
            "Suburbs (node 5) should have lowest demand on easy"
        )

    def test_hard_airport_high_demand(self):
        weights = TASKS["hard"].zone_demand_weights
        assert weights[0] >= 2.0, "Airport should have high demand weight on hard"

    def test_hard_suburbs_low_demand(self):
        weights = TASKS["hard"].zone_demand_weights
        assert weights[10] <= 0.8, "Suburbs_S should have low demand weight on hard"


class TestDifficultyProgression:
    def test_nodes_increase_with_difficulty(self):
        assert TASKS["hard"].node_count > TASKS["medium"].node_count
        assert TASKS["medium"].node_count > TASKS["easy"].node_count

    def test_horizon_increases_with_difficulty(self):
        assert TASKS["hard"].horizon_steps > TASKS["medium"].horizon_steps
        assert TASKS["medium"].horizon_steps > TASKS["easy"].horizon_steps

    def test_lateness_budget_decreases_with_difficulty(self):
        assert TASKS["hard"].lateness_budget < TASKS["medium"].lateness_budget
        assert TASKS["medium"].lateness_budget < TASKS["easy"].lateness_budget

    def test_task_order_correct(self):
        assert TASK_ORDER == ["easy", "medium", "hard"]

    def test_all_tasks_registered(self):
        assert set(TASK_ORDER) == set(TASKS.keys())
