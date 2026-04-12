"""Unit tests for the per-task graders."""

import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from grader import score_episode, get_grader, GRADERS, _score_easy, _score_medium, _score_hard


# ── Strict (0, 1) range — all tasks, all edge cases ──────────────────────────

class TestScoreRange:
    @pytest.mark.parametrize("fn,name", [
        (_score_easy, "easy"), (_score_medium, "medium"), (_score_hard, "hard")
    ])
    @pytest.mark.parametrize("args", [
        (0, 0, 0, 0.0),
        (1, 0, 0, 0.0),
        (1, 1, 0, 3.5),
        (60, 10, 0, 15.0),
        (84, 14, 2, 22.0),
        (120, 8, 5, -3.0),
        (10_000, 0, 0, 0.0),     # huge step count — must clamp
        (120, 500, 0, 1000.0),   # huge rides — must clamp
        (1, 0, 1000, -999.0),    # massive late penalty — must floor
    ])
    def test_strictly_in_0_1(self, fn, name, args):
        s = fn(*args)
        assert 0.0 < s < 1.0, f"{name} grader returned {s} for args {args}"


# ── Graders are DISTINCT — different tasks produce different scores ───────────

class TestGradersAreDistinct:
    def test_same_inputs_yield_different_scores(self):
        """The three graders must NOT return the same value for the same inputs."""
        args = (60, 10, 2, 15.0)
        scores = {
            "easy":   _score_easy(*args),
            "medium": _score_medium(*args),
            "hard":   _score_hard(*args),
        }
        # All three must differ
        assert len(set(scores.values())) == 3, (
            f"Graders returned identical scores: {scores}"
        )

    def test_hard_scores_lower_than_easy_for_same_inputs(self):
        """Hard task penalises the same trajectory more heavily than easy."""
        args = (60, 10, 5, 15.0)  # 5 late rides — light on easy, severe on hard
        assert _score_hard(*args) < _score_easy(*args)

    def test_hard_scores_lower_than_medium_for_same_inputs(self):
        args = (60, 10, 5, 15.0)
        assert _score_hard(*args) < _score_medium(*args)


# ── Monotonicity ──────────────────────────────────────────────────────────────

class TestMonotonicity:
    @pytest.mark.parametrize("fn", [_score_easy, _score_medium, _score_hard])
    def test_more_rides_higher_score(self, fn):
        assert fn(60, 15, 0, 10.0) >= fn(60, 5, 0, 10.0)

    @pytest.mark.parametrize("fn", [_score_easy, _score_medium, _score_hard])
    def test_late_rides_penalise(self, fn):
        assert fn(60, 10, 0, 15.0) >= fn(60, 10, 5, 15.0)

    @pytest.mark.parametrize("fn", [_score_easy, _score_medium, _score_hard])
    def test_higher_reward_higher_score(self, fn):
        assert fn(60, 10, 0, 30.0) >= fn(60, 10, 0, 5.0)


# ── Hard task design property: greedy fails, SLA-aware passes ────────────────

class TestHardTaskDesignProperty:
    def test_greedy_fails_threshold(self):
        """Greedy always-accept agent on hard task scores below 0.5."""
        greedy_score = _score_hard(120, 40, 14, 18.0)
        assert greedy_score < 0.5, (
            f"Greedy agent should fail 0.5 threshold on hard, got {greedy_score:.4f}"
        )

    def test_sla_aware_passes_threshold(self):
        """Agent that avoids high-wait rides on hard task scores above 0.5."""
        sla_aware_score = _score_hard(120, 38, 2, 32.0)
        assert sla_aware_score > 0.5, (
            f"SLA-aware agent should pass 0.5 threshold on hard, got {sla_aware_score:.4f}"
        )

    def test_sla_aware_beats_greedy(self):
        greedy = _score_hard(120, 40, 14, 18.0)
        sla_aware = _score_hard(120, 38, 2, 32.0)
        assert sla_aware > greedy

    def test_zero_late_rides_dramatically_better_on_hard(self):
        """Eliminating late rides on hard task gives a large score jump."""
        with_late    = _score_hard(120, 40, 14, 28.0)
        without_late = _score_hard(120, 40,  0, 28.0)
        assert without_late - with_late > 0.15, (
            f"Expected >0.15 score delta for eliminating late rides on hard, "
            f"got {without_late - with_late:.4f}"
        )

    def test_easy_greedy_passes_threshold(self):
        """Easy task is easy — greedy agent should pass 0.5."""
        greedy_score = _score_easy(60, 20, 1, 15.0)
        assert greedy_score > 0.5


# ── Registry ─────────────────────────────────────────────────────────────────

class TestGraderRegistry:
    def test_all_tasks_registered(self):
        for task in ("easy", "medium", "hard"):
            assert task in GRADERS

    def test_get_grader_returns_callable(self):
        for task in ("easy", "medium", "hard"):
            assert callable(get_grader(task))

    def test_get_grader_unknown_raises(self):
        with pytest.raises(KeyError):
            get_grader("legendary")

    def test_registry_routes_to_correct_fn(self):
        """Registry grader must match the direct function call."""
        args = (50, 8, 1, 12.0)
        assert get_grader("easy")(*args)   == _score_easy(*args)
        assert get_grader("medium")(*args) == _score_medium(*args)
        assert get_grader("hard")(*args)   == _score_hard(*args)

    def test_score_episode_routes_by_task(self):
        """score_episode(task_name=X) must match _score_X directly."""
        args = (60, 10, 2, 15.0)
        assert score_episode(*args, task_name="easy")   == _score_easy(*args)
        assert score_episode(*args, task_name="medium") == _score_medium(*args)
        assert score_episode(*args, task_name="hard")   == _score_hard(*args)

    def test_score_episode_defaults_to_easy(self):
        args = (60, 10, 2, 15.0)
        assert score_episode(*args) == _score_easy(*args)
