"""Tests for divergence and overlap metrics."""

import pytest

from code_divergence.metrics import (
    AgentComparator,
    DivergenceCalculator,
    MetricResult,
    OverlapCalculator,
    _jaccard_distance,
    _jaccard_similarity,
    _text_similarity,
)
from code_divergence.tracker import AgentTracker, EventType


def build_two_agents(
    files_a: list[str],
    files_b: list[str],
    output_a: str = "",
    output_b: str = "",
) -> tuple:
    tracker = AgentTracker()
    aid_a = tracker.new_session("agent-a", agent_id="A")
    aid_b = tracker.new_session("agent-b", agent_id="B")
    for f in files_a:
        tracker.track_file_change("A", f, EventType.FILE_MODIFIED)
    for f in files_b:
        tracker.track_file_change("B", f, EventType.FILE_MODIFIED)
    if output_a:
        tracker.track_output("A", output_a)
    if output_b:
        tracker.track_output("B", output_b)
    return tracker.get_session("A"), tracker.get_session("B")


class TestJaccardHelpers:
    def test_identical_sets_zero_distance(self):
        s = {"a", "b", "c"}
        assert _jaccard_distance(s, s) == 0.0

    def test_disjoint_sets_distance_one(self):
        assert _jaccard_distance({"a"}, {"b"}) == 1.0

    def test_partial_overlap(self):
        assert _jaccard_distance({"a", "b"}, {"b", "c"}) == pytest.approx(2 / 3)

    def test_empty_sets(self):
        assert _jaccard_distance(set(), set()) == 0.0

    def test_similarity_complement_of_distance(self):
        a, b = {"x", "y"}, {"y", "z"}
        assert _jaccard_similarity(a, b) == pytest.approx(1 - _jaccard_distance(a, b))


class TestTextSimilarity:
    def test_identical_strings(self):
        assert _text_similarity("hello world", "hello world") == pytest.approx(1.0)

    def test_completely_different(self):
        assert _text_similarity("abc", "xyz") == pytest.approx(0.0)

    def test_empty_strings(self):
        assert _text_similarity("", "") == pytest.approx(1.0)

    def test_one_empty(self):
        assert _text_similarity("abc", "") == pytest.approx(0.0)


class TestDivergenceCalculator:
    def setup_method(self):
        self.calc = DivergenceCalculator()

    def test_identical_files_zero_divergence(self):
        a, b = build_two_agents(["a.py", "b.py"], ["a.py", "b.py"])
        assert self.calc.file_divergence(a, b) == pytest.approx(0.0)

    def test_disjoint_files_max_divergence(self):
        a, b = build_two_agents(["a.py"], ["b.py"])
        assert self.calc.file_divergence(a, b) == pytest.approx(1.0)

    def test_partial_file_divergence(self):
        a, b = build_two_agents(["a.py", "b.py"], ["b.py", "c.py"])
        # intersection={b.py}, union={a,b,c}.py → jaccard_dist = 1 - 1/3 = 2/3
        assert self.calc.file_divergence(a, b) == pytest.approx(2 / 3)

    def test_identical_output_zero_divergence(self):
        a, b = build_two_agents([], [], "the quick brown fox", "the quick brown fox")
        assert self.calc.output_divergence(a, b) == pytest.approx(0.0)

    def test_empty_outputs_zero_divergence(self):
        a, b = build_two_agents([], [], "", "")
        assert self.calc.output_divergence(a, b) == pytest.approx(0.0)

    def test_overall_is_weighted(self):
        a, b = build_two_agents(["a.py"], ["b.py"], "foo bar", "baz qux")
        d = self.calc.overall(a, b)
        assert 0.0 <= d <= 1.0


class TestOverlapCalculator:
    def setup_method(self):
        self.calc = OverlapCalculator()

    def test_identical_files_max_overlap(self):
        a, b = build_two_agents(["a.py", "b.py"], ["a.py", "b.py"])
        assert self.calc.file_overlap(a, b) == pytest.approx(1.0)

    def test_disjoint_files_zero_overlap(self):
        a, b = build_two_agents(["a.py"], ["b.py"])
        assert self.calc.file_overlap(a, b) == pytest.approx(0.0)

    def test_output_overlap_identical(self):
        a, b = build_two_agents([], [], "hello world foo", "hello world foo")
        assert self.calc.output_overlap(a, b) == pytest.approx(1.0)

    def test_overall_in_range(self):
        a, b = build_two_agents(["x.py"], ["x.py"], "same output", "same output")
        o = self.calc.overall(a, b)
        assert 0.0 <= o <= 1.0


class TestAgentComparator:
    def setup_method(self):
        self.cmp = AgentComparator()

    def test_compare_returns_metric_result(self):
        a, b = build_two_agents(["f.py"], ["g.py"])
        result = self.cmp.compare(a, b)
        assert isinstance(result, MetricResult)
        assert result.agent_a == "A"
        assert result.agent_b == "B"

    def test_scores_in_range(self):
        a, b = build_two_agents(["f.py", "g.py"], ["g.py", "h.py"], "foo", "bar")
        r = self.cmp.compare(a, b)
        for score in (r.divergence, r.overlap, r.file_divergence, r.file_overlap,
                      r.output_divergence, r.output_overlap):
            assert 0.0 <= score <= 1.0, f"Score out of range: {score}"

    def test_compare_all_pairwise(self):
        tracker = AgentTracker()
        for name in ("alpha", "beta", "gamma"):
            tracker.new_session(name, agent_id=name)
        sessions = tracker.sessions
        results = self.cmp.compare_all(sessions)
        # 3 agents → 3 pairs
        assert len(results) == 3

    def test_high_overlap_detection(self):
        a, b = build_two_agents(
            ["a.py", "b.py", "c.py"],
            ["a.py", "b.py", "c.py"],
            "identical output text here",
            "identical output text here",
        )
        r = self.cmp.compare(a, b)
        assert r.is_over_overlapping  # overlap should be ~1.0

    def test_high_divergence_detection(self):
        a, b = build_two_agents(
            ["auth/login.py", "auth/token.py", "auth/session.py"],
            ["ui/button.py", "ui/modal.py", "ui/sidebar.py"],
            "authenticate users with tokens session management",
            "render button modal sidebar component layout",
        )
        r = self.cmp.compare(a, b)
        assert r.is_over_diverged  # completely different files + outputs

    def test_details_contain_expected_keys(self):
        a, b = build_two_agents(["shared.py", "only_a.py"], ["shared.py", "only_b.py"])
        r = self.cmp.compare(a, b)
        assert "shared_files" in r.details
        assert "only_in_a" in r.details
        assert "only_in_b" in r.details
        assert "shared.py" in r.details["shared_files"]
        assert "only_a.py" in r.details["only_in_a"]
        assert "only_b.py" in r.details["only_in_b"]
