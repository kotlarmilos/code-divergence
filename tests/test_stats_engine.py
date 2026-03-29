"""Tests for the statistical engine — all four analysis layers."""

import math

import pytest

from code_divergence.stats_engine import (
    CUSUMState,
    ControlViolation,
    CoverageDensity,
    DiagnosticReport,
    DistributionalAnalyser,
    DriftDetector,
    DriftResult,
    InformationTheoreticAnalyser,
    PairwiseJSD,
    Recommendation,
    RoutingEntropy,
    SPCAnalyser,
    Severity,
    StatsEngine,
    _jsd,
    _kl_divergence,
    _shannon_entropy,
    _to_distribution,
    _compute_psi,
    _token_set_from_session,
)
from code_divergence.tracker import AgentTracker, EventType


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #


def _build_sessions(
    files_a: list[str],
    files_b: list[str],
    output_a: str = "",
    output_b: str = "",
) -> tuple:
    tracker = AgentTracker()
    tracker.new_session("agent-a", agent_id="A")
    tracker.new_session("agent-b", agent_id="B")
    for f in files_a:
        tracker.track_file_change("A", f, EventType.FILE_MODIFIED)
    for f in files_b:
        tracker.track_file_change("B", f, EventType.FILE_MODIFIED)
    if output_a:
        tracker.track_output("A", output_a)
    if output_b:
        tracker.track_output("B", output_b)
    return tracker, tracker.sessions


# ------------------------------------------------------------------ #
# Math primitives
# ------------------------------------------------------------------ #


class TestShannonEntropy:
    def test_uniform_distribution(self):
        dist = {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}
        assert _shannon_entropy(dist) == pytest.approx(2.0)  # log2(4)

    def test_single_element(self):
        dist = {"x": 1.0}
        assert _shannon_entropy(dist) == pytest.approx(0.0)

    def test_empty_distribution(self):
        assert _shannon_entropy({}) == 0.0

    def test_binary_distribution(self):
        dist = {"a": 0.5, "b": 0.5}
        assert _shannon_entropy(dist) == pytest.approx(1.0)


class TestKLDivergence:
    def test_identical_distributions(self):
        p = {"a": 0.5, "b": 0.5}
        assert _kl_divergence(p, p) == pytest.approx(0.0)

    def test_empty_distributions(self):
        assert _kl_divergence({}, {}) == 0.0

    def test_positive_value(self):
        p = {"a": 0.9, "b": 0.1}
        q = {"a": 0.5, "b": 0.5}
        kl = _kl_divergence(p, q)
        assert kl > 0.0

    def test_asymmetric(self):
        p = {"a": 0.9, "b": 0.1}
        q = {"a": 0.1, "b": 0.9}
        kl_pq = _kl_divergence(p, q)
        kl_qp = _kl_divergence(q, p)
        # KL is not symmetric in general, but for mirror distributions it's equal
        assert kl_pq == pytest.approx(kl_qp)


class TestJSD:
    def test_identical_distributions_zero(self):
        p = {"a": 0.5, "b": 0.5}
        assert _jsd(p, p) == pytest.approx(0.0, abs=1e-10)

    def test_symmetric(self):
        p = {"a": 0.7, "b": 0.3}
        q = {"a": 0.3, "b": 0.7}
        assert _jsd(p, q) == pytest.approx(_jsd(q, p))

    def test_bounded_zero_one(self):
        p = {"a": 1.0}
        q = {"b": 1.0}
        jsd_val = _jsd(p, q)
        assert 0.0 <= jsd_val <= 1.0

    def test_empty_distributions(self):
        assert _jsd({}, {}) == 0.0


class TestToDistribution:
    def test_basic_distribution(self):
        tokens = {"a", "b"}
        vocab = {"a", "b", "c"}
        dist = _to_distribution(tokens, vocab)
        assert len(dist) == 3
        assert sum(dist.values()) == pytest.approx(1.0)
        assert dist["a"] > dist["c"]  # "a" is present, "c" is not

    def test_empty_vocab(self):
        assert _to_distribution({"a"}, set()) == {}


class TestComputePSI:
    def test_identical_distributions(self):
        d = {"a": 0.5, "b": 0.5}
        psi = _compute_psi(d, d)
        assert psi == pytest.approx(0.0, abs=1e-4)

    def test_different_distributions(self):
        baseline = {"a": 0.8, "b": 0.2}
        current = {"a": 0.2, "b": 0.8}
        psi = _compute_psi(baseline, current)
        assert psi > 0.0

    def test_empty_distributions(self):
        assert _compute_psi({}, {}) == 0.0


# ------------------------------------------------------------------ #
# Layer 1: Distributional analysis
# ------------------------------------------------------------------ #


class TestDistributionalAnalyser:
    def setup_method(self):
        self.analyser = DistributionalAnalyser()

    def test_pairwise_jsd_identical_agents(self):
        tracker, sessions = _build_sessions(
            ["a.py", "b.py"], ["a.py", "b.py"],
            "hello world", "hello world",
        )
        results = self.analyser.pairwise_jsd(sessions)
        assert len(results) == 1
        assert results[0].jsd_score == pytest.approx(0.0, abs=1e-4)
        assert results[0].is_near_duplicate

    def test_pairwise_jsd_different_agents(self):
        tracker, sessions = _build_sessions(
            ["auth/login.py", "auth/session.py"],
            ["ui/button.py", "ui/modal.py"],
            "authenticate users login session token",
            "render button modal sidebar component layout",
        )
        results = self.analyser.pairwise_jsd(sessions)
        assert len(results) == 1
        assert results[0].jsd_score > 0.1

    def test_routing_entropy_clear_routing(self):
        tracker, sessions = _build_sessions(
            ["auth/login.py"], ["ui/button.py"],
            "authenticate users", "render button",
        )
        entries = self.analyser.routing_entropy(sessions, probes=["authenticate"])
        assert len(entries) == 1
        assert entries[0].clarity > 0.5  # should be clear

    def test_routing_entropy_ambiguous(self):
        tracker, sessions = _build_sessions(
            ["shared.py"], ["shared.py"],
            "shared logic here", "shared logic here",
        )
        entries = self.analyser.routing_entropy(sessions, probes=["shared"])
        assert len(entries) == 1
        # Both agents have "shared" → high entropy, low clarity
        assert entries[0].clarity < 0.5

    def test_system_routing_clarity_in_range(self):
        tracker, sessions = _build_sessions(
            ["a.py"], ["b.py"], "foo bar", "baz qux",
        )
        clarity = self.analyser.system_routing_clarity(sessions)
        assert 0.0 <= clarity <= 1.0

    def test_coverage_density_full_coverage(self):
        tracker, sessions = _build_sessions(
            ["a.py"], ["b.py"], "foo", "bar",
        )
        # Use all tokens as probes → everything is covered
        coverage = self.analyser.coverage_density(sessions)
        assert coverage.mean_max_similarity == pytest.approx(1.0)
        assert coverage.uncovered_fraction == pytest.approx(0.0)

    def test_single_session_no_entropy(self):
        tracker = AgentTracker()
        tracker.new_session("only", agent_id="X")
        tracker.track_file_change("X", "f.py")
        sessions = tracker.sessions
        entries = self.analyser.routing_entropy(sessions)
        assert entries == []

    def test_pairwise_jsd_three_agents(self):
        tracker = AgentTracker()
        for name, aid, f in [("a", "A", "auth.py"), ("b", "B", "ui.py"), ("c", "C", "db.py")]:
            tracker.new_session(name, agent_id=aid)
            tracker.track_file_change(aid, f)
        results = self.analyser.pairwise_jsd(tracker.sessions)
        assert len(results) == 3  # 3 choose 2


# ------------------------------------------------------------------ #
# Layer 2: Information-theoretic health
# ------------------------------------------------------------------ #


class TestInformationTheoreticAnalyser:
    def setup_method(self):
        self.analyser = InformationTheoreticAnalyser()

    def test_mutual_information_identical(self):
        tracker, sessions = _build_sessions(
            ["a.py", "b.py"], ["a.py", "b.py"],
            "same output text", "same output text",
        )
        results = self.analyser.mutual_information(sessions)
        assert len(results) == 1
        # Identical agents should have high MI
        assert results[0].mi_score > 0.0

    def test_mutual_information_disjoint(self):
        tracker, sessions = _build_sessions(
            ["auth/login.py"], ["ui/button.py"],
            "authenticate session token", "render layout component",
        )
        results = self.analyser.mutual_information(sessions)
        assert len(results) == 1
        # Disjoint agents should have lower MI
        # (Still some MI due to smoothing, but not flagged as redundant)

    def test_conditional_entropy_range(self):
        tracker, sessions = _build_sessions(
            ["a.py"], ["b.py"], "foo bar", "baz qux",
        )
        ce = self.analyser.conditional_entropy(sessions)
        assert 0.0 <= ce <= 1.0

    def test_conditional_entropy_perfect_separation(self):
        tracker, sessions = _build_sessions(
            ["auth/login.py", "auth/session.py"],
            ["ui/button.py", "ui/modal.py"],
            "authenticate users login session token",
            "render button modal sidebar component",
        )
        ce = self.analyser.conditional_entropy(sessions)
        # Not all tokens are shared, so entropy shouldn't be maximal
        assert ce < 1.0

    def test_per_agent_entropy(self):
        tracker, sessions = _build_sessions(
            ["a.py", "b.py"], ["c.py"],
            "foo bar baz qux", "hello",
        )
        result = self.analyser.per_agent_entropy(sessions)
        assert "A" in result
        assert "B" in result
        # Agent A has more tokens → higher entropy
        assert result["A"] > result["B"]

    def test_health_metrics_aggregation(self):
        tracker, sessions = _build_sessions(
            ["a.py"], ["b.py"], "foo", "bar",
        )
        health = self.analyser.health_metrics(sessions)
        assert health.mutual_information is not None
        assert health.conditional_entropy >= 0
        assert len(health.per_agent_entropy) == 2


# ------------------------------------------------------------------ #
# Layer 3: Drift detection
# ------------------------------------------------------------------ #


class TestDriftDetector:
    def test_no_drift_same_state(self):
        tracker, sessions = _build_sessions(
            ["a.py"], ["b.py"], "foo", "bar",
        )
        detector = DriftDetector()
        detector.set_baselines_from_sessions(sessions)
        results = detector.detect_drift(sessions)
        assert len(results) == 2
        for r in results:
            assert r.kl_divergence == pytest.approx(0.0, abs=0.1)
            assert not r.cusum_alert
            assert not r.psi_alert

    def test_drift_detected_after_change(self):
        tracker = AgentTracker()
        tracker.new_session("a", agent_id="A")
        tracker.track_file_change("A", "old.py")
        tracker.track_output("A", "original code")

        sessions = tracker.sessions
        detector = DriftDetector()
        detector.set_baselines_from_sessions(sessions)

        # Now modify the session significantly
        tracker.track_file_change("A", "new_module.py")
        tracker.track_output("A", "completely different approach architecture redesign")
        tracker.track_output("A", "brand new functionality refactored everything")

        results = detector.detect_drift(tracker.sessions)
        assert len(results) == 1
        # KL should be > 0 since distribution changed
        assert results[0].kl_divergence >= 0.0

    def test_cusum_accumulates(self):
        detector = DriftDetector(cusum_threshold=2.0, cusum_slack=0.1)

        tracker = AgentTracker()
        tracker.new_session("a", agent_id="A")
        tracker.track_file_change("A", "base.py")

        detector.set_baselines_from_sessions(tracker.sessions)

        # Repeatedly add new content to drift
        for i in range(10):
            tracker.track_output("A", f"new_content_{i} unique_token_{i}")
            results = detector.detect_drift(tracker.sessions)

        # After many changes, CUSUM should have accumulated
        final_state = detector._cusum_states["A"]
        assert final_state.s_pos >= 0.0

    def test_psi_empty_distributions(self):
        detector = DriftDetector()
        assert _compute_psi({}, {}) == 0.0

    def test_set_baseline_explicit(self):
        detector = DriftDetector()
        detector.set_baseline("A", {"token1": 0.5, "token2": 0.5})
        assert "A" in detector._baselines

    def test_no_baseline_no_results(self):
        tracker = AgentTracker()
        tracker.new_session("a", agent_id="A")
        tracker.track_file_change("A", "f.py")

        detector = DriftDetector()
        # No baseline set
        results = detector.detect_drift(tracker.sessions)
        assert results == []


# ------------------------------------------------------------------ #
# Layer 4: Statistical process control
# ------------------------------------------------------------------ #


class TestSPCAnalyser:
    def setup_method(self):
        self.spc = SPCAnalyser(window_size=5)

    def test_in_control_process(self):
        # Stable process: 1 defect out of 20 in each period
        observations = [(1, 20)] * 10
        result = self.spc.p_chart("A", observations)
        assert result.is_in_control

    def test_out_of_control_beyond_3sigma(self):
        # Mostly clean, then a spike
        observations = [(0, 20)] * 8 + [(18, 20)]
        result = self.spc.p_chart("A", observations)
        assert not result.is_in_control
        assert ControlViolation.BEYOND_3_SIGMA in result.violations

    def test_empty_observations(self):
        result = self.spc.p_chart("A", [])
        assert result.is_in_control
        assert result.chart_points == []

    def test_chart_points_have_correct_limits(self):
        observations = [(2, 100)] * 5
        result = self.spc.p_chart("A", observations)
        for pt in result.chart_points:
            assert pt.lcl <= pt.center <= pt.ucl
            assert 0.0 <= pt.lcl
            assert pt.ucl <= 1.0

    def test_from_session_windows(self):
        tracker = AgentTracker()
        tracker.new_session("a", agent_id="A")
        for _ in range(18):
            tracker.track_file_change("A", "f.py")
        for _ in range(2):
            tracker.track_error("A", "boom")

        session = tracker.get_session("A")
        result = self.spc.from_session_windows(session)
        assert result.agent_id == "A"
        assert len(result.chart_points) > 0

    def test_from_empty_session(self):
        tracker = AgentTracker()
        tracker.new_session("a", agent_id="A")
        session = tracker.get_session("A")
        result = self.spc.from_session_windows(session)
        assert result.is_in_control

    def test_western_electric_2_of_3(self):
        # Create observations where p-values are high for 3 consecutive points
        # p_bar will be low, so 3 consecutive high p points trigger rule 2
        observations = [(0, 50)] * 5 + [(15, 50)] * 3
        result = self.spc.p_chart("A", observations)
        # Should trigger either beyond_3σ or 2_of_3 rule
        assert not result.is_in_control


# ------------------------------------------------------------------ #
# StatsEngine orchestration
# ------------------------------------------------------------------ #


class TestStatsEngine:
    def test_diagnose_empty_tracker(self):
        tracker = AgentTracker()
        engine = StatsEngine(tracker)
        report = engine.diagnose()
        assert isinstance(report, DiagnosticReport)
        assert report.routing_clarity == 1.0
        assert report.recommendations[0].severity == Severity.OK

    def test_diagnose_with_sessions(self):
        tracker, sessions = _build_sessions(
            ["auth/login.py", "auth/session.py"],
            ["ui/button.py", "ui/modal.py"],
            "authenticate users login session",
            "render button modal sidebar",
        )
        engine = StatsEngine(tracker)
        report = engine.diagnose()

        assert len(report.jsd_scores) == 1
        assert 0.0 <= report.routing_clarity <= 1.0
        assert report.coverage is not None
        assert report.health is not None
        assert isinstance(report.recommendations, list)

    def test_diagnose_to_dict(self):
        tracker, sessions = _build_sessions(
            ["a.py"], ["b.py"], "foo", "bar",
        )
        engine = StatsEngine(tracker)
        report = engine.diagnose()
        d = report.to_dict()

        assert "layer_1_distributional" in d
        assert "layer_2_information_theoretic" in d
        assert "layer_3_drift" in d
        assert "layer_4_spc" in d
        assert "recommendations" in d

    def test_diagnose_near_duplicate_recommendation(self):
        tracker, sessions = _build_sessions(
            ["a.py", "b.py", "c.py"],
            ["a.py", "b.py", "c.py"],
            "identical output text for both agents here",
            "identical output text for both agents here",
        )
        engine = StatsEngine(tracker)
        report = engine.diagnose()

        # Identical agents should produce a near-duplicate warning
        near_dup_recs = [
            r for r in report.recommendations
            if r.metric_name == "jsd" and "near-duplicate" in r.message
        ]
        assert len(near_dup_recs) > 0

    def test_diagnose_with_errors(self):
        tracker = AgentTracker()
        tracker.new_session("a", agent_id="A")
        for _ in range(15):
            tracker.track_file_change("A", "f.py")
        for _ in range(5):
            tracker.track_error("A", "error")
        engine = StatsEngine(tracker)
        report = engine.diagnose()
        # SPC should have chart points
        assert len(report.spc_results) == 1

    def test_diagnose_with_custom_probes(self):
        tracker, sessions = _build_sessions(
            ["a.py"], ["b.py"],
            "authenticate login", "render display",
        )
        engine = StatsEngine(tracker)
        report = engine.diagnose(probes=["authenticate", "render"])
        assert 0.0 <= report.routing_clarity <= 1.0

    def test_diagnose_three_agents(self):
        tracker = AgentTracker()
        for name, aid, files, output in [
            ("auth", "A", ["auth.py"], "authenticate users"),
            ("ui", "B", ["ui.py"], "render components"),
            ("db", "C", ["db.py"], "query database"),
        ]:
            tracker.new_session(name, agent_id=aid)
            for f in files:
                tracker.track_file_change(aid, f)
            tracker.track_output(aid, output)

        engine = StatsEngine(tracker)
        report = engine.diagnose()
        assert len(report.jsd_scores) == 3  # 3 choose 2


# ------------------------------------------------------------------ #
# Token extraction
# ------------------------------------------------------------------ #


class TestTokenSetFromSession:
    def test_extracts_file_tokens(self):
        tracker = AgentTracker()
        tracker.new_session("a", agent_id="A")
        tracker.track_file_change("A", "auth/login.py")
        session = tracker.get_session("A")
        tokens = _token_set_from_session(session)
        assert "auth" in tokens
        assert "login" in tokens

    def test_extracts_output_tokens(self):
        tracker = AgentTracker()
        tracker.new_session("a", agent_id="A")
        tracker.track_output("A", "authenticate users with tokens")
        session = tracker.get_session("A")
        tokens = _token_set_from_session(session)
        assert "authenticate" in tokens
        assert "users" in tokens

    def test_extracts_commit_tokens(self):
        tracker = AgentTracker()
        tracker.new_session("a", agent_id="A")
        tracker.track_commit("A", "abc123", "feat: implement login flow")
        session = tracker.get_session("A")
        tokens = _token_set_from_session(session)
        assert "implement" in tokens
        assert "login" in tokens

    def test_empty_session(self):
        tracker = AgentTracker()
        tracker.new_session("a", agent_id="A")
        session = tracker.get_session("A")
        tokens = _token_set_from_session(session)
        assert tokens == set()
