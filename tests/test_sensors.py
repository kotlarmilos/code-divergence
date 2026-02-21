"""Tests for the sensor / alerting system."""

import time

import pytest

from code_divergence.sensors import Alert, AlertKind, AlertLevel, PerformanceSensor, SensorConfig
from code_divergence.tracker import AgentTracker, EventType


def make_sensor(
    tracker: AgentTracker,
    divergence_warn: float = 0.70,
    divergence_crit: float = 0.85,
    overlap_warn: float = 0.60,
    overlap_crit: float = 0.75,
    error_rate_warn: float = 0.10,
    error_rate_crit: float = 0.25,
    min_events: int = 1,
) -> PerformanceSensor:
    config = SensorConfig(
        divergence_warn=divergence_warn,
        divergence_crit=divergence_crit,
        overlap_warn=overlap_warn,
        overlap_crit=overlap_crit,
        error_rate_warn=error_rate_warn,
        error_rate_crit=error_rate_crit,
        min_events_for_alert=min_events,
    )
    return PerformanceSensor(tracker, config)


class TestDivergenceAlerts:
    def test_no_alert_below_threshold(self):
        tracker = AgentTracker()
        tracker.new_session("a", agent_id="A")
        tracker.new_session("b", agent_id="B")
        for f in ["shared.py"]:
            tracker.track_file_change("A", f)
            tracker.track_file_change("B", f)

        sensor = make_sensor(tracker)
        alerts = sensor.evaluate()
        divergence_alerts = [a for a in alerts if a.kind == AlertKind.OVER_DIVERGED]
        assert not divergence_alerts

    def test_alert_on_high_divergence(self):
        tracker = AgentTracker()
        tracker.new_session("a", agent_id="A")
        tracker.new_session("b", agent_id="B")
        # Completely different files → divergence = 1.0
        for f in ["auth/a.py", "auth/b.py", "auth/c.py"]:
            tracker.track_file_change("A", f)
        for f in ["ui/x.py", "ui/y.py", "ui/z.py"]:
            tracker.track_file_change("B", f)

        # divergence=1.0 exceeds crit=0.95 → CRIT alert (not WARN)
        sensor = make_sensor(tracker, divergence_warn=0.5, divergence_crit=0.95)
        alerts = sensor.evaluate()
        divergence_alerts = [a for a in alerts if a.kind == AlertKind.OVER_DIVERGED]
        assert divergence_alerts, "Expected a divergence alert"
        assert any(a.level == AlertLevel.CRIT for a in divergence_alerts)

    def test_crit_alert_on_extreme_divergence(self):
        tracker = AgentTracker()
        tracker.new_session("a", agent_id="A")
        tracker.new_session("b", agent_id="B")
        for f in ["auth/a.py", "auth/b.py", "auth/c.py"]:
            tracker.track_file_change("A", f)
        for f in ["ui/x.py", "ui/y.py", "ui/z.py"]:
            tracker.track_file_change("B", f)

        sensor = make_sensor(tracker, divergence_warn=0.5, divergence_crit=0.5)
        alerts = sensor.evaluate()
        crit_alerts = [a for a in alerts if a.kind == AlertKind.OVER_DIVERGED and a.level == AlertLevel.CRIT]
        assert crit_alerts, "Expected a CRIT divergence alert"


class TestOverlapAlerts:
    def test_warn_alert_on_high_overlap(self):
        tracker = AgentTracker()
        tracker.new_session("a", agent_id="A")
        tracker.new_session("b", agent_id="B")
        files = ["main.py", "utils.py", "config.py"]
        for f in files:
            tracker.track_file_change("A", f)
            tracker.track_file_change("B", f)
        tracker.track_output("A", "identical output here for both agents")
        tracker.track_output("B", "identical output here for both agents")

        sensor = make_sensor(tracker, overlap_warn=0.3, overlap_crit=0.95)
        alerts = sensor.evaluate()
        overlap_alerts = [a for a in alerts if a.kind == AlertKind.OVER_OVERLAPPING]
        assert overlap_alerts, "Expected an overlap alert"


class TestErrorRateAlerts:
    def test_warn_on_high_error_rate(self):
        tracker = AgentTracker()
        tracker.new_session("x", agent_id="X")
        for _ in range(8):
            tracker.track_file_change("X", "f.py")
        for _ in range(2):
            tracker.track_error("X", "boom")

        sensor = make_sensor(tracker, error_rate_warn=0.15, error_rate_crit=0.50, min_events=1)
        alerts = sensor.evaluate()
        err_alerts = [a for a in alerts if a.kind == AlertKind.HIGH_ERROR_RATE]
        assert err_alerts

    def test_crit_on_very_high_error_rate(self):
        tracker = AgentTracker()
        tracker.new_session("y", agent_id="Y")
        for _ in range(3):
            tracker.track_file_change("Y", "f.py")
        for _ in range(7):
            tracker.track_error("Y", "oops")

        sensor = make_sensor(tracker, error_rate_warn=0.10, error_rate_crit=0.50, min_events=1)
        alerts = sensor.evaluate()
        crit_alerts = [a for a in alerts if a.kind == AlertKind.HIGH_ERROR_RATE and a.level == AlertLevel.CRIT]
        assert crit_alerts


class TestAlertProperties:
    def test_alert_str_contains_level_and_kind(self):
        alert = Alert(
            level=AlertLevel.WARN,
            kind=AlertKind.OVER_DIVERGED,
            message="test alert",
            agent_ids=["a", "b"],
            metric_value=0.8,
            threshold=0.7,
        )
        s = str(alert)
        assert "WARN" in s
        assert "over_diverged" in s

    def test_alert_history_accumulates(self):
        tracker = AgentTracker()
        tracker.new_session("a", agent_id="A")
        tracker.new_session("b", agent_id="B")
        # force divergence
        for f in ["a1.py", "a2.py", "a3.py"]:
            tracker.track_file_change("A", f)
        for f in ["b1.py", "b2.py", "b3.py"]:
            tracker.track_file_change("B", f)

        sensor = make_sensor(tracker, divergence_warn=0.5, divergence_crit=0.95)
        sensor.evaluate()
        sensor.evaluate()
        # History accumulates across calls
        assert len(sensor.alert_history) >= 2
