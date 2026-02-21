"""
Performance sensors: continuous monitors that fire alerts when thresholds are crossed.

A sensor wraps an AgentTracker + AgentComparator and evaluates the current
state on demand or on a schedule.  Each evaluation produces a list of Alerts,
which can be inspected, logged, or acted on by the caller.

Thresholds (all configurable):
  divergence_warn   – warn when pairwise divergence ≥ this value
  divergence_crit   – critical alert when divergence ≥ this value
  overlap_warn      – warn when pairwise overlap ≥ this value
  overlap_crit      – critical alert when overlap ≥ this value
  error_rate_warn   – warn when a single agent error_rate ≥ this value
  inactivity_secs   – warn when a session has no new events for this long
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from .metrics import AgentComparator, MetricResult
from .tracker import AgentTracker, AgentSession


# ------------------------------------------------------------------ #
# Alert types
# ------------------------------------------------------------------ #


class AlertLevel(str, Enum):
    INFO = "INFO"
    WARN = "WARN"
    CRIT = "CRIT"


class AlertKind(str, Enum):
    OVER_DIVERGED = "over_diverged"
    OVER_OVERLAPPING = "over_overlapping"
    HIGH_ERROR_RATE = "high_error_rate"
    AGENT_INACTIVE = "agent_inactive"
    NO_PROGRESS = "no_progress"


@dataclass
class Alert:
    level: AlertLevel
    kind: AlertKind
    message: str
    timestamp: float = field(default_factory=time.time)
    agent_ids: list[str] = field(default_factory=list)
    metric_value: float | None = None
    threshold: float | None = None

    def __str__(self) -> str:
        ts = time.strftime("%H:%M:%S", time.localtime(self.timestamp))
        agents = ",".join(self.agent_ids) if self.agent_ids else "—"
        value_str = f"  value={self.metric_value:.3f}" if self.metric_value is not None else ""
        threshold_str = f" threshold={self.threshold:.3f}" if self.threshold is not None else ""
        return f"[{ts}] {self.level.value:4s} {self.kind.value:20s} agents=[{agents}]{value_str}{threshold_str}  {self.message}"


# ------------------------------------------------------------------ #
# Sensor configuration
# ------------------------------------------------------------------ #


@dataclass
class SensorConfig:
    # Pairwise divergence thresholds
    divergence_warn: float = 0.70
    divergence_crit: float = 0.85

    # Pairwise overlap thresholds
    overlap_warn: float = 0.60
    overlap_crit: float = 0.75

    # Per-agent thresholds
    error_rate_warn: float = 0.10   # 10% events are errors → warn
    error_rate_crit: float = 0.25   # 25% → crit

    # Inactivity detection (seconds since last event)
    inactivity_warn_secs: float = 300.0   # 5 min
    inactivity_crit_secs: float = 900.0   # 15 min

    # Minimum events before we start raising alerts
    min_events_for_alert: int = 3


# ------------------------------------------------------------------ #
# Main sensor class
# ------------------------------------------------------------------ #


class PerformanceSensor:
    """
    Evaluates the current tracker state and returns a list of Alerts.

    Usage::

        sensor = PerformanceSensor(tracker)
        alerts = sensor.evaluate()
        for alert in alerts:
            print(alert)

        # Continuous monitoring (blocks):
        sensor.monitor(interval=60, on_alert=lambda a: print(a))
    """

    def __init__(
        self,
        tracker: AgentTracker,
        config: SensorConfig | None = None,
    ) -> None:
        self.tracker = tracker
        self.config = config or SensorConfig()
        self._comparator = AgentComparator()
        self._alert_history: list[Alert] = []

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def evaluate(self) -> list[Alert]:
        """Run all sensor checks and return fresh alerts."""
        alerts: list[Alert] = []
        sessions = self.tracker.sessions

        # Per-agent checks
        for session in sessions:
            alerts.extend(self._check_error_rate(session))
            alerts.extend(self._check_inactivity(session))
            alerts.extend(self._check_no_progress(session))

        # Pairwise checks
        for i in range(len(sessions)):
            for j in range(i + 1, len(sessions)):
                a, b = sessions[i], sessions[j]
                if a.event_count < self.config.min_events_for_alert:
                    continue
                if b.event_count < self.config.min_events_for_alert:
                    continue
                result = self._comparator.compare(a, b)
                alerts.extend(self._check_divergence(result))
                alerts.extend(self._check_overlap(result))

        self._alert_history.extend(alerts)
        return alerts

    def monitor(
        self,
        interval: float = 60.0,
        on_alert: Callable[[Alert], None] | None = None,
        max_iterations: int | None = None,
    ) -> None:
        """
        Continuously evaluate at `interval` seconds.
        Calls `on_alert` for each new alert (default: print to stdout).
        Stops after `max_iterations` if provided (useful for tests).
        """
        if on_alert is None:
            on_alert = lambda a: print(str(a))

        iteration = 0
        while True:
            alerts = self.evaluate()
            for alert in alerts:
                on_alert(alert)
            iteration += 1
            if max_iterations is not None and iteration >= max_iterations:
                break
            time.sleep(interval)

    @property
    def alert_history(self) -> list[Alert]:
        return list(self._alert_history)

    def clear_history(self) -> None:
        self._alert_history.clear()

    # ------------------------------------------------------------------ #
    # Individual checks
    # ------------------------------------------------------------------ #

    def _check_divergence(self, result: MetricResult) -> list[Alert]:
        alerts = []
        div = result.divergence
        cfg = self.config

        if div >= cfg.divergence_crit:
            alerts.append(Alert(
                level=AlertLevel.CRIT,
                kind=AlertKind.OVER_DIVERGED,
                message=(
                    f"Agents {result.agent_a} and {result.agent_b} are critically diverged "
                    f"(divergence={div:.2f}). They may be working on incompatible solutions."
                ),
                agent_ids=[result.agent_a, result.agent_b],
                metric_value=div,
                threshold=cfg.divergence_crit,
            ))
        elif div >= cfg.divergence_warn:
            alerts.append(Alert(
                level=AlertLevel.WARN,
                kind=AlertKind.OVER_DIVERGED,
                message=(
                    f"Agents {result.agent_a} and {result.agent_b} are diverging "
                    f"(divergence={div:.2f}). Consider synchronising their work."
                ),
                agent_ids=[result.agent_a, result.agent_b],
                metric_value=div,
                threshold=cfg.divergence_warn,
            ))
        return alerts

    def _check_overlap(self, result: MetricResult) -> list[Alert]:
        alerts = []
        ovl = result.overlap
        cfg = self.config

        if ovl >= cfg.overlap_crit:
            alerts.append(Alert(
                level=AlertLevel.CRIT,
                kind=AlertKind.OVER_OVERLAPPING,
                message=(
                    f"Agents {result.agent_a} and {result.agent_b} are critically overlapping "
                    f"(overlap={ovl:.2f}). They are likely duplicating work."
                ),
                agent_ids=[result.agent_a, result.agent_b],
                metric_value=ovl,
                threshold=cfg.overlap_crit,
            ))
        elif ovl >= cfg.overlap_warn:
            alerts.append(Alert(
                level=AlertLevel.WARN,
                kind=AlertKind.OVER_OVERLAPPING,
                message=(
                    f"Agents {result.agent_a} and {result.agent_b} have significant overlap "
                    f"(overlap={ovl:.2f}). Shared files: {result.details.get('shared_files', [])}."
                ),
                agent_ids=[result.agent_a, result.agent_b],
                metric_value=ovl,
                threshold=cfg.overlap_warn,
            ))
        return alerts

    def _check_error_rate(self, session: AgentSession) -> list[Alert]:
        alerts = []
        if session.event_count < self.config.min_events_for_alert:
            return alerts
        rate = session.error_rate
        cfg = self.config

        if rate >= cfg.error_rate_crit:
            alerts.append(Alert(
                level=AlertLevel.CRIT,
                kind=AlertKind.HIGH_ERROR_RATE,
                message=f"Agent {session.name} ({session.agent_id}) has critical error rate {rate:.1%}.",
                agent_ids=[session.agent_id],
                metric_value=rate,
                threshold=cfg.error_rate_crit,
            ))
        elif rate >= cfg.error_rate_warn:
            alerts.append(Alert(
                level=AlertLevel.WARN,
                kind=AlertKind.HIGH_ERROR_RATE,
                message=f"Agent {session.name} ({session.agent_id}) has elevated error rate {rate:.1%}.",
                agent_ids=[session.agent_id],
                metric_value=rate,
                threshold=cfg.error_rate_warn,
            ))
        return alerts

    def _check_inactivity(self, session: AgentSession) -> list[Alert]:
        """Alert if no events have been recorded recently."""
        if not session.events or session.end_time is not None:
            return []

        last_event_time = session.events[-1].timestamp
        idle_secs = time.time() - last_event_time
        cfg = self.config

        if idle_secs >= cfg.inactivity_crit_secs:
            return [Alert(
                level=AlertLevel.CRIT,
                kind=AlertKind.AGENT_INACTIVE,
                message=(
                    f"Agent {session.name} ({session.agent_id}) has been inactive "
                    f"for {idle_secs:.0f}s (threshold={cfg.inactivity_crit_secs:.0f}s)."
                ),
                agent_ids=[session.agent_id],
                metric_value=idle_secs,
                threshold=cfg.inactivity_crit_secs,
            )]
        elif idle_secs >= cfg.inactivity_warn_secs:
            return [Alert(
                level=AlertLevel.WARN,
                kind=AlertKind.AGENT_INACTIVE,
                message=(
                    f"Agent {session.name} ({session.agent_id}) has been inactive "
                    f"for {idle_secs:.0f}s."
                ),
                agent_ids=[session.agent_id],
                metric_value=idle_secs,
                threshold=cfg.inactivity_warn_secs,
            )]
        return []

    def _check_no_progress(self, session: AgentSession) -> list[Alert]:
        """Alert if the agent has been running a long time with no commits."""
        if session.end_time is not None:
            return []
        duration = session.duration
        # Only flag if running >10min and zero commits
        if duration > 600 and session.commit_count == 0 and session.event_count >= self.config.min_events_for_alert:
            return [Alert(
                level=AlertLevel.WARN,
                kind=AlertKind.NO_PROGRESS,
                message=(
                    f"Agent {session.name} ({session.agent_id}) has been running "
                    f"{duration:.0f}s with no commits."
                ),
                agent_ids=[session.agent_id],
                metric_value=duration,
            )]
        return []
