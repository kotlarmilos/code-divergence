"""
Reporting: generate human-readable and machine-readable summaries of agent activity.
"""

from __future__ import annotations

import json
import time
from typing import TextIO
import sys

from .metrics import AgentComparator, MetricResult
from .sensors import Alert, AlertLevel, PerformanceSensor, SensorConfig
from .tracker import AgentTracker, AgentSession


_BAR_WIDTH = 30


def _bar(value: float, width: int = _BAR_WIDTH) -> str:
    """ASCII progress bar for 0.0–1.0 values."""
    filled = round(value * width)
    return "[" + "#" * filled + "." * (width - filled) + f"] {value:.2f}"


def _level_color(level: AlertLevel) -> str:
    return {
        AlertLevel.INFO: "",
        AlertLevel.WARN: "! ",
        AlertLevel.CRIT: "!! ",
    }.get(level, "")


class Reporter:
    """
    Generates reports from tracker state.

    All output methods write to a TextIO (default stdout) so they are
    easy to redirect to files or capture in tests.
    """

    def __init__(self, tracker: AgentTracker, sensor_config: SensorConfig | None = None) -> None:
        self.tracker = tracker
        self._comparator = AgentComparator()
        self._sensor_config = sensor_config or SensorConfig()

    # ------------------------------------------------------------------ #
    # Text report
    # ------------------------------------------------------------------ #

    def print_report(self, out: TextIO = sys.stdout) -> None:
        """Full human-readable report."""
        sessions = self.tracker.sessions
        _w = out.write

        _w("=" * 70 + "\n")
        _w("  CODE DIVERGENCE — Agent Performance Report\n")
        _w(f"  Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        _w("=" * 70 + "\n\n")

        if not sessions:
            _w("No agent sessions recorded.\n")
            return

        # --- Per-agent summary ---
        _w(f"{'AGENT SESSIONS':=<70}\n\n")
        for s in sessions:
            status = "active" if s.end_time is None else f"ended after {s.duration:.0f}s"
            _w(f"  Agent: {s.name}  (id={s.agent_id})  [{status}]\n")
            _w(f"    Events     : {s.event_count}\n")
            _w(f"    Files      : {len(s.files_touched)}  {sorted(s.files_touched) or '—'}\n")
            _w(f"    Commits    : {s.commit_count}\n")
            _w(f"    Tasks      : {s.task_count}\n")
            _w(f"    Errors     : {len(s._errors)} (rate={s.error_rate:.1%})\n")
            _w(f"    Outputs    : {len(s._outputs)}\n")
            _w("\n")

        if len(sessions) < 2:
            _w("(Need ≥ 2 agents to compute pairwise metrics)\n")
            return

        # --- Pairwise metrics ---
        results = self._comparator.compare_all(sessions)
        _w(f"{'PAIRWISE METRICS':=<70}\n\n")
        for r in results:
            self._print_pair_result(r, out)

        # --- Sensor alerts ---
        sensor = PerformanceSensor(self.tracker, self._sensor_config)
        alerts = sensor.evaluate()
        _w(f"{'SENSOR ALERTS':=<70}\n\n")
        if not alerts:
            _w("  No alerts — all agents within thresholds.\n\n")
        else:
            for alert in alerts:
                prefix = _level_color(alert.level)
                _w(f"  {prefix}{alert}\n")
            _w("\n")

        _w("=" * 70 + "\n")

    def _print_pair_result(self, r: MetricResult, out: TextIO) -> None:
        _w = out.write
        _w(f"  {r.agent_a}  vs  {r.agent_b}\n")
        _w(f"    Divergence  {_bar(r.divergence)}\n")
        _w(f"    Overlap     {_bar(r.overlap)}\n")
        _w(f"    ├─ File divergence  : {r.file_divergence:.3f}\n")
        _w(f"    ├─ File overlap     : {r.file_overlap:.3f}\n")
        _w(f"    ├─ Output divergence: {r.output_divergence:.3f}\n")
        _w(f"    ├─ Output overlap   : {r.output_overlap:.3f}\n")
        _w(f"    └─ Commit divergence: {r.commit_divergence:.3f}\n")
        shared = r.details.get("shared_files", [])
        only_a = r.details.get("only_in_a", [])
        only_b = r.details.get("only_in_b", [])
        if shared:
            _w(f"    Shared files   : {shared}\n")
        if only_a:
            _w(f"    Only in {r.agent_a}: {only_a}\n")
        if only_b:
            _w(f"    Only in {r.agent_b}: {only_b}\n")
        status = r.summary().split("status=")[1]
        _w(f"    Status: {status}\n")
        _w("\n")

    # ------------------------------------------------------------------ #
    # JSON export
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        sessions = self.tracker.sessions
        results = self._comparator.compare_all(sessions)
        sensor = PerformanceSensor(self.tracker, self._sensor_config)
        alerts = sensor.evaluate()

        # Stats engine diagnostic (Layer 1-4)
        from .stats_engine import StatsEngine
        engine = StatsEngine(self.tracker)
        diagnostic = engine.diagnose()

        return {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "agents": [s.summary() for s in sessions],
            "pairwise": [
                {
                    "agent_a": r.agent_a,
                    "agent_b": r.agent_b,
                    "divergence": r.divergence,
                    "overlap": r.overlap,
                    "file_divergence": r.file_divergence,
                    "file_overlap": r.file_overlap,
                    "output_divergence": r.output_divergence,
                    "output_overlap": r.output_overlap,
                    "commit_divergence": r.commit_divergence,
                    "tool_divergence": r.tool_divergence,
                    "details": r.details,
                    "flags": {
                        "over_diverged": r.is_over_diverged,
                        "over_overlapping": r.is_over_overlapping,
                    },
                }
                for r in results
            ],
            "alerts": [
                {
                    "level": a.level.value,
                    "kind": a.kind.value,
                    "message": a.message,
                    "agent_ids": a.agent_ids,
                    "metric_value": a.metric_value,
                    "threshold": a.threshold,
                    "timestamp": a.timestamp,
                }
                for a in alerts
            ],
            "diagnostic": diagnostic.to_dict(),
        }

    def print_json(self, out: TextIO = sys.stdout, indent: int = 2) -> None:
        json.dump(self.to_dict(), out, indent=indent)
        out.write("\n")
