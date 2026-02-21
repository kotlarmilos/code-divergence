"""
code-divergence: Track AI agent performance across divergence and overlap dimensions.

Core concepts:
- Divergence: agents working in incompatible or too-different directions
- Overlap: agents duplicating each other's work
- Sensors: continuous monitors that fire alerts when thresholds are crossed
"""

from .tracker import AgentTracker, AgentSession, Event, EventType
from .metrics import DivergenceCalculator, OverlapCalculator, MetricResult
from .sensors import PerformanceSensor, Alert, AlertLevel
from .reporter import Reporter
from .git_ingester import GitIngester, find_symbol_conflicts, SymbolConflict

__all__ = [
    "AgentTracker",
    "AgentSession",
    "Event",
    "EventType",
    "DivergenceCalculator",
    "OverlapCalculator",
    "MetricResult",
    "PerformanceSensor",
    "Alert",
    "AlertLevel",
    "Reporter",
    "GitIngester",
    "find_symbol_conflicts",
    "SymbolConflict",
]

__version__ = "0.1.0"
