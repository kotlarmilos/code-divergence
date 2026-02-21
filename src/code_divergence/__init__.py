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
]

__version__ = "0.1.0"
