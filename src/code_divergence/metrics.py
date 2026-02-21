"""
Divergence and overlap metrics for comparing agent sessions.

Divergence (0.0 – 1.0):
  0.0 = agents working in identical direction
  1.0 = agents working on completely separate things

Overlap (0.0 – 1.0):
  0.0 = no shared work
  1.0 = agents doing exactly the same work

Both dimensions matter independently:
  - High divergence + low overlap → agents badly misaligned (should coordinate)
  - Low divergence + high overlap → agents duplicating work (wasteful)
  - Moderate divergence + low overlap → healthy parallel work
"""

from __future__ import annotations

import difflib
import math
import re
from dataclasses import dataclass

from .tracker import AgentSession


# ------------------------------------------------------------------ #
# Core result type
# ------------------------------------------------------------------ #


@dataclass
class MetricResult:
    """Pairwise comparison result between two agent sessions."""

    agent_a: str
    agent_b: str
    divergence: float          # 0 = same direction, 1 = completely different
    overlap: float             # 0 = no shared work, 1 = identical work
    file_divergence: float
    file_overlap: float
    output_divergence: float
    output_overlap: float
    commit_divergence: float
    tool_divergence: float
    details: dict

    @property
    def is_over_diverged(self) -> bool:
        """True when divergence meets the default warning threshold (0.70)."""
        return self.divergence >= 0.70

    @property
    def is_over_overlapping(self) -> bool:
        """True when overlap meets the default warning threshold (0.60)."""
        return self.overlap >= 0.60

    def summary(self) -> str:
        flags = []
        if self.is_over_diverged:
            flags.append("OVER-DIVERGED")
        if self.is_over_overlapping:
            flags.append("OVER-OVERLAPPING")
        status = " | ".join(flags) if flags else "OK"
        return (
            f"[{self.agent_a} vs {self.agent_b}] "
            f"divergence={self.divergence:.2f} overlap={self.overlap:.2f} "
            f"status={status}"
        )


# ------------------------------------------------------------------ #
# Utilities
# ------------------------------------------------------------------ #


def _jaccard_distance(a: set, b: set) -> float:
    """1 - |a ∩ b| / |a ∪ b|.  Returns 1.0 if both sets are empty."""
    union = a | b
    if not union:
        return 0.0  # both empty → no divergence
    return 1.0 - len(a & b) / len(union)


def _jaccard_similarity(a: set, b: set) -> float:
    """1 - jaccard_distance. 1.0 when sets are identical."""
    return 1.0 - _jaccard_distance(a, b)


def _text_similarity(a: str, b: str) -> float:
    """
    Similarity ratio between two strings using difflib SequenceMatcher.
    Returns 0.0 – 1.0, where 1.0 = identical.
    Handles empty strings gracefully.
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def _text_divergence(a: str, b: str) -> float:
    return 1.0 - _text_similarity(a, b)


def _normalise_text(text: str) -> str:
    """Lowercase, collapse whitespace – makes similarity more meaningful."""
    return re.sub(r"\s+", " ", text.lower().strip())


def _token_set(text: str) -> set[str]:
    """Set of alphanumeric tokens from text (for overlap estimation)."""
    return set(re.findall(r"\w+", text.lower()))


def _weighted_average(values: list[tuple[float, float]]) -> float:
    """Compute weighted average of (value, weight) pairs."""
    total_weight = sum(w for _, w in values)
    if total_weight == 0:
        return 0.0
    return sum(v * w for v, w in values) / total_weight


# ------------------------------------------------------------------ #
# Divergence calculator
# ------------------------------------------------------------------ #


class DivergenceCalculator:
    """
    Measures how different two agents' work directions are.

    High divergence means agents are working on very different things,
    which may indicate misalignment or coordination problems.
    """

    def file_divergence(self, a: AgentSession, b: AgentSession) -> float:
        """Jaccard distance between file sets touched by each agent."""
        return _jaccard_distance(a.files_touched, b.files_touched)

    def output_divergence(self, a: AgentSession, b: AgentSession) -> float:
        """
        Text divergence between combined outputs.
        Uses token-level comparison to be robust against whitespace differences.
        """
        tokens_a = _token_set(a.all_output_text)
        tokens_b = _token_set(b.all_output_text)
        return _jaccard_distance(tokens_a, tokens_b)

    def commit_divergence(self, a: AgentSession, b: AgentSession) -> float:
        """
        How different are the commit messages of the two agents?
        Proxy for strategic direction divergence.
        """
        msg_a = " ".join(c.get("message", "") for c in a._commits)
        msg_b = " ".join(c.get("message", "") for c in b._commits)
        tokens_a = _token_set(msg_a)
        tokens_b = _token_set(msg_b)
        if not tokens_a and not tokens_b:
            return 0.0
        return _jaccard_distance(tokens_a, tokens_b)

    def tool_divergence(self, a: AgentSession, b: AgentSession) -> float:
        """How different are the tools each agent uses?"""
        return _jaccard_distance(set(a._tool_calls), set(b._tool_calls))

    def overall(self, a: AgentSession, b: AgentSession) -> float:
        """
        Weighted composite divergence score.

        File divergence carries the most weight because it most directly
        reflects whether agents are working on the same problem area.

        Dimensions with no data in either session are excluded from the
        weighted average so that the absence of commits/outputs does not
        dilute a clear file-based divergence signal.
        """
        components: list[tuple[float, float]] = []

        # Files – always included (even empty file sets produce a signal)
        components.append((self.file_divergence(a, b), 0.45))

        # Outputs – only include when at least one agent has produced output
        if a._outputs or b._outputs:
            components.append((self.output_divergence(a, b), 0.30))

        # Commits – only include when at least one agent has commits
        if a._commits or b._commits:
            components.append((self.commit_divergence(a, b), 0.15))

        # Tools – only include when at least one agent has used tools
        if a._tool_calls or b._tool_calls:
            components.append((self.tool_divergence(a, b), 0.10))

        return _weighted_average(components)


# ------------------------------------------------------------------ #
# Overlap calculator
# ------------------------------------------------------------------ #


class OverlapCalculator:
    """
    Measures how much two agents' work duplicates each other.

    High overlap means agents are doing the same work, wasting compute.
    """

    def file_overlap(self, a: AgentSession, b: AgentSession) -> float:
        """Jaccard similarity between file sets."""
        return _jaccard_similarity(a.files_touched, b.files_touched)

    def output_overlap(self, a: AgentSession, b: AgentSession) -> float:
        """
        Token-level Jaccard similarity between outputs.
        High overlap = agents generating very similar content.
        """
        tokens_a = _token_set(a.all_output_text)
        tokens_b = _token_set(b.all_output_text)
        return _jaccard_similarity(tokens_a, tokens_b)

    def commit_overlap(self, a: AgentSession, b: AgentSession) -> float:
        """How much do commit messages overlap in vocabulary?"""
        msg_a = " ".join(c.get("message", "") for c in a._commits)
        msg_b = " ".join(c.get("message", "") for c in b._commits)
        tokens_a = _token_set(msg_a)
        tokens_b = _token_set(msg_b)
        if not tokens_a and not tokens_b:
            return 0.0
        return _jaccard_similarity(tokens_a, tokens_b)

    def task_overlap(self, a: AgentSession, b: AgentSession) -> float:
        """How similar are the task descriptions?"""
        desc_a = " ".join(t.get("description", "") for t in a._tasks)
        desc_b = " ".join(t.get("description", "") for t in b._tasks)
        tokens_a = _token_set(desc_a)
        tokens_b = _token_set(desc_b)
        if not tokens_a and not tokens_b:
            return 0.0
        return _jaccard_similarity(tokens_a, tokens_b)

    def overall(self, a: AgentSession, b: AgentSession) -> float:
        """
        Weighted composite overlap score.

        Only dimensions with data in at least one session are included,
        so missing commits/tasks don't deflate a clear file-level overlap.
        """
        components: list[tuple[float, float]] = []

        components.append((self.file_overlap(a, b), 0.40))

        if a._outputs or b._outputs:
            components.append((self.output_overlap(a, b), 0.35))

        if a._commits or b._commits:
            components.append((self.commit_overlap(a, b), 0.15))

        if a._tasks or b._tasks:
            components.append((self.task_overlap(a, b), 0.10))

        return _weighted_average(components)


# ------------------------------------------------------------------ #
# Combined comparison
# ------------------------------------------------------------------ #


class AgentComparator:
    """Runs both divergence and overlap calculations in one pass."""

    def __init__(self) -> None:
        self._div = DivergenceCalculator()
        self._ovl = OverlapCalculator()

    def compare(self, a: AgentSession, b: AgentSession) -> MetricResult:
        fd = self._div.file_divergence(a, b)
        od = self._div.output_divergence(a, b)
        cd = self._div.commit_divergence(a, b)
        td = self._div.tool_divergence(a, b)
        div = self._div.overall(a, b)

        fo = self._ovl.file_overlap(a, b)
        oo = self._ovl.output_overlap(a, b)
        ovl = self._ovl.overall(a, b)

        shared_files = sorted(a.files_touched & b.files_touched)
        only_a = sorted(a.files_touched - b.files_touched)
        only_b = sorted(b.files_touched - a.files_touched)

        return MetricResult(
            agent_a=a.agent_id,
            agent_b=b.agent_id,
            divergence=round(div, 4),
            overlap=round(ovl, 4),
            file_divergence=round(fd, 4),
            file_overlap=round(fo, 4),
            output_divergence=round(od, 4),
            output_overlap=round(oo, 4),
            commit_divergence=round(cd, 4),
            tool_divergence=round(td, 4),
            details={
                "shared_files": shared_files,
                "only_in_a": only_a,
                "only_in_b": only_b,
                "commits_a": len(a._commits),
                "commits_b": len(b._commits),
                "errors_a": len(a._errors),
                "errors_b": len(b._errors),
            },
        )

    def compare_all(self, sessions: list[AgentSession]) -> list[MetricResult]:
        """Compare every pair of sessions."""
        results = []
        for i in range(len(sessions)):
            for j in range(i + 1, len(sessions)):
                results.append(self.compare(sessions[i], sessions[j]))
        return results
