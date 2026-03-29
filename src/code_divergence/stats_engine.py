"""
Statistical engine: four-layer diagnostic analysis for agent routing quality.

Layers:
  1. Distributional analysis (static) — JSD, routing entropy, coverage density
  2. Information-theoretic health — mutual information, conditional entropy
  3. Drift detection (temporal) — KL divergence, CUSUM, PSI
  4. Statistical process control — p-chart, Western Electric rules

Every metric maps to a conditional recommendation in the final diagnostic report.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .tracker import AgentSession, AgentTracker

# ------------------------------------------------------------------ #
# Shared math utilities
# ------------------------------------------------------------------ #


def _to_distribution(tokens: set[str], vocab: set[str], smoothing: float = 1e-10) -> dict[str, float]:
    """Convert a token set into a probability distribution over a shared vocabulary."""
    if not vocab:
        return {}
    counts: dict[str, float] = {}
    for t in vocab:
        counts[t] = (1.0 if t in tokens else 0.0) + smoothing
    total = sum(counts.values())
    return {t: c / total for t, c in counts.items()}


def _shannon_entropy(dist: dict[str, float]) -> float:
    """H(X) = -Σ p(x) log2 p(x).  Returns 0.0 for empty distributions."""
    if not dist:
        return 0.0
    return -sum(p * math.log2(p) for p in dist.values() if p > 0)


def _kl_divergence(p: dict[str, float], q: dict[str, float]) -> float:
    """
    D_KL(P || Q) = Σ P(x) log2(P(x) / Q(x)).

    Requires that Q(x) > 0 wherever P(x) > 0.  Caller must ensure
    distributions share the same support (use smoothing).
    """
    if not p or not q:
        return 0.0
    total = 0.0
    for x, px in p.items():
        qx = q.get(x, 0.0)
        if px > 0 and qx > 0:
            total += px * math.log2(px / qx)
    return max(total, 0.0)


def _jsd(p: dict[str, float], q: dict[str, float]) -> float:
    """
    Jensen-Shannon Divergence: JSD(P, Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
    where M = 0.5*(P + Q).  Bounded [0, 1] when using log base 2.
    """
    if not p and not q:
        return 0.0
    all_keys = set(p) | set(q)
    m: dict[str, float] = {}
    for k in all_keys:
        m[k] = 0.5 * p.get(k, 0.0) + 0.5 * q.get(k, 0.0)
    return 0.5 * _kl_divergence(p, m) + 0.5 * _kl_divergence(q, m)


def _token_set_from_session(session: AgentSession) -> set[str]:
    """Collect all meaningful tokens from a session's outputs, files, and commits."""
    import re
    tokens: set[str] = set()
    # File paths → path segments
    for f in session.files_touched:
        tokens.update(re.findall(r"\w+", f.lower()))
    # Output text tokens
    for output in session._outputs:
        tokens.update(re.findall(r"\b[a-zA-Z_]\w{2,}\b", output.lower()))
    # Commit message tokens
    for commit in session._commits:
        msg = commit.get("message", "")
        tokens.update(re.findall(r"\b[a-zA-Z_]\w{2,}\b", msg.lower()))
    # Task description tokens
    for task in session._tasks:
        desc = task.get("description", "")
        tokens.update(re.findall(r"\b[a-zA-Z_]\w{2,}\b", desc.lower()))
    # Tool call names
    for tool in session._tool_calls:
        tokens.update(re.findall(r"\w+", tool.lower()))
    return tokens


# ------------------------------------------------------------------ #
# Layer 1: Distributional analysis
# ------------------------------------------------------------------ #


@dataclass
class PairwiseJSD:
    """JSD result between two agent sessions."""
    agent_a: str
    agent_b: str
    jsd_score: float  # 0 = identical, 1 = maximally different

    @property
    def is_near_duplicate(self) -> bool:
        return self.jsd_score < 0.1

    @property
    def is_well_separated(self) -> bool:
        return self.jsd_score > 0.5


@dataclass
class RoutingEntropy:
    """Routing clarity for a single probe token."""
    probe: str
    entropy: float
    max_entropy: float
    assignment: dict[str, float]  # P(agent | probe)

    @property
    def clarity(self) -> float:
        """1.0 = perfectly clear routing, 0.0 = maximum confusion."""
        if self.max_entropy == 0:
            return 1.0
        return 1.0 - (self.entropy / self.max_entropy)


@dataclass
class CoverageDensity:
    """Coverage analysis across a probe set."""
    mean_max_similarity: float
    std_max_similarity: float
    uncovered_fraction: float  # fraction of probes with max_sim < threshold
    coverage_uniformity: float  # 1 = uniform, 0 = very patchy


class DistributionalAnalyser:
    """Layer 1: static distributional analysis of agent sessions."""

    def pairwise_jsd(self, sessions: list[AgentSession]) -> list[PairwiseJSD]:
        """Compute JSD between every pair of sessions."""
        results: list[PairwiseJSD] = []
        token_sets = {s.agent_id: _token_set_from_session(s) for s in sessions}
        all_tokens = set()
        for ts in token_sets.values():
            all_tokens |= ts

        dists = {
            aid: _to_distribution(ts, all_tokens)
            for aid, ts in token_sets.items()
        }

        for i in range(len(sessions)):
            for j in range(i + 1, len(sessions)):
                a_id = sessions[i].agent_id
                b_id = sessions[j].agent_id
                score = _jsd(dists[a_id], dists[b_id])
                results.append(PairwiseJSD(
                    agent_a=a_id,
                    agent_b=b_id,
                    jsd_score=round(score, 6),
                ))
        return results

    def routing_entropy(
        self,
        sessions: list[AgentSession],
        probes: list[str] | None = None,
    ) -> list[RoutingEntropy]:
        """
        For each probe token, compute P(agent | probe) and the entropy of that
        distribution.  High entropy = the probe is ambiguous (could go to any agent).

        If probes are not provided, uses the union of all agent tokens as the
        probe set (self-analysis).
        """
        if len(sessions) < 2:
            return []

        token_sets = {s.agent_id: _token_set_from_session(s) for s in sessions}

        if probes is None:
            all_tokens: set[str] = set()
            for ts in token_sets.values():
                all_tokens |= ts
            probes = sorted(all_tokens)

        if not probes:
            return []

        max_ent = math.log2(len(sessions)) if len(sessions) > 1 else 0.0
        results: list[RoutingEntropy] = []

        for probe in probes:
            # P(agent | probe) ∝ 1 if probe in agent's token set, else ε
            raw: dict[str, float] = {}
            for aid, ts in token_sets.items():
                raw[aid] = 1.0 if probe in ts else 1e-6
            total = sum(raw.values())
            assignment = {aid: v / total for aid, v in raw.items()}
            ent = _shannon_entropy(assignment)
            results.append(RoutingEntropy(
                probe=probe,
                entropy=round(ent, 6),
                max_entropy=round(max_ent, 6),
                assignment=assignment,
            ))

        return results

    def system_routing_clarity(self, sessions: list[AgentSession], probes: list[str] | None = None) -> float:
        """
        Aggregate routing clarity: 1.0 = every probe maps cleanly to one agent,
        0.0 = maximum confusion.
        """
        entries = self.routing_entropy(sessions, probes)
        if not entries:
            return 1.0
        return sum(e.clarity for e in entries) / len(entries)

    def coverage_density(
        self,
        sessions: list[AgentSession],
        probes: list[str] | None = None,
        coverage_threshold: float = 0.5,
    ) -> CoverageDensity:
        """
        For each probe, compute max similarity to any agent's token set.
        Reports mean, std, uncovered fraction, and uniformity.
        """
        token_sets = [_token_set_from_session(s) for s in sessions]

        if probes is None:
            all_tokens: set[str] = set()
            for ts in token_sets:
                all_tokens |= ts
            probes = sorted(all_tokens)

        if not probes:
            return CoverageDensity(
                mean_max_similarity=0.0,
                std_max_similarity=0.0,
                uncovered_fraction=1.0,
                coverage_uniformity=0.0,
            )

        max_sims: list[float] = []
        for probe in probes:
            best = 0.0
            for ts in token_sets:
                sim = 1.0 if probe in ts else 0.0
                best = max(best, sim)
            max_sims.append(best)

        n = len(max_sims)
        mean_sim = sum(max_sims) / n
        variance = sum((x - mean_sim) ** 2 for x in max_sims) / n
        std_sim = math.sqrt(variance)
        uncovered = sum(1 for s in max_sims if s < coverage_threshold) / n

        # Uniformity: 1 - normalised std (0 = patchy, 1 = uniform)
        max_possible_std = 0.5  # max std for binary values
        uniformity = max(0.0, 1.0 - std_sim / max_possible_std) if max_possible_std > 0 else 1.0

        return CoverageDensity(
            mean_max_similarity=round(mean_sim, 6),
            std_max_similarity=round(std_sim, 6),
            uncovered_fraction=round(uncovered, 6),
            coverage_uniformity=round(uniformity, 6),
        )


# ------------------------------------------------------------------ #
# Layer 2: Information-theoretic health metrics
# ------------------------------------------------------------------ #


@dataclass
class MutualInformationResult:
    """Mutual information between two agent descriptions."""
    agent_a: str
    agent_b: str
    mi_score: float  # higher = more redundant

    @property
    def is_redundant(self) -> bool:
        """High MI means knowing one agent's description tells you a lot about the other."""
        return self.mi_score > 0.5


@dataclass
class HealthMetrics:
    """Layer 2 aggregate results."""
    mutual_information: list[MutualInformationResult]
    conditional_entropy: float  # H(agent | token) averaged over probes
    per_agent_entropy: dict[str, float]  # individual description entropy


class InformationTheoreticAnalyser:
    """Layer 2: mutual information and conditional entropy."""

    def mutual_information(self, sessions: list[AgentSession]) -> list[MutualInformationResult]:
        """
        MI(A; B) = H(A) + H(B) - H(A, B)

        Here H(A) is the entropy of agent A's token distribution and H(A,B)
        is the joint entropy over the union of both token sets.
        """
        results: list[MutualInformationResult] = []
        token_sets = {s.agent_id: _token_set_from_session(s) for s in sessions}

        for i in range(len(sessions)):
            for j in range(i + 1, len(sessions)):
                a_id = sessions[i].agent_id
                b_id = sessions[j].agent_id
                ts_a = token_sets[a_id]
                ts_b = token_sets[b_id]
                all_tokens = ts_a | ts_b

                if not all_tokens:
                    results.append(MutualInformationResult(a_id, b_id, 0.0))
                    continue

                dist_a = _to_distribution(ts_a, all_tokens)
                dist_b = _to_distribution(ts_b, all_tokens)

                h_a = _shannon_entropy(dist_a)
                h_b = _shannon_entropy(dist_b)

                # Joint distribution: P(x in A, x in B) — represents co-occurrence
                joint: dict[str, float] = {}
                smoothing = 1e-10
                for t in all_tokens:
                    in_a = 1.0 if t in ts_a else 0.0
                    in_b = 1.0 if t in ts_b else 0.0
                    # Encode as 4 possible states: (0,0), (0,1), (1,0), (1,1)
                    joint[f"{t}_11"] = (in_a * in_b) + smoothing
                    joint[f"{t}_10"] = (in_a * (1 - in_b)) + smoothing
                    joint[f"{t}_01"] = ((1 - in_a) * in_b) + smoothing
                    joint[f"{t}_00"] = ((1 - in_a) * (1 - in_b)) + smoothing

                total_j = sum(joint.values())
                joint_dist = {k: v / total_j for k, v in joint.items()}
                h_joint = _shannon_entropy(joint_dist)

                mi = max(0.0, h_a + h_b - h_joint)
                # Normalise by min(H(A), H(B)) to get [0, 1] range
                normaliser = min(h_a, h_b) if min(h_a, h_b) > 0 else 1.0
                mi_normalised = min(mi / normaliser, 1.0)

                results.append(MutualInformationResult(
                    agent_a=a_id,
                    agent_b=b_id,
                    mi_score=round(mi_normalised, 6),
                ))
        return results

    def conditional_entropy(self, sessions: list[AgentSession], probes: list[str] | None = None) -> float:
        """
        H(agent | token) averaged over probe tokens.

        Low = clean routing (tokens map clearly to agents).
        High = the system is guessing.
        """
        if len(sessions) < 2:
            return 0.0

        token_sets = {s.agent_id: _token_set_from_session(s) for s in sessions}

        if probes is None:
            all_tokens: set[str] = set()
            for ts in token_sets.values():
                all_tokens |= ts
            probes = sorted(all_tokens)

        if not probes:
            return 0.0

        max_ent = math.log2(len(sessions)) if len(sessions) > 1 else 0.0
        total_entropy = 0.0

        for probe in probes:
            raw: dict[str, float] = {}
            for aid, ts in token_sets.items():
                raw[aid] = 1.0 if probe in ts else 1e-6
            total = sum(raw.values())
            assignment = {aid: v / total for aid, v in raw.items()}
            total_entropy += _shannon_entropy(assignment)

        avg = total_entropy / len(probes)
        # Normalise to [0, 1]
        if max_ent > 0:
            return round(avg / max_ent, 6)
        return 0.0

    def per_agent_entropy(self, sessions: list[AgentSession]) -> dict[str, float]:
        """Entropy of each agent's token distribution (proxy for description perplexity)."""
        result: dict[str, float] = {}
        for s in sessions:
            tokens = _token_set_from_session(s)
            if not tokens:
                result[s.agent_id] = 0.0
                continue
            dist = _to_distribution(tokens, tokens)
            result[s.agent_id] = round(_shannon_entropy(dist), 6)
        return result

    def health_metrics(self, sessions: list[AgentSession]) -> HealthMetrics:
        return HealthMetrics(
            mutual_information=self.mutual_information(sessions),
            conditional_entropy=self.conditional_entropy(sessions),
            per_agent_entropy=self.per_agent_entropy(sessions),
        )


# ------------------------------------------------------------------ #
# Layer 3: Drift detection (temporal)
# ------------------------------------------------------------------ #


@dataclass
class DriftSnapshot:
    """A point-in-time distribution snapshot for an agent."""
    agent_id: str
    timestamp: float
    distribution: dict[str, float]


@dataclass
class DriftResult:
    """Drift detection result for one agent."""
    agent_id: str
    kl_divergence: float  # D_KL(current || baseline)
    cusum_value: float
    cusum_alert: bool
    psi: float
    psi_alert: bool  # PSI > 0.2 = significant shift


@dataclass
class CUSUMState:
    """CUSUM accumulator state."""
    s_pos: float = 0.0
    s_neg: float = 0.0
    alert: bool = False


def _compute_psi(baseline: dict[str, float], current: dict[str, float], n_bins: int = 10) -> float:
    """
    Population Stability Index.

    PSI = Σ (P_i - Q_i) * ln(P_i / Q_i) across bins.
    PSI < 0.1: no significant shift
    PSI 0.1–0.2: moderate shift
    PSI > 0.2: significant shift
    """
    if not baseline or not current:
        return 0.0

    all_keys = sorted(set(baseline) | set(current))
    if not all_keys:
        return 0.0

    # Bin the distributions
    bin_size = max(1, len(all_keys) // n_bins)
    bins_b: list[float] = []
    bins_c: list[float] = []

    for i in range(0, len(all_keys), bin_size):
        chunk = all_keys[i:i + bin_size]
        bins_b.append(sum(baseline.get(k, 0.0) for k in chunk))
        bins_c.append(sum(current.get(k, 0.0) for k in chunk))

    # Normalise bins
    total_b = sum(bins_b) or 1.0
    total_c = sum(bins_c) or 1.0
    bins_b = [b / total_b for b in bins_b]
    bins_c = [c / total_c for c in bins_c]

    # Compute PSI with smoothing
    smoothing = 1e-6
    psi = 0.0
    for pb, pc in zip(bins_b, bins_c):
        pb = max(pb, smoothing)
        pc = max(pc, smoothing)
        psi += (pc - pb) * math.log(pc / pb)

    return max(psi, 0.0)


class DriftDetector:
    """
    Layer 3: temporal drift detection.

    Compares current agent distributions against a baseline snapshot.
    Uses rolling KL divergence, CUSUM, and PSI.
    """

    def __init__(self, cusum_threshold: float = 5.0, cusum_slack: float = 0.5) -> None:
        self.cusum_threshold = cusum_threshold
        self.cusum_slack = cusum_slack
        self._cusum_states: dict[str, CUSUMState] = {}
        self._baselines: dict[str, dict[str, float]] = {}

    def set_baseline(self, agent_id: str, distribution: dict[str, float]) -> None:
        """Set the baseline distribution for an agent."""
        self._baselines[agent_id] = dict(distribution)
        self._cusum_states[agent_id] = CUSUMState()

    def set_baselines_from_sessions(self, sessions: list[AgentSession]) -> None:
        """Convenience: set baselines from current session state."""
        for s in sessions:
            tokens = _token_set_from_session(s)
            dist = _to_distribution(tokens, tokens)
            self.set_baseline(s.agent_id, dist)

    def detect_drift(self, sessions: list[AgentSession]) -> list[DriftResult]:
        """Compare current session state against baselines."""
        results: list[DriftResult] = []
        for s in sessions:
            if s.agent_id not in self._baselines:
                continue

            baseline = self._baselines[s.agent_id]
            current_tokens = _token_set_from_session(s)
            all_tokens = set(baseline) | current_tokens
            current_dist = _to_distribution(current_tokens, all_tokens)
            baseline_smoothed = _to_distribution(
                {k for k, v in baseline.items() if v > 1e-8},
                all_tokens,
            )

            # KL divergence
            kl = _kl_divergence(current_dist, baseline_smoothed)

            # CUSUM update
            state = self._cusum_states.get(s.agent_id, CUSUMState())
            state.s_pos = max(0.0, state.s_pos + kl - self.cusum_slack)
            state.s_neg = max(0.0, state.s_neg - kl + self.cusum_slack)
            state.alert = state.s_pos > self.cusum_threshold or state.s_neg > self.cusum_threshold
            self._cusum_states[s.agent_id] = state

            # PSI
            psi = _compute_psi(baseline_smoothed, current_dist)

            results.append(DriftResult(
                agent_id=s.agent_id,
                kl_divergence=round(kl, 6),
                cusum_value=round(state.s_pos, 6),
                cusum_alert=state.alert,
                psi=round(psi, 6),
                psi_alert=psi > 0.2,
            ))

        return results


# ------------------------------------------------------------------ #
# Layer 4: Statistical process control
# ------------------------------------------------------------------ #


class ControlViolation(str, Enum):
    """Types of Western Electric rule violations."""
    BEYOND_3_SIGMA = "beyond_3_sigma"
    TWO_OF_THREE_BEYOND_2_SIGMA = "2_of_3_beyond_2_sigma"
    FOUR_OF_FIVE_BEYOND_1_SIGMA = "4_of_5_beyond_1_sigma"
    EIGHT_CONSECUTIVE_SAME_SIDE = "8_consecutive_same_side"


@dataclass
class ControlChartPoint:
    """A single point on a p-chart."""
    agent_id: str
    period: int
    p_value: float  # observed proportion
    n: int  # sample size
    ucl: float  # upper control limit
    lcl: float  # lower control limit
    center: float  # center line (p̄)
    violations: list[ControlViolation] = field(default_factory=list)

    @property
    def in_control(self) -> bool:
        return len(self.violations) == 0


@dataclass
class SPCResult:
    """Statistical process control result for one agent."""
    agent_id: str
    chart_points: list[ControlChartPoint]
    is_in_control: bool
    violations: list[ControlViolation]


class SPCAnalyser:
    """
    Layer 4: statistical process control.

    Applies p-chart with binomial control limits and Western Electric rules
    to agent error rates over time windows.
    """

    def __init__(self, window_size: int = 10) -> None:
        self.window_size = window_size

    def p_chart(self, agent_id: str, observations: list[tuple[int, int]]) -> SPCResult:
        """
        Build a p-chart from (defects, total) observations.

        Each observation is a (number_of_defects, sample_size) tuple
        representing one time period.

        Returns control chart points with UCL/LCL and violation flags.
        """
        if not observations:
            return SPCResult(agent_id=agent_id, chart_points=[], is_in_control=True, violations=[])

        total_defects = sum(d for d, _ in observations)
        total_n = sum(n for _, n in observations)
        p_bar = total_defects / total_n if total_n > 0 else 0.0

        points: list[ControlChartPoint] = []
        for period, (defects, n) in enumerate(observations):
            if n == 0:
                continue
            p_val = defects / n
            sigma = math.sqrt(p_bar * (1 - p_bar) / n) if 0 < p_bar < 1 and n > 0 else 0.0
            ucl = min(1.0, p_bar + 3 * sigma)
            lcl = max(0.0, p_bar - 3 * sigma)

            points.append(ControlChartPoint(
                agent_id=agent_id,
                period=period,
                p_value=round(p_val, 6),
                n=n,
                ucl=round(ucl, 6),
                lcl=round(lcl, 6),
                center=round(p_bar, 6),
            ))

        # Apply Western Electric rules
        self._apply_western_electric(points, p_bar)

        all_violations: list[ControlViolation] = []
        for pt in points:
            all_violations.extend(pt.violations)

        return SPCResult(
            agent_id=agent_id,
            chart_points=points,
            is_in_control=len(all_violations) == 0,
            violations=list(set(all_violations)),
        )

    def _apply_western_electric(self, points: list[ControlChartPoint], p_bar: float) -> None:
        """Apply Western Electric rules for pattern detection."""
        if not points:
            return

        for i, pt in enumerate(points):
            # Rule 1: any point beyond 3σ limits
            if pt.p_value > pt.ucl or pt.p_value < pt.lcl:
                pt.violations.append(ControlViolation.BEYOND_3_SIGMA)

            # Calculate sigma for zone boundaries
            sigma = math.sqrt(p_bar * (1 - p_bar) / pt.n) if 0 < p_bar < 1 and pt.n > 0 else 0.0
            two_sigma_upper = p_bar + 2 * sigma
            two_sigma_lower = p_bar - 2 * sigma
            one_sigma_upper = p_bar + sigma
            one_sigma_lower = p_bar - sigma

            # Rule 2: 2 of 3 consecutive beyond 2σ on same side
            if i >= 2:
                window = points[i - 2:i + 1]
                above_2s = sum(1 for p in window if p.p_value > two_sigma_upper)
                below_2s = sum(1 for p in window if p.p_value < two_sigma_lower)
                if above_2s >= 2 or below_2s >= 2:
                    pt.violations.append(ControlViolation.TWO_OF_THREE_BEYOND_2_SIGMA)

            # Rule 3: 4 of 5 consecutive beyond 1σ on same side
            if i >= 4:
                window = points[i - 4:i + 1]
                above_1s = sum(1 for p in window if p.p_value > one_sigma_upper)
                below_1s = sum(1 for p in window if p.p_value < one_sigma_lower)
                if above_1s >= 4 or below_1s >= 4:
                    pt.violations.append(ControlViolation.FOUR_OF_FIVE_BEYOND_1_SIGMA)

            # Rule 4: 8 consecutive points on same side of center
            if i >= 7:
                window = points[i - 7:i + 1]
                above = sum(1 for p in window if p.p_value > p_bar)
                below = sum(1 for p in window if p.p_value < p_bar)
                if above == 8 or below == 8:
                    pt.violations.append(ControlViolation.EIGHT_CONSECUTIVE_SAME_SIDE)

    def from_session_windows(self, session: AgentSession) -> SPCResult:
        """
        Build a p-chart from a session's events, split into time windows.

        Each window of `window_size` events becomes one observation:
        (error_count, event_count).
        """
        events = session.events
        if not events:
            return SPCResult(
                agent_id=session.agent_id,
                chart_points=[],
                is_in_control=True,
                violations=[],
            )

        observations: list[tuple[int, int]] = []
        for start in range(0, len(events), self.window_size):
            chunk = events[start:start + self.window_size]
            n = len(chunk)
            errors = sum(1 for e in chunk if e.type.value == "error")
            observations.append((errors, n))

        return self.p_chart(session.agent_id, observations)


# ------------------------------------------------------------------ #
# Diagnostic report
# ------------------------------------------------------------------ #


class Severity(str, Enum):
    OK = "ok"
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Recommendation:
    """A single recommendation from the diagnostic report."""
    severity: Severity
    category: str
    message: str
    metric_name: str
    metric_value: float


@dataclass
class DiagnosticReport:
    """Complete diagnostic output from the statistical engine."""
    # Layer 1
    jsd_scores: list[PairwiseJSD]
    routing_clarity: float
    coverage: CoverageDensity
    # Layer 2
    health: HealthMetrics
    # Layer 3
    drift_results: list[DriftResult]
    # Layer 4
    spc_results: list[SPCResult]
    # Recommendations
    recommendations: list[Recommendation]

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_1_distributional": {
                "jsd_scores": [
                    {"agent_a": j.agent_a, "agent_b": j.agent_b, "jsd": j.jsd_score,
                     "near_duplicate": j.is_near_duplicate, "well_separated": j.is_well_separated}
                    for j in self.jsd_scores
                ],
                "routing_clarity": self.routing_clarity,
                "coverage": {
                    "mean_max_similarity": self.coverage.mean_max_similarity,
                    "std_max_similarity": self.coverage.std_max_similarity,
                    "uncovered_fraction": self.coverage.uncovered_fraction,
                    "coverage_uniformity": self.coverage.coverage_uniformity,
                },
            },
            "layer_2_information_theoretic": {
                "mutual_information": [
                    {"agent_a": m.agent_a, "agent_b": m.agent_b, "mi": m.mi_score, "redundant": m.is_redundant}
                    for m in self.health.mutual_information
                ],
                "conditional_entropy": self.health.conditional_entropy,
                "per_agent_entropy": self.health.per_agent_entropy,
            },
            "layer_3_drift": [
                {
                    "agent_id": d.agent_id,
                    "kl_divergence": d.kl_divergence,
                    "cusum_value": d.cusum_value,
                    "cusum_alert": d.cusum_alert,
                    "psi": d.psi,
                    "psi_alert": d.psi_alert,
                }
                for d in self.drift_results
            ],
            "layer_4_spc": [
                {
                    "agent_id": r.agent_id,
                    "in_control": r.is_in_control,
                    "violations": [v.value for v in r.violations],
                    "points": len(r.chart_points),
                }
                for r in self.spc_results
            ],
            "recommendations": [
                {
                    "severity": r.severity.value,
                    "category": r.category,
                    "message": r.message,
                    "metric": r.metric_name,
                    "value": r.metric_value,
                }
                for r in self.recommendations
            ],
        }


# ------------------------------------------------------------------ #
# Orchestrator
# ------------------------------------------------------------------ #


class StatsEngine:
    """
    Orchestrates all four analysis layers and produces a DiagnosticReport.

    Usage::

        engine = StatsEngine(tracker)
        report = engine.diagnose()
        print(report.recommendations)
    """

    def __init__(self, tracker: AgentTracker) -> None:
        self.tracker = tracker
        self._dist = DistributionalAnalyser()
        self._info = InformationTheoreticAnalyser()
        self._drift = DriftDetector()
        self._spc = SPCAnalyser()

    def diagnose(
        self,
        probes: list[str] | None = None,
        baseline_sessions: list[AgentSession] | None = None,
    ) -> DiagnosticReport:
        """
        Run all four analysis layers and return a complete diagnostic report.

        Args:
            probes: optional list of probe tokens for routing analysis.
            baseline_sessions: optional baseline sessions for drift detection.
                               If not provided, current sessions serve as their own baseline.
        """
        sessions = self.tracker.sessions
        if not sessions:
            return DiagnosticReport(
                jsd_scores=[],
                routing_clarity=1.0,
                coverage=CoverageDensity(0.0, 0.0, 1.0, 0.0),
                health=HealthMetrics([], 0.0, {}),
                drift_results=[],
                spc_results=[],
                recommendations=[Recommendation(
                    severity=Severity.OK,
                    category="overall",
                    message="No sessions to analyse.",
                    metric_name="overall",
                    metric_value=0.0,
                )],
            )

        # Layer 1
        jsd_scores = self._dist.pairwise_jsd(sessions)
        routing_clarity = self._dist.system_routing_clarity(sessions, probes)
        coverage = self._dist.coverage_density(sessions, probes)

        # Layer 2
        health = self._info.health_metrics(sessions)

        # Layer 3
        if baseline_sessions:
            self._drift.set_baselines_from_sessions(baseline_sessions)
        elif not self._drift._baselines:
            self._drift.set_baselines_from_sessions(sessions)
        drift_results = self._drift.detect_drift(sessions)

        # Layer 4
        spc_results = [self._spc.from_session_windows(s) for s in sessions]

        # Generate recommendations
        recommendations = self._generate_recommendations(
            jsd_scores, routing_clarity, coverage, health, drift_results, spc_results
        )

        return DiagnosticReport(
            jsd_scores=jsd_scores,
            routing_clarity=routing_clarity,
            coverage=coverage,
            health=health,
            drift_results=drift_results,
            spc_results=spc_results,
            recommendations=recommendations,
        )

    def _generate_recommendations(
        self,
        jsd_scores: list[PairwiseJSD],
        routing_clarity: float,
        coverage: CoverageDensity,
        health: HealthMetrics,
        drift_results: list[DriftResult],
        spc_results: list[SPCResult],
    ) -> list[Recommendation]:
        recs: list[Recommendation] = []

        # Layer 1 recommendations
        for j in jsd_scores:
            if j.is_near_duplicate:
                recs.append(Recommendation(
                    severity=Severity.WARNING,
                    category="distributional",
                    message=f"Agents {j.agent_a} and {j.agent_b} are near-duplicates "
                            f"(JSD={j.jsd_score:.3f} < 0.1). Consider merging them.",
                    metric_name="jsd",
                    metric_value=j.jsd_score,
                ))

        if routing_clarity < 0.5:
            recs.append(Recommendation(
                severity=Severity.WARNING,
                category="distributional",
                message=f"System routing clarity is low ({routing_clarity:.2f}). "
                        "Many tokens are ambiguous — agents may route incorrectly.",
                metric_name="routing_clarity",
                metric_value=routing_clarity,
            ))
        elif routing_clarity < 0.8:
            recs.append(Recommendation(
                severity=Severity.INFO,
                category="distributional",
                message=f"Routing clarity is moderate ({routing_clarity:.2f}). "
                        "Some tokens are shared across agents.",
                metric_name="routing_clarity",
                metric_value=routing_clarity,
            ))

        if coverage.uncovered_fraction > 0.3:
            recs.append(Recommendation(
                severity=Severity.WARNING,
                category="distributional",
                message=f"Coverage is patchy: {coverage.uncovered_fraction:.0%} of probes are uncovered. "
                        "Consider adding agents or broadening existing ones.",
                metric_name="uncovered_fraction",
                metric_value=coverage.uncovered_fraction,
            ))

        # Layer 2 recommendations
        for m in health.mutual_information:
            if m.is_redundant:
                recs.append(Recommendation(
                    severity=Severity.WARNING,
                    category="information_theoretic",
                    message=f"Agents {m.agent_a} and {m.agent_b} have high mutual information "
                            f"(MI={m.mi_score:.3f}). They may be redundant.",
                    metric_name="mutual_information",
                    metric_value=m.mi_score,
                ))

        if health.conditional_entropy > 0.7:
            recs.append(Recommendation(
                severity=Severity.CRITICAL,
                category="information_theoretic",
                message=f"Conditional entropy is very high ({health.conditional_entropy:.2f}). "
                        "The system is essentially guessing which agent to route to.",
                metric_name="conditional_entropy",
                metric_value=health.conditional_entropy,
            ))
        elif health.conditional_entropy > 0.5:
            recs.append(Recommendation(
                severity=Severity.WARNING,
                category="information_theoretic",
                message=f"Conditional entropy is elevated ({health.conditional_entropy:.2f}). "
                        "Agent descriptions need better differentiation.",
                metric_name="conditional_entropy",
                metric_value=health.conditional_entropy,
            ))

        # Layer 3 recommendations
        for d in drift_results:
            if d.cusum_alert:
                recs.append(Recommendation(
                    severity=Severity.CRITICAL,
                    category="drift",
                    message=f"CUSUM alert for agent {d.agent_id}: cumulative drift detected "
                            f"(CUSUM={d.cusum_value:.3f}). The agent's behavior is shifting.",
                    metric_name="cusum",
                    metric_value=d.cusum_value,
                ))
            if d.psi_alert:
                recs.append(Recommendation(
                    severity=Severity.WARNING,
                    category="drift",
                    message=f"Population stability alert for agent {d.agent_id} "
                            f"(PSI={d.psi:.3f} > 0.2). Distribution has shifted significantly.",
                    metric_name="psi",
                    metric_value=d.psi,
                ))

        # Layer 4 recommendations
        for r in spc_results:
            if not r.is_in_control:
                violation_strs = [v.value for v in r.violations]
                recs.append(Recommendation(
                    severity=Severity.CRITICAL,
                    category="spc",
                    message=f"Agent {r.agent_id} is out of statistical control. "
                            f"Violations: {', '.join(violation_strs)}. "
                            "Error rate shows statistically significant regression.",
                    metric_name="spc_violations",
                    metric_value=float(len(r.violations)),
                ))

        # If nothing is wrong, say so
        if not recs:
            recs.append(Recommendation(
                severity=Severity.OK,
                category="overall",
                message="All metrics within healthy ranges. No action needed.",
                metric_name="overall",
                metric_value=0.0,
            ))

        return recs
