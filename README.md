# code-divergence

Track and diagnose AI agent routing quality — detect divergence, overlap, and drift in multi-agent workflows using only the Python standard library.

## What it does

When multiple AI agents work on a codebase, things go wrong in predictable ways: agents duplicate each other's work, drift apart in incompatible directions, or silently degrade over time. **code-divergence** catches all of this.

It treats every agent setup as a routing function `f(query) → skill` and measures how good that function is — without ever running the agent — purely from configuration, file changes, outputs, and commit history.

**Core signals:**
- **Divergence** — agents working in incompatible or too-different directions
- **Overlap** — agents duplicating the same work
- **Drift** — behavior silently shifting over time
- **Control** — error rates breaking out of statistical norms

## Install

```bash
pip install -e .

# with dev tools (pytest, coverage)
pip install -e ".[dev]"
```

Requires **Python ≥ 3.10**. Zero runtime dependencies.

## Quick start

### Python API

```python
from code_divergence import AgentTracker, StatsEngine

tracker = AgentTracker()

# Record agent work
tracker.new_session("agent-alpha", agent_id="alpha")
tracker.track_file_change("alpha", "src/auth/login.py")
tracker.track_output("alpha", "def login(user, password): ...")
tracker.track_commit("alpha", "a1b2c3", "feat: implement login")

tracker.new_session("agent-beta", agent_id="beta")
tracker.track_file_change("beta", "src/auth/login.py")  # same file!
tracker.track_output("beta", "def authenticate(credentials): ...")

# Run the statistical engine
engine = StatsEngine(tracker)
report = engine.diagnose()

for rec in report.recommendations:
    print(f"[{rec.severity.value}] {rec.message}")
```

### CLI

```bash
# Record events
code-divergence track --agent-id alpha --event file_modified --file src/auth/login.py
code-divergence track --agent-id alpha --event code_output --content "def login(): ..."

# Full diagnostic (4-layer statistical analysis)
code-divergence diagnose
code-divergence diagnose --json
code-divergence diagnose --json --out report.json

# Reports and status
code-divergence report
code-divergence report --json
code-divergence status

# Git-based passive ingestion (no instrumentation needed)
code-divergence git-sync                          # auto-discover claude/* branches
code-divergence git-sync --prefix agent/ feature-a feature-b

# Symbol conflict detection
code-divergence symbols
code-divergence symbols --json

# Compare two agents directly
code-divergence compare alpha beta --json

# Live monitoring
code-divergence monitor --interval 30
```

## The statistical engine

The core of the project is a four-layer statistical diagnostic that analyzes agent routing quality.

### Layer 1 — Distributional analysis

Each agent's work (files, outputs, commits) defines a token distribution. Layer 1 compares these distributions:

| Metric | What it measures | Key threshold |
|--------|-----------------|---------------|
| **Pairwise JSD** | Jensen-Shannon divergence between agent pairs | `< 0.1` = near-duplicate |
| **Routing entropy** | How ambiguous query→agent assignment is | High entropy = system is guessing |
| **Routing clarity** | 1 − normalized entropy, aggregated over probes | `< 0.5` triggers a warning |
| **Coverage density** | Are all query types handled, or are there gaps? | `> 30%` uncovered triggers warning |

### Layer 2 — Information-theoretic health

| Metric | What it measures | Key threshold |
|--------|-----------------|---------------|
| **Mutual information** | How much knowing one agent tells you about another | `> 0.5` = redundant agents |
| **Conditional entropy** | H(agent\|token) — how clean is the routing? | `> 0.7` = critical, `> 0.5` = warning |
| **Per-agent entropy** | Perplexity proxy for each agent's description | Higher = more confusing to route |

### Layer 3 — Drift detection

When you have temporal data (multiple sessions, production logs):

| Metric | What it measures | Alert condition |
|--------|-----------------|-----------------|
| **Rolling KL divergence** | D_KL(current \|\| baseline) | Magnitude of shift |
| **CUSUM** | Cumulative sum control chart on KL values | Catches slow drift that thresholds miss |
| **PSI** | Population Stability Index (symmetrized KL with binning) | `> 0.2` = significant shift |

### Layer 4 — Statistical process control

For pass/fail outcomes (error rates):

| Metric | What it measures | Alert condition |
|--------|-----------------|-----------------|
| **p-chart** | Binomial control limits: UCL/LCL = p̄ ± 3√(p̄(1−p̄)/n) | Any point outside limits |
| **Western Electric rules** | Pattern detection on the control chart | 2-of-3 beyond 2σ, 4-of-5 beyond 1σ, 8 consecutive same side |

### Diagnostic output

Every metric maps to a `Recommendation` with severity levels:

- **OK** — nothing wrong
- **INFO** — worth noting
- **WARNING** — needs attention
- **CRITICAL** — statistically significant problem

```bash
$ code-divergence diagnose

======================================================================
  STATISTICAL ENGINE — Diagnostic Report
======================================================================

─── Layer 1: Distributional Analysis ───

  JSD(alpha, beta) = 0.0412 ⚠ NEAR-DUPLICATE

  Routing clarity : 0.623
  Coverage mean   : 0.812
  Uncovered       : 18.8%
  Uniformity      : 0.741

─── Layer 2: Information-Theoretic Health ───

  MI(alpha, beta) = 0.6210 ⚠ REDUNDANT

  Conditional entropy : 0.445

─── Layer 3: Drift Detection ───

  alpha: KL=0.0023  CUSUM=0.0000  PSI=0.0100

─── Layer 4: Statistical Process Control ───

  alpha: IN CONTROL  (10 points)

─── Recommendations ───

  ⚠ [WARNING] Near-duplicate agents: alpha ↔ beta (JSD=0.041)
  ⚠ [WARNING] Redundant agents: alpha ↔ beta (MI=0.621)
  ✓ [OK] Routing clarity is good.

======================================================================
```

## Architecture

```
src/code_divergence/
├── __init__.py          # Package exports
├── tracker.py           # Event-sourced agent session storage
├── metrics.py           # Pairwise divergence & overlap calculators
├── sensors.py           # Live alerting (thresholds, monitoring)
├── reporter.py          # Human-readable & JSON report generation
├── git_ingester.py      # Passive git branch ingestion + symbol conflicts
├── stats_engine.py      # Four-layer statistical diagnostic engine
└── cli.py               # CLI entry point (all subcommands)

tests/
├── test_tracker.py
├── test_metrics.py
├── test_sensors.py
├── test_git_ingester.py
└── test_stats_engine.py  # 55 tests covering all 4 layers

examples/
└── demo.py              # Simulates complementary + duplicate agent work

hooks/                   # Git/Claude Code integration hooks
├── git-post-commit.sh   # Auto-ingest on commit
├── post-tool-use.sh     # Track Claude Code tool usage
├── post-tool-use-output.sh
└── stop.sh              # Session-end checkpoint
```

## Key classes

| Class | Module | Purpose |
|-------|--------|---------|
| `AgentTracker` | tracker | Record agent events, persist to JSON |
| `AgentSession` | tracker | Single agent's event history |
| `DivergenceCalculator` | metrics | File/output/commit divergence scores |
| `OverlapCalculator` | metrics | File/output/commit/task overlap scores |
| `PerformanceSensor` | sensors | Live threshold-based alerting |
| `Reporter` | reporter | Combined reports (text + JSON) |
| `GitIngester` | git_ingester | Ingest git branches without instrumentation |
| `StatsEngine` | stats_engine | Four-layer statistical diagnostic orchestrator |
| `DiagnosticReport` | stats_engine | Structured output from `StatsEngine.diagnose()` |

## Git hook integration

Install the post-commit hook to automatically track agent work:

```bash
./install-hooks.sh            # current repo
./install-hooks.sh /path/to/repo  # another repo
```

This installs a `post-commit` hook that runs `git-sync` and `symbols` after every commit. Configure with environment variables:

```bash
export CODE_DIVERGENCE_STORE=divergence_state.json
export CODE_DIVERGENCE_BASE=main
```

## Passive git ingestion

No instrumentation needed — just point at existing branches:

```bash
# Auto-discover branches with a prefix
code-divergence git-sync --prefix claude/

# Or specify branches explicitly
code-divergence git-sync feature-auth feature-ui --base main
```

This reconstructs agent sessions from commit history, diffs, and file changes.

## Testing

```bash
# Run all 125 tests
pytest

# With coverage
pytest --cov=code_divergence

# Just the stats engine tests (55 tests)
pytest tests/test_stats_engine.py -v
```

## License

MIT
