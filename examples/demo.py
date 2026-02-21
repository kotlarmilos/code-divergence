"""
Demo: simulate two AI agents working on a codebase and check their divergence/overlap.
Run with:  python examples/demo.py
"""

from code_divergence import AgentTracker, AgentComparator, PerformanceSensor, Reporter, SensorConfig
from code_divergence.tracker import EventType

# ── Setup ──────────────────────────────────────────────────────────────────
tracker = AgentTracker()

alpha = tracker.new_session("agent-alpha", agent_id="alpha")
beta  = tracker.new_session("agent-beta",  agent_id="beta")

# ── Agent Alpha: working on auth subsystem ─────────────────────────────────
tracker.track_task_start(alpha, "implement authentication")
tracker.track_file_change(alpha, "src/auth/login.py",   EventType.FILE_MODIFIED)
tracker.track_file_change(alpha, "src/auth/session.py", EventType.FILE_CREATED)
tracker.track_file_change(alpha, "src/auth/tokens.py",  EventType.FILE_CREATED)
tracker.track_output(alpha, "def login(user, password): ...", task="implement login")
tracker.track_output(alpha, "def create_session(user_id): ...", task="session mgmt")
tracker.track_tool_call(alpha, "Read", {"file": "src/auth/login.py"})
tracker.track_tool_call(alpha, "Edit", {"file": "src/auth/login.py"})
tracker.track_commit(alpha, "a1b2c3", "feat(auth): implement login and session management",
                     files=["src/auth/login.py", "src/auth/session.py", "src/auth/tokens.py"])
tracker.track_task_end(alpha, "auth complete")

# ── Agent Beta: working on the UI layer ────────────────────────────────────
tracker.track_task_start(beta, "build dashboard UI")
tracker.track_file_change(beta, "src/ui/dashboard.py", EventType.FILE_CREATED)
tracker.track_file_change(beta, "src/ui/components.py", EventType.FILE_MODIFIED)
tracker.track_file_change(beta, "src/ui/layout.py",    EventType.FILE_CREATED)
tracker.track_output(beta, "def render_dashboard(): ...", task="dashboard widget")
tracker.track_output(beta, "class NavBar: ...", task="navigation component")
tracker.track_tool_call(beta, "Read",  {"file": "src/ui/components.py"})
tracker.track_tool_call(beta, "Write", {"file": "src/ui/dashboard.py"})
tracker.track_commit(beta, "d4e5f6", "feat(ui): add dashboard and navigation components",
                     files=["src/ui/dashboard.py", "src/ui/components.py", "src/ui/layout.py"])
tracker.track_task_end(beta, "UI complete")

tracker.end_session(alpha)
tracker.end_session(beta)

print("=" * 60)
print("SCENARIO 1: Complementary work (expected: moderate divergence, low overlap)")
print("=" * 60)

reporter = Reporter(tracker)
reporter.print_report()

# ── Scenario 2: Duplicate work ─────────────────────────────────────────────
print()
print("=" * 60)
print("SCENARIO 2: Duplicate work (expected: high overlap alert)")
print("=" * 60)

tracker2 = AgentTracker()
x = tracker2.new_session("agent-x", agent_id="X")
y = tracker2.new_session("agent-y", agent_id="Y")

shared_files = ["src/models/user.py", "src/models/post.py", "src/models/comment.py"]
shared_output = "class User: id name email\nclass Post: id title content author_id\nclass Comment: id body post_id"

for f in shared_files:
    tracker2.track_file_change(x, f, EventType.FILE_MODIFIED)
    tracker2.track_file_change(y, f, EventType.FILE_MODIFIED)
tracker2.track_output(x, shared_output, task="data models")
tracker2.track_output(y, shared_output, task="data models")
tracker2.track_commit(x, "aaa111", "feat: define data models")
tracker2.track_commit(y, "bbb222", "feat: define data models")

config = SensorConfig(overlap_warn=0.5, overlap_crit=0.75)
sensor2 = PerformanceSensor(tracker2, config)
alerts = sensor2.evaluate()
print(f"Alerts ({len(alerts)}):")
for alert in alerts:
    print(f"  {alert}")

print()
print("Done. See Reporter.print_json() for machine-readable output.")
