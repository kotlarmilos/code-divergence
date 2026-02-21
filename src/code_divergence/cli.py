"""
CLI entry point.

Commands:
  track       – record an agent event from the command line
  report      – print a full report
  monitor     – continuously evaluate and print alerts
  compare     – compare two specific agents
  status      – quick health summary
  git-sync    – ingest agent branches from git (passive, no instrumentation needed)
  symbols     – show which symbols (functions/classes) are being implemented by multiple agents
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .reporter import Reporter
from .sensors import PerformanceSensor, SensorConfig
from .tracker import AgentTracker, EventType


DEFAULT_STORE = "divergence_state.json"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="code-divergence",
        description="Track and monitor AI agent performance: divergence and overlap sensors.",
    )
    parser.add_argument(
        "--store",
        default=DEFAULT_STORE,
        metavar="PATH",
        help=f"Path to the state file (default: {DEFAULT_STORE})",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── track ──────────────────────────────────────────────────────────
    p_track = sub.add_parser("track", help="Record an event for an agent")
    p_track.add_argument("--agent-id", required=True, help="Agent session ID")
    p_track.add_argument("--event", required=True, choices=[e.value for e in EventType], help="Event type")
    p_track.add_argument("--file", metavar="PATH", help="File path (for file events)")
    p_track.add_argument("--content", help="Output content (for output events)")
    p_track.add_argument("--commit", metavar="HASH", help="Commit hash")
    p_track.add_argument("--message", help="Commit or task message")
    p_track.add_argument("--tool", help="Tool name (for tool_call events)")

    # ── new-session ─────────────────────────────────────────────────────
    p_new = sub.add_parser("new-session", help="Start a new agent session")
    p_new.add_argument("name", help="Human-readable agent name")
    p_new.add_argument("--agent-id", help="Optional explicit session ID")

    # ── end-session ─────────────────────────────────────────────────────
    p_end = sub.add_parser("end-session", help="Mark an agent session as ended")
    p_end.add_argument("agent_id", help="Agent session ID")

    # ── report ──────────────────────────────────────────────────────────
    p_report = sub.add_parser("report", help="Print the full performance report")
    p_report.add_argument("--json", action="store_true", help="Output as JSON")
    p_report.add_argument("--out", metavar="FILE", help="Write report to file instead of stdout")

    # ── monitor ─────────────────────────────────────────────────────────
    p_monitor = sub.add_parser("monitor", help="Continuously monitor and print alerts")
    p_monitor.add_argument("--interval", type=float, default=30.0, metavar="SECS", help="Check interval in seconds (default: 30)")
    p_monitor.add_argument("--divergence-warn", type=float, default=0.70)
    p_monitor.add_argument("--divergence-crit", type=float, default=0.85)
    p_monitor.add_argument("--overlap-warn", type=float, default=0.60)
    p_monitor.add_argument("--overlap-crit", type=float, default=0.75)

    # ── compare ─────────────────────────────────────────────────────────
    p_compare = sub.add_parser("compare", help="Compare two specific agents")
    p_compare.add_argument("agent_a", help="First agent ID")
    p_compare.add_argument("agent_b", help="Second agent ID")
    p_compare.add_argument("--json", action="store_true", help="Output as JSON")

    # ── status ──────────────────────────────────────────────────────────
    sub.add_parser("status", help="Quick one-line health summary")

    # ── git-sync ─────────────────────────────────────────────────────────
    p_git = sub.add_parser(
        "git-sync",
        help="Ingest agent branches from git — no agent instrumentation needed",
    )
    p_git.add_argument(
        "branches",
        nargs="*",
        metavar="BRANCH",
        help="Branches to ingest (e.g. claude/agent-auth-XYZ). "
             "Omit to auto-discover all branches matching --prefix.",
    )
    p_git.add_argument("--prefix", default="claude/", help="Branch prefix for auto-discovery (default: claude/)")
    p_git.add_argument("--base", default="main", help="Base branch to diff against (default: main)")
    p_git.add_argument("--repo", default=".", help="Path to git repo (default: current dir)")
    p_git.add_argument("--json", action="store_true", help="Output ingestion summary as JSON")

    # ── symbols ──────────────────────────────────────────────────────────
    p_sym = sub.add_parser(
        "symbols",
        help="Show functions/classes being implemented by multiple agents (strongest overlap signal)",
    )
    p_sym.add_argument("--json", action="store_true", help="Output as JSON")

    return parser


# ------------------------------------------------------------------ #
# Command handlers
# ------------------------------------------------------------------ #


def cmd_new_session(args, tracker: AgentTracker) -> int:
    agent_id = tracker.new_session(name=args.name, agent_id=args.agent_id)
    tracker.save(args.store)
    print(f"Session created: id={agent_id}  name={args.name}")
    return 0


def cmd_end_session(args, tracker: AgentTracker) -> int:
    try:
        tracker.end_session(args.agent_id)
    except KeyError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    tracker.save(args.store)
    print(f"Session {args.agent_id} ended.")
    return 0


def cmd_track(args, tracker: AgentTracker) -> int:
    etype = EventType(args.event)
    aid = args.agent_id

    # Make sure session exists; auto-create if not (convenience)
    try:
        tracker.get_session(aid)
    except KeyError:
        tracker.new_session(name=aid, agent_id=aid)

    if etype in (EventType.FILE_MODIFIED, EventType.FILE_CREATED, EventType.FILE_DELETED):
        if not args.file:
            print("Error: --file required for file events", file=sys.stderr)
            return 1
        tracker.track_file_change(aid, args.file, etype)

    elif etype in (EventType.CODE_OUTPUT, EventType.TEXT_OUTPUT):
        content = args.content or ""
        tracker.track_output(aid, content, output_type=etype)

    elif etype == EventType.COMMIT:
        tracker.track_commit(aid, args.commit or "", args.message or "")

    elif etype == EventType.TASK_START:
        tracker.track_task_start(aid, args.message or "")

    elif etype == EventType.TASK_END:
        tracker.track_task_end(aid, args.message or "")

    elif etype == EventType.TOOL_CALL:
        tracker.track_tool_call(aid, args.tool or "unknown")

    elif etype == EventType.ERROR:
        tracker.track_error(aid, args.message or "")

    tracker.save(args.store)
    print(f"Event recorded: {etype.value} for agent {aid}")
    return 0


def cmd_report(args, tracker: AgentTracker) -> int:
    reporter = Reporter(tracker)
    out_stream = sys.stdout
    out_file = None

    if args.out:
        out_file = open(args.out, "w")
        out_stream = out_file

    try:
        if args.json:
            reporter.print_json(out_stream)
        else:
            reporter.print_report(out_stream)
    finally:
        if out_file:
            out_file.close()

    return 0


def cmd_monitor(args, tracker: AgentTracker) -> int:
    config = SensorConfig(
        divergence_warn=args.divergence_warn,
        divergence_crit=args.divergence_crit,
        overlap_warn=args.overlap_warn,
        overlap_crit=args.overlap_crit,
    )
    sensor = PerformanceSensor(tracker, config)

    print(f"Monitoring every {args.interval}s — press Ctrl+C to stop.")
    try:
        sensor.monitor(interval=args.interval)
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
    return 0


def cmd_compare(args, tracker: AgentTracker) -> int:
    from .metrics import AgentComparator
    try:
        a = tracker.get_session(args.agent_a)
        b = tracker.get_session(args.agent_b)
    except KeyError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    comparator = AgentComparator()
    result = comparator.compare(a, b)

    if args.json:
        data = {
            "agent_a": result.agent_a,
            "agent_b": result.agent_b,
            "divergence": result.divergence,
            "overlap": result.overlap,
            "file_divergence": result.file_divergence,
            "file_overlap": result.file_overlap,
            "output_divergence": result.output_divergence,
            "output_overlap": result.output_overlap,
            "commit_divergence": result.commit_divergence,
            "tool_divergence": result.tool_divergence,
            "details": result.details,
        }
        print(json.dumps(data, indent=2))
    else:
        print(result.summary())
        print(f"  file_divergence   = {result.file_divergence:.3f}")
        print(f"  file_overlap      = {result.file_overlap:.3f}")
        print(f"  output_divergence = {result.output_divergence:.3f}")
        print(f"  output_overlap    = {result.output_overlap:.3f}")
        print(f"  commit_divergence = {result.commit_divergence:.3f}")
        if result.details.get("shared_files"):
            print(f"  shared_files      = {result.details['shared_files']}")

    return 0


def cmd_git_sync(args, tracker: AgentTracker) -> int:
    from .git_ingester import GitIngester

    ingester = GitIngester(repo_path=args.repo)

    if args.branches:
        agent_ids = [
            ingester.ingest_branch(tracker, branch=b, base=args.base)
            for b in args.branches
        ]
    else:
        agent_ids = ingester.ingest_all_agent_branches(
            tracker, prefix=args.prefix, base=args.base
        )

    tracker.save(args.store)

    if not agent_ids:
        print(f"No branches found matching prefix '{args.prefix}'.", file=sys.stderr)
        return 1

    if args.json:
        summary = [tracker.get_session(aid).summary() for aid in agent_ids]
        print(json.dumps(summary, indent=2))
    else:
        print(f"Ingested {len(agent_ids)} branch(es):")
        for aid in agent_ids:
            s = tracker.get_session(aid)
            branch = s.metadata.get("branch", aid)
            print(f"  {branch}  →  agent_id={aid}  files={len(s.files_touched)}  commits={s.commit_count}  symbols={len(s.metadata.get('symbols_defined', []))}")

    return 0


def cmd_symbols(args, tracker: AgentTracker) -> int:
    from .git_ingester import find_symbol_conflicts

    conflicts = find_symbol_conflicts(tracker)

    if not conflicts:
        print("No shared symbols — agents are implementing distinct things.")
        return 0

    if args.json:
        print(json.dumps([
            {"symbol": c.symbol, "agent_ids": c.agent_ids}
            for c in conflicts
        ], indent=2))
    else:
        print(f"Shared symbols ({len(conflicts)}) — agents implementing the same thing:\n")
        for c in conflicts:
            print(f"  !! {c}")
        print(f"\nRun `code-divergence report` for full divergence/overlap metrics.")

    return 0


def cmd_status(args, tracker: AgentTracker) -> int:
    sessions = tracker.sessions
    if not sessions:
        print("No sessions. Use `code-divergence new-session <name>` to start one.")
        return 0

    active = [s for s in sessions if s.end_time is None]
    print(f"Sessions: {len(sessions)} total, {len(active)} active")
    for s in sessions:
        state = "active" if s.end_time is None else "ended"
        print(f"  [{state}] {s.name} ({s.agent_id})  events={s.event_count}  files={len(s.files_touched)}  commits={s.commit_count}")

    if len(sessions) >= 2:
        from .metrics import AgentComparator
        comparator = AgentComparator()
        results = comparator.compare_all(sessions)
        print("\nPairwise health:")
        for r in results:
            print(f"  {r.summary()}")

    return 0


# ------------------------------------------------------------------ #
# Entry point
# ------------------------------------------------------------------ #


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    tracker = AgentTracker(store_path=args.store if Path(args.store).exists() else None)
    if Path(args.store).exists():
        tracker.load(args.store)

    dispatch = {
        "new-session": cmd_new_session,
        "end-session": cmd_end_session,
        "track": cmd_track,
        "report": cmd_report,
        "monitor": cmd_monitor,
        "compare": cmd_compare,
        "status": cmd_status,
        "git-sync": cmd_git_sync,
        "symbols": cmd_symbols,
    }

    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    return handler(args, tracker)


if __name__ == "__main__":
    sys.exit(main())
