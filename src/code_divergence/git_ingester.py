"""
Git-based passive ingestion.

Reads existing git branches to populate AgentSessions without any manual
instrumentation.  Agents don't need to know this tool exists — it just reads
git history the way Entero HQ reads conversation checkpoints.

Usage::

    from code_divergence.git_ingester import GitIngester

    tracker = AgentTracker()
    ingester = GitIngester(repo_path=".")

    # One call per agent branch — no agent cooperation needed
    ingester.ingest_branch(tracker, branch="claude/agent-auth-XYZ",  agent_id="auth-agent",  name="Auth Agent")
    ingester.ingest_branch(tracker, branch="claude/agent-ui-ABC",    agent_id="ui-agent",    name="UI Agent")
    ingester.ingest_branch(tracker, branch="claude/agent-tests-DEF", agent_id="test-agent",  name="Test Agent")

    # Now compare them normally
    reporter = Reporter(tracker)
    reporter.print_report()
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .tracker import AgentTracker, EventType


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #


def _git(args: list[str], cwd: str | Path) -> str:
    """Run a git command and return stdout, empty string on error."""
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return ""


def _extract_symbols(diff_text: str) -> set[str]:
    """
    Extract defined function and class names from a unified diff.

    Looks at added lines (+) for Python/JS/TS/Go style definitions so we can
    detect when two branches define the same symbol (the strongest signal that
    they are implementing the same thing).
    """
    symbols: set[str] = set()
    patterns = [
        # Python
        r"^\+\s*(?:async\s+)?def\s+(\w+)\s*\(",
        r"^\+\s*class\s+(\w+)\s*[:\(]",
        # JS / TS
        r"^\+\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(",
        r"^\+\s*(?:export\s+)?class\s+(\w+)\s*[{\(]",
        r"^\+\s*(?:export\s+)?const\s+(\w+)\s*=\s*(?:async\s*)?\(",
        # Go
        r"^\+\s*func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)\s*\(",
        # Rust
        r"^\+\s*(?:pub\s+)?fn\s+(\w+)\s*[<\(]",
        r"^\+\s*(?:pub\s+)?struct\s+(\w+)\s*[{\(]",
    ]
    for line in diff_text.splitlines():
        for pat in patterns:
            m = re.match(pat, line)
            if m:
                symbols.add(m.group(1))
                break
    return set(symbols)


def _extract_tokens(diff_text: str) -> set[str]:
    """Token set from added lines only (for overlap comparison)."""
    added = "\n".join(
        line[1:] for line in diff_text.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    return set(re.findall(r"\b[a-zA-Z_]\w{2,}\b", added.lower()))


# ------------------------------------------------------------------ #
# Checkpoint
# ------------------------------------------------------------------ #


@dataclass
class BranchSnapshot:
    """Point-in-time snapshot of a branch's state."""
    branch: str
    base: str
    commit_count: int
    files_changed: list[str]
    symbols_defined: set[str]
    diff_tokens: set[str]
    commits: list[dict]
    full_diff: str


# ------------------------------------------------------------------ #
# Ingester
# ------------------------------------------------------------------ #


class GitIngester:
    """
    Reads git history to populate an AgentTracker passively.

    No agent instrumentation needed — just point it at branches.
    """

    def __init__(self, repo_path: str | Path = ".") -> None:
        self.repo_path = Path(repo_path).resolve()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def ingest_branch(
        self,
        tracker: AgentTracker,
        branch: str,
        agent_id: str | None = None,
        name: str | None = None,
        base: str = "main",
    ) -> str:
        """
        Read all commits and diffs on `branch` since it diverged from `base`,
        create (or update) an AgentSession in `tracker`, and return the agent_id.

        Safe to call repeatedly — if a session with the same agent_id already
        exists, new events are appended (acts like a checkpoint refresh).
        """
        agent_id = agent_id or branch.replace("/", "-")
        name = name or agent_id

        # Ensure session exists
        try:
            tracker.get_session(agent_id)
        except KeyError:
            tracker.new_session(name=name, agent_id=agent_id)

        snapshot = self._snapshot(branch, base)

        # Record file-level events
        for path in snapshot.files_changed:
            tracker.track_file_change(agent_id, path, EventType.FILE_MODIFIED)

        # Record each commit
        for commit in snapshot.commits:
            tracker.track_commit(
                agent_id,
                commit_hash=commit["hash"],
                message=commit["message"],
                files=commit.get("files", []),
            )

        # Record diff content as a code output — this is what the metrics module
        # uses to detect semantic overlap between branches.
        if snapshot.full_diff:
            tracker.track_output(
                agent_id,
                content=snapshot.full_diff,
                output_type=EventType.CODE_OUTPUT,
            )

        # Store symbol list in session metadata so sensors can compare directly
        session = tracker.get_session(agent_id)
        existing = set(session.metadata.get("symbols_defined", []))
        session.metadata["symbols_defined"] = sorted(existing | snapshot.symbols_defined)
        session.metadata["branch"] = branch
        session.metadata["base"] = base

        return agent_id

    def ingest_all_agent_branches(
        self,
        tracker: AgentTracker,
        prefix: str = "claude/",
        base: str = "main",
    ) -> list[str]:
        """
        Auto-discover all branches matching `prefix` and ingest them.
        Returns list of agent_ids created.
        """
        branches = self._list_branches(prefix)
        return [
            self.ingest_branch(tracker, branch=b, base=base)
            for b in branches
        ]

    def snapshot(self, branch: str, base: str = "main") -> BranchSnapshot:
        """Public access to a branch snapshot (useful for custom analysis)."""
        return self._snapshot(branch, base)

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _snapshot(self, branch: str, base: str) -> BranchSnapshot:
        merge_base = _git(["merge-base", base, branch], self.repo_path) or base

        # Files changed since divergence
        files_raw = _git(
            ["diff", "--name-only", f"{merge_base}..{branch}"],
            self.repo_path,
        )
        files_changed = [f for f in files_raw.splitlines() if f]

        # Full diff (used for token + symbol extraction)
        full_diff = _git(
            ["diff", f"{merge_base}..{branch}"],
            self.repo_path,
        )

        # Commits
        log_raw = _git(
            ["log", "--format=%H%x00%s%x00%aI", f"{merge_base}..{branch}"],
            self.repo_path,
        )
        commits = []
        for line in log_raw.splitlines():
            parts = line.split("\x00")
            if len(parts) >= 2:
                h, msg = parts[0], parts[1]
                commit_files_raw = _git(
                    ["diff-tree", "--no-commit-id", "-r", "--name-only", h],
                    self.repo_path,
                )
                commits.append({
                    "hash": h,
                    "message": msg,
                    "files": [f for f in commit_files_raw.splitlines() if f],
                })

        return BranchSnapshot(
            branch=branch,
            base=base,
            commit_count=len(commits),
            files_changed=files_changed,
            symbols_defined=_extract_symbols(full_diff),
            diff_tokens=_extract_tokens(full_diff),
            commits=commits,
            full_diff=full_diff,
        )

    def _list_branches(self, prefix: str) -> list[str]:
        # Local branches
        local = _git(["branch", "--format=%(refname:short)"], self.repo_path)
        branches = [b for b in local.splitlines() if b.startswith(prefix)]
        # Remote branches (origin/...)
        remote = _git(
            ["branch", "-r", "--format=%(refname:short)"],
            self.repo_path,
        )
        for b in remote.splitlines():
            # strip "origin/" prefix
            short = re.sub(r"^origin/", "", b)
            if short.startswith(prefix) and short not in branches:
                branches.append(short)
        return branches


# ------------------------------------------------------------------ #
# Symbol-level overlap detector
# ------------------------------------------------------------------ #


@dataclass
class SymbolConflict:
    """Two agents are implementing the same named symbol."""
    symbol: str
    agent_ids: list[str]

    def __str__(self) -> str:
        return f"Symbol '{self.symbol}' implemented by: {', '.join(self.agent_ids)}"


def find_symbol_conflicts(tracker: AgentTracker) -> list[SymbolConflict]:
    """
    Scan all sessions for shared symbol definitions — the strongest signal
    that two agents are implementing the same thing.

    Returns a SymbolConflict for every symbol defined in more than one session.
    """
    symbol_map: dict[str, list[str]] = {}
    for session in tracker.sessions:
        for sym in session.metadata.get("symbols_defined", []):
            symbol_map.setdefault(sym, []).append(session.agent_id)

    conflicts = []
    for sym, agent_ids in symbol_map.items():
        if len(agent_ids) > 1:
            conflicts.append(SymbolConflict(symbol=sym, agent_ids=agent_ids))

    # Sort by number of conflicts (most contentious first)
    return sorted(conflicts, key=lambda c: len(c.agent_ids), reverse=True)
