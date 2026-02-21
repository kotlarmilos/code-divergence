"""Tests for git_ingester — passive branch ingestion and symbol conflict detection."""

import subprocess
from pathlib import Path

import pytest

from code_divergence.git_ingester import (
    GitIngester,
    SymbolConflict,
    _extract_symbols,
    _extract_tokens,
    find_symbol_conflicts,
)
from code_divergence.tracker import AgentTracker


# ------------------------------------------------------------------ #
# Unit tests for diff parsing helpers
# ------------------------------------------------------------------ #


class TestExtractSymbols:
    def test_python_function(self):
        diff = "+def authenticate(user, password):\n+    pass\n"
        assert "authenticate" in _extract_symbols(diff)

    def test_python_async_function(self):
        diff = "+async def fetch_user(user_id: int):\n+    ...\n"
        assert "fetch_user" in _extract_symbols(diff)

    def test_python_class(self):
        diff = "+class UserService:\n+    pass\n"
        assert "UserService" in _extract_symbols(diff)

    def test_js_function(self):
        diff = "+function renderDashboard() {\n+  return null;\n+}\n"
        assert "renderDashboard" in _extract_symbols(diff)

    def test_js_arrow_const(self):
        diff = "+const handleLogin = async (req, res) => {\n+};\n"
        assert "handleLogin" in _extract_symbols(diff)

    def test_ts_class(self):
        diff = "+export class AuthService {\n+}\n"
        assert "AuthService" in _extract_symbols(diff)

    def test_go_function(self):
        diff = "+func CreateUser(db *DB, user User) error {\n+}\n"
        assert "CreateUser" in _extract_symbols(diff)

    def test_rust_fn(self):
        diff = "+pub fn parse_token(token: &str) -> Result<Claims, Error> {\n+}\n"
        assert "parse_token" in _extract_symbols(diff)

    def test_rust_struct(self):
        diff = "+pub struct UserClaims {\n+    pub user_id: u64,\n+}\n"
        assert "UserClaims" in _extract_symbols(diff)

    def test_removed_lines_ignored(self):
        diff = "-def old_function():\n-    pass\n"
        assert "old_function" not in _extract_symbols(diff)

    def test_context_lines_ignored(self):
        diff = " def context_function():\n     pass\n"
        assert "context_function" not in _extract_symbols(diff)

    def test_multiple_symbols(self):
        diff = (
            "+def login(user, pw):\n"
            "+    pass\n"
            "+class SessionManager:\n"
            "+    pass\n"
            "+async def logout(session_id):\n"
            "+    pass\n"
        )
        syms = _extract_symbols(diff)
        assert "login" in syms
        assert "SessionManager" in syms
        assert "logout" in syms

    def test_empty_diff(self):
        assert _extract_symbols("") == set()


class TestExtractTokens:
    def test_extracts_added_words(self):
        diff = "+def authenticate_user(email, password):\n+    return True\n"
        tokens = _extract_tokens(diff)
        assert "authenticate_user" in tokens
        assert "email" in tokens
        assert "password" in tokens

    def test_ignores_removed_lines(self):
        diff = "-old_function()\n+new_function()\n"
        tokens = _extract_tokens(diff)
        assert "old_function" not in tokens
        assert "new_function" in tokens

    def test_short_words_excluded(self):
        # Only words with 3+ chars are kept (regex: \b[a-zA-Z_]\w{2,}\b)
        diff = "+if x == y:\n+    do_something()\n"
        tokens = _extract_tokens(diff)
        assert "do_something" in tokens


# ------------------------------------------------------------------ #
# Integration tests using a real temporary git repo
# ------------------------------------------------------------------ #


@pytest.fixture
def git_repo(tmp_path):
    """Create a minimal git repo with two agent branches."""
    repo = tmp_path / "repo"
    repo.mkdir()

    def run(*args):
        subprocess.run(
            list(args), cwd=str(repo), capture_output=True, check=True,
            env={"GIT_AUTHOR_NAME": "Test", "GIT_AUTHOR_EMAIL": "t@t.com",
                 "GIT_COMMITTER_NAME": "Test", "GIT_COMMITTER_EMAIL": "t@t.com",
                 "HOME": str(tmp_path), "PATH": "/usr/bin:/bin"}
        )

    # Init and create main branch with a base file
    run("git", "init", "-b", "main")
    run("git", "config", "user.email", "test@test.com")
    run("git", "config", "user.name", "Test")
    (repo / "README.md").write_text("# Project\n")
    run("git", "add", ".")
    run("git", "commit", "-m", "initial commit")

    # Branch for agent-alpha: auth work
    run("git", "checkout", "-b", "claude/agent-alpha-AAA")
    (repo / "auth.py").write_text(
        "def login(user, password):\n    return True\n\n"
        "class AuthService:\n    pass\n"
    )
    run("git", "add", ".")
    run("git", "commit", "-m", "feat(auth): implement login and AuthService")

    # Back to main
    run("git", "checkout", "main")

    # Branch for agent-beta: also implements login (overlap!)
    run("git", "checkout", "-b", "claude/agent-beta-BBB")
    (repo / "services.py").write_text(
        "def login(username, pwd):\n    return authenticate(username, pwd)\n\n"
        "class UserService:\n    pass\n"
    )
    run("git", "add", ".")
    run("git", "commit", "-m", "feat: add login and UserService")

    run("git", "checkout", "main")

    return repo


class TestGitIngester:
    def test_ingest_branch_creates_session(self, git_repo):
        tracker = AgentTracker()
        ingester = GitIngester(repo_path=git_repo)
        aid = ingester.ingest_branch(tracker, branch="claude/agent-alpha-AAA", base="main")
        session = tracker.get_session(aid)
        assert session is not None

    def test_ingest_captures_files(self, git_repo):
        tracker = AgentTracker()
        ingester = GitIngester(repo_path=git_repo)
        ingester.ingest_branch(tracker, branch="claude/agent-alpha-AAA", base="main")
        session = tracker.get_session("claude-agent-alpha-AAA")
        assert "auth.py" in session.files_touched

    def test_ingest_captures_commits(self, git_repo):
        tracker = AgentTracker()
        ingester = GitIngester(repo_path=git_repo)
        ingester.ingest_branch(tracker, branch="claude/agent-alpha-AAA", base="main")
        session = tracker.get_session("claude-agent-alpha-AAA")
        assert session.commit_count == 1
        assert "auth" in session._commits[0]["message"].lower()

    def test_ingest_extracts_symbols(self, git_repo):
        tracker = AgentTracker()
        ingester = GitIngester(repo_path=git_repo)
        ingester.ingest_branch(tracker, branch="claude/agent-alpha-AAA", base="main")
        session = tracker.get_session("claude-agent-alpha-AAA")
        symbols = set(session.metadata.get("symbols_defined", []))
        assert "login" in symbols
        assert "AuthService" in symbols

    def test_auto_discover_branches(self, git_repo):
        tracker = AgentTracker()
        ingester = GitIngester(repo_path=git_repo)
        agent_ids = ingester.ingest_all_agent_branches(tracker, prefix="claude/", base="main")
        assert len(agent_ids) == 2

    def test_ingest_stores_branch_metadata(self, git_repo):
        tracker = AgentTracker()
        ingester = GitIngester(repo_path=git_repo)
        ingester.ingest_branch(tracker, branch="claude/agent-alpha-AAA", base="main")
        session = tracker.get_session("claude-agent-alpha-AAA")
        assert session.metadata.get("branch") == "claude/agent-alpha-AAA"

    def test_snapshot_returns_data(self, git_repo):
        ingester = GitIngester(repo_path=git_repo)
        snap = ingester.snapshot("claude/agent-alpha-AAA", base="main")
        assert snap.commit_count == 1
        assert "auth.py" in snap.files_changed
        assert snap.full_diff != ""


class TestFindSymbolConflicts:
    def test_detects_shared_symbol(self, git_repo):
        tracker = AgentTracker()
        ingester = GitIngester(repo_path=git_repo)
        ingester.ingest_all_agent_branches(tracker, prefix="claude/", base="main")
        conflicts = find_symbol_conflicts(tracker)
        # Both branches define 'login'
        shared_names = {c.symbol for c in conflicts}
        assert "login" in shared_names

    def test_no_conflict_for_unique_symbols(self, git_repo):
        tracker = AgentTracker()
        ingester = GitIngester(repo_path=git_repo)
        ingester.ingest_all_agent_branches(tracker, prefix="claude/", base="main")
        conflicts = find_symbol_conflicts(tracker)
        conflict_names = {c.symbol for c in conflicts}
        # AuthService is only in alpha, UserService only in beta
        assert "AuthService" not in conflict_names
        assert "UserService" not in conflict_names

    def test_conflict_lists_both_agents(self, git_repo):
        tracker = AgentTracker()
        ingester = GitIngester(repo_path=git_repo)
        ingester.ingest_all_agent_branches(tracker, prefix="claude/", base="main")
        conflicts = find_symbol_conflicts(tracker)
        login_conflict = next(c for c in conflicts if c.symbol == "login")
        assert len(login_conflict.agent_ids) == 2

    def test_no_conflicts_when_distinct(self):
        tracker = AgentTracker()
        tracker.new_session("a", agent_id="A")
        tracker.new_session("b", agent_id="B")
        tracker.get_session("A").metadata["symbols_defined"] = ["func_a", "ClassA"]
        tracker.get_session("B").metadata["symbols_defined"] = ["func_b", "ClassB"]
        assert find_symbol_conflicts(tracker) == []

    def test_symbol_conflict_str(self):
        c = SymbolConflict(symbol="login", agent_ids=["alpha", "beta"])
        assert "login" in str(c)
        assert "alpha" in str(c)
        assert "beta" in str(c)
