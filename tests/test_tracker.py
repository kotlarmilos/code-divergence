"""Tests for tracker module."""

import json
import tempfile
from pathlib import Path

import pytest

from code_divergence.tracker import AgentTracker, AgentSession, Event, EventType


def make_tracker(tmp_path=None) -> AgentTracker:
    if tmp_path:
        return AgentTracker(store_path=tmp_path / "state.json")
    return AgentTracker()


class TestAgentSession:
    def test_files_touched_includes_created_and_modified(self):
        tracker = make_tracker()
        aid = tracker.new_session("alpha")
        tracker.track_file_change(aid, "src/a.py", EventType.FILE_MODIFIED)
        tracker.track_file_change(aid, "src/b.py", EventType.FILE_CREATED)
        tracker.track_file_change(aid, "src/c.py", EventType.FILE_DELETED)
        s = tracker.get_session(aid)
        assert "src/a.py" in s.files_touched
        assert "src/b.py" in s.files_touched
        assert "src/c.py" not in s.files_touched  # deleted not in touched

    def test_error_rate(self):
        tracker = make_tracker()
        aid = tracker.new_session("beta")
        for _ in range(8):
            tracker.track_file_change(aid, "x.py")
        for _ in range(2):
            tracker.track_error(aid, "oops")
        s = tracker.get_session(aid)
        assert abs(s.error_rate - 0.2) < 1e-9

    def test_commit_tracking(self):
        tracker = make_tracker()
        aid = tracker.new_session("gamma")
        tracker.track_commit(aid, "abc123", "feat: add thing")
        s = tracker.get_session(aid)
        assert s.commit_count == 1
        assert s._commits[0]["hash"] == "abc123"

    def test_task_tracking(self):
        tracker = make_tracker()
        aid = tracker.new_session("delta")
        tracker.track_task_start(aid, "write tests")
        tracker.track_task_end(aid, "done")
        s = tracker.get_session(aid)
        assert s.task_count == 1

    def test_output_accumulation(self):
        tracker = make_tracker()
        aid = tracker.new_session("epsilon")
        tracker.track_output(aid, "def foo(): pass", task="task1")
        tracker.track_output(aid, "def bar(): pass", task="task2")
        s = tracker.get_session(aid)
        assert len(s._outputs) == 2
        assert "def foo" in s.all_output_text


class TestAgentTrackerPersistence:
    def test_save_and_load(self, tmp_path):
        store = tmp_path / "state.json"
        tracker = AgentTracker(store_path=store)
        aid = tracker.new_session("alpha", agent_id="a1")
        tracker.track_file_change("a1", "main.py", EventType.FILE_MODIFIED)
        tracker.save(store)

        tracker2 = AgentTracker()
        tracker2.load(store)
        s = tracker2.get_session("a1")
        assert s.name == "alpha"
        assert "main.py" in s.files_touched

    def test_round_trip_events(self, tmp_path):
        store = tmp_path / "state.json"
        tracker = AgentTracker(store_path=store)
        aid = tracker.new_session("z", agent_id="z1")
        tracker.track_commit("z1", "deadbeef", "initial commit", ["README.md"])
        tracker.track_error("z1", "something failed")
        tracker.save(store)

        tracker2 = AgentTracker()
        tracker2.load(store)
        s = tracker2.get_session("z1")
        assert s.commit_count == 1
        assert len(s._errors) == 1

    def test_session_summary_keys(self):
        tracker = make_tracker()
        aid = tracker.new_session("x")
        summary = tracker.get_session(aid).summary()
        expected_keys = {
            "agent_id", "name", "duration_seconds", "event_count",
            "files_touched", "commit_count", "task_count", "error_count", "error_rate",
        }
        assert expected_keys.issubset(summary.keys())


class TestTrackerMissingSession:
    def test_get_missing_session_raises(self):
        tracker = make_tracker()
        with pytest.raises(KeyError):
            tracker.get_session("nonexistent")
