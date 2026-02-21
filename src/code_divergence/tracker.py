"""
Agent session tracking.

Records what each agent does: files touched, code written, commits made, task outputs.
Persists state to JSON so monitoring can resume across process restarts.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any


class EventType(str, Enum):
    FILE_MODIFIED = "file_modified"
    FILE_CREATED = "file_created"
    FILE_DELETED = "file_deleted"
    CODE_OUTPUT = "code_output"
    TEXT_OUTPUT = "text_output"
    COMMIT = "commit"
    TASK_START = "task_start"
    TASK_END = "task_end"
    TOOL_CALL = "tool_call"
    ERROR = "error"


@dataclass
class Event:
    type: EventType
    timestamp: float
    data: dict[str, Any]
    agent_id: str

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "timestamp": self.timestamp,
            "data": self.data,
            "agent_id": self.agent_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Event":
        return cls(
            type=EventType(d["type"]),
            timestamp=d["timestamp"],
            data=d["data"],
            agent_id=d["agent_id"],
        )


@dataclass
class AgentSession:
    """All recorded activity for one agent."""

    agent_id: str
    name: str
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    events: list[Event] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Derived sets, rebuilt from events on load
    _files_modified: set[str] = field(default_factory=set, repr=False)
    _files_created: set[str] = field(default_factory=set, repr=False)
    _files_deleted: set[str] = field(default_factory=set, repr=False)
    _outputs: list[str] = field(default_factory=list, repr=False)
    _commits: list[dict] = field(default_factory=list, repr=False)
    _tool_calls: list[str] = field(default_factory=list, repr=False)
    _tasks: list[dict] = field(default_factory=list, repr=False)
    _errors: list[str] = field(default_factory=list, repr=False)

    def add_event(self, event: Event) -> None:
        self.events.append(event)
        self._apply(event)

    def _apply(self, event: Event) -> None:
        t = event.type
        d = event.data
        if t == EventType.FILE_MODIFIED:
            self._files_modified.add(d["path"])
        elif t == EventType.FILE_CREATED:
            self._files_created.add(d["path"])
            self._files_modified.add(d["path"])
        elif t == EventType.FILE_DELETED:
            self._files_deleted.add(d["path"])
        elif t in (EventType.CODE_OUTPUT, EventType.TEXT_OUTPUT):
            self._outputs.append(d.get("content", ""))
        elif t == EventType.COMMIT:
            self._commits.append(d)
        elif t == EventType.TOOL_CALL:
            self._tool_calls.append(d.get("tool", "unknown"))
        elif t == EventType.TASK_START:
            self._tasks.append({"start": event.timestamp, "description": d.get("description", ""), "end": None})
        elif t == EventType.TASK_END:
            if self._tasks:
                self._tasks[-1]["end"] = event.timestamp
        elif t == EventType.ERROR:
            self._errors.append(d.get("message", ""))

    @property
    def files_touched(self) -> set[str]:
        """All files the agent has touched (created + modified)."""
        return self._files_modified | self._files_created

    @property
    def all_output_text(self) -> str:
        return "\n".join(self._outputs)

    @property
    def duration(self) -> float:
        end = self.end_time or time.time()
        return end - self.start_time

    @property
    def event_count(self) -> int:
        return len(self.events)

    @property
    def commit_count(self) -> int:
        return len(self._commits)

    @property
    def error_rate(self) -> float:
        if not self.events:
            return 0.0
        return len(self._errors) / len(self.events)

    @property
    def task_count(self) -> int:
        return len(self._tasks)

    def summary(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "duration_seconds": round(self.duration, 2),
            "event_count": self.event_count,
            "files_touched": sorted(self.files_touched),
            "files_modified": sorted(self._files_modified),
            "files_created": sorted(self._files_created),
            "files_deleted": sorted(self._files_deleted),
            "commit_count": self.commit_count,
            "task_count": self.task_count,
            "error_count": len(self._errors),
            "error_rate": round(self.error_rate, 4),
            "tool_calls": self._tool_calls,
            "outputs_count": len(self._outputs),
        }

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "events": [e.to_dict() for e in self.events],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AgentSession":
        session = cls(
            agent_id=d["agent_id"],
            name=d["name"],
            start_time=d["start_time"],
            end_time=d.get("end_time"),
            metadata=d.get("metadata", {}),
        )
        for event_data in d.get("events", []):
            event = Event.from_dict(event_data)
            session.events.append(event)
            session._apply(event)
        return session


class AgentTracker:
    """
    Central registry for all agent sessions.

    Usage::

        tracker = AgentTracker()
        agent_id = tracker.new_session("agent-alpha")
        tracker.track_file_change(agent_id, "src/foo.py", EventType.FILE_MODIFIED)
        tracker.track_output(agent_id, "def foo(): ...", task="write function")
        tracker.track_commit(agent_id, "abc123", "feat: add foo")
        tracker.end_session(agent_id)
        tracker.save("tracker_state.json")
    """

    def __init__(self, store_path: str | Path | None = None) -> None:
        self._sessions: dict[str, AgentSession] = {}
        self.store_path = Path(store_path) if store_path else None
        if self.store_path and self.store_path.exists():
            self.load(self.store_path)

    # ------------------------------------------------------------------ #
    # Session lifecycle
    # ------------------------------------------------------------------ #

    def new_session(
        self,
        name: str,
        agent_id: str | None = None,
        metadata: dict | None = None,
    ) -> str:
        """Create a new agent session and return its ID."""
        agent_id = agent_id or str(uuid.uuid4())[:8]
        session = AgentSession(
            agent_id=agent_id,
            name=name,
            metadata=metadata or {},
        )
        self._sessions[agent_id] = session
        return agent_id

    def end_session(self, agent_id: str) -> None:
        session = self._get(agent_id)
        session.end_time = time.time()
        if self.store_path:
            self.save(self.store_path)

    def get_session(self, agent_id: str) -> AgentSession:
        return self._get(agent_id)

    @property
    def sessions(self) -> list[AgentSession]:
        return list(self._sessions.values())

    @property
    def active_sessions(self) -> list[AgentSession]:
        return [s for s in self._sessions.values() if s.end_time is None]

    # ------------------------------------------------------------------ #
    # Event recording helpers
    # ------------------------------------------------------------------ #

    def _record(self, agent_id: str, event_type: EventType, data: dict) -> None:
        session = self._get(agent_id)
        event = Event(
            type=event_type,
            timestamp=time.time(),
            data=data,
            agent_id=agent_id,
        )
        session.add_event(event)
        if self.store_path:
            self.save(self.store_path)

    def track_file_change(self, agent_id: str, path: str, change_type: EventType = EventType.FILE_MODIFIED, diff: str = "") -> None:
        """Record that the agent modified, created, or deleted a file."""
        self._record(agent_id, change_type, {"path": path, "diff": diff})

    def track_output(self, agent_id: str, content: str, task: str = "", output_type: EventType = EventType.CODE_OUTPUT) -> None:
        """Record agent-generated content (code or text)."""
        self._record(agent_id, output_type, {"content": content, "task": task})

    def track_commit(self, agent_id: str, commit_hash: str, message: str, files: list[str] | None = None) -> None:
        """Record a git commit made by the agent."""
        self._record(agent_id, EventType.COMMIT, {
            "hash": commit_hash,
            "message": message,
            "files": files or [],
        })

    def track_task_start(self, agent_id: str, description: str) -> None:
        self._record(agent_id, EventType.TASK_START, {"description": description})

    def track_task_end(self, agent_id: str, result: str = "") -> None:
        self._record(agent_id, EventType.TASK_END, {"result": result})

    def track_tool_call(self, agent_id: str, tool: str, args: dict | None = None) -> None:
        self._record(agent_id, EventType.TOOL_CALL, {"tool": tool, "args": args or {}})

    def track_error(self, agent_id: str, message: str) -> None:
        self._record(agent_id, EventType.ERROR, {"message": message})

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 1,
            "sessions": {sid: s.to_dict() for sid, s in self._sessions.items()},
        }
        path.write_text(json.dumps(data, indent=2))

    def load(self, path: str | Path) -> None:
        path = Path(path)
        if not path.exists():
            return
        data = json.loads(path.read_text())
        for sid, session_data in data.get("sessions", {}).items():
            self._sessions[sid] = AgentSession.from_dict(session_data)

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _get(self, agent_id: str) -> AgentSession:
        if agent_id not in self._sessions:
            raise KeyError(f"No session with id '{agent_id}'")
        return self._sessions[agent_id]
