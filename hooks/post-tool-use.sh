#!/usr/bin/env bash
# Claude Code PostToolUse hook — tracks every file edit passively.
#
# Claude Code passes a JSON object on stdin with shape:
#   { "session_id": "...", "tool_name": "Edit", "tool_input": { "file_path": "..." }, ... }
#
# Wired in via .claude/settings.json — no agent instrumentation needed.

set -euo pipefail

STORE="${CODE_DIVERGENCE_STORE:-divergence_state.json}"

# Parse session_id, tool_name, and file_path from stdin
read -r -d '' INPUT || true
INPUT="${INPUT:-$(cat)}"

eval "$(echo "$INPUT" | python3 - <<'PYEOF'
import sys, json
try:
    d = json.loads(sys.stdin.read())
    sid   = d.get("session_id", "unknown-session")
    tool  = d.get("tool_name", "unknown")
    inp   = d.get("tool_input", {})
    fpath = inp.get("file_path") or inp.get("path") or ""
    # Emit shell variable assignments
    print(f"AGENT_ID={sid!r}")
    print(f"TOOL={tool!r}")
    print(f"FILE_PATH={fpath!r}")
except Exception as e:
    print("AGENT_ID='unknown-session'")
    print("TOOL='unknown'")
    print("FILE_PATH=''")
PYEOF
)"

[ -z "$FILE_PATH" ] && exit 0

# Auto-create session on first use (idempotent — safe to call repeatedly)
code-divergence --store "$STORE" new-session "$AGENT_ID" --agent-id "$AGENT_ID" 2>/dev/null || true

code-divergence --store "$STORE" track \
    --agent-id "$AGENT_ID" \
    --event file_modified \
    --file "$FILE_PATH"

code-divergence --store "$STORE" track \
    --agent-id "$AGENT_ID" \
    --event tool_call \
    --tool "$TOOL"
