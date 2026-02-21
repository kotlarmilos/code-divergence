#!/usr/bin/env bash
# Claude Code PostToolUse hook — captures written content for overlap detection.
# Attach this to Write tool calls alongside post-tool-use.sh.

set -euo pipefail

STORE="${CODE_DIVERGENCE_STORE:-divergence_state.json}"

INPUT="$(cat)"

eval "$(echo "$INPUT" | python3 - <<'PYEOF'
import sys, json
try:
    d = json.loads(sys.stdin.read())
    sid     = d.get("session_id", "unknown-session")
    inp     = d.get("tool_input", {})
    content = inp.get("content") or inp.get("new_string") or ""
    print(f"AGENT_ID={sid!r}")
    # Truncate to 8KB to keep the state file manageable
    content = content[:8192]
    # Escape for shell assignment
    import shlex
    print(f"CONTENT={shlex.quote(content)}")
except Exception:
    print("AGENT_ID='unknown-session'")
    print("CONTENT=''")
PYEOF
)"

[ -z "$CONTENT" ] && exit 0

code-divergence --store "$STORE" new-session "$AGENT_ID" --agent-id "$AGENT_ID" 2>/dev/null || true

code-divergence --store "$STORE" track \
    --agent-id "$AGENT_ID" \
    --event code_output \
    --content "$CONTENT"
