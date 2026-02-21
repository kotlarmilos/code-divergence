#!/usr/bin/env bash
# Claude Code PostToolUse hook — passively tracks every file edit without
# the agent knowing it's being monitored.
#
# Install in .claude/settings.json:
#
#   {
#     "hooks": {
#       "PostToolUse": [{
#         "matcher": "Edit|Write|MultiEdit",
#         "hooks": [{
#           "type": "command",
#           "command": "/path/to/hooks/post-tool-use.sh"
#         }]
#       }]
#     }
#   }
#
# The hook receives tool input as JSON on stdin.
# Required env vars (set by Claude Code):
#   CLAUDE_SESSION_ID  — unique ID for this Claude session
#   TOOL_NAME          — which tool was called

set -euo pipefail

STORE="${CODE_DIVERGENCE_STORE:-divergence_state.json}"
AGENT_ID="${CLAUDE_SESSION_ID:-unknown-session}"
TOOL="${TOOL_NAME:-unknown}"

# Read stdin (tool input JSON)
INPUT="$(cat)"

# Extract file path from the tool input (works for Edit, Write, MultiEdit)
FILE_PATH="$(echo "$INPUT" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d.get('file_path') or d.get('path') or '')
except Exception:
    print('')
" 2>/dev/null || true)"

# Nothing to track if no file path
[ -z "$FILE_PATH" ] && exit 0

# Auto-create session on first use (idempotent)
code-divergence --store "$STORE" new-session "$AGENT_ID" --agent-id "$AGENT_ID" 2>/dev/null || true

# Record the file modification
code-divergence --store "$STORE" track \
    --agent-id "$AGENT_ID" \
    --event file_modified \
    --file "$FILE_PATH"

# Record the tool call
code-divergence --store "$STORE" track \
    --agent-id "$AGENT_ID" \
    --event tool_call \
    --tool "$TOOL"
