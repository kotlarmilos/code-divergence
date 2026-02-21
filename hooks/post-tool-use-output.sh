#!/usr/bin/env bash
# Claude Code PostToolUse hook — captures agent *outputs* (code written to files).
# Pair this with post-tool-use.sh to also track the content being generated.
#
# Install alongside post-tool-use.sh in .claude/settings.json:
#
#   "matcher": "Write"   ← capture full file writes as output events

set -euo pipefail

STORE="${CODE_DIVERGENCE_STORE:-divergence_state.json}"
AGENT_ID="${CLAUDE_SESSION_ID:-unknown-session}"

INPUT="$(cat)"

CONTENT="$(echo "$INPUT" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d.get('content') or d.get('new_string') or '')
except Exception:
    print('')
" 2>/dev/null || true)"

[ -z "$CONTENT" ] && exit 0

code-divergence --store "$STORE" new-session "$AGENT_ID" --agent-id "$AGENT_ID" 2>/dev/null || true

code-divergence --store "$STORE" track \
    --agent-id "$AGENT_ID" \
    --event code_output \
    --content "$CONTENT"
