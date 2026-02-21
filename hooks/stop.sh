#!/usr/bin/env bash
# Claude Code Stop hook — checkpoint triggered when an agent session ends.
#
# Runs git-sync to ingest the current branch's full commit + diff history,
# then prints a divergence/overlap summary. This mirrors how Entero HQ creates
# checkpoints at the end of a conversation — we checkpoint at end of session.

set -euo pipefail

STORE="${CODE_DIVERGENCE_STORE:-divergence_state.json}"
BASE="${CODE_DIVERGENCE_BASE:-main}"
PREFIX="${CODE_DIVERGENCE_PREFIX:-claude/}"
REPO="$(git rev-parse --show-toplevel 2>/dev/null || echo '.')"

echo "[code-divergence] Session ended — running git-sync checkpoint..."

# Ingest all agent branches by prefix
code-divergence --store "$STORE" git-sync \
    --repo "$REPO" \
    --prefix "$PREFIX" \
    --base "$BASE" 2>/dev/null || true

# Print symbol conflicts (clearest overlap signal)
CONFLICTS="$(code-divergence --store "$STORE" symbols 2>/dev/null || true)"
if echo "$CONFLICTS" | grep -q '!!'; then
    echo ""
    echo "[code-divergence] ⚠ Symbol conflicts detected:"
    echo "$CONFLICTS"
fi

# Print pairwise status
code-divergence --store "$STORE" status 2>/dev/null || true
