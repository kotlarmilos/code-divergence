#!/usr/bin/env bash
# Git post-commit hook — re-ingests the current branch after every commit.
#
# Install via:  bash install-hooks.sh
# Or manually:  ln -sf ../../hooks/git-post-commit.sh .git/hooks/post-commit

set -euo pipefail

STORE="${CODE_DIVERGENCE_STORE:-divergence_state.json}"
BASE="${CODE_DIVERGENCE_BASE:-main}"

BRANCH="$(git branch --show-current 2>/dev/null || echo '')"
[ -z "$BRANCH" ] && exit 0

REPO="$(git rev-parse --show-toplevel)"

# Re-ingest just this branch so metrics stay current after each commit
code-divergence --store "$STORE" git-sync \
    --repo "$REPO" \
    --branches "$BRANCH" \
    --base "$BASE" 2>/dev/null || true

# Quick symbol conflict check — warn if another agent branch shares a symbol
CONFLICTS="$(code-divergence --store "$STORE" symbols 2>/dev/null || true)"
if echo "$CONFLICTS" | grep -q '!!'; then
    echo ""
    echo "code-divergence: symbol overlap detected after commit on $BRANCH:"
    echo "$CONFLICTS"
    echo ""
fi
