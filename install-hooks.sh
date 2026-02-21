#!/usr/bin/env bash
# Installs code-divergence git hooks into .git/hooks/.
#
# Usage:
#   bash install-hooks.sh            # install into the current repo
#   bash install-hooks.sh /path/repo # install into a specific repo

set -euo pipefail

REPO="${1:-$(git rev-parse --show-toplevel 2>/dev/null || echo '.')}"
HOOKS_DIR="$REPO/.git/hooks"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -d "$HOOKS_DIR" ]; then
    echo "Error: $HOOKS_DIR does not exist. Is $REPO a git repository?"
    exit 1
fi

# post-commit — re-ingest branch + symbol conflict check after each commit
install_hook() {
    local name="$1"
    local source="$2"
    local target="$HOOKS_DIR/$name"

    if [ -f "$target" ] && [ ! -L "$target" ]; then
        echo "  Backing up existing $name → $name.bak"
        mv "$target" "$target.bak"
    fi

    ln -sf "$source" "$target"
    chmod +x "$target"
    echo "  Installed $name → $target"
}

echo "Installing code-divergence git hooks into $HOOKS_DIR"
echo ""

install_hook "post-commit" "$SCRIPT_DIR/hooks/git-post-commit.sh"

echo ""
echo "Done. Each commit will now:"
echo "  1. Re-ingest the current branch via 'code-divergence git-sync'"
echo "  2. Print a warning if any symbol is being implemented by multiple branches"
echo ""
echo "Optional: set these env vars to customise behaviour:"
echo "  CODE_DIVERGENCE_STORE  — state file path (default: divergence_state.json)"
echo "  CODE_DIVERGENCE_BASE   — base branch to diff against (default: main)"
