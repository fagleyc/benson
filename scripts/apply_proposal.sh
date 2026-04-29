#!/usr/bin/env bash
# apply_proposal.sh <branch>
#
# Fast-forward-merges a proposal/<...> branch into main, syntax-checks the
# Python sources, restarts benson, and rolls back if startup fails.
#
# Run by: the benson middleware (via subprocess) when Casey clicks "merge"
# on the /admin/proposals page. Requires sudoers carveout for
# `systemctl restart benson.service`.

set -euo pipefail

BRANCH="${1:-}"
if [[ -z "$BRANCH" ]]; then
    echo "usage: apply_proposal.sh <branch>" >&2
    exit 2
fi

# Only proposal/* branches may be applied. Belt-and-suspenders: even if
# the dashboard is somehow tricked, this guards the critical path.
if [[ "$BRANCH" != proposal/* ]]; then
    echo "refusing: branch '$BRANCH' does not start with 'proposal/'" >&2
    exit 3
fi

cd /opt/benson

# Capture pre-merge state so we can roll back cleanly.
PRE_SHA="$(git rev-parse main)"
echo "[apply] pre-merge main = $PRE_SHA"

# Verify the proposal branch exists.
if ! git show-ref --quiet --verify "refs/heads/$BRANCH"; then
    echo "[apply] branch '$BRANCH' not found" >&2
    exit 4
fi

# Switch to main and fast-forward merge. Reject non-FF merges — proposal
# branches must be rebased onto main before they can apply.
git checkout main >/dev/null 2>&1
if ! git merge --ff-only "$BRANCH"; then
    echo "[apply] non-FF merge — proposal branch must rebase onto main" >&2
    exit 5
fi

POST_SHA="$(git rev-parse main)"
echo "[apply] post-merge main = $POST_SHA"

# Syntax check every staged Python source. Cheap, catches the obvious
# class of breakage before we restart the service.
echo "[apply] python syntax check"
if ! find /opt/benson/middleware -maxdepth 2 -name '*.py' -not -path '*/venv/*' -not -path '*/__pycache__/*' \
        -exec /opt/benson/middleware/venv/bin/python -m py_compile {} \;; then
    echo "[apply] syntax check failed — rolling back" >&2
    git reset --hard "$PRE_SHA"
    exit 6
fi

# Restart benson. The sudoers carveout limits this to exactly this command.
echo "[apply] restarting benson.service"
if ! sudo -n /bin/systemctl restart benson.service; then
    echo "[apply] systemctl restart failed — rolling back" >&2
    git reset --hard "$PRE_SHA"
    sudo -n /bin/systemctl restart benson.service || true
    exit 7
fi

# Wait a few seconds and confirm it came up.
sleep 4
if ! sudo -n /bin/systemctl is-active --quiet benson.service; then
    echo "[apply] service is not active after restart — rolling back" >&2
    git reset --hard "$PRE_SHA"
    sudo -n /bin/systemctl restart benson.service || true
    exit 8
fi

# Delete the proposal branch — it's been merged.
git branch -d "$BRANCH" >/dev/null 2>&1 || true

echo "[apply] OK: $BRANCH merged to $POST_SHA, benson restarted"
