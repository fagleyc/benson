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

# Clean the proposal-meta file out of main — it was branch-only context
# (rationale + instructions for Casey's review). No reason to ship it.
if [ -f /opt/benson/.benson-proposal-meta.json ]; then
    git rm -q .benson-proposal-meta.json >/dev/null 2>&1 || true
    GIT_AUTHOR_NAME=Benson GIT_AUTHOR_EMAIL=benson@fagley.home \
    GIT_COMMITTER_NAME=Benson GIT_COMMITTER_EMAIL=benson@fagley.home \
    git commit -q -m "[proposal-meta] cleanup after merge of $BRANCH" >/dev/null 2>&1 || true
fi

# Delete the proposal branch — it's been merged.
git branch -d "$BRANCH" >/dev/null 2>&1 || true

# Restart benson. We can't `systemctl restart` directly here because this
# script was invoked as a subprocess of benson — the restart would SIGTERM
# our own parent before we finish. Detach via setsid + background, with a
# 2s delay so the response (this stdout) reaches the caller first.
#
# Rollback safety net: a separate watcher (scripts/benson_watchdog.sh)
# spawned in the same detached chain checks 8s after restart whether
# benson came up healthy, and if not reverts main to the pre-merge SHA
# and restarts again.
echo "[apply] OK: $BRANCH merged to $POST_SHA — restarting (detached)"

PRE_SHA_FILE="/tmp/benson-apply-${POST_SHA}.presha"
echo "$PRE_SHA" > "$PRE_SHA_FILE"

setsid bash -c "
    sleep 2
    sudo -n /bin/systemctl restart benson.service
    sleep 8
    if ! sudo -n /bin/systemctl is-active --quiet benson.service; then
        cd /opt/benson
        git reset --hard \"$PRE_SHA\" >/dev/null 2>&1
        sudo -n /bin/systemctl restart benson.service
        echo \"[watchdog] benson failed to come up — rolled back to $PRE_SHA\" \
            | systemd-cat -t benson-apply -p err
    else
        echo \"[watchdog] benson healthy on $POST_SHA\" \
            | systemd-cat -t benson-apply -p info
    fi
    rm -f \"$PRE_SHA_FILE\"
" </dev/null >/dev/null 2>&1 &

exit 0
