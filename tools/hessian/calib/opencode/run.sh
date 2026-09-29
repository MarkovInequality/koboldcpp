#!/bin/bash
# Runs OpenCode tasks from tasks.jsonl one at a time against koboldcpp (serve.sh opencode, port 5003) through
# the logging proxy (port 5004), then renders each capture into its document. Every run is sandboxed: a
# user + mount namespace overlays $HOME with a throwaway layer in /tmp, so nothing the agent writes, its working
# copy included, reaches the real files; only the isolated XDG dirs are mounted through. The layer is deleted
# once the run is rendered, also on failure or interrupt.
#
# usage: run.sh [--user-config] [TASK_ID ...]   (default: every task not yet rendered without failure)
#   --user-config: one request with the user's own OpenCode config instead, for the prompt comparison
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../../.." && pwd)
OC=$ROOT/build-hq/calib/work/opencode
PY=$ROOT/build-hq/calib/work/venv/bin/python
TASKS=$HERE/tasks.jsonl
PROXY_PID=
LAYER=

cleanup() {
    [[ -n $PROXY_PID ]] && kill "$PROXY_PID" 2>/dev/null
    # overlayfs leaves its work dir mode 000
    [[ -n $LAYER && -d $LAYER ]] && { chmod -R u+rwx "$LAYER" 2>/dev/null; rm -rf "$LAYER"; }
    PROXY_PID=; LAYER=
}
trap 'cleanup; exit 130' INT TERM
trap cleanup EXIT

start_proxy() {
    "$PY" "$HERE/proxy.py" 5004 "${UPSTREAM:-http://127.0.0.1:5003}" "$1" 131072 &
    PROXY_PID=$!
    for _ in $(seq 50); do (echo > /dev/tcp/127.0.0.1/5004) 2>/dev/null && return 0; sleep 0.2; done
    echo "proxy did not start"; return 1
}

# sandbox XDG_DIR COMMAND...: runs COMMAND with $HOME overlaid by a throwaway layer, XDG_DIR mounted through
sandbox() {
    local xdg=$1; shift
    LAYER=$(mktemp -d /tmp/hessian-oc.XXXXXX)
    mkdir -p "$LAYER"/{upper,work,xdg}
    unshare --user --map-root-user --mount -- bash -c '
        set -e
        mount --bind "$2" "$1/xdg"
        mount -t overlay overlay -o lowerdir="$HOME",upperdir="$1/upper",workdir="$1/work" "$HOME"
        mount --bind "$1/xdg" "$2"
        shift 2
        exec unshare --user --map-user='"$(id -u)"' --map-group='"$(id -g)"' -- "$@"
    ' _ "$LAYER" "$xdg" "$@"
}

run_task() {
    local id=$1
    local task
    task=$(grep -F "\"id\": \"$id\"" "$TASKS") || { echo "no task $id"; return 1; }
    local cap=$OC/captures/$id.jsonl
    rm -f "$cap"
    echo "[$(date +%T)] $id"
    start_proxy "$cap" || return 1
    sandbox "$OC/xdg" bash "$HERE/inside.sh" "$task" > "$OC/runs/$id.events.jsonl" 2> "$OC/runs/$id.log"
    local rc=$?
    cleanup
    echo "[$(date +%T)] $id: opencode exit $rc, $(wc -l < "$cap") requests captured"
    "$PY" "$HERE/render.py" "$id"
}

if [[ ${1:-} == --user-config ]]; then
    cap=$OC/captures/_user-config.jsonl
    rm -f "$cap"
    start_proxy "$cap"
    task='{"id": "_user-config", "kind": "probe", "repo": "public:psf/requests", "variant": "", "prompt": "hi", "followups": [], "user_config": true}'
    sandbox "$OC/xdg-user" bash "$HERE/inside.sh" "$task" > "$OC/runs/_user-config.events.jsonl" 2> "$OC/runs/_user-config.log"
    cleanup
    echo "captured $(wc -l < "$cap") requests with the user's config"
    exit 0
fi

ids=("$@")
if [[ ${#ids[@]} -eq 0 ]]; then
    mapfile -t ids < <("$PY" -c "
import json, sys
sys.path.insert(0, '$HERE/..')
from common import WORK
checks = WORK / 'opencode' / 'checks'
for l in open('$TASKS'):
    t = json.loads(l)
    p = checks / (t['id'] + '.json')
    if not p.exists() or json.loads(p.read_text()).get('failed'):
        print(t['id'])")
fi
for id in "${ids[@]}"; do
    run_task "$id"
done
