#!/bin/bash
# One OpenCode task, inside run.sh's sandbox: a fresh clone of the task's repo with no remotes (and the task's
# injected change committed, if any), then opencode run and its follow-up turns in the same session. The event
# stream goes to stdout.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../../.." && pwd)
OC=$ROOT/tools/hessian/calib/work/opencode
OPENCODE=${OPENCODE:-$(command -v opencode || true)}
[[ -x $OPENCODE ]] || { echo "opencode not found: put it on PATH or set OPENCODE" >&2; exit 1; }
TASK=$1

field() { python3 -c "import json,sys; t=json.loads(sys.argv[1]); v=t.get(sys.argv[2]); print(v if isinstance(v, str) else json.dumps(v))" "$TASK" "$1"; }
id=$(field id)
repo=$(field repo)
variant=$(field variant)
prompt=$(field prompt)

case $repo in
    user:*)   src=${USER_REPOS:?set USER_REPOS to the directory holding the user repos}/${repo#user:}; depth=50 ;;
    public:*) r=${repo#public:}; src=$ROOT/tools/hessian/calib/work/sources/repos/${r/\//__}; depth=1 ;;
esac
copy=$OC/copies/$id
rm -rf "$copy"
mkdir -p "$OC/copies"
git clone -q --depth "$depth" "file://$src" "$copy" >&2
cd "$copy"
git remote remove origin
python3 - "$TASK" <<'PY' >&2
import json, subprocess, sys
t = json.loads(sys.argv[1])
m = t.get("mutate")
if m:
    s = open(m["file"]).read()
    assert s.count(m["from"]) == 1, "mutation site not found"
    open(m["file"], "w").write(s.replace(m["from"], m["to"]))
    subprocess.run(["git", "-c", "user.name=dev", "-c", "user.email=dev@localhost", "commit", "-qam", m["message"]], check=True)
PY

export XDG_CONFIG_HOME=$OC/xdg/config XDG_DATA_HOME=$OC/xdg/data XDG_CACHE_HOME=$OC/xdg/cache XDG_STATE_HOME=$OC/xdg/state
if [[ $(field user_config) == true ]]; then
    export XDG_CONFIG_HOME=$OC/xdg-user/config XDG_DATA_HOME=$OC/xdg-user/data XDG_CACHE_HOME=$OC/xdg-user/cache XDG_STATE_HOME=$OC/xdg-user/state
fi
export CARGO_NET_OFFLINE=true npm_config_offline=true PIP_NO_INDEX=1 OPENCODE_DISABLE_AUTOUPDATE=1

args=(run --pure --format json --dir "$copy" --auto ${OPENCODE_DEBUG:+--print-logs --log-level DEBUG})
[[ $(field user_config) == true ]] || args+=(-m Koboldcpp/Qwen3.8-27B-UD-Q4_K_XL.kcpps)
[[ -n $variant ]] && args+=(--variant "$variant")
events=$(mktemp)
# opencode run reads a piped stdin into the message and waits for its EOF
timeout "${OPENCODE_TIMEOUT:-5400}" "$OPENCODE" "${args[@]}" "$prompt" < /dev/null | tee -a "$events"
sid=$(grep -o '"sessionID":"[^"]*"' "$events" | head -1 | cut -d'"' -f4 || true)
python3 -c "import json,sys; [print(f) for f in json.loads(sys.argv[1]).get('followups', [])]" "$TASK" | while IFS= read -r follow; do
    [[ -z $sid ]] && { echo "no session id for follow-ups" >&2; break; }
    timeout "${OPENCODE_TIMEOUT:-5400}" "$OPENCODE" "${args[@]}" -s "$sid" "$follow" < /dev/null | tee -a "$events"
done
