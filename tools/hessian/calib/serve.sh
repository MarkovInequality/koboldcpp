#!/bin/bash
# koboldcpp serving Qwen3.8-27B-HQ8_0 for the calibration set.
#   serve.sh gen       raw completions, 8 parallel requests, port 5002
#   serve.sh opencode  the user's OpenCode settings (jinja, tools, q5_1 KV, MTP), one session, 128k context, port 5003
#   serve.sh stop
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
WORK=$ROOT/build-hq/calib/work
MODEL=${MODEL:-$HOME/Sandbox/unquantized/Qwen3.8-27B-gguf/Qwen3.8-27B-HQ8_0.gguf}
PIDFILE=$WORK/server.pid
mkdir -p "$WORK"

stop() {
    if [[ -f $PIDFILE ]]; then
        pid=$(cat "$PIDFILE")
        kill "$pid" 2>/dev/null || true
        for _ in $(seq 60); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
        rm -f "$PIDFILE"
    fi
}

start() {
    local log=$WORK/server-$1.log port=$2
    shift 2
    stop
    cd "$ROOT"
    nohup python3 koboldcpp.py --model "$MODEL" --usecuda normal 0 --gpulayers 99 --skiplauncher --quiet \
        --threads 7 --port "$port" "$@" > "$log" 2>&1 &
    echo $! > "$PIDFILE"
    for _ in $(seq 600); do
        grep -q "Please connect to custom endpoint" "$log" && { echo "ready on port $port"; return 0; }
        kill -0 "$(cat "$PIDFILE")" 2>/dev/null || { tail -20 "$log"; echo "server died"; return 1; }
        sleep 2
    done
    echo "timed out"; return 1
}

case ${1:-} in
    gen)
        # FFN weights of the last 7 layers stay in RAM, which leaves ~1 GB of VRAM
        start gen 5002 --overridetensors 'blk\.(5[7-9]|6[0-3])\.ffn_(gate|up|down)\.weight=CPU' \
            --contextsize 65536 --parallelrequests 8 --quantkv q8_0 --batchsize 512 --noshift ;;
    opencode)
        start opencode 5003 --overridetensors "${OT:-blk\.(5[0-9]|6[0-3])\.ffn_(gate|up|down)\.weight=CPU}" \
            --contextsize 131072 --quantkv q5_1 --jinja --jinja_tools --batchsize 1024 --usemtp ;;
    stop) stop ;;
    *) echo "usage: $0 gen|opencode|stop"; exit 1 ;;
esac
