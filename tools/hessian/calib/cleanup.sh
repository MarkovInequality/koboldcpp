#!/bin/bash
# Removes build-hq/calib/work/ (downloads, generation state, OpenCode dirs and captures, diagnostic Grams) and
# lists anything left under build-hq/calib/ outside the kept dataset.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
CALIB=$ROOT/build-hq/calib
if [[ -f $CALIB/work/server.pid ]] && kill -0 "$(cat "$CALIB/work/server.pid")" 2>/dev/null; then
    "$(dirname "$0")/serve.sh" stop
fi
rm -rf "$CALIB/work"
chmod -R u+rwx /tmp/hessian-oc.* 2>/dev/null || true
rm -rf /tmp/hessian-oc.* 2>/dev/null || true
echo "removed $CALIB/work"
left=$(find "$CALIB" -mindepth 1 -maxdepth 2 ! -path "$CALIB/qwen38-calib-v1" ! -path "$CALIB/qwen38-calib-v1/*")
if [[ -n $left ]]; then
    echo "left outside the kept paths:"
    echo "$left"
fi
ls -la "$CALIB/qwen38-calib-v1" 2>/dev/null || true
