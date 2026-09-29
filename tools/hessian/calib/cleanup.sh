#!/bin/bash
# Removes work/ (downloads, generation state, OpenCode dirs and captures, diagnostic Grams) and leftover sandbox
# layers, then lists the kept dataset in dataset/.
set -euo pipefail
CALIB=$(cd "$(dirname "$0")" && pwd)
WORK=$CALIB/work
if [[ -f $WORK/server.pid ]] && kill -0 "$(cat "$WORK/server.pid")" 2>/dev/null; then
    "$CALIB/serve.sh" stop
fi
rm -rf "$WORK"
chmod -R u+rwx "${TMPDIR:-/tmp}"/hessian-oc.* 2>/dev/null || true
rm -rf "${TMPDIR:-/tmp}"/hessian-oc.* 2>/dev/null || true
echo "removed $WORK"
ls -la "$CALIB/dataset"
