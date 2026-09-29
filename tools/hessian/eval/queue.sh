#!/bin/bash
# queue.sh: runs the lines of WORK/queue.txt ("NAME ALPHA DAMP [NOTE]") through iter.sh one at a time, removing each
# as it starts, so lines can be appended while it runs. Each run's output goes to WORK/iter-NAME.out.
set -uo pipefail
source "$(dirname "$0")/common.sh"
Q=$WORK/queue.txt
while line=$(head -1 "$Q" 2>/dev/null) && [[ -n $line ]]; do
    sed -i 1d "$Q"
    echo "$(date +%H:%M) start: $line"
    eval "set -- $line"
    if "$HERE/iter.sh" "$@" > "$WORK/iter-$1.out" 2>&1; then echo "$(date +%H:%M) done: $1"; else echo "$(date +%H:%M) FAILED: $1"; fi
done
echo "queue empty"
