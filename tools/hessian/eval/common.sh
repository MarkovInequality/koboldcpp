# Sourced by the eval scripts: paths, the required settings and the three evaluation sets.
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
WORK=${WORK:-$HERE/work}     # caches, logs, quantized files: disposable
TEXTS=${TEXTS:-$HERE/texts}  # the chat eval texts, kept (texts.py rebuilds them if missing)
PPL=${PPL:-$ROOT/test-hadamard-ppl-cuda}
QUANTIZE=${QUANTIZE:-$ROOT/quantize_gguf}
mkdir -p "$WORK"
[[ -f $TEXTS/heldout.txt && -f $TEXTS/opencode.txt ]] || python3 "$HERE/texts.py" "$TEXTS"

need() { # need VAR WHAT: exits with a message unless VAR is set
    [[ -n ${!1:-} ]] || { echo "set $1 to $2" >&2; exit 1; }
}
need REF "the BF16 reference model, which the quantizations are also made from"
need WIKI "wikitext-2's wiki.test.raw"

SETS="wiki heldout opencode"

eval_set() { # eval_set NAME: sets TEXT CTX CHUNKS PAR and CHAT (1: chat text, scored with --chat-mask)
    case $1 in
        wiki)     TEXT=$WIKI;              CTX=512;  CHUNKS=80; PAR=4; CHAT=0 ;;
        heldout)  TEXT=$TEXTS/heldout.txt;  CTX=1024; CHUNKS=64; PAR=2; CHAT=1 ;;
        opencode) TEXT=$TEXTS/opencode.txt; CTX=2048; CHUNKS=36; PAR=1; CHAT=1 ;;
        *) echo "no eval set $1" >&2; exit 1 ;;
    esac
}
export REF WIKI WORK TEXTS PPL QUANTIZE

cells() { # cells EVAL_OUTPUT: "bpw | wiki KL | wiki top-1 % | heldout KL | ... | opencode top-1 %", for a log row
    local bpw
    bpw=$(echo "$1" | awk '$1 == "wiki" { print $2 }')
    printf "%s" "${bpw:-?}"
    for s in $SETS; do
        echo "$1" | awk -v s=$s '$1 == s { if (NF < 7) printf " | FAILED | FAILED"; else printf " | %s | %s", $5, $7 }'
    done
}
