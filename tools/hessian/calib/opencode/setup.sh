#!/bin/bash
# Isolated OpenCode setup for the calibration sessions: XDG dirs under tools/hessian/calib/work/opencode/, a config
# that mirrors the user's (the same provider and model names, which OpenCode writes into its system prompt, and
# the same reasoning-effort variants) pointed at the logging proxy, and a snapshot of the user's OpenCode data and
# original repos to compare against afterwards. The provider's timeouts (5 min each by default) are off or an hour:
# koboldcpp sends a tool-mode reply only once it is complete, HQ8_0 decodes ~15 tokens/s, and OpenCode would retry a
# longer reply from scratch.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../../.." && pwd)
OC=$ROOT/tools/hessian/calib/work/opencode
mkdir -p "$OC"/xdg/{config/opencode,data,cache,state} "$OC"/captures "$OC"/runs

cat > "$OC/xdg/config/opencode/opencode.json" <<'JSON'
{
  "$schema": "https://opencode.ai/config.json",
  "model": "Koboldcpp/Qwen3.8-27B-UD-Q4_K_XL.kcpps",
  "autoupdate": false,
  "share": "disabled",
  "provider": {
    "Koboldcpp": {
      "npm": "@ai-sdk/openai-compatible",
      "options": { "baseURL": "http://127.0.0.1:5004/v1", "apiKey": "dummy", "timeout": false, "headerTimeout": 3600000, "chunkTimeout": 3600000 },
      "models": {
        "Qwen3.8-27B-UD-Q4_K_XL.kcpps": {
          "limit": { "context": 131072, "output": 8192 },
          "options": { "reasoningEffort": "medium" },
          "variants": {
            "xhigh": { "reasoningEffort": "xhigh" },
            "medium": { "reasoningEffort": "medium" },
            "low": { "reasoningEffort": "low" }
          }
        }
      }
    }
  },
  "permission": {
    "edit": "allow",
    "external_directory": "deny",
    "bash": {
      "*": "allow",
      "git push*": "deny", "sudo *": "deny", "curl *": "deny", "wget *": "deny",
      "npm install*": "deny", "npm i *": "deny", "npm add*": "deny", "npm ci*": "deny",
      "pnpm install*": "deny", "pnpm i *": "deny", "pnpm add*": "deny",
      "yarn install*": "deny", "yarn add*": "deny", "bun install*": "deny", "bun add*": "deny",
      "pip install*": "deny", "pip3 install*": "deny", "python -m pip*": "deny", "python3 -m pip*": "deny",
      "uv pip*": "deny", "uv add*": "deny", "uv sync*": "deny",
      "cargo install*": "deny", "cargo add*": "deny"
    }
  }
}
JSON

# the user's own config, pointed at the proxy, for the system-prompt/tool-list comparison
mkdir -p "$OC/xdg-user/config/opencode" "$OC/xdg-user/data" "$OC/xdg-user/cache" "$OC/xdg-user/state"
sed 's#"baseURL": *"[^"]*"#"baseURL": "http://127.0.0.1:5004/v1"#' "${XDG_CONFIG_HOME:-$HOME/.config}/opencode/opencode.json" > "$OC/xdg-user/config/opencode/opencode.json"

[[ -f $OC/snapshot-before.json ]] || python3 "$HERE/snapshot.py" "$OC/snapshot-before.json"
echo "set up $OC"
