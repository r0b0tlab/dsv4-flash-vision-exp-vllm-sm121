#!/usr/bin/env bash
# Gates: arithmetic 4/4 + max_tokens 1-vs-2 probe + health. Host-side, hits API.
set -u
B=http://192.168.3.2:30000
M=deepseek-v4-flash-vision-exp
pass=0; fail=0
q() { # prompt max_tokens
  curl -s -m 120 "$B/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"$1\"}],\"max_tokens\":$2,\"temperature\":0}"
}
check() { # expr want
  local r txt
  r=$(q "Compute exactly: $1. Answer with only the number, nothing else." 200)
  txt=$(echo "$r" | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])' 2>/dev/null)
  if echo "$txt" | grep -q "$2"; then echo "OK   $1 -> $txt"; pass=$((pass+1)); else echo "FAIL $1 -> $txt"; fail=$((fail+1)); fi
}
check "17*23" "391"
check "12*12" "144"
check "144/12" "12"
check "100-27" "73"
r1=$(q "Continue exactly: ALP" 1); r2=$(q "Continue exactly: ALP" 2)
t1=$(echo "$r1" | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])' 2>/dev/null)
t2=$(echo "$r2" | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])' 2>/dev/null)
echo "mt1='$t1' mt2='$t2'"
if [ -n "$t2" ] && [ "${t2:0:${#t1}}" = "$t1" ]; then echo "OK   max_tokens prefix"; pass=$((pass+1)); else echo "FAIL max_tokens prefix"; fail=$((fail+1)); fi
h=$(curl -s -m 8 "$B/v1/models" | head -c 200)
echo "health: $h"
echo "GATES: pass=$pass fail=$fail"
