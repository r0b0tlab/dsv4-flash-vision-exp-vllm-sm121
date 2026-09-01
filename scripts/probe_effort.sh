#!/usr/bin/env bash
# Proves server-side thinking/effort defaults via prompt_tokens deltas (encoder prepends the effort text at index 0).
set -u
B="${1:-http://192.168.3.2:30000}"; M="${2:-deepseek-v4-flash-vision-exp}"
pt() { curl -s -m 120 "$B/v1/chat/completions" -H 'Content-Type: application/json' -d "$1" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["usage"]["prompt_tokens"])'; }
base='{"model":"'$M'","messages":[{"role":"user","content":"hi"}],"max_tokens":1'
d=$(pt "$base}")
lo=$(pt "$base,\"chat_template_kwargs\":{\"thinking\":true,\"reasoning_effort\":\"low\"}}")
hi=$(pt "$base,\"chat_template_kwargs\":{\"thinking\":true,\"reasoning_effort\":\"high\"}}")
mx=$(pt "$base,\"chat_template_kwargs\":{\"thinking\":true,\"reasoning_effort\":\"max\"}}")
echo "prompt_tokens default=$d low=$lo high=$hi max=$mx"
r=$(curl -s -m 600 "$B/v1/chat/completions" -H 'Content-Type: application/json' -d '{"model":"'$M'","messages":[{"role":"user","content":"What is 17*23? Answer with the number."}],"max_tokens":2048,"temperature":0}')
echo "$r" | python3 -c 'import json,sys; d=json.load(sys.stdin)["choices"][0]["message"]; print("reasoning_content_len=",len(d.get("reasoning_content") or ""),"content=",repr((d.get("content") or "")[:80]))'
pass=1
[[ "$d" == "$mx" ]] || { echo "FAIL: default != max ($d vs $mx)"; pass=0; }
[[ "$mx" -gt "$hi" && "$hi" -gt "$lo" ]] || { echo "FAIL: effort ordering low<high<max not seen"; pass=0; }
[[ $pass == 1 ]] && echo "EFFORT PROBE: PASS" || echo "EFFORT PROBE: FAIL"
