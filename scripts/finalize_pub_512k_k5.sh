#!/usr/bin/env bash
# Finalize the k=5 adaptive publication: grade Q200v2 text180 local lanes,
# emit public JSONs, rebuild the site. Run after run-pub-512k-k5-evals.sh
# completes (Q200 + BFCL + NIAH raw evidence in
# evidence/vision-opt/V0/prod-512k-k5-adapt/).
set -euo pipefail
ROOT="${ROOT:?set ROOT to the repo root}"
EV="${EV:-${ROOT}/evidence/vision-opt/V0/prod-512k-k5-adapt}"
export Q200V2_RUNNER="${Q200V2_RUNNER:?set Q200V2_RUNNER to the frozen q200v2 runner root}"
export Q200V2_DATASET="${Q200V2_RUNNER}/artifacts/quality-text-180-v2.jsonl"
cd "${EV}"

echo "== humaneval local grade =="
Q200_ROWS="${EV}/q200v2-text180.rows.jsonl" Q200_HE_OUT="${EV}/humaneval-local-grade.json" \
  python3 "${ROOT}/scripts/grade_humaneval_local.py"

echo "== niah public copy =="
python3 - <<'PY'
import json
from pathlib import Path
ev = Path(".")
src = ev / "niah-524288.json"
if src.exists():
    d = json.loads(src.read_text())
    pub = {"verdict": d.get("verdict"), "target_tokens": d.get("target_tokens")}
    pub["results"] = {
        k: {k2: v.get(k2) for k2 in ("retrieved", "api_prompt_tokens", "elapsed_s", "multikey")}
        for k, v in (d.get("results") or {}).items()
    }
    (ev / "niah-public.json").write_text(json.dumps(pub, indent=2) + "\n")
    print("niah", pub.get("verdict"))
else:
    print("niah raw absent")
PY

echo "== bfcl public copy =="
python3 - <<'PY'
import json
from pathlib import Path
src = next(Path(".").glob("bfcl-hard20-run*/bfcl-hard20-summary.json"), None)
if src:
    d = json.loads(src.read_text())
    score = d.get("score") or {}
    pub = {
        "status": "SCORED" if score else "PENDING",
        "family": "bfcl_hard20",
        "score": {k: score.get(k) for k in ("accuracy", "correct_count", "incorrect_count", "total_count")},
        "thinking": "high",
    }
    Path("bfcl-hard20-public.json").write_text(json.dumps(pub, indent=2) + "\n")
    print("bfcl", pub["score"])
else:
    print("bfcl summary absent")
PY

echo "== q200 e2e =="
python3 - <<'PY'
import json
from pathlib import Path
summary = Path("q200v2-text180.summary.json")
if summary.exists():
    d = json.loads(summary.read_text())
    e2e = d.get("e2e_tok_s") or {}
    # local-graded lanes (grader INCOMPLETE on humaneval sandbox + hard by design)
    local = {}
    he = Path("humaneval-local-grade.json")
    hr = Path("hard-reasoning-grade.json")
    if he.exists():
        h = json.loads(he.read_text())
        local["humaneval_local_subprocess"] = f"{h['passed']}/{h['n']}"
    if hr.exists():
        h = json.loads(hr.read_text())
        local["hard_reasoning_manual"] = f"{h['passed']}/{h['n']}"
    out = {
        "profile": "524288 k=5 adaptive thinking=high vision-54631-fi512b-k5adapt",
        "status_from_grader": d.get("status"),
        "rows": d.get("rows"),
        "correct_count": d.get("correct_count"),
        "incorrect_count": d.get("incorrect_count"),
        "ungraded_count": d.get("ungraded_count"),
        "grader_error_count": d.get("grader_error_count"),
        "e2e_tok_s": {k: e2e.get(k) for k in ("n", "mean", "min", "max", "aggregate_completion_over_wall")},
        "campaign_wall_seconds": d.get("campaign_wall_seconds"),
        "local_graded": local,
        "note": "thinking=high, thinking_token_budget=2048. k=5 adaptive. 512k.",
    }
    Path("Q200V2-E2E.json").write_text(json.dumps(out, indent=2) + "\n")
    print("e2e", out["e2e_tok_s"])
else:
    print("q200 summary absent")
PY

echo "== rebuild site =="
python3 "${ROOT}/scripts/build_prod_512k_site.py"

echo "== PII gate on public site =="
# Generic private-pattern scan: LAN ranges, absolute home paths, token shapes.
# Patterns are string-built so this file itself never contains the literals.
python3 - "${ROOT}/publication/html/index.html" <<'PY'
import re
import sys
from pathlib import Path
t = Path(sys.argv[1]).read_text()
patterns = [
    "192" + "." + "168" + ".",
    "10" + "." + r"\d+" + "." + r"\d+" + "." + r"\d+",
    "172" + "." + r"(1[6-9]|2\d|3[01])" + ".",
    "/ho" + "me" + "/" + r"[A-Za-z0-9_.-]+",
    "ghp" + "_" + r"[A-Za-z0-9]{20,}",
    "github" + "_pat" + "_" + r"[A-Za-z0-9_]{20,}",
    "NIAH" + "_SLOT",
    r"[A-Za-z0-9-]+" + ".ts" + ".net",
]
bad = [p for p in patterns if re.search(p, t)]
assert not bad, f"PII leak: {bad}"
print("site PII clean")
PY
echo "finalize done"