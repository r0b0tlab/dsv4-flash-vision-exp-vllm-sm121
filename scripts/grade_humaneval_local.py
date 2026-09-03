#!/usr/bin/env python3
"""Local HumanEval grade via subprocess+timeout. No exec() of model code in-process."""
from __future__ import annotations

import ast
import json
import re
import subprocess
import tempfile
from pathlib import Path

ROWS = Path("/home/r0b0tdgx/projects/DeepSeek-V4-Flash-Vision-Exp/evidence/vision-opt/V0/prod-512k-k3/q200v2-text180.rows.jsonl")
DS = Path("/home/r0b0tdgx/qwen38-flash-next-w4a16/q200v2ar-20260829T141533Z-runner/artifacts/quality-text-180-v2.jsonl")
OUT = Path("/home/r0b0tdgx/projects/DeepSeek-V4-Flash-Vision-Exp/evidence/vision-opt/V0/prod-512k-k3/humaneval-local-grade.json")


def extract_code(text: str) -> str:
    m = re.search(r"```(?:python|py)?\s*(.*?)```", text, flags=re.I | re.S)
    if m:
        return m.group(1).strip() + "\n"
    return text.strip() + "\n"


def prelude(prompt: str, entry: str) -> str:
    m = re.search(r"```(?:python|py)?\s*(.*?)```", prompt, flags=re.I | re.S)
    if not m:
        return ""
    source = m.group(1)
    tree = ast.parse(source)
    targets = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == entry]
    if len(targets) != 1:
        return ""
    t = targets[0]
    start = min([t.lineno, *(d.lineno for d in t.decorator_list)])
    pre = "".join(source.splitlines(True)[: start - 1]).strip()
    return (pre + "\n\n") if pre else ""


def grade_one(code: str, pre: str, entry: str, test: str, timeout: float = 12.0) -> dict:
    body = pre + code + "\n" + test + f"\ncheck({entry})\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(body)
        path = f.name
    try:
        p = subprocess.run(["python3", path], capture_output=True, text=True, timeout=timeout)
        ok = p.returncode == 0
        return {
            "passed": ok,
            "status": "scored" if ok else "failed",
            "returncode": p.returncode,
            "stderr": str(p.stderr or "")[-400:],
        }
    except subprocess.TimeoutExpired:
        return {"passed": False, "status": "timeout"}
    finally:
        Path(path).unlink(missing_ok=True)


def main() -> None:
    ds = {json.loads(l)["id"]: json.loads(l) for l in DS.read_text().splitlines()}
    results = []
    for line in ROWS.read_text().splitlines():
        r = json.loads(line)
        if r.get("family") != "humaneval":
            continue
        src = ds[r["id"]]
        ref = src["reference"]
        entry = ref["entry_point"]
        test = ref["test"]
        pre = prelude(src["prompt"], entry)
        code = extract_code(r.get("content") or "")
        g = grade_one(code, pre, entry, test)
        g["id"] = r["id"]
        results.append(g)
    passed = sum(1 for x in results if x.get("passed") is True)
    failed = sum(1 for x in results if x.get("passed") is False)
    out = {"n": len(results), "passed": passed, "failed": failed, "results": results}
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"n": out["n"], "passed": passed, "failed": failed}))


if __name__ == "__main__":
    main()
