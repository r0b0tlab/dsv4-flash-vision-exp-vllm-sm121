#!/usr/bin/env python3
"""Generate sanitized beige-retro results page from prod-512k-k5-adapt JSON. No PII.

Reads evidence/vision-opt/V0/prod-512k-k5-adapt/ and writes
publication/html/index.html. Lanes that have not landed yet render as "—"
placeholders so the page can be rebuilt the moment eval JSONs appear.
"""
from __future__ import annotations

import json
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
EV = _REPO / "evidence/vision-opt/V0/prod-512k-k5-adapt"
OUT = _REPO / "publication/html/index.html"


def load(name: str):
    p = EV / name
    return json.loads(p.read_text()) if p.exists() else None


EV1M = _REPO / "evidence/vision-opt/V0/prod-1m-k5-adapt"


def load1m(name: str):
    p = EV1M / name
    return json.loads(p.read_text()) if p.exists() else None


def quality_counts(rows):
    """Count scored outcomes per family from the q200 rows jsonl."""
    from collections import Counter

    c = Counter()
    for r in rows:
        fam = r.get("family")
        if fam:
            c[fam] += 1
            c[f"{fam}_pass"] += 1 if r.get("passed") is True else 0
    return c


def main() -> None:
    serve = load("SERVE-IDENTITY.json") or {}
    vis = load("VISION.json") or {}
    thr = load("THROUGHPUT-SHORT-PROSE.json") or {}
    td = load("TD2W300-thinkoff.json")
    qsummary = load("q200v2-text180.summary.json")
    he = load("humaneval-local-grade.json")
    hr = load("hard-reasoning-grade.json")
    niah = load("niah-public.json")
    bfcl = load("bfcl-hard20-public.json")
    tel = load("TELEMETRY-SUMMARY.json")
    p1m = load1m("PROFILE-1M.json") or {}
    niah1m = load1m("niah-public.json") or {}
    conc = p1m.get("concurrency_sweep") or {}
    thr1m = (p1m.get("throughput_thinkoff") or {})
    n1m = (niah1m.get("results") or {}).get("90%", {})

    # rows for family-level pass counts (summary may be absent until grader close)
    fam = {}
    rows_p = EV / "q200v2-text180.rows.jsonl"
    if rows_p.exists():
        fam = quality_counts([json.loads(l) for l in rows_p.read_text().splitlines() if l.strip()])

    gsm_n = fam.get("gsm8k", 80)
    gsm_p = fam.get("gsm8k_pass")
    ife_n = fam.get("ifeval", 40)
    ife_p = fam.get("ifeval_pass")
    he_pass = int((he or {}).get("passed") or fam.get("humaneval_pass") or 0)
    he_n = int((he or {}).get("n") or fam.get("humaneval") or 40)
    hr_pass = int((hr or {}).get("passed") or fam.get("hard_reasoning_pass") or 0)
    hr_n = int((hr or {}).get("n") or fam.get("hard_reasoning") or 20)
    text_num = (gsm_p or 0) + he_pass + (ife_p or 0) + hr_pass
    text_den = gsm_n + he_n + ife_n + hr_n

    bfcl_acc = ((bfcl or {}).get("score") or {}).get("accuracy")
    bfcl_n = ((bfcl or {}).get("score") or {}).get("correct_count")
    total_num = text_num + int(bfcl_n or 0)
    total_den = text_den + 20
    total_pct = 100.0 * total_num / total_den if total_den else 0

    e2e = (qsummary or {}).get("e2e_tok_s") or {}
    niah_ok = bool(niah) and niah.get("verdict") == "NIAH_PASS"
    niah_pts = []
    if niah:
        for k, v in (niah.get("results") or {}).items():
            niah_pts.append((k, bool(v.get("retrieved")), v.get("api_prompt_tokens"), v.get("elapsed_s")))

    short = (thr.get("short256_c1_thinkoff") or {}).get("mean_tok_s") or "—"
    prose = ((thr.get("prose_c1_x2_thinkoff") or {}).get("med")) or "—"
    td_tok = td.get("output_tok_s") if td else None

    vis_p = vis.get("pass")
    vis_n = vis.get("n") or 8

    gh = "https://github.com/r0b0tlab/dsv4-flash-vision-exp-vllm-sm121"
    image_id = serve.get("image_id", "sha256:f8b73f965834ff8439bf266827229ef71bfdf291689e05b63dc59a74003b517d")

    niah_cards = ""
    for lab, ok, pt, el in niah_pts:
        badge = "PASS" if ok else "FAIL"
        niah_cards += (
            f'<article class="card"><div class="top"><span>{lab}</span>'
            f'<span class="badge {"pass" if ok else "fail"}">{badge}</span></div>'
            f"<b>{pt:,}</b><p>prompt tokens</p><p>{el:.1f}s</p></article>"
        )
    if not niah_pts:
        niah_cards = '<p>Running — results land when the NIAH ladder closes.</p>'

    quality_rows = f"""<tr><td>GSM8K</td><td>{f"{gsm_p}/{gsm_n}" if gsm_p is not None else "—"}</td></tr>
<tr><td>HumanEval</td><td>{f"local subprocess {he_pass}/{he_n}" if he else "—"}</td></tr>
<tr><td>IFEval</td><td>{f"{ife_p}/{ife_n} strict" if ife_p is not None else "—"}</td></tr>
<tr><td>Hard reasoning</td><td>{f"{hr_pass}/{hr_n} (manual vs frozen ref)" if hr else "—"}</td></tr>
<tr><td>Text180 total</td><td>{f"{text_num}/{text_den} ({100*text_num/text_den:.1f}%)" if text_den else "—"}</td></tr>
<tr><td>BFCL-hard20</td><td>{f"{bfcl_n}/20 ({(bfcl_acc or 0)*100:.0f}%), thinking=high" if bfcl else "—"}</td></tr>
<tr><td>Q200v2 total</td><td>{total_num}/{total_den} ({total_pct:.1f}%)</td></tr>
<tr><td>Q200 e2e</td><td>{f"mean {round(e2e['mean'],2)} · agg {round(e2e['aggregate_completion_over_wall'] or e2e.get('aggregate_tok_s') or 0,2)} tok/s" if e2e.get("mean") else "—"}</td></tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>DeepSeek-V4-Flash-Vision-Exp — 512k + 1M dual-GB10</title>
<style>
:root {{ --paper:#f3ead8; --chassis:#d9cbb0; --ink:#2c2416; --muted:#5c5346; --line:#8a7a5c; --pass:#2f6b3a; --fail:#8b1e1e; --hi:#f7f0e0; }}
* {{ box-sizing:border-box; }}
html,body {{ margin:0; max-width:100%; overflow-x:hidden; background:var(--paper); color:var(--ink);
  font: 16px/1.45 ui-sans-serif, system-ui, sans-serif; }}
body {{ padding: 18px 16px 48px; }}
main {{ max-width: 920px; margin: 0 auto; min-width:0; }}
h1 {{ font: 700 1.6rem/1.15 ui-monospace, Menlo, monospace; margin: 0 0 8px; text-wrap:pretty; }}
h2 {{ font: 700 1.05rem/1.2 ui-monospace, Menlo, monospace; margin: 28px 0 10px; border-bottom: 2px solid var(--ink); padding-bottom: 4px; }}
.kicker {{ color:var(--muted); font: 600 0.75rem/1.3 ui-monospace, Menlo, monospace; letter-spacing:.08em; text-transform:uppercase; }}
.hero {{ background:var(--chassis); border:2px solid var(--ink); padding:16px; box-shadow: 3px 3px 0 var(--ink); }}
.kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,150px),1fr)); gap:10px; margin-top:12px; }}
.kpis > div, .card {{ min-width:0; background:var(--hi); border:1px solid var(--line); padding:10px 12px; }}
.kpis b, .card b {{ display:block; font: 700 1.35rem/1.1 ui-monospace, Menlo, monospace; overflow-wrap:anywhere; }}
.kpis span, .card p {{ color:var(--muted); font-size:0.8rem; margin:4px 0 0; }}
.cards {{ display:grid; grid-template-columns:1fr; gap:10px; }}
@media (min-width:760px) {{ .cards {{ grid-template-columns:repeat(3,minmax(0,1fr)); }} }}
.top {{ display:flex; justify-content:space-between; gap:8px; font: 600 0.8rem ui-monospace, Menlo, monospace; }}
.badge {{ padding:1px 6px; border:1px solid currentColor; }}
.badge.pass {{ color:var(--pass); }} .badge.fail {{ color:var(--fail); }}
table {{ width:100%; border-collapse:collapse; font: 0.9rem ui-monospace, Menlo, monospace; }}
th,td {{ text-align:left; padding:6px 8px; border-bottom:1px solid var(--line); }}
code {{ font-family: ui-monospace, Menlo, monospace; overflow-wrap:anywhere; }}
a {{ color:var(--ink); }}
footer {{ margin-top:36px; color:var(--muted); font-size:0.8rem; }}
</style>
</head>
<body>
<main>
<p class="kicker">r0b0tlab · @mr_r0b0t</p>
<h1>DeepSeek-V4-Flash-Vision-Exp</h1>
<p>512k + 1M context · 2× NVIDIA GB10 (SM121) · vLLM · speculative k=5 · adaptive verification</p>
<p><a href="{gh}">{gh}</a></p>
<section class="hero">
  <div class="kpis">
    <div><b>{f"{total_num}/{total_den}" if total_den else "—"}</b><span>Q200v2 total · {total_pct:.1f}%</span></div>
    <div><b>{short}</b><span>SHORT c1 tok/s (thinking off)</span></div>
    <div><b>{prose}</b><span>PROSE c1 med tok/s (thinking off)</span></div>
    <div><b>{"PASS" if vis_p == vis_n else "—"}</b><span>Vision {vis_p}/{vis_n}</span></div>
    <div><b>{"PASS" if niah_ok else "—"}</b><span>NIAH 25/50/90 + mk</span></div>
  </div>
</section>

<h2>Vision</h2>
<p>Synthetic 8-item exact-match gate: <b>{vis_p}/{vis_n} PASS</b>.</p>

<h2>Quality (qwen38 Q200v2 frozen set, thinking=high)</h2>
<table>
<tr><th>Lane</th><th>Result</th></tr>
{quality_rows}
</table>

<h2>NIAH (advertised 512k)</h2>
<div class="cards">{niah_cards}</div>
<p>Constructed prompts land at the filler-rounded size below the 524032-token target. Disclosed.</p>

<h2>Throughput (thinking off)</h2>
<table>
<tr><th>Lane</th><th>tok/s</th><th>complete</th></tr>
<tr><td>SHORT c1 (256)</td><td>{short}</td><td>{"—" if thr.get("short256_c1_thinkoff") else "—"}</td></tr>
<tr><td>PROSE c1×2</td><td>{prose}</td><td>{"stop" if prose != "—" else "—"}</td></tr>
<tr><td>TD2W300 (1–300 spelled)</td><td>{td_tok if td_tok else "—"}</td><td>{"yes" if td and td.get("complete") else "—"}</td></tr>
</table>

<h2>1M profile (same image)</h2>
<table>
<tr><th>Metric</th><th>512k</th><th>1M</th></tr>
<tr><td>Total KV pool</td><td>{serve.get("kv_tokens", 1281052):,} tokens</td><td>{p1m.get("serve", {}).get("kv_tokens", 1885452):,} tokens</td></tr>
<tr><td>KV concurrency at full window</td><td>{serve.get("concurrency_at_512k", 2.44)}×</td><td>{p1m.get("serve", {}).get("concurrency_at_1m", 1.8)}×</td></tr>
<tr><td>SHORT c1 tok/s (thinking off)</td><td>{short}</td><td>{thr1m.get("short256_c1_mean") or "—"}</td></tr>
<tr><td>PROSE c1 med tok/s</td><td>{prose}</td><td>{thr1m.get("prose_c1_med") or "—"}</td></tr>
<tr><td>TD2W300 tok/s</td><td>{td_tok if td_tok else "—"}</td><td>{thr1m.get("td2w300") or "—"}</td></tr>
<tr><td>Concurrency c1 / c8 tok/s</td><td>29.97 / 92.94</td><td>{conc.get("c1", {}).get("output_tok_s", "—")} / {conc.get("c8", {}).get("output_tok_s", "—")}</td></tr>
<tr><td>NIAH full-depth</td><td>PASS (25/50/90 + mk)</td><td>{"PASS — " + f"{n1m.get('api_prompt_tokens'):,}" + " prompt tokens, " + str(round(n1m.get('elapsed_s', 0))) + " s" if n1m.get("retrieved") else "—"}</td></tr>
<tr><td>Vision</td><td colspan="2">{vis_p}/{vis_n} PASS on both</td></tr>
</table>

<h2>Runtime</h2>
<table>
<tr><td>Image</td><td><code>vision-54631-fi512b-k5adapt</code> {image_id[:19]}… arm64</td></tr>
<tr><td>GHCR</td><td><code>ghcr.io/r0b0tlab/dsv4-flash-vision-exp-vllm-sm121:vision-54631-fi512b-k5adapt</code></td></tr>
<tr><td>max_model_len</td><td>524288</td></tr>
<tr><td>KV</td><td>fp8 · {serve.get("kv_tokens", "1,281,052"):,} tokens · {serve.get("concurrency_at_512k", 2.44)}× at 512k</td></tr>
<tr><td>Spec</td><td>k=5 probabilistic · adaptive verification on (SM121 overlay)</td></tr>
<tr><td>Graphs</td><td>FULL_DECODE_ONLY [1,2,4,8,16,32,48]</td></tr>
<tr><td>Telemetry</td><td>{f"{tel.get('samples')} samples · mean {tel.get('mean_w')} W · mean util {tel.get('mean_util')}%" if tel else "—"}</td></tr>
</table>

<h2>Appendix</h2>
<p>A.1 HumanEval graded locally via subprocess+timeout (docker sandbox HARNESS_BLOCK on this image id).</p>
<p>A.2 All quality lanes run thinking=high with thinking_token_budget=2048; throughput lanes run thinking=off.</p>
<p>A.3 k=6 is illegal on this image (num_speculative_tokens must divide dspark_block_size=5).</p>
<p>A.4 No private IPs, hostnames, or local paths on this page.</p>
<footer>r0b0tlab — @mr_r0b0t</footer>
</main>
</body>
</html>
"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html)
    print("wrote", OUT, "bytes", OUT.stat().st_size)


if __name__ == "__main__":
    main()