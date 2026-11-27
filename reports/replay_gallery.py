"""Render audited, fixed scenario ordinals and build a local video catalog.

python reports/replay_gallery.py --render --workers 4
The renderer is a separate process; training is never invoked.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import html
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def prepare():
    with (ROOT/"experiments/registry.csv").open(encoding="utf-8", newline="") as file:
        rows = {r["run_id"]: r for r in csv.DictReader(file)}
    jobs = []
    for marker in sorted((ROOT/"artifacts/evaluation").glob("*/*/audit.json")):
        if not json.loads(marker.read_text(encoding="utf-8")).get("passed"):
            continue
        folder = marker.parent
        run = folder.parent.name
        meta = json.loads((folder/"evaluation.json").read_text(encoding="utf-8"))
        row = rows[run]
        with np.load(folder/"endpoints.npz", allow_pickle=False) as d:
            for ordinal in (6, 7):
                index = ordinal-1
                if index >= len(d["scenario_id"]):
                    continue
                scenario = int(d["scenario_id"][index])
                output = ROOT/"artifacts/videos"/run/folder.name/f"scenario_{scenario}.mp4"
                jobs.append({"directory": str(folder.relative_to(ROOT)).replace(chr(92), "/"),
                    "output": str(output.relative_to(ROOT)).replace(chr(92), "/"),
                    "run_id": run, "case": folder.name, "scenario_id": scenario, "ordinal": ordinal,
                    "algorithm": row["algorithm"], "architecture": row["architecture"],
                    "repetition": row["repetition"], "training_horizon": row["training_horizon"],
                    "comparison_ids": row["comparison_ids"], "initialization_seed": row["initialization_seed"],
                    "outcome": ("Censored", "Ownership", "Collision", "Deadline", "Starvation")[int(d["outcome"][index])],
                    "ticks": int(d["time"][index]), "cash": float(d["cash"][index]),
                    "car_debt": float(d["car_debt"][index]), "micro_debt": float(d["micro_debt"][index]),
                    "return": float(d["external_return"][index]), "parameters": meta["config"],
                    "audit": str(marker.relative_to(ROOT)).replace(chr(92), "/")})
    return jobs


def render(job):
    output = ROOT/job["output"]
    if not output.exists() or not output.with_suffix(".events.json").exists():
        subprocess.run([sys.executable, str(ROOT/"src/render_replay.py"), str(ROOT/job["directory"]),
                        "--scenario-id", str(job["scenario_id"]), "--output", str(output)], check=True,
                       cwd=ROOT, stdout=subprocess.DEVNULL)
    # A separate readable complete ledger uses exact final per-ID balances from
    # endpoints, including zero for historical IDs whose paid slot was reused.
    ledger = json.loads(output.with_suffix(".loans.json").read_text(encoding="utf-8"))["loans"]
    with np.load(ROOT/job["directory"]/"endpoints.npz", allow_pickle=False) as d:
        i = int(np.flatnonzero(d["scenario_id"] == job["scenario_id"])[0])
        balances = dict(zip(map(int, d["loan_ids"][i]), map(float, d["loan_balance"][i])))
    table = []
    for key, loan in ledger.items():
        table.append(f"<tr><td>{key}</td><td>${loan['principal']:.2f}</td><td>${balances.get(int(key),0.):.2f}</td><td>{loan['opened']}</td><td>{len(loan['payments'])}</td></tr>")
    ledger_path = output.with_suffix(".ledger.html")
    ledger_path.write_text("<!doctype html><meta charset='utf-8'><title>Individual loan ledger</title><style>body{font:16px system-ui;max-width:900px;margin:40px auto}td,th{padding:8px;text-align:right;border-bottom:1px solid #ddd}</style>"
        f"<h1>Individual microloans, scenario {job['scenario_id']}</h1><p>{html.escape(job['run_id'])}, {html.escape(job['case'])}</p>"
        "<p>Principal and payment receipts come from the trajectory. Final per-ID balances come from exact endpoint arrays. A paid loan whose storage slot was reused has final balance zero.</p>"
        "<table><thead><tr><th>ID</th><th>Principal</th><th>Final balance</th><th>Opened tick</th><th>Payments</th></tr></thead><tbody>"+"".join(table)+"</tbody></table>", encoding="utf-8")
    job["video_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    return job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    arguments = parser.parse_args()
    jobs = prepare()
    if arguments.render:
        with ThreadPoolExecutor(max_workers=arguments.workers) as pool:
            jobs = list(pool.map(render, jobs))
    (ROOT/"reports/video_catalog.json").write_text(json.dumps(jobs, indent=2), encoding="utf-8")
    cards = []
    for j in jobs:
        relative = "../"+j["output"]
        tags = " ".join(str(j[k]) for k in ("run_id", "case", "algorithm", "architecture", "repetition", "training_horizon", "comparison_ids", "scenario_id", "outcome"))
        links = [(relative.removesuffix(".mp4")+".ledger.html", "Individual loan ledger"),
                 (relative.removesuffix(".mp4")+".events.json", "Event receipts"),
                 ("../"+j["directory"]+"/evaluation.json", "Parameters and aggregate outcomes"),
                 ("../"+j["audit"], "Replay audit")]
        cards.append(f"<article data-search='{html.escape(tags.lower(),quote=True)}'><h2>{html.escape(j['algorithm'])} / {j['architecture']}, {j['training_horizon']}, {j['repetition']}</h2>"
            f"<p>{html.escape(j['comparison_ids'])}, {html.escape(j['case'])}, scenario {j['scenario_id']} (fixed ordinal {j['ordinal']})</p>"
            f"<video controls preload='none' src='{html.escape(relative,quote=True)}'></video>"
            f"<p><strong>{j['outcome']}</strong>, {j['ticks']} ticks, cash ${j['cash']:.2f}, car ${j['car_debt']:.2f}, microloans ${j['micro_debt']:.2f}, return {j['return']:.4f}</p>"
            +"<p>"+", ".join(f"<a href='{html.escape(u,quote=True)}'>{t}</a>" for u,t in links)+"</p></article>")
    document = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Gambler's Taxi, Recorded policies</title>
<style>body{font:15px system-ui;color:#183041;background:#f1f4f6;max-width:1440px;margin:auto;padding:32px}h1{font-size:32px}h2{font-size:19px}header{max-width:1000px}input{font:inherit;padding:12px;width:min(600px,90%);border:1px solid #aab8c4;border-radius:6px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:20px;margin-top:24px}article{background:white;border:1px solid #dce3e8;padding:20px;border-radius:9px}video{width:100%;border-radius:5px;background:#15202a}a{color:#176a8c}p{line-height:1.6}article[hidden]{display:none}</style>
<header><h1>Gambler's Taxi, Recorded policies</h1><p>Offline playback of audited numeric trajectories. Every evaluation uses prespecified scenario ordinals 6 and 7; the displayed ID identifies its bank. These examples are fixed in advance, including unsuccessful episodes. Full 256-scenario outcomes, not these clips, support comparisons. All models are learned policies; reference demonstrations are labeled separately in the reports.</p><p>Search by algorithm, memory, horizon, repetition, case, scenario, outcome or comparison ID.</p><input id="search" type="search" placeholder="e.g. ppo gru 10d repA standard"><p id="count"></p></header><main>"""+"\n".join(cards)+"""</main><script>const s=document.querySelector('#search'),cards=[...document.querySelectorAll('article')];function filter(){let q=s.value.toLowerCase().split(/\\s+/).filter(Boolean),n=0;cards.forEach(c=>{c.hidden=!q.every(x=>c.dataset.search.includes(x));if(!c.hidden)n++});document.querySelector('#count').textContent=n+' / '+cards.length+' recorded clips'}s.addEventListener('input',filter);filter();</script></html>"""
    (ROOT/"reports/replay_gallery.html").write_text(document, encoding="utf-8")
    print(f"Cataloged {len(jobs)} fixed-scenario clips; rendered={arguments.render}.")


if __name__ == "__main__":
    main()
