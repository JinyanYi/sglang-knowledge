#!/usr/bin/env python3
"""Check sgl-project/sglang Nightly Test NPU for qwen3_6_35b_a3b jobs only."""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from typing import Any

REPO = "sgl-project/sglang"
WORKFLOW_FILE = "nightly-test-npu.yml"
MODEL_KEY = "qwen3_6_35b_a3b"
UA = {"User-Agent": "nightly-npu-qwen36-35b-checker", "Accept": "application/vnd.github+json"}


def get_json(url: str) -> Any:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def get_text(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read().decode("utf-8", errors="replace")


def list_schedule_runs(per_page: int = 20) -> list[dict]:
    url = (
        f"https://api.github.com/repos/{REPO}/actions/workflows/{WORKFLOW_FILE}/runs"
        f"?event=schedule&per_page={per_page}"
    )
    return get_json(url).get("workflow_runs", [])


def pick_run(runs: list[dict], date: str | None, run_id: int | None) -> dict:
    if run_id is not None:
        for r in runs:
            if r["id"] == run_id:
                return r
        # fetch directly if not in recent page
        return get_json(f"https://api.github.com/repos/{REPO}/actions/runs/{run_id}")
    if date:
        matched = [r for r in runs if r.get("created_at", "").startswith(date)]
        if not matched:
            raise SystemExit(f"No schedule run found for date={date}")
        return matched[0]
    if not runs:
        raise SystemExit("No schedule runs returned")
    return runs[0]


def list_jobs(run_id: int) -> list[dict]:
    jobs: list[dict] = []
    page = 1
    while True:
        url = (
            f"https://api.github.com/repos/{REPO}/actions/runs/{run_id}/jobs"
            f"?per_page=100&page={page}"
        )
        batch = get_json(url).get("jobs", [])
        jobs.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return jobs


def short_case(name: str) -> str:
    if " / " in name:
        return name.split(" / ")[-1].strip()
    m = re.search(rf"{MODEL_KEY}[\w\-]*", name)
    return m.group(0) if m else name


def failure_snippet(job: dict) -> str:
    job_id = job["id"]
    run_id = job["run_id"]
    # Prefer public HTML AssertionError scrape; logs API often 403.
    html_url = f"https://github.com/{REPO}/actions/runs/{run_id}/job/{job_id}"
    try:
        html = get_text(html_url)
    except Exception as e:
        return f"(could not fetch job page: {e})"
    m = re.search(r"AssertionError:.{0,200}", html)
    if m:
        return re.sub(r"<[^>]+>", "", m.group(0))
    m = re.search(r"FAILED.{0,200}", html)
    if m:
        return re.sub(r"<[^>]+>", "", m.group(0))
    fails = [s["name"] for s in job.get("steps", []) if s.get("conclusion") == "failure"]
    return f"failed steps: {fails}" if fails else "(no snippet)"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", help="UTC date prefix, e.g. 2026-07-22")
    ap.add_argument("--run-id", type=int, help="Exact workflow run id")
    ap.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = ap.parse_args()

    runs = list_schedule_runs()
    run = pick_run(runs, args.date, args.run_id)
    jobs = [j for j in list_jobs(run["id"]) if MODEL_KEY in j.get("name", "")]

    rows = []
    for j in jobs:
        row = {
            "case": short_case(j["name"]),
            "conclusion": j.get("conclusion"),
            "url": j.get("html_url"),
        }
        if j.get("conclusion") == "failure":
            row["reason"] = failure_snippet(j)
        rows.append(row)

    payload = {
        "run_id": run["id"],
        "created_at": run.get("created_at"),
        "conclusion": run.get("conclusion"),
        "url": run.get("html_url"),
        "model_jobs": rows,
    }

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    print(f"Nightly: {payload['url']}")
    print(f"created {payload['created_at']}  conclusion {payload['conclusion']}")
    print()
    print("| case | result | job |")
    print("|------|--------|-----|")
    for r in rows:
        print(f"| `{r['case']}` | {r['conclusion']} | {r['url']} |")
    fails = [r for r in rows if r.get("conclusion") == "failure"]
    print()
    if not fails:
        print("Failures: none for qwen3_6_35b_a3b")
    else:
        print("Failures:")
        for r in fails:
            print(f"- `{r['case']}`: {r.get('reason', '')}")


if __name__ == "__main__":
    main()
