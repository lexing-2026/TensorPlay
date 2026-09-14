import json
import os
import subprocess
import sys
import time

RUN = sys.argv[1]
INTERVAL = int(sys.argv[2]) if len(sys.argv) > 2 else 30
BUDGET = int(sys.argv[3]) if len(sys.argv) > 3 else 1800
REPO = "lexing-2026/TensorPlay"


def gh(*args):
    return subprocess.run(["gh", *args], capture_output=True, text=True).stdout


def jobs():
    out = gh("api", f"repos/{REPO}/actions/runs/{RUN}/jobs", "--jq",
             '.jobs[] | [.id, .name, .status, (.conclusion // "-"), (.steps[-1].name // "-")] | @tsv')
    rows = []
    for line in out.splitlines():
        p = line.split("\t")
        if len(p) == 5:
            rows.append(p)
    return rows


def fail_log(job_id):
    out = subprocess.run(["gh", "api", f"repos/{REPO}/actions/jobs/{job_id}/logs"],
                         capture_output=True, text=True).stdout
    lines = [l for l in out.splitlines() if l.strip()]
    # keep error lines + last context
    err = [l for l in lines if "##[error]" in l or "error:" in l.lower() or "fatal" in l.lower()]
    return "\n".join((err[-8:] if err else lines[-12:]))


prev = {}
t0 = time.time()
while time.time() - t0 < BUDGET:
    rows = jobs()
    for jid, name, st, concl, step in rows:
        key = (name, st, concl, step)
        if prev.get(name) != key:
            print(f"[{time.strftime('%H:%M:%S')}] {name[:60]} | {st} | {concl} | step: {step}", flush=True)
    for jid, name, st, concl, step in rows:
        if st == "completed" and concl == "failure":
            print(f"\n=== FAILURE in {name} (job {jid}) ===", flush=True)
            print(fail_log(jid), flush=True)
            sys.exit(2)
    prev = {r[1]: (r[1], r[2], r[3], r[4]) for r in rows}
    if rows and all(r[2] == "completed" for r in rows):
        concl = {r[3] for r in rows}
        print(f"RUN COMPLETE: {concl}", flush=True)
        sys.exit(0 if concl <= {"success", "skipped"} else 2)
    time.sleep(INTERVAL)
print("watch budget exhausted; run still active", flush=True)
