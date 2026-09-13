import json
import os, urllib.request, sys

TOKEN = os.environ.get("CIRCLECI_TOKEN", "")
BASE = "https://circleci.com/api/v2"


def get(url):
    req = urllib.request.Request(url, headers={"Circle-Token": TOKEN})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def pipelines(n=6):
    d = get(f"{BASE}/project/gh/lexing-2026/TensorPlay/pipeline?limit={n}")
    out = []
    for p in d["items"]:
        out.append((p["number"], (p.get("vcs") or {}).get("revision", "")[:8], p["id"]))
    return out


def workflows(pid):
    d = get(f"{BASE}/pipeline/{pid}/workflow")
    return [(w["name"], w["status"], w["id"]) for w in d.get("items", [])]


def jobs(wid):
    d = get(f"{BASE}/workflow/{wid}/job")
    return [(j["job_number"], j["name"], j["status"], j.get("started_at")) for j in d.get("items", [])]


if __name__ == "__main__":
    for num, rev, pid in pipelines(int(sys.argv[1]) if len(sys.argv) > 1 else 6):
        for name, st, wid in workflows(pid):
            print(f"p{num} {rev} wf:{name} {st}")
            if st in ("running",):
                for jn, jnname, jst, started in jobs(wid):
                    print(f"    job {jn} {jst} started {started}")
