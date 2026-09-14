import json
import os
import sys
import urllib.request

TOKEN = os.environ.get("CIRCLECI_TOKEN", "")


def get(url):
    req = urllib.request.Request(url, headers={"Circle-Token": TOKEN})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


build = int(sys.argv[1])
d = json.loads(get(f"https://circleci.com/api/v1.1/project/gh/lexing-2026/TensorPlay/{build}?circle-token={TOKEN}"))
print("build", build, "status:", d.get("status"))
for s in d.get("steps", []):
    for a in s.get("actions", []):
        st = a.get("status")
        if st == "failed" or "Build the cu124" in s["name"]:
            print(f"step: {s['name']} | {st} | exit {a.get('exit_code')}")
            url = a.get("output_url")
            if url:
                logs = json.loads(get(url))
                msgs = [m.get("message", "") for m in logs]
                print(f"--- log lines: {len(msgs)}; LAST 40 ---")
                for m in msgs[-40:]:
                    sys.stdout.write(m[:220] if m.endswith("\n") else m[:220] + "\n")
            else:
                print("(no output_url)")
