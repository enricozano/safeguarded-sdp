#!/usr/bin/env python3
"""Progress of the experiments in benchmark/results/: finished / timed-out / failed searches.

    python benchmark/hpc/status.py
"""
import collections
import csv
import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

for d in sorted(glob.glob(os.path.join(ROOT, "results", "*", ""))):
    rows = {}
    if os.path.exists(os.path.join(d, "summary.csv")):
        with open(os.path.join(d, "summary.csv"), newline="") as fh:
            rows = {r["job_key"]: r for r in csv.DictReader(fh)}      # last row per search
    total = "?"
    if os.path.exists(os.path.join(d, "jobs.json")):
        with open(os.path.join(d, "jobs.json")) as fh:
            total = json.load(fh)["total"]
    count = collections.Counter(r["status"] for r in rows.values())
    print(f"{os.path.basename(d.rstrip('/')):18s} {count['done']:4d} done / {total} "
          f"(timeout {count['timeout']}, error {count['error']})")
