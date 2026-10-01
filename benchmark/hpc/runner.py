#!/usr/bin/env python3
"""
Optional: run an experiment on many cores at once, to save time. Every search runs
exactly the code of the experiment script (its `run_search`) in a pool of worker
processes, longest expected search first; results go to the same CSV files and are
the same as running the script directly.

    python benchmark/hpc/runner.py precision_scaling [script options] [--workers N]
    python benchmark/hpc/runner.py logistics --problem vrp --workers 6
    python benchmark/hpc/runner.py portfolio

--budget-hours caps the wall time of one search (it is then recorded as status=timeout
and rerun with --retry-timeouts). Already finished searches are skipped.
"""
from __future__ import annotations

import argparse
import importlib
import json
import multiprocessing as mp
import os
import platform
import signal
import socket
import sys
import time

BENCHMARK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BENCHMARK, "experiments"))

import common  # noqa: E402

MODULES = {"precision_scaling": "precision_scaling", "logistics": "logistics",
           "portfolio": "portfolio"}


class Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise Timeout()


def _work(job):
    """Worker: one search -> (summary row, query rows)."""
    module_name, experiment, spec, budget_s, quiet = job
    module = importlib.import_module(module_name)
    key = spec["key"]

    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {key} {msg.strip()}", flush=True)

    if budget_s > 0:
        signal.signal(signal.SIGALRM, _alarm)
        signal.setitimer(signal.ITIMER_REAL, budget_s)
    try:
        return common.execute(experiment, spec, module.run_search, log=None if quiet else log)
    except Timeout:
        status, error = "timeout", ""
    except Exception as exc:                     # keep going; the search is retried next time
        status, error = "error", repr(exc)[:500]
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    row = {"experiment": experiment, "run_id": key, "job_key": key, "solver": spec["solver"],
           "eps": spec["eps"], "status": status, "error": error, **spec["instance"]}
    return row, []


def run_info(workers, todo):
    info = {"host": socket.gethostname(), "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "python": sys.version.split()[0], "platform": platform.platform(),
            "cpus": available_cpus(), "workers": workers, "to_run": todo,
            "blas_threads": os.environ.get("OMP_NUM_THREADS", "unset")}
    for m in ("numpy", "scipy", "pandas", "cvxpy", "clarabel"):
        try:
            info[m] = importlib.import_module(m).__version__
        except ImportError:
            pass
    return info


def available_cpus():
    """CPUs this process may use: the affinity set, capped by a cgroup CPU quota (inside
    containers os.cpu_count() can report the whole host)."""
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    try:
        with open("/sys/fs/cgroup/cpu.max") as fh:
            quota, period = fh.read().split()
        if quota != "max":
            n = min(n, max(1, int(quota) // int(period)))
    except (OSError, ValueError):
        pass
    return n


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("experiment", choices=sorted(MODULES))
    p.add_argument("--workers", type=int, default=available_cpus())
    p.add_argument("--budget-hours", type=float, default=0.0, help="per search; 0 = no limit")
    p.add_argument("--retry-timeouts", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="list the pending searches")
    args, rest = p.parse_known_args()

    module = importlib.import_module(MODULES[args.experiment])
    opts = module.parse(rest) if hasattr(module, "parse") else module.parser().parse_args(rest)
    experiment = f"logistics_{opts.problem}" if args.experiment == "logistics" else args.experiment
    out = opts.out or os.path.join(common.RESULTS, experiment)
    os.makedirs(out, exist_ok=True)

    specs = module.searches(opts)
    finished = common.finished_keys(out, include_timeouts=not args.retry_timeouts)
    todo = sorted((s for s in specs if s["key"] not in finished), key=lambda s: -s["cost"])
    print(f"== {experiment}: {len(specs)} searches, {len(specs) - len(todo)} finished, "
          f"{len(todo)} to run on {args.workers} workers -> {out}", flush=True)
    if args.dry_run:
        print("\n".join("   " + s["key"] for s in todo))
        return
    with open(os.path.join(out, "jobs.json"), "w") as fh:
        json.dump({"total": len(specs)}, fh)
    with open(os.path.join(out, "run_info.jsonl"), "a") as fh:
        fh.write(json.dumps(run_info(args.workers, len(todo))) + "\n")
    if hasattr(module, "prepare"):
        module.prepare(todo)                     # e.g. parse the logistics CSVs once

    fields = module.EXTRA_FIELDS
    jobs = [(MODULES[args.experiment], experiment, s, args.budget_hours * 3600, opts.quiet)
            for s in todo]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for i, (summary, queries) in enumerate(pool.imap_unordered(_work, jobs), 1):
            common.write_rows(os.path.join(out, "queries.csv"), common.QUERY_FIELDS + fields, queries)
            common.write_rows(os.path.join(out, "summary.csv"), common.SUMMARY_FIELDS + fields, [summary])
            print(f"[{time.strftime('%H:%M:%S')}] == {i}/{len(todo)} {summary['job_key']} -> "
                  f"{summary['status']}, {summary.get('n_oracle', '')} oracle calls, "
                  f"{summary.get('wall_s', '')}s", flush=True)
    print(f"== {experiment} finished", flush=True)


if __name__ == "__main__":
    main()
