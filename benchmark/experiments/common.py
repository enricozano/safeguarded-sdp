"""
Shared plumbing of the three experiment scripts: parse the grids, run a list of
binary searches one after another, and append the results to two CSV files

    benchmark/results/<experiment>/summary.csv   one row per binary search
    benchmark/results/<experiment>/queries.csv   one row per feasibility query

A search already in summary.csv is skipped, so an interrupted run continues where it
stopped when the same command is run again.
"""
from __future__ import annotations

import csv
import math
import os
import time

import numpy as np

from solvers import BENCHMARK, totals

DATA = os.path.join(BENCHMARK, "data")
RESULTS = os.path.join(BENCHMARK, "results")

SUMMARY_FIELDS = ["experiment", "run_id", "job_key", "solver", "eps", "status", "n_queries",
                  "n_uncertified_queries", "lo", "hi", "estimate", "n_oracle", "n_iter",
                  "n_prop_attempts", "n_prop_accepted", "n_fallback", "wall_s", "error"]
QUERY_FIELDS = ["experiment", "run_id", "job_key", "solver", "eps", "q", "phase", "mu",
                "lo_before", "hi_before", "query_eps", "verdict", "feasible", "reason",
                "n_oracle", "n_iter", "n_prop_attempts", "n_prop_accepted", "n_fallback",
                "f_min", "grad_inf", "U_final", "ham_op_norm", "hu_overshoots", "wall_s"]


def int_list(spec):
    """'4-9' -> [4, ..., 9];  '0,2,5' -> [0, 2, 5]."""
    out = []
    for tok in str(spec).split(","):
        a, _, b = tok.strip().partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def eps_list(spec):
    """'1e-2:1e-4:6' -> 6 log-spaced values from 1e-2 to 1e-4;  '1e-2,1e-3' -> [1e-2, 1e-3]."""
    if ":" in spec:
        a, b, n = spec.split(":")
        return [float(x) for x in np.geomspace(float(a), float(b), int(n))]
    return [float(x) for x in spec.split(",")]


MIN_ETA = 1e-7   # smallest query tolerance: below it, double precision limits the line search


def search_key(solver, instance, eps):
    return f"{solver}|{instance}|eps={eps:.6e}"


def bisection_steps(width, eps):
    return max(0, math.ceil(math.log2(width / eps)))


def execute(experiment, spec, run_search, log=None):
    """Run one binary search; return (summary row, query rows)."""
    base = {"experiment": experiment, "run_id": spec["key"], "job_key": spec["key"],
            "solver": spec["solver"], "eps": spec["eps"], **spec["instance"]}
    t0 = time.perf_counter()
    result = run_search(spec, log)
    base.update(result.get("instance", {}))
    records = result.get("records", [])
    lo, hi = result.get("lo"), result.get("hi")
    summary = {**base, **totals(records), "status": "done", "lo": lo, "hi": hi,
               "estimate": 0.5 * (lo + hi) if lo is not None else "",
               "wall_s": round(time.perf_counter() - t0, 3), **result.get("extra", {})}
    queries = [{**base, "query_eps": spec["eps"], **r} for r in records]
    return summary, queries


def write_rows(path, fields, rows):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows({k: (float(v) if isinstance(v, np.floating) else v) for k, v in r.items()}
                    for r in rows)


def finished_keys(out, include_timeouts=False):
    path = os.path.join(out, "summary.csv")
    if not os.path.exists(path):
        return set()
    ok = {"done", "timeout"} if include_timeouts else {"done"}
    with open(path, newline="") as fh:
        return {r["job_key"] for r in csv.DictReader(fh) if r["status"] in ok}


def run_searches(experiment, searches, run_search, extra_fields, out=None, quiet=False):
    """Run every search that is not yet in <out>/summary.csv, one after another."""
    out = out or os.path.join(RESULTS, experiment)
    os.makedirs(out, exist_ok=True)
    done = finished_keys(out)
    todo = [s for s in searches if s["key"] not in done]
    print(f"== {experiment}: {len(searches)} searches, {len(searches) - len(todo)} already done, "
          f"results in {out}", flush=True)
    for i, spec in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {spec['key']}", flush=True)
        summary, queries = execute(experiment, spec, run_search, log=None if quiet else print)
        write_rows(os.path.join(out, "queries.csv"), QUERY_FIELDS + extra_fields, queries)
        write_rows(os.path.join(out, "summary.csv"), SUMMARY_FIELDS + extra_fields, [summary])
        print(f"    -> {summary['n_queries']} queries, {summary['n_oracle']} oracle calls, "
              f"{summary['wall_s']:.1f}s", flush=True)
