#!/usr/bin/env python3
"""
Figures and tables of the paper from the experiment results.

    python benchmark/make_figures.py                          # benchmark/results/
    python benchmark/make_figures.py --results other/dir      # another results directory

Writes benchmark/figures/*.{pdf,png} and benchmark/figures/tables/*.csv:

    precision_N128_hu_vs_lbfgs  oracle calls vs. eps at N = 128: HU and safeguarded L-BFGS
    dimension_qubo              safeguarded L-BFGS: oracle calls vs. N at fixed eps (synthetic QUBO)
    precision_qubo_all_N        safeguarded L-BFGS: oracle calls vs. eps for every N
    precision_portfolio         portfolio: oracle calls vs. eps for every number of assets
    dimension_portfolio         portfolio: oracle calls vs. SDP dimension at fixed eps
    tables/precision_exponents.csv, logistics_table.csv, portfolio_table.csv
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import NullFormatter

BENCHMARK = os.path.dirname(os.path.abspath(__file__))
_p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_p.add_argument("--results", default=os.path.join(BENCHMARK, "results"))
_p.add_argument("--out", default=os.path.join(BENCHMARK, "figures"))
_args = _p.parse_args()
RES = os.path.abspath(_args.results)
FIG = _args.out
TAB = os.path.join(FIG, "tables")
os.makedirs(TAB, exist_ok=True)

BLUE, ORANGE, INK2, GRID = "#2a78d6", "#eb6834", "#52514e", "#e4e3df"
RAMP = LinearSegmentedColormap.from_list("blues", ["#9ec2ee", BLUE, "#0d3a73"])
plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm", "font.size": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.edgecolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 6.5, "legend.handlelength": 1.4,
    "legend.labelspacing": 0.3, "legend.borderaxespad": 0.3, "lines.linewidth": 1.4,
    "savefig.bbox": "tight", "savefig.dpi": 200,
})
EPS, CALLS = r"precision $\varepsilon$", "oracle calls per binary search"
SAFEGUARDED = "safeguarded L-BFGS"


def load(e):
    s = pd.read_csv(os.path.join(RES, e, "summary.csv"))
    return s[s.status == "done"].drop_duplicates("job_key", keep="last")


def stats(d, x):
    g = d.groupby(x).n_oracle
    return pd.DataFrame({"m": g.median(), "q1": g.quantile(.25), "q3": g.quantile(.75)}).sort_index()


def power_fit(x, y):
    a, b = np.polyfit(np.log(np.asarray(x, float)), np.log(np.asarray(y, float)), 1)
    return a, b


def fitted(ax, x, y, color, label, marker="o", band=None, fit_line=True):
    """Data, interquartile band and fitted power law (drawn if fit_line, else the data are
    connected); `label(a)` gets the fitted exponent."""
    a, b = power_fit(x, y)
    if band is not None:
        ax.fill_between(x, band[0], band[1], color=color, alpha=0.18, lw=0)
    if fit_line:
        ax.plot(x, y, marker, color=color, mec="white", mew=0.6, ms=4.2)
        xs = np.geomspace(min(x), max(x), 60)
        ax.plot(xs, np.exp(b) * xs ** a, color=color, lw=1.2, label=label(a))
    else:
        ax.plot(x, y, marker + "-", color=color, mec="white", mew=0.6, ms=4.2, label=label(a))
    return a


VAREPS = r"\varepsilon"


def law(var, a):
    return rf"$\propto {var}^{{{a:.2f}}}$"


def style(ax, xlabel, title, eps_axis=True, yticks=None, xticks=None):
    ax.grid(axis="y", which="major", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_yscale("log")
    if xticks is None:
        ax.set_xscale("log")
    else:
        ax.set_xscale("log", base=2)
        ax.set_xticks(xticks, [str(t) for t in xticks])
        ax.xaxis.set_minor_formatter(NullFormatter())
    if eps_axis:
        ax.invert_xaxis()
    if yticks:
        ax.set_yticks(yticks, [str(t) for t in yticks])
        ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(xlabel)
    ax.set_ylabel(CALLS)
    ax.set_title(title, fontsize=8, color=INK2, loc="left")


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"))
    plt.close(fig)
    print("wrote", name)


# ----------------------------------------------------------------- synthetic QUBO-SDPs
ps = load("precision_scaling")
lbfgs, hu = ps[ps.solver == "lbfgs_scipy"], ps[ps.solver == "hu"]

fig, ax = plt.subplots(figsize=(3.4, 2.7))
for d, col, mk, name in ((hu, ORANGE, "s", "Hamiltonian Updates"), (lbfgs, BLUE, "o", SAFEGUARDED)):
    st = stats(d[d.k == 7], "eps")
    fitted(ax, st.index, st.m, col, lambda a: f"{name} ({law(VAREPS, a)})", mk, (st.q1, st.q3))
style(ax, EPS, r"$N=128$, median and interquartile range of 10 instances")
ax.legend(loc="upper left")
save(fig, "precision_N128_hu_vs_lbfgs")

fig, ax = plt.subplots(figsize=(3.4, 2.7))
Ns = sorted(lbfgs.N.unique())
shared = sorted(set.intersection(*[set(lbfgs[lbfgs.N == N].eps.round(12)) for N in Ns]), reverse=True)
for i, e in enumerate(shared):
    st = stats(lbfgs[np.isclose(lbfgs.eps, e)], "N")
    fitted(ax, st.index, st.m, RAMP(i / max(1, len(shared) - 1)),
           lambda a: rf"$\varepsilon$ = {e:.1e} ({law('N', a)})", "o", (st.q1, st.q3), fit_line=False)
style(ax, "dimension $N$", f"{SAFEGUARDED}, synthetic QUBO-SDPs", eps_axis=False,
      yticks=[100, 200, 300, 500], xticks=Ns)
ax.legend(loc="upper left")
save(fig, "dimension_qubo")

fig, ax = plt.subplots(figsize=(3.6, 2.8))
expo = []
for i, N in enumerate(Ns):
    st = stats(lbfgs[lbfgs.N == N], "eps")
    a = fitted(ax, st.index, st.m, RAMP(i / (len(Ns) - 1)), lambda a: f"$N={N}$ ({law(VAREPS, a)})")
    expo.append(dict(solver="lbfgs_scipy", N=N, eps_min=st.index.min(), eps_max=st.index.max(), exponent=a))
for N in sorted(hu.N.unique()):
    st = stats(hu[hu.N == N], "eps")
    expo.append(dict(solver="hu", N=N, eps_min=st.index.min(), eps_max=st.index.max(),
                     exponent=power_fit(st.index, st.m)[0]))
pd.DataFrame(expo).to_csv(os.path.join(TAB, "precision_exponents.csv"), index=False)
style(ax, EPS, f"{SAFEGUARDED}, median of 10 instances", yticks=[100, 200, 500])
ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
save(fig, "precision_qubo_all_N")

# ----------------------------------------------------------------- portfolio
pf = load("portfolio")
po = pf[pf.solver == "lbfgs_scipy"]
ipm = pf[pf.solver == "clarabel"].set_index("n")

fig, ax = plt.subplots(figsize=(3.4, 2.7))
ns = sorted(po.n.unique())
for i, n in enumerate(ns):
    d = po[po.n == n].sort_values("eps")
    fitted(ax, d.eps.values, d.n_oracle.values, RAMP(i / (len(ns) - 1)),
           lambda a: f"$n={n}$, $D={int(d.D.iloc[0])}$ ({law(VAREPS, a)})")
a_pool = power_fit(po.eps, po.n_oracle)[0]
style(ax, EPS, rf"portfolio SDP, {SAFEGUARDED}; pooled fit $\propto\varepsilon^{{{a_pool:.2f}}}$")
ax.legend(loc="upper left")
save(fig, "precision_portfolio")

fig, ax = plt.subplots(figsize=(3.4, 2.7))
pe = sorted(set.intersection(*[set(po[po.n == n].eps.round(12)) for n in ns]), reverse=True)
for i, e in enumerate(pe):
    d = po[np.isclose(po.eps, e)].sort_values("D")
    fitted(ax, d.D.values, d.n_oracle.values, RAMP(i / max(1, len(pe) - 1)),
           lambda a: rf"$\varepsilon$ = {e:.0e} ({law('D', a)})", "o", fit_line=False)
Ds = sorted(po.D.unique())
style(ax, "SDP dimension $D$", f"portfolio SDP, {SAFEGUARDED}", eps_axis=False,
      yticks=[200, 500, 1000, 2000, 5000, 10000])
ax.set_xticks(Ds, [str(int(x)) for x in Ds])
ax.xaxis.set_minor_formatter(NullFormatter())
ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.47))
save(fig, "dimension_portfolio")

# ----------------------------------------------------------------- tables
rows = []
for e, key in (("logistics_asp", "size"), ("logistics_vrp", "instance")):
    for (inst, solver, eps), g in load(e).groupby([key, "solver", "eps"]):
        rows.append(dict(instance=str(inst).split("_p")[0], solver=solver, eps=eps, n=len(g),
                         rel_error=g.rel_error.median(), oracle=g.n_oracle.median(), wall_s=g.wall_s.median()))
pd.DataFrame(rows).sort_values(["instance", "solver", "eps"], ascending=[True, True, False]).to_csv(
    os.path.join(TAB, "logistics_table.csv"), index=False)
t4 = po.assign(mu_ipm=po.n.map(ipm.mu_ipm), ipm_wall_s=po.n.map(ipm.ipm_wall_s))
t4["abs_err"] = (t4.mu_star - t4.mu_ipm).abs()
t4[["n", "D", "m_constraints", "R_tr", "gamma_port", "eps", "mu_star", "cert_lo", "cert_hi", "mu_ipm",
    "abs_err", "n_oracle", "n_queries", "wall_s", "ipm_wall_s"]].sort_values(["n", "eps"], ascending=[True, False]).to_csv(
    os.path.join(TAB, "portfolio_table.csv"), index=False)
print("wrote tables")
