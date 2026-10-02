#!/usr/bin/env python3
"""
The paper's figures from the experiment results.

    python benchmark/figures/make_figures.py                       # benchmark/results/
    python benchmark/figures/make_figures.py --results other/dir   # another results directory

Writes benchmark/figures/*.png:

    precision_qubo      (a) oracle calls vs. eps at N = 128, HU and safeguarded L-BFGS;
                        (b) safeguarded L-BFGS for every N
    dimension_qubo      oracle calls vs. N at fixed eps
    portfolio           (a) oracle calls vs. eps for every number of assets;
                        (b) oracle calls vs. SDP dimension at fixed eps
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
from matplotlib.ticker import NullFormatter

FIGURES = os.path.dirname(os.path.abspath(__file__))
BENCHMARK = os.path.dirname(FIGURES)
_p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_p.add_argument("--results", default=os.path.join(BENCHMARK, "results"))
_p.add_argument("--out", default=FIGURES)
_args = _p.parse_args()
RES = os.path.abspath(_args.results)
FIG = _args.out
os.makedirs(FIG, exist_ok=True)

BLUE, ORANGE, INK2 = "#2a78d6", "#eb6834", "#52514e"
RAMP = LinearSegmentedColormap.from_list("blues", ["#9ec2ee", BLUE, "#0d3a73"])
COL1, COL2, HEIGHT = 3.3, 6.9, 2.25            # single column, full width, panel height (in)
plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm", "font.size": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "legend.fontsize": 6.5, "legend.handlelength": 1.3, "legend.labelspacing": 0.25,
    "legend.borderaxespad": 0.2, "lines.linewidth": 1.2, "lines.markersize": 3.5,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "savefig.dpi": 300,
})
EPS, CALLS = r"precision $\varepsilon$", "oracle calls"


def load(e):
    s = pd.read_csv(os.path.join(RES, e, "summary.csv"))
    return s[s.status == "done"].drop_duplicates("job_key", keep="last")


def stats(d, x):
    g = d.groupby(x).n_oracle
    return pd.DataFrame({"m": g.median(), "q1": g.quantile(.25), "q3": g.quantile(.75)}).sort_index()


def exponent(x, y):
    return np.polyfit(np.log(np.asarray(x, float)), np.log(np.asarray(y, float)), 1)


def sci(e):
    """1e-2 -> $10^{-2}$, 3.59e-3 -> $3.6\\times10^{-3}$."""
    m, k = f"{e:.1e}".split("e")
    k = int(k)
    return rf"$10^{{{k}}}$" if m == "1.0" else rf"${m}\times10^{{{k}}}$"


def log_axes(ax, invert_x=True):
    ax.set_xscale("log")
    ax.set_yscale("log")
    if invert_x:
        ax.invert_xaxis()


def panel(ax, letter):
    ax.text(-0.17, 1.02, f"({letter})", transform=ax.transAxes, fontsize=8, fontweight="bold",
            va="bottom")


def save(fig, name):
    fig.savefig(os.path.join(FIG, f"{name}.png"))
    plt.close(fig)
    print("wrote", name)


ps = load("precision_scaling")
lbfgs, hu = ps[ps.solver == "lbfgs_scipy"], ps[ps.solver == "hu"]
Ns = sorted(lbfgs.N.unique())
norm_N = BoundaryNorm(np.arange(len(Ns) + 1) - 0.5, RAMP.N)

# ------------------------------------------------- precision scaling, synthetic QUBO-SDPs
fig, (ax, bx) = plt.subplots(1, 2, figsize=(COL2, HEIGHT), gridspec_kw={"wspace": 0.32})
for d, col, mk, name in ((hu, ORANGE, "s", "Hamiltonian Updates"), (lbfgs, BLUE, "o", "safeguarded L-BFGS")):
    st = stats(d[d.N == 128], "eps")
    a, b = exponent(st.index, st.m)
    ax.fill_between(st.index, st.q1, st.q3, color=col, alpha=0.2, lw=0)
    ax.plot(st.index, st.m, mk, color=col, mec="white", mew=0.5)
    xs = np.geomspace(st.index.min(), st.index.max(), 50)
    ax.plot(xs, np.exp(b) * xs ** a, "-", color=col, lw=1.0,
            label=rf"{name}, $\propto\varepsilon^{{{a:.2f}}}$")
log_axes(ax)
ax.set(xlabel=EPS, ylabel=CALLS)
ax.legend(loc="upper left")
ax.text(0.97, 0.04, r"$N=128$", transform=ax.transAxes, ha="right", fontsize=7, color=INK2)
panel(ax, "a")

for i, N in enumerate(Ns):
    st = stats(lbfgs[lbfgs.N == N], "eps")
    col = RAMP(norm_N(i))
    bx.fill_between(st.index, st.q1, st.q3, color=col, alpha=0.15, lw=0)
    bx.plot(st.index, st.m, "o-", color=col, mec="white", mew=0.4, ms=3)
log_axes(bx)
bx.set(xlabel=EPS, ylabel=CALLS)
sm = plt.cm.ScalarMappable(cmap=RAMP, norm=norm_N)
cb = fig.colorbar(sm, ax=bx, ticks=range(len(Ns)), pad=0.02, fraction=0.05, aspect=25)
cb.ax.set_yticklabels([str(N) for N in Ns])
cb.ax.tick_params(labelsize=6.5, width=0.5, length=2)
cb.set_label("$N$", fontsize=8, rotation=0, labelpad=6)
cb.outline.set_linewidth(0.5)
panel(bx, "b")
save(fig, "precision_qubo")

# ------------------------------------------------- dimension scaling, synthetic QUBO-SDPs
fig, ax = plt.subplots(figsize=(COL1, HEIGHT))
shared = sorted(set.intersection(*[set(lbfgs[lbfgs.N == N].eps.round(12)) for N in Ns]), reverse=True)
for i, e in enumerate(shared):
    st = stats(lbfgs[np.isclose(lbfgs.eps, e)], "N")
    a = exponent(st.index, st.m)[0]
    col = RAMP(i / max(1, len(shared) - 1))
    ax.fill_between(st.index, st.q1, st.q3, color=col, alpha=0.15, lw=0)
    ax.plot(st.index, st.m, "o-", color=col, mec="white", mew=0.4, ms=3,
            label=rf"$\varepsilon=${sci(e)}, $\propto N^{{{a:.2f}}}$")
st = stats(hu[np.isclose(hu.eps, 1e-2)], "N")
a = exponent(st.index, st.m)[0]
ax.fill_between(st.index, st.q1, st.q3, color=ORANGE, alpha=0.15, lw=0)
ax.plot(st.index, st.m, "s--", color=ORANGE, mec="white", mew=0.4, ms=3,
        label=rf"HU, $\varepsilon=10^{{-2}}$, $\propto N^{{{a:.2f}}}$")
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xticks(Ns, [str(N) for N in Ns])
ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_yticks([100, 200, 500], ["100", "200", "500"])
ax.yaxis.set_minor_formatter(NullFormatter())
ax.set(xlabel="dimension $N$", ylabel=CALLS)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, columnspacing=1.0, fontsize=6)
save(fig, "dimension_qubo")

# ------------------------------------------------- portfolio
pf = load("portfolio")
po = pf[pf.solver == "lbfgs_scipy"]
ns = sorted(po.n.unique())
fig, (ax, bx) = plt.subplots(1, 2, figsize=(COL2, HEIGHT), gridspec_kw={"wspace": 0.32})
for i, n in enumerate(ns):
    d = po[po.n == n].sort_values("eps")
    col = RAMP(i / (len(ns) - 1))
    ax.plot(d.eps, d.n_oracle, "o-", color=col, mec="white", mew=0.4, ms=3,
            label=rf"$n={n}$ ($D={int(d.D.iloc[0])}$)")
a_pool, b_pool = exponent(po.eps, po.n_oracle)
xs = np.geomspace(po.eps.min(), po.eps.max(), 50)
ax.plot(xs, np.exp(b_pool) * xs ** a_pool, ":", color=INK2, lw=1.0,
        label=rf"pooled fit, $\propto\varepsilon^{{{a_pool:.2f}}}$")
log_axes(ax)
ax.set(xlabel=EPS, ylabel=CALLS)
ax.legend(loc="upper left")
panel(ax, "a")

pe = sorted(set.intersection(*[set(po[po.n == n].eps.round(12)) for n in ns]), reverse=True)
for i, e in enumerate(pe):
    d = po[np.isclose(po.eps, e)].sort_values("D")
    bx.plot(d.D, d.n_oracle, "o-", color=RAMP(i / max(1, len(pe) - 1)), mec="white", mew=0.4, ms=3,
            label=rf"$\varepsilon=${sci(e)}")
Ds = sorted(po.D.unique())
bx.set_xscale("log")
bx.set_yscale("log")
bx.set_xticks(Ds, [str(int(x)) for x in Ds])
bx.xaxis.set_minor_formatter(NullFormatter())
bx.set_yticks([200, 500, 1000, 2000, 5000], ["200", "500", "1000", "2000", "5000"])
bx.yaxis.set_minor_formatter(NullFormatter())
bx.set(xlabel="SDP dimension $D$", ylabel=CALLS)
bx.legend(loc="center right", bbox_to_anchor=(1.0, 0.47))
panel(bx, "b")
save(fig, "portfolio")
