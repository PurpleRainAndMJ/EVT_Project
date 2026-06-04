import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cyber_extremes import (
    simulate_portfolio, to_frechet, chi_u, chibar_u, chi_empirical,
    pickands_A, pickands_A_theoretical, spectral_measure,
    diversification_curve,
)

os.makedirs("figures", exist_ok=True)
plt.rcParams.update({"font.size": 9, "figure.dpi": 150,
                     "axes.grid": True, "grid.alpha": 0.3})

N_YEARS = 60_000
P_LIST = [0.0, 0.3, 0.6]
COLORS = {0.0: "#2a7ab9", 0.3: "#e07b39", 0.6: "#c23b3b", 0.05: "#5aa469", 0.2: "#8e5aa4", 0.5: "#c23b3b"}

print("Simulation des portefeuilles bivaries...")
DATA = {p: simulate_portfolio(n_years=N_YEARS, d=2, p=p, seed=42) for p in P_LIST}

print("Fig 1 : scatter Frechet")
fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.2), sharex=True, sharey=True)
for ax, p in zip(axes, P_LIST):
    X = DATA[p]
    fx, fy = to_frechet(X[:, 0]), to_frechet(X[:, 1])
    ax.loglog(fx, fy, ".", ms=1, alpha=0.25, color=COLORS[p])
    ax.set_title(rf"$p={p}$  ($\hat\chi(0.99)={chi_empirical(X[:,0],X[:,1]):.2f}$)")
    ax.set_xlabel("$X_1$ (Fréchet)")
axes[0].set_ylabel("$X_2$ (Fréchet)")
fig.suptitle("Excès joints en marginales Fréchet : la diagonale apparaît avec la contagion $p$", y=1.02)
fig.tight_layout()
fig.savefig("figures/fig1_scatter.png", bbox_inches="tight")
plt.close(fig)

print("Fig 2 : chi(u), chibar(u)")
u_grid = np.linspace(0.80, 0.995, 60)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.4))
for p in P_LIST:
    X = DATA[p]
    ax1.plot(u_grid, chi_u(X[:, 0], X[:, 1], u_grid), color=COLORS[p], label=rf"$p={p}$")
    ax2.plot(u_grid, chibar_u(X[:, 0], X[:, 1], u_grid), color=COLORS[p], label=rf"$p={p}$")
    ax1.axhline(p, color=COLORS[p], ls=":", lw=1)
ax1.set(xlabel="$u$", ylabel=r"$\chi(u)$", title=r"$\chi(u)$ — pointillés : valeur théorique $\chi=p$")
ax2.axhline(1.0, color="k", ls=":", lw=1)
ax2.set(xlabel="$u$", ylabel=r"$\bar\chi(u)$", title=r"$\bar\chi(u)$ — $\bar\chi=1$ : dépendance asymptotique")
ax1.legend(); ax2.legend()
fig.tight_layout()
fig.savefig("figures/fig2_chi.png", bbox_inches="tight")
plt.close(fig)

print("Fig 3 : fonction de Pickands (maxima par composantes, blocs de 20 ans)")
def block_maxima(X, m=20):
    n = (len(X) // m) * m
    return X[:n].reshape(-1, m, X.shape[1]).max(axis=1)

w = np.linspace(0.0, 1.0, 101)
fig, ax = plt.subplots(figsize=(5.2, 4.0))
for p in P_LIST:
    M = block_maxima(DATA[p], m=20)
    ax.plot(w, pickands_A(M[:, 0], M[:, 1], w, "cfg"), color=COLORS[p], label=rf"CFG, $p={p}$")
    ax.plot(w, pickands_A_theoretical(w, p), color=COLORS[p], ls="--", lw=1)
ax.plot(w, np.maximum(w, 1 - w), "k:", lw=1, label="comonotonie")
ax.plot([0, 1], [1, 1], "k-", lw=0.8, label="indépendance")
ax.set(xlabel="$w$", ylabel="$A(w)$", ylim=(0.48, 1.02),
       title="Fonction de Pickands sur maxima par composantes :\nCFG (plein) vs Marshall–Olkin $A(w)=1-p\\min(w,1-w)$ (tirets)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("figures/fig3_pickands.png", bbox_inches="tight")
plt.close(fig)

print("Fig 4 : mesure spectrale")
fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.0), sharey=True)
for ax, p in zip(axes, P_LIST):
    Wang = spectral_measure(DATA[p][:, 0], DATA[p][:, 1], k_frac=0.02)
    ax.hist(Wang, bins=30, range=(0, 1), density=True, color=COLORS[p], alpha=0.85)
    ax.set_title(rf"$p={p}$")
    ax.set_xlabel("$W = X_1/(X_1+X_2)$")
axes[0].set_ylabel("densité angulaire")
fig.suptitle("Mesure spectrale empirique : la masse migre des coins (diversification réelle)\nvers le centre (extrêmes simultanés)", y=1.06)
fig.tight_layout()
fig.savefig("figures/fig4_spectral.png", bbox_inches="tight")
plt.close(fig)

print("Fig 5 : courbe de diversification (long)")
d_grid = np.array([2, 5, 10, 20, 50, 100, 200])
fig, ax = plt.subplots(figsize=(5.6, 4.0))
for p in [0.0, 0.05, 0.2, 0.5]:
    dr = diversification_curve(d_grid, p=p, n_years=25_000)
    ax.plot(d_grid, dr, "o-", color=COLORS[p], label=rf"$p={p}$")
ax.set(xscale="log", xlabel="taille du portefeuille $d$",
       ylabel=r"$DR(d)=\mathrm{TVaR}_{99\%}(\Sigma X_i)\,/\,\Sigma\,\mathrm{TVaR}_{99\%}(X_i)$",
       title="Anti-diversification : $DR(d)$ plafonne dès que $p>0$")
ax.legend()
fig.tight_layout()
fig.savefig("figures/fig5_diversification.png", bbox_inches="tight")
plt.close(fig)

print("OK — figures dans ./figures/")
