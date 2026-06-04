"""
app.py — Le cyber comme cas limite de dependance multivariee.
Lancer :  streamlit run app.py
"""
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from cyber_extremes import (
    simulate_portfolio, to_frechet, chi_u, chibar_u, chi_empirical,
    pickands_A, pickands_A_theoretical, spectral_measure,
    diversification_curve,
)

st.set_page_config(page_title="Cyber & extrêmes multivariés", layout="wide")
st.title("Le cyber comme cas limite de dépendance multivariée")
st.caption(
    "Modèle à choc commun  X_i = Z_i + I·B_i·S  (d'après Zeller & Scherer, 2022). "
    "Résultat clé : χ = p — le taux de contagion *est* la dépendance de queue."
)

st.sidebar.header("Paramètres du modèle")
preset = st.sidebar.radio("Préréglage", ["Cyber", "Nat-cat multi-régions", "Libre"])
defaults = {"Cyber": 0.30, "Nat-cat multi-régions": 0.0, "Libre": 0.30}

p = st.sidebar.slider("p — contagion (part du portefeuille exposée à la "
                      "vulnérabilité commune)", 0.0, 0.9, defaults[preset], 0.05,
                      disabled=(preset != "Libre"))
if preset != "Libre":
    p = defaults[preset]

q = st.sidebar.slider("q — fréquence annuelle de l'événement systémique", 0.05, 0.9, 0.40, 0.05)
alpha_s = st.sidebar.slider("α_s — indice de queue systémique (petit = lourd)", 1.1, 3.0, 1.5, 0.1)
alpha_z = st.sidebar.slider("α_z — indice de queue idiosyncratique", 2.0, 6.0, 4.0, 0.5)
n_years = st.sidebar.select_slider("Années simulées", [10_000, 30_000, 60_000], 30_000)
level = st.sidebar.select_slider("Niveau TVaR", [0.95, 0.99, 0.995], 0.99)

if alpha_s >= alpha_z:
    st.sidebar.warning("α_s ≥ α_z : la queue systémique ne domine plus → "
                       "régime d'indépendance asymptotique (χ → 0, χ̄ < 1).")

@st.cache_data(show_spinner="Simulation du portefeuille bivarié...")
def get_data(n_years, p, q, alpha_s, alpha_z):
    return simulate_portfolio(n_years=n_years, d=2, p=p, q=q,
                              alpha_s=alpha_s, alpha_z=alpha_z, seed=42)

@st.cache_data(show_spinner="Courbe de diversification (peut prendre ~1 min)...")
def get_dr_curves(p, q, alpha_s, alpha_z, level):
    d_grid = np.array([2, 5, 10, 20, 50, 100])
    cur = diversification_curve(d_grid, p=p, q=q, alpha_s=alpha_s,
                                alpha_z=alpha_z, level=level, n_years=15_000)
    ref = diversification_curve(d_grid, p=0.0, q=q, alpha_s=alpha_s,
                                alpha_z=alpha_z, level=level, n_years=15_000)
    return d_grid, cur, ref

X = get_data(n_years, p, q, alpha_s, alpha_z)
x, y = X[:, 0], X[:, 1]
chi99 = chi_empirical(x, y, 0.99)

c1, c2, c3 = st.columns(3)
c1.metric("χ théorique (= p, si α_s < α_z)", f"{p:.2f}")
c2.metric("χ̂(0.99) empirique", f"{chi99:.2f}")
c3.metric("λ_U Marshall–Olkin = 2(1 − A(½))", f"{p:.2f}")

colA, colB = st.columns(2)

with colA:
    st.subheader("1 — Excès joints (marginales Fréchet)")
    fx, fy = to_frechet(x), to_frechet(y)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.loglog(fx, fy, ".", ms=1.5, alpha=0.3, color="#c23b3b" if p > 0 else "#2a7ab9")
    ax.set(xlabel="$X_1$", ylabel="$X_2$",
           title="La diagonale = chocs communs frappant les deux polices")
    ax.grid(alpha=0.3)
    st.pyplot(fig, clear_figure=True)

    st.subheader("3 — Mesure spectrale empirique")
    W = spectral_measure(x, y, k_frac=0.02)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.hist(W, bins=30, range=(0, 1), density=True, color="#8e5aa4", alpha=0.9)
    ax.set(xlabel="$W = X_1/(X_1+X_2)$", ylabel="densité angulaire",
           title="Coins = diversification réelle · Centre = extrêmes simultanés")
    ax.grid(alpha=0.3)
    st.pyplot(fig, clear_figure=True)

with colB:
    st.subheader("2 — Coefficients χ(u) et χ̄(u)")
    u_grid = np.linspace(0.80, 0.995, 50)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(u_grid, chi_u(x, y, u_grid), color="#e07b39", label=r"$\chi(u)$")
    ax.plot(u_grid, chibar_u(x, y, u_grid), color="#2a7ab9", label=r"$\bar\chi(u)$")
    ax.axhline(p, ls=":", color="#e07b39", label=rf"$\chi$ théorique $= p = {p}$")
    ax.axhline(1.0, ls=":", color="k", lw=0.8)
    ax.set(xlabel="$u$", ylim=(-0.1, 1.05),
           title="χ > 0 : dépendance asymptotique")
    ax.legend(); ax.grid(alpha=0.3)
    st.pyplot(fig, clear_figure=True)

    st.subheader("4 — Fonction de Pickands (maxima par blocs)")
    m = 20
    n = (len(X) // m) * m
    M = X[:n].reshape(-1, m, 2).max(axis=1)
    w = np.linspace(0, 1, 81)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(w, pickands_A(M[:, 0], M[:, 1], w, "cfg"), color="#c23b3b", label="CFG empirique")
    ax.plot(w, pickands_A_theoretical(w, p), "k--", lw=1,
            label=rf"$1 - p\min(w,1-w)$, $p={p}$")
    ax.plot(w, np.maximum(w, 1 - w), "k:", lw=0.8)
    ax.set(xlabel="$w$", ylabel="$A(w)$", ylim=(0.48, 1.02))
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    st.pyplot(fig, clear_figure=True)

st.subheader("5 — Anti-diversification : DR(d) = TVaR(Σ Xᵢ) / Σ TVaR(Xᵢ)")
if st.button("Calculer la courbe de diversification"):
    d_grid, cur, ref = get_dr_curves(p, q, alpha_s, alpha_z, level)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(d_grid, ref, "o-", color="#2a7ab9", label="référence p = 0 (nat-cat idéal)")
    ax.plot(d_grid, cur, "o-", color="#c23b3b", label=rf"p = {p}")
    ax.set(xscale="log", xlabel="taille du portefeuille d", ylabel="DR(d)",
           title="Dès que p > 0, DR plafonne : grossir le portefeuille n'use plus le risque")
    ax.legend(); ax.grid(alpha=0.3)
    st.pyplot(fig, clear_figure=True)
    st.info(
        f"À d = 100 : DR = {cur[-1]:.2f} contre {ref[-1]:.2f} pour le nat-cat idéal — "
        f"soit ≈ {cur[-1]/max(ref[-1],1e-9):.1f}× plus de capital relatif par police."
    )
