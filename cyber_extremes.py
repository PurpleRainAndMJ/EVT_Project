from __future__ import annotations

import numpy as np

EULER_GAMMA = 0.5772156649015329

def pareto(rng: np.random.Generator, alpha: float, size) -> np.ndarray:
    return rng.uniform(size=size) ** (-1.0 / alpha)


def simulate_portfolio(
    n_years: int = 50_000,
    d: int = 2,
    p: float = 0.3,
    q: float = 0.4,
    lambda_z: float = 1.0,
    alpha_z: float = 4.0,
    alpha_s: float = 1.5,
    scale_s: float = 3.0,
    seed: int | None = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    N = rng.poisson(lambda_z, size=(n_years, d))
    Z = np.zeros((n_years, d))
    total = int(N.sum())
    if total > 0:
        sev = pareto(rng, alpha_z, total)
        flat = np.repeat(np.arange(n_years * d), N.ravel())
        Z = np.bincount(flat, weights=sev, minlength=n_years * d).reshape(n_years, d)

    I = rng.uniform(size=n_years) < q                      
    B = rng.uniform(size=(n_years, d)) < p                 
    S = scale_s * pareto(rng, alpha_s, n_years)            
    X = Z + (I[:, None] & B) * S[:, None]
    return X

def to_uniform(x: np.ndarray) -> np.ndarray:
    n = len(x)
    ranks = np.argsort(np.argsort(x)) + 1.0
    return ranks / (n + 1.0)


def to_frechet(x: np.ndarray) -> np.ndarray:
    return -1.0 / np.log(to_uniform(x))

def chi_u(x: np.ndarray, y: np.ndarray, u_grid: np.ndarray):
    U, V = to_uniform(x), to_uniform(y)
    out = np.empty_like(u_grid)
    for k, u in enumerate(u_grid):
        c = np.mean((U <= u) & (V <= u))
        out[k] = 2.0 - np.log(max(c, 1e-12)) / np.log(u)
    return out


def chibar_u(x: np.ndarray, y: np.ndarray, u_grid: np.ndarray):
    U, V = to_uniform(x), to_uniform(y)
    out = np.empty_like(u_grid)
    for k, u in enumerate(u_grid):
        s = np.mean((U > u) & (V > u))
        out[k] = 2.0 * np.log(1.0 - u) / np.log(max(s, 1e-12)) - 1.0
    return out


def chi_empirical(x: np.ndarray, y: np.ndarray, u: float = 0.99) -> float:
    U, V = to_uniform(x), to_uniform(y)
    denom = np.mean(U > u)
    return float(np.mean((U > u) & (V > u)) / denom) if denom > 0 else 0.0

def pickands_A(x: np.ndarray, y: np.ndarray, w_grid: np.ndarray,
               method: str = "cfg") -> np.ndarray:
    xi = -np.log(to_uniform(x))
    eta = -np.log(to_uniform(y))
    n = len(xi)
    A = np.empty_like(w_grid)
    for k, w in enumerate(w_grid):
        if w <= 0 or w >= 1:
            A[k] = 1.0
            continue
        m = np.minimum(xi / (1.0 - w), eta / w)
        if method == "pickands":
            A[k] = n / m.sum()
        else:  # CFG
            A[k] = np.exp(-EULER_GAMMA - np.mean(np.log(m)))
    if method == "cfg":
        w = w_grid
        A0 = np.interp(0.0, w, A)
        A1 = np.interp(1.0, w, A)
        A = A * np.exp(-((1 - w) * np.log(A0) + w * np.log(A1)))
    return np.clip(A, np.maximum(w_grid, 1.0 - w_grid), 1.0)


def pickands_A_theoretical(w_grid: np.ndarray, chi: float) -> np.ndarray:
    return 1.0 - chi * np.minimum(w_grid, 1.0 - w_grid)

def spectral_measure(x: np.ndarray, y: np.ndarray, k_frac: float = 0.02):
    fx, fy = to_frechet(x), to_frechet(y)
    r = fx + fy
    k = max(int(k_frac * len(x)), 50)
    idx = np.argsort(r)[-k:]
    return fx[idx] / r[idx]

def tvar(losses: np.ndarray, level: float = 0.99) -> float:
    var = np.quantile(losses, level)
    tail = losses[losses >= var]
    return float(tail.mean()) if len(tail) else float(var)


def diversification_ratio(X: np.ndarray, level: float = 0.99) -> float:
    num = tvar(X.sum(axis=1), level)
    den = sum(tvar(X[:, j], level) for j in range(X.shape[1]))
    return num / den


def diversification_curve(d_grid, p, level=0.99, n_years=30_000, seed=0, **kw):
    out = []
    for j, d in enumerate(d_grid):
        X = simulate_portfolio(n_years=n_years, d=int(d), p=p,
                               seed=seed + 1000 * j, **kw)
        out.append(diversification_ratio(X, level))
    return np.array(out)
