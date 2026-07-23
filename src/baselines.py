"""Baselines no neuronales: NOTEARS (lineal), PC y GES.

- NOTEARS: optimización continua lineal (Zheng et al. 2018), implementación
  numpy/scipy autocontenida.
- PC y GES: métodos clásicos basados en restricciones/score vía causal-learn.
"""
import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt

from utils import threshold_W


# --------------------------- NOTEARS lineal ---------------------------------
def notears_linear(X, lambda1=0.1, loss_type="l2", max_iter=100, h_tol=1e-8,
                   rho_max=1e16, w_threshold=0.3):
    """Aprende una matriz de pesos DAG minimizando MSE + penalización L1 bajo la
    restricción de aciclicidad diferenciable h(W)=tr(e^{W∘W})-d (Zheng et al. 2018)."""
    n, d = X.shape

    def _loss(W):
        M = X @ W
        R = X - M
        loss = 0.5 / n * (R ** 2).sum()
        G_loss = -1.0 / n * X.T @ R
        return loss, G_loss

    def _h(W):
        E = slin.expm(W * W)
        h = np.trace(E) - d
        G_h = E.T * W * 2
        return h, G_h

    def _adj(w):
        return (w[: d * d] - w[d * d:]).reshape([d, d])

    def _func(w):
        W = _adj(w)
        loss, G_loss = _loss(W)
        h, G_h = _h(W)
        obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w.sum()
        G_smooth = G_loss + (rho * h + alpha) * G_h
        g_obj = np.concatenate((G_smooth + lambda1, -G_smooth + lambda1), axis=None)
        return obj, g_obj

    w_est, rho, alpha, h = np.zeros(2 * d * d), 1.0, 0.0, np.inf
    bnds = [(0, 0) if i == j else (0, None)
            for _ in range(2) for i in range(d) for j in range(d)]
    for _ in range(max_iter):
        w_new, h_new = None, None
        while rho < rho_max:
            sol = sopt.minimize(_func, w_est, method="L-BFGS-B", jac=True, bounds=bnds)
            w_new = sol.x
            h_new, _ = _h(_adj(w_new))
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        w_est, h = w_new, h_new
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            break
    W_est = _adj(w_est)
    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est


# --------------------------- PC y GES (causal-learn) ------------------------
def _cpdag_to_B(G):
    """Convierte una matriz causal-learn (con codificación -1/1) a adyacencia binaria.

    En causal-learn: g[i,j]=-1 y g[j,i]=1  => i -> j ; g[i,j]=g[j,i]=-1 => i - j (no dirigida).
    """
    d = G.shape[0]
    B = np.zeros((d, d), dtype=int)
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            if G[i, j] == -1 and G[j, i] == 1:
                B[i, j] = 1
            elif G[i, j] == -1 and G[j, i] == -1:
                B[i, j] = 1
                B[j, i] = 1
    return B


def run_pc(X, alpha=0.05):
    from causallearn.search.ConstraintBased.PC import pc
    cg = pc(X, alpha, indep_test="fisherz", show_progress=False)
    return _cpdag_to_B(cg.G.graph)


def run_ges(X):
    from causallearn.search.ScoreBased.GES import ges
    rec = ges(X, score_func="local_score_BIC")
    return _cpdag_to_B(rec["G"].graph)


def run_notears(X, w_threshold=0.3):
    W = notears_linear(X, w_threshold=w_threshold)
    return threshold_W(W, thr=1e-9), W
