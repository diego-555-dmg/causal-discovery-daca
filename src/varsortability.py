"""Varsortability y baseline sortnregress.

Función `varsortability` tomada de Reisach et al. (2021) con documentación
ampliada; `sortnregress` reimplementado con la misma lógica (ordenar por
varianza marginal y regresar cada nodo sobre los de menor varianza).
"""
import numpy as np
from sklearn.linear_model import LinearRegression, LassoLarsIC


def varsortability(X: np.ndarray, W: np.ndarray, tol: float = 1e-9) -> float:
    """Grado de acuerdo entre el orden de varianzas marginales y el orden causal.

    X: datos n x d ; W: adyacencia ponderada d x d (i->j).
    Retorna un valor en [0, 1]: 1 => la varianza crece perfectamente con el
    orden causal (el orden causal es "leíble" desde la escala); 0.5 => sin señal.
    """
    E = W != 0
    Ek = E.copy()
    var = np.var(X, axis=0, keepdims=True)
    n_paths = 0
    n_correctly_ordered_paths = 0
    for _ in range(E.shape[0] - 1):
        n_paths += Ek.sum()
        n_correctly_ordered_paths += (Ek * var / var.T > 1 + tol).sum()
        n_correctly_ordered_paths += 0.5 * (
            (Ek * var / var.T <= 1 + tol) * (Ek * var / var.T > 1 - tol)
        ).sum()
        Ek = Ek.dot(E)
    return n_correctly_ordered_paths / n_paths if n_paths else 0.5


def sortnregress(X: np.ndarray, rng=None, tie_decimals: int = 8) -> np.ndarray:
    """Baseline trivial que explota la varsortability.

    Ordena los nodos por varianza marginal creciente y regresa cada nodo sobre
    los de menor varianza; los coeficientes (con selección Lasso-BIC) son la
    estructura estimada. Iguala a métodos sofisticados cuando varsortability≈1.

    Nota de implementación: sobre datos estandarizados las varianzas son ~1 y
    difieren solo por ruido de punto flotante (~1e-13) que está sistemáticamente
    correlacionado con la escala original; ordenar por ese ruido "filtra"
    espuriamente el orden causal. Para respetar la intención del método
    (la varianza no aporta información cuando los datos están estandarizados),
    redondeamos las varianzas antes de ordenar y rompemos empates al azar.
    """
    rng = rng or np.random.default_rng(0)
    LR = LinearRegression()
    LL = LassoLarsIC(criterion="bic")
    d = X.shape[1]
    W = np.zeros((d, d))
    v = np.round(np.var(X, axis=0), tie_decimals)
    increasing = np.lexsort((rng.random(d), v))  # empates -> orden aleatorio
    for k in range(1, d):
        cov = increasing[:k]
        target = increasing[k]
        LR.fit(X[:, cov], X[:, target].ravel())
        weight = np.abs(LR.coef_)
        LL.fit(X[:, cov] * weight, X[:, target].ravel())
        W[cov, target] = LL.coef_ * weight
    return W
