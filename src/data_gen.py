"""Generación de datos: DAGs sintéticos y modelos aditivos de ruido (ANM) lineales.

Permite controlar la *varsortability* de los datos a través de la política de
escalas de ruido, para estudiar cuándo la ventaja de un método neuronal es real
o instrumental (artefacto de escala), siguiendo a Reisach et al. (2021).
"""
import numpy as np
import networkx as nx


def simulate_dag(d: int, s0: int, graph_type: str = "ER", rng=None) -> np.ndarray:
    """Genera una matriz de adyacencia binaria DAG (i->j).

    d: número de nodos; s0: número esperado de aristas; graph_type: 'ER' o 'SF'.
    """
    rng = rng or np.random.default_rng()

    def _random_permutation(M):
        P = rng.permutation(np.eye(M.shape[0]))
        return P.T @ M @ P

    def _random_acyclic_orientation(B_und):
        return np.tril(_random_permutation(B_und), k=-1)

    if graph_type == "ER":
        p = float(s0) / (d * (d - 1) / 2)
        B_und = (rng.random((d, d)) < p).astype(int)
        B_und = np.triu(B_und, k=1)
        B_und = B_und + B_und.T
        B = _random_acyclic_orientation(B_und)
    elif graph_type == "SF":
        m = max(int(round(s0 / d)), 1)
        G = nx.barabasi_albert_graph(d, m, seed=int(rng.integers(1e6)))
        B_und = nx.to_numpy_array(G)
        B = _random_acyclic_orientation(B_und)
    else:
        raise ValueError(graph_type)
    # Reordena aleatoriamente para que el orden causal no coincida con el índice.
    B = _random_permutation(B)
    return (B != 0).astype(int)


def simulate_linear_sem(
    B: np.ndarray,
    n: int,
    noise: str = "gauss",
    w_range=(0.5, 2.0),
    noise_scale_policy: str = "increasing",
    rng=None,
):
    """Muestrea datos de un SEM lineal X = X W + E respetando el orden topológico.

    noise_scale_policy:
      - 'equal'      : todas las varianzas de ruido iguales (=1)  -> varsortability baja/moderada
      - 'increasing' : varianzas crecientes con el orden causal   -> varsortability alta (típico benchmark)
      - 'random'     : varianzas aleatorias                        -> varsortability variable
    Devuelve (X, W) con W la matriz de pesos ponderada (i->j).
    """
    rng = rng or np.random.default_rng()
    d = B.shape[0]
    # Pesos con signo aleatorio.
    W = np.zeros((d, d))
    idx = np.where(B != 0)
    mags = rng.uniform(*w_range, size=len(idx[0]))
    signs = rng.choice([-1.0, 1.0], size=len(idx[0]))
    W[idx] = mags * signs

    G = nx.DiGraph(B)
    order = list(nx.topological_sort(G))

    # Escalas de ruido según política.
    if noise_scale_policy == "equal":
        scales = np.ones(d)
    elif noise_scale_policy == "increasing":
        pos = np.empty(d)
        for rank, node in enumerate(order):
            pos[node] = rank
        scales = 0.5 + 1.5 * (pos / max(d - 1, 1))  # de 0.5 a 2.0 según profundidad
    elif noise_scale_policy == "random":
        scales = rng.uniform(0.5, 2.0, size=d)
    else:
        raise ValueError(noise_scale_policy)

    X = np.zeros((n, d))
    for node in order:
        parents = list(G.predecessors(node))
        eta = X[:, parents] @ W[parents, node] if parents else np.zeros(n)
        if noise == "gauss":
            e = rng.normal(0.0, scales[node], size=n)
        elif noise == "exp":
            e = (rng.exponential(scales[node], size=n) - scales[node])
        elif noise == "uniform":
            e = rng.uniform(-scales[node], scales[node], size=n)
        else:
            raise ValueError(noise)
        X[:, node] = eta + e
    return X, W
