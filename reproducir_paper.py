#!/usr/bin/env python3
# -*- coding: utf-8 -*-
###############################################################################
#                                                                             #
#   ¿MEJORAS REALES O INSTRUMENTALES?                                         #
#   Evaluación crítica del descubrimiento causal neuronal (DAG-GNN)           #
#   frente a baselines no neuronales (NOTEARS, PC, GES, sortnregress).        #
#                                                                             #
#   Autor : Diego Alonso Córdova Ayala  <diego.cordova@dmg-pe.com>            #
#   Curso : Introducción a Deep Learning (Doctorado) — Proyecto Final         #
#   Base  : Reisach et al., "Beware of the Simulated DAG!" (NeurIPS 2021)     #
#           https://github.com/Scriddie/Varsortability  (código mejorado)     #
#                                                                             #
#   ---------------------------------------------------------------------     #
#   ARCHIVO ÚNICO Y AUTOCONTENIDO PARA REPRODUCCIÓN INMEDIATA.                #
#   Ejecuta TODO el pipeline del paper de principio a fin en un solo script:  #
#     1) genera los datos sintéticos,                                         #
#     2) corre los 5 métodos,                                                 #
#     3) calcula las métricas (SHD, esqueleto, orientación),                  #
#     4) guarda resultados en CSV y                                           #
#     5) produce las figuras del paper.                                       #
#                                                                             #
#   USO:                                                                      #
#     python reproducir_paper.py               # corrida completa (paper)     #
#     python reproducir_paper.py --config quick # demo rápida (~1 min)        #
#     python reproducir_paper.py --no-figuras   # omite las figuras           #
#                                                                             #
#   DEPENDENCIAS: numpy(<2), scipy, pandas, networkx, scikit-learn,           #
#                 matplotlib, causal-learn, torch  (ver requirements.txt)     #
#                                                                             #
#   REPRODUCIBILIDAD: se fijan semillas por condición; los hiperparámetros    #
#                 están documentados abajo; el driver es RESUMIBLE (si se     #
#                 interrumpe, al reejecutarlo continúa donde quedó).          #
###############################################################################

import argparse
import json
import os
import time

import numpy as np
import networkx as nx
import scipy.linalg as slin
import scipy.optimize as sopt
import pandas as pd
from sklearn.linear_model import LinearRegression, LassoLarsIC

# torch se importa aquí; si no está instalado, el script avisa con claridad.
try:
    import torch
    import torch.nn as nn
except Exception as _e:  # pragma: no cover
    raise SystemExit(
        "Falta PyTorch. Instala las dependencias con:\n"
        "    pip install -r requirements.txt\n"
        f"(detalle: {_e})"
    )


# ===========================================================================
# 0. CONFIGURACIÓN GLOBAL E HIPERPARÁMETROS
# ---------------------------------------------------------------------------
# Todo lo que define un experimento vive aquí, para que el lector vea de un
# vistazo qué se está corriendo y pueda reproducir/variar el estudio.
# La configuración "full" es la que produce los números del paper.
# ===========================================================================
CONFIGS = {
    # demo rápida para verificar que todo corre (grafos y entrenamiento chicos)
    "quick": dict(d=8, s0=8, n=500, noise="gauss",
                  policies=["increasing", "equal"], seeds=[0, 1],
                  scales=["raw", "std"], gnn_epochs=150, gnn_outer=6),
    # corrida del paper: 3 políticas × 2 escalas × 3 semillas = 18 condiciones
    "full": dict(d=12, s0=15, n=1000, noise="gauss",
                 policies=["increasing", "equal", "random"], seeds=[0, 1, 2],
                 scales=["raw", "std"], gnn_epochs=250, gnn_outer=8),
}

# Directorios de salida (se crean junto a este archivo).
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
FIGURES_DIR = os.path.join(HERE, "figures")

# Columnas del CSV de resultados (orden fijo para lectura estable).
FIELDS = ["policy", "scale", "seed", "method", "varsortability", "time_s",
          "shd", "skeleton_precision", "skeleton_recall", "skeleton_f1",
          "orientation_accuracy", "n_shared_edges"]


# ===========================================================================
# 1. UTILIDADES BÁSICAS
# ---------------------------------------------------------------------------
# Semillas (reproducibilidad), comprobación de aciclicidad, estandarización y
# umbralización de una matriz de pesos a una adyacencia binaria.
# ===========================================================================
def set_seed(seed: int) -> None:
    """Fija las semillas de numpy y torch.

    PASO CLAVE de reproducibilidad: sin esto, cada corrida generaría grafos y
    datos distintos y los resultados no serían comparables ni replicables.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_dag(W: np.ndarray) -> bool:
    """¿La matriz de adyacencia representa un DAG (grafo sin ciclos)?

    Usa el truco de la traza de la exponencial de matriz: para una matriz de
    adyacencia binaria B, tr(e^B) == d  si y solo si  no hay ciclos dirigidos.
    """
    B = (np.abs(W) > 0).astype(float)
    return np.isclose(np.trace(slin.expm(B)), B.shape[0])


def standardize(X: np.ndarray) -> np.ndarray:
    """Estandariza cada columna a media 0 y varianza 1.

    PASO CLAVE del estudio: al poner todas las varianzas a 1 se ELIMINA la
    "señal de escala". Es la intervención que revela si un método dependía de
    la varsortability (ver sección 3) para acertar.
    """
    return (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-12)


def threshold_W(W: np.ndarray, thr: float = 0.3) -> np.ndarray:
    """Convierte una matriz de pesos en una adyacencia binaria dirigida.

    Los métodos continuos (NOTEARS, DAG-GNN) devuelven pesos reales; un peso
    pequeño ≈ "sin arista". Umbralizamos en |w|>thr y borramos la diagonal.
    """
    B = (np.abs(W) > thr).astype(int)
    np.fill_diagonal(B, 0)
    return B


# ===========================================================================
# 2. GENERACIÓN DE DATOS: DAGs y modelos aditivos de ruido (ANM) lineales
# ---------------------------------------------------------------------------
# Aquí construimos el "mundo verdadero": un DAG y un SEM lineal que lo respeta.
# La política de escalas del ruido nos permite CONTROLAR la varsortability,
# que es la variable independiente central del experimento.
# ===========================================================================
def simulate_dag(d: int, s0: int, graph_type: str = "ER", rng=None) -> np.ndarray:
    """Genera una matriz de adyacencia binaria de un DAG (entrada i,j = arista i->j).

    d: número de nodos (variables); s0: número esperado de aristas.
    Estrategia: se crea un grafo NO dirigido aleatorio, se le da una orientación
    ACÍCLICA (tomando el triángulo inferior tras una permutación) y finalmente se
    permutan los nodos para que el orden causal NO coincida con el índice
    (evita "pistas" espurias por el orden de las columnas).
    """
    rng = rng or np.random.default_rng()

    def _random_permutation(M):
        P = rng.permutation(np.eye(M.shape[0]))
        return P.T @ M @ P

    def _random_acyclic_orientation(B_und):
        # tril(., -1) = triángulo inferior estricto => grafo garantizadamente acíclico
        return np.tril(_random_permutation(B_und), k=-1)

    if graph_type == "ER":  # Erdős–Rényi
        p = float(s0) / (d * (d - 1) / 2)          # prob. de arista para ~s0 aristas
        B_und = (rng.random((d, d)) < p).astype(int)
        B_und = np.triu(B_und, k=1)
        B_und = B_und + B_und.T                     # simetriza -> no dirigido
        B = _random_acyclic_orientation(B_und)
    elif graph_type == "SF":  # Scale-Free (Barabási–Albert)
        m = max(int(round(s0 / d)), 1)
        G = nx.barabasi_albert_graph(d, m, seed=int(rng.integers(1e6)))
        B_und = nx.to_numpy_array(G)
        B = _random_acyclic_orientation(B_und)
    else:
        raise ValueError(graph_type)

    B = _random_permutation(B)                      # desalinea orden causal vs índice
    return (B != 0).astype(int)


def simulate_linear_sem(B, n, noise="gauss", w_range=(0.5, 2.0),
                        noise_scale_policy="increasing", rng=None):
    """Muestrea datos de un SEM lineal  X = X·W + E  respetando el orden topológico.

    Devuelve (X, W): la matriz de datos n×d y la matriz de pesos ponderada d×d.

    ------------------------------------------------------------------------
    PUNTO CENTRAL DEL ESTUDIO — la política de escalas del ruido:
      · 'increasing' : la varianza del ruido CRECE con la profundidad causal.
                       Como la varianza se acumula hacia abajo del DAG, esto
                       produce VARSORTABILITY ALTA (≈1): el orden de varianzas
                       marginales coincide con el orden causal. Es el caso
                       "cómodo" típico de los benchmarks sintéticos.
      · 'equal'      : todas las varianzas de ruido iguales.
      · 'random'     : varianzas de ruido aleatorias.
    Estandarizar luego los datos (standardize) DESTRUYE esta señal (vs -> 0.5).
    ------------------------------------------------------------------------
    """
    rng = rng or np.random.default_rng()
    d = B.shape[0]

    # (a) Asignamos pesos con signo aleatorio a cada arista del DAG.
    W = np.zeros((d, d))
    idx = np.where(B != 0)
    mags = rng.uniform(*w_range, size=len(idx[0]))
    signs = rng.choice([-1.0, 1.0], size=len(idx[0]))
    W[idx] = mags * signs

    # (b) Orden topológico: garantiza que al generar un nodo ya existan sus padres.
    G = nx.DiGraph(B)
    order = list(nx.topological_sort(G))

    # (c) Escalas de ruido por nodo según la política elegida.
    if noise_scale_policy == "equal":
        scales = np.ones(d)
    elif noise_scale_policy == "increasing":
        pos = np.empty(d)
        for rank, node in enumerate(order):
            pos[node] = rank
        scales = 0.5 + 1.5 * (pos / max(d - 1, 1))  # de 0.5 (fuentes) a 2.0 (hojas)
    elif noise_scale_policy == "random":
        scales = rng.uniform(0.5, 2.0, size=d)
    else:
        raise ValueError(noise_scale_policy)

    # (d) Generación nodo por nodo en orden topológico: X_j = sum_padres X·W + ruido.
    X = np.zeros((n, d))
    for node in order:
        parents = list(G.predecessors(node))
        eta = X[:, parents] @ W[parents, node] if parents else np.zeros(n)
        if noise == "gauss":
            e = rng.normal(0.0, scales[node], size=n)
        elif noise == "exp":
            e = rng.exponential(scales[node], size=n) - scales[node]
        elif noise == "uniform":
            e = rng.uniform(-scales[node], scales[node], size=n)
        else:
            raise ValueError(noise)
        X[:, node] = eta + e
    return X, W


# ===========================================================================
# 3. VARSORTABILITY y el baseline trivial SORTNREGRESS
# ---------------------------------------------------------------------------
# La varsortability mide cuánta información sobre el orden causal está "filtrada"
# en el orden de las varianzas marginales. sortnregress explota exactamente eso.
# ===========================================================================
def varsortability(X, W, tol=1e-9):
    """Acuerdo entre el orden de varianzas marginales y el orden causal, en [0,1].

    1.0 => la varianza crece perfectamente a lo largo del orden causal (el orden
    causal es "leíble" desde la escala). 0.5 => la varianza no aporta información.
    Se recorre cada camino dirigido y se cuenta cuántos van de menor a mayor
    varianza. (Función original de Reisach et al., documentada.)
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
            (Ek * var / var.T <= 1 + tol) * (Ek * var / var.T > 1 - tol)).sum()
        Ek = Ek.dot(E)                              # caminos de longitud +1
    return n_correctly_ordered_paths / n_paths if n_paths else 0.5


def sortnregress(X, rng=None, tie_decimals=8):
    """Baseline TRIVIAL que solo explota la varsortability (Reisach et al.).

    Idea: ordena los nodos por varianza marginal creciente y regresa cada nodo
    sobre los de MENOR varianza; los coeficientes (con selección Lasso-BIC) son
    la estructura estimada. Cuando varsortability≈1 este truco IGUALA a métodos
    sofisticados, sin razonar causalmente en absoluto.

    ------------------------------------------------------------------------
    MEJORA PROPIA (corrección de una fuga numérica):
      Sobre datos ESTANDARIZADOS las varianzas valen ~1 y difieren solo por
      ruido de punto flotante (~1e-13) que, sorprendentemente, está
      CORRELACIONADO con la escala original (ρ≈0.95 con la profundidad causal).
      Ordenar por ese ruido "recupera" espuriamente el orden causal y hace que
      sortnregress parezca funcionar aun estandarizado. Lo corregimos
      redondeando las varianzas (colapsa el ruido a empates) y rompiendo empates
      AL AZAR: así, cuando los datos están estandarizados, la varianza no aporta
      información, que es justo la intención del método.
    ------------------------------------------------------------------------
    """
    rng = rng or np.random.default_rng(0)
    LR = LinearRegression()
    LL = LassoLarsIC(criterion="bic")
    d = X.shape[1]
    W = np.zeros((d, d))
    v = np.round(np.var(X, axis=0), tie_decimals)
    increasing = np.lexsort((rng.random(d), v))     # empates -> orden aleatorio
    for k in range(1, d):
        cov = increasing[:k]                         # covariables = nodos de menor varianza
        target = increasing[k]
        LR.fit(X[:, cov], X[:, target].ravel())
        weight = np.abs(LR.coef_)
        LL.fit(X[:, cov] * weight, X[:, target].ravel())
        W[cov, target] = LL.coef_ * weight
    return W


# ===========================================================================
# 4. BASELINES NO NEURONALES: NOTEARS (lineal), PC y GES
# ===========================================================================
def notears_linear(X, lambda1=0.1, max_iter=100, h_tol=1e-8,
                   rho_max=1e16, w_threshold=0.3):
    """NOTEARS lineal (Zheng et al. 2018): optimización continua de la estructura.

    Minimiza  MSE(X, XW) + λ‖W‖₁  sujeto a la restricción de ACICLICIDAD
    diferenciable  h(W)=tr(e^{W∘W}) − d = 0, resuelta con Lagrangiano aumentado.
    Es "no neuronal" pero comparte con DAG-GNN la idea de optimización continua;
    por eso también depende de la escala (varsortability).
    """
    n, d = X.shape

    def _loss(W):                                    # MSE y su gradiente
        R = X - X @ W
        return 0.5 / n * (R ** 2).sum(), -1.0 / n * X.T @ R

    def _h(W):                                       # restricción de aciclicidad
        E = slin.expm(W * W)
        return np.trace(E) - d, E.T * W * 2

    def _adj(w):                                     # w (2d²) -> W (d×d), truco w=w+ - w-
        return (w[:d * d] - w[d * d:]).reshape([d, d])

    def _func(w):                                    # objetivo aumentado + gradiente
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
    for _ in range(max_iter):                        # bucle externo del Lagrangiano
        w_new, h_new = None, None
        while rho < rho_max:
            sol = sopt.minimize(_func, w_est, method="L-BFGS-B", jac=True, bounds=bnds)
            w_new = sol.x
            h_new, _ = _h(_adj(w_new))
            if h_new > 0.25 * h:                      # si no baja la aciclicidad, sube rho
                rho *= 10
            else:
                break
        w_est, h = w_new, h_new
        alpha += rho * h                             # actualiza el multiplicador
        if h <= h_tol or rho >= rho_max:
            break
    W_est = _adj(w_est)
    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est


def _cpdag_to_B(G):
    """Convierte la matriz de causal-learn (codifica -1/1) a adyacencia binaria.

    PC y GES devuelven un CPDAG (clase de equivalencia): algunas aristas quedan
    SIN orientar. Codificación causal-learn: g[i,j]=-1 y g[j,i]=1 => i->j ;
    g[i,j]=g[j,i]=-1 => i–j (no dirigida, se marca en ambos sentidos).
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
    """PC (Spirtes et al.): método por RESTRICCIONES (tests de independencia).

    Invariante a escala (usa correlaciones parciales, Fisher-z), por lo que su
    desempeño NO cambia al estandarizar: es la referencia "honesta" del estudio.
    """
    from causallearn.search.ConstraintBased.PC import pc
    cg = pc(X, alpha, indep_test="fisherz", show_progress=False)
    return _cpdag_to_B(cg.G.graph)


def run_ges(X):
    """GES (Chickering): método por SCORE (búsqueda voraz con BIC).

    También invariante a escala. Junto con PC, se vuelven los mejores métodos
    cuando los datos están estandarizados.
    """
    from causallearn.search.ScoreBased.GES import ges
    rec = ges(X, score_func="local_score_BIC")
    return _cpdag_to_B(rec["G"].graph)


def run_notears(X, w_threshold=0.3):
    """Envoltorio de NOTEARS que devuelve (adyacencia binaria, pesos)."""
    W = notears_linear(X, w_threshold=w_threshold)
    return threshold_W(W, thr=1e-9), W


# ===========================================================================
# 5. DAG-GNN: el método NEURONAL bajo evaluación (Yu et al. 2019)
# ---------------------------------------------------------------------------
# Autoencoder variacional en grafo con una adyacencia ponderada aprendible A.
# La única vía por la que la información pasa entre variables es A (las MLPs son
# por-nodo), así que A debe capturar la estructura causal. Optimizamos una cota
# variacional (reconstrucción + KL) con la restricción de aciclicidad de DAG-GNN
# vía Lagrangiano aumentado.
# ===========================================================================
class NodeMLP(nn.Module):
    """MLP compartida aplicada por separado a la característica escalar de cada nodo.

    Al ser "por nodo", NO mezcla variables entre sí: eso obliga a que toda la
    dependencia entre variables pase por la matriz A (la estructura causal).
    """
    def __init__(self, hidden=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):                            # x: (n, d)
        n, d = x.shape
        return self.net(x.reshape(-1, 1)).reshape(n, d)


class DAGGNN(nn.Module):
    """Modelo DAG-GNN.

    Codificador:  Z = (I − Aᵀ)·g_enc(X)      (residuos = codificación latente)
    Decodificador: X̂ = g_dec((I − Aᵀ)⁻¹·Z)
    En modo lineal (nonlinear=False) g_enc = g_dec = identidad, que es la
    configuración estable usada en el paper para benchmarks de ANM lineales.
    """
    def __init__(self, d, hidden=16, nonlinear=False):
        super().__init__()
        self.d = d
        self.nonlinear = nonlinear
        self.A = nn.Parameter(torch.zeros(d, d))     # adyacencia ponderada aprendible
        self.z_logvar = nn.Parameter(torch.zeros(d)) # log-varianza del latente por nodo
        if nonlinear:
            self.enc = NodeMLP(hidden)
            self.dec = NodeMLP(hidden)

    def _Am(self):
        """A con diagonal forzada a 0 (un nodo no puede ser su propio padre)."""
        return self.A * (1.0 - torch.eye(self.d, device=self.A.device))

    def h_acyclic(self):
        """Restricción de aciclicidad de DAG-GNN: h(A)=tr[(I + A∘A/d)^d] − d = 0."""
        A = self._Am()
        M = torch.eye(self.d, device=A.device) + (1.0 / self.d) * (A * A)
        return torch.matrix_power(M, self.d).trace() - self.d

    def forward(self, X):
        d = self.d
        Am = self._Am()
        ImAt = torch.eye(d, device=Am.device) - Am.t()   # (I − Aᵀ)
        H = self.enc(X) if self.nonlinear else X
        z_mean = H @ ImAt.t()                             # codificación (residuos)
        var = torch.exp(self.z_logvar)
        z = z_mean + torch.sqrt(var) * torch.randn_like(z_mean)  # reparametrización VAE
        Hdec = z @ torch.inverse(ImAt).t()                # decodifica pasando por A⁻¹
        X_hat = self.dec(Hdec) if self.nonlinear else Hdec
        return X_hat, z_mean, var


def train_dag_gnn(X, hidden=16, nonlinear=False, epochs=250, lr=1e-2,
                  lambda_l1=0.02, w_threshold=0.3, max_outer=8, seed=0,
                  h_tol=1e-8):
    """Entrena DAG-GNN con Lagrangiano aumentado. Devuelve (B, W, history).

    PASO A PASO de cada iteración:
      1) forward: reconstruye X̂ y calcula la codificación latente.
      2) pérdida = reconstrucción(MSE) + KL(q(Z)‖N(0,I)) + λ‖A‖₁
                   + α·h(A) + ½ρ·h(A)²      (los dos últimos: aciclicidad).
      3) backward + Adam actualizan A y las MLPs.
      4) al terminar el bucle interno, se actualiza el multiplicador α y, si la
         aciclicidad no bajó lo suficiente, se aumenta la penalización ρ.
    El resultado se umbraliza (|A|>w_threshold) para obtener el DAG estimado.
    """
    torch.manual_seed(seed)
    n, d = X.shape
    Xt = torch.tensor(X, dtype=torch.float32)
    model = DAGGNN(d, hidden=hidden, nonlinear=nonlinear)

    rho, alpha, h_val = 1.0, 0.0, np.inf
    rho_max = 1e18
    history = {"recon": [], "h": []}

    for _ in range(max_outer):                       # bucle EXTERNO (Lagrangiano)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for _ in range(epochs):                      # bucle INTERNO (descenso)
            opt.zero_grad()
            X_hat, z_mean, var = model(Xt)
            recon = ((X_hat - Xt) ** 2).mean()                       # reconstrucción
            kl = 0.5 * (z_mean ** 2 + var - torch.log(var + 1e-12) - 1).mean()  # KL
            h = model.h_acyclic()                                    # aciclicidad
            l1 = model.A.abs().sum()                                 # dispersión
            loss = recon + kl + lambda_l1 * l1 + alpha * h + 0.5 * rho * h * h
            loss.backward()
            opt.step()
            history["recon"].append(float(recon.item()))
            history["h"].append(float(h.item()))
        with torch.no_grad():
            h_new = float(model.h_acyclic().item())
        alpha += rho * h_new                          # actualiza multiplicador
        if h_new > 0.25 * h_val and rho < rho_max:    # endurece penalización si hace falta
            rho *= 10
        h_val = h_new
        if h_val <= h_tol:                            # ya es (casi) acíclico -> paramos
            break

    W = model._Am().detach().numpy()
    B = (np.abs(W) > w_threshold).astype(int)
    np.fill_diagonal(B, 0)
    return B, W, history


# ===========================================================================
# 6. MÉTRICAS: SHD + (aporte del proyecto) SEPARAR esqueleto de orientación
# ---------------------------------------------------------------------------
# No basta el SHD: el hallazgo del paper solo se ve con claridad al separar la
# calidad del ESQUELETO (qué variables están conectadas) de la de ORIENTACIÓN
# (en qué dirección), porque el colapso al estandarizar ocurre en la orientación.
# ===========================================================================
def _skeleton(B):
    """Esqueleto = versión no dirigida (hay arista i–j si hay i->j o j->i)."""
    S = ((B + B.T) != 0).astype(int)
    np.fill_diagonal(S, 0)
    return S


def structural_hamming_distance(B_true, B_est):
    """SHD: número de aristas faltantes + sobrantes + con dirección equivocada."""
    d = B_true.shape[0]
    shd = 0
    for i in range(d):
        for j in range(i + 1, d):
            t = (B_true[i, j], B_true[j, i])
            e = (B_est[i, j], B_est[j, i])
            if t == e:
                continue
            shd += 1
    return shd


def skeleton_prf(B_true, B_est):
    """Precisión, recall y F1 del ESQUELETO (presencia de arista, sin dirección)."""
    St, Se = _skeleton(B_true), _skeleton(B_est)
    iu = np.triu_indices_from(St, k=1)
    t, e = St[iu], Se[iu]
    tp = int(np.sum((t == 1) & (e == 1)))
    fp = int(np.sum((t == 0) & (e == 1)))
    fn = int(np.sum((t == 1) & (e == 0)))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return dict(skeleton_precision=prec, skeleton_recall=rec, skeleton_f1=f1)


def orientation_accuracy(B_true, B_est):
    """Fracción de aristas bien ORIENTADAS, solo entre las bien detectadas.

    Una arista no dirigida del CPDAG (PC/GES) cuenta 0.5 (empate). Un valor ≈0.5
    significa "orientación al azar": es la firma del colapso al estandarizar.
    """
    d = B_true.shape[0]
    correct, total = 0.0, 0
    for i in range(d):
        for j in range(i + 1, d):
            true_edge = B_true[i, j] or B_true[j, i]
            est_edge = B_est[i, j] or B_est[j, i]
            if not (true_edge and est_edge):
                continue
            total += 1
            if B_true[i, j] and not B_true[j, i]:
                td = "ij"
            elif B_true[j, i] and not B_true[i, j]:
                td = "ji"
            else:
                td = "both"
            if B_est[i, j] and B_est[j, i]:
                correct += 0.5                        # no dirigida => medio acierto
            elif B_est[i, j] and not B_est[j, i]:
                correct += 1.0 if td == "ij" else 0.0
            elif B_est[j, i] and not B_est[i, j]:
                correct += 1.0 if td == "ji" else 0.0
    return dict(orientation_accuracy=(correct / total if total else 0.0),
                n_shared_edges=total)


def evaluate(B_true, B_est):
    """Reúne todas las métricas para un par (verdad, estimación)."""
    out = {"shd": structural_hamming_distance(B_true, B_est)}
    out.update(skeleton_prf(B_true, B_est))
    out.update(orientation_accuracy(B_true, B_est))
    return out


# ===========================================================================
# 7. DRIVER DEL EXPERIMENTO (resumible con checkpoints)
# ---------------------------------------------------------------------------
# Recorre todas las condiciones (política × escala × semilla), corre los 5
# métodos, evalúa y guarda cada condición al terminarla. Si se interrumpe, al
# reejecutar continúa donde quedó (útil en equipos con límite de tiempo/proc).
# ===========================================================================
def _load_done(csv_path):
    if not os.path.exists(csv_path):
        return set()
    df = pd.read_csv(csv_path)
    return set(zip(df.policy, df.scale, df.seed))


def _append_rows(csv_path, rows):
    df = pd.DataFrame(rows)[FIELDS]
    df.to_csv(csv_path, mode="a", header=not os.path.exists(csv_path), index=False)


def run_experiment(config_name="full"):
    """Ejecuta el estudio completo y devuelve (df_detalle, df_resumen)."""
    cfg = CONFIGS[config_name]
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, f"results_{config_name}.csv")
    curves_path = os.path.join(RESULTS_DIR, f"curves_{config_name}.json")
    done = _load_done(csv_path)
    curves = json.load(open(curves_path)) if os.path.exists(curves_path) else {}

    for policy in cfg["policies"]:
        for seed in cfg["seeds"]:
            for scale in cfg["scales"]:
                if (policy, scale, seed) in done:
                    continue                          # ya calculado (resumible)

                # --- (1) Mundo verdadero: DAG + datos ---
                set_seed(seed)
                rng = np.random.default_rng(seed)
                B_true = simulate_dag(cfg["d"], cfg["s0"], "ER", rng=rng)
                X_raw, W_true = simulate_linear_sem(
                    B_true, cfg["n"], noise=cfg["noise"],
                    noise_scale_policy=policy, rng=rng)

                # --- (2) Condición de escala: cruda o estandarizada ---
                X = X_raw if scale == "raw" else standardize(X_raw)
                vs = varsortability(X, W_true)        # varsortability efectiva

                # --- (3) Corremos los 5 métodos y medimos tiempos ---
                methods = {}
                t = time.time()
                Wsr = sortnregress(X, rng=np.random.default_rng(1000 + seed))
                Bsr = (np.abs(Wsr) > 1e-6).astype(int); np.fill_diagonal(Bsr, 0)
                methods["sortnregress"] = (Bsr, time.time() - t)

                t = time.time(); Bnt, _ = run_notears(X)
                methods["NOTEARS"] = (Bnt, time.time() - t)

                try:
                    t = time.time(); methods["PC"] = (run_pc(X), time.time() - t)
                except Exception as e:
                    print("PC falló:", e)
                try:
                    t = time.time(); methods["GES"] = (run_ges(X), time.time() - t)
                except Exception as e:
                    print("GES falló:", e)

                t = time.time()
                Bg, _, hist = train_dag_gnn(
                    X, nonlinear=False, epochs=cfg["gnn_epochs"],
                    max_outer=cfg["gnn_outer"], seed=seed)
                methods["DAG-GNN"] = (Bg, time.time() - t)
                curves[f"{policy}_{scale}_seed{seed}"] = {
                    "recon": hist["recon"], "h": hist["h"]}

                # --- (4) Evaluación y checkpoint en disco ---
                rows = []
                for name, (B_est, dt) in methods.items():
                    m = evaluate(B_true, B_est)
                    rows.append(dict(policy=policy, scale=scale, seed=seed,
                                     method=name, varsortability=round(vs, 3),
                                     time_s=round(dt, 2), **m))
                    print(f"[{policy:>10}|{scale:>3}|s{seed}] vs={vs:.2f} "
                          f"{name:>13}: SHD={m['shd']:>2} "
                          f"skF1={m['skeleton_f1']:.2f} "
                          f"orient={m['orientation_accuracy']:.2f}")
                _append_rows(csv_path, rows)
                json.dump(curves, open(curves_path, "w"))

    # --- (5) Resumen agregado (promedio sobre políticas y semillas) ---
    df = pd.read_csv(csv_path)
    agg = (df.groupby(["scale", "method"])
             [["varsortability", "shd", "skeleton_f1", "orientation_accuracy"]]
             .mean().round(3).reset_index())
    agg.to_csv(os.path.join(RESULTS_DIR, f"summary_{config_name}.csv"), index=False)
    n_cond = len(cfg["policies"]) * len(cfg["seeds"]) * len(cfg["scales"])
    print(f"\nCompletado {len(_load_done(csv_path))}/{n_cond} condiciones.")
    print("\n===== RESUMEN (promedio por escala) =====")
    print(agg.to_string(index=False))
    return df, agg


# ===========================================================================
# 8. FIGURAS DEL PAPER
# ===========================================================================
def make_figures(config_name="full"):
    """Genera las 4 figuras del paper a partir del CSV de resultados."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIGURES_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, f"results_{config_name}.csv")
    df = pd.read_csv(csv_path)
    METHODS = ["sortnregress", "NOTEARS", "DAG-GNN", "PC", "GES"]
    np.random.seed(0)

    # (Fig. 1) Precisión de orientación: crudo vs estandarizado -> figura central.
    g = df.groupby(["scale", "method"])["orientation_accuracy"].agg(["mean", "std"])
    x = np.arange(len(METHODS)); w = 0.38
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    for i, sc in enumerate(["raw", "std"]):
        means = [g.loc[(sc, m), "mean"] if (sc, m) in g.index else 0 for m in METHODS]
        errs = [g.loc[(sc, m), "std"] if (sc, m) in g.index else 0 for m in METHODS]
        ax.bar(x + (i - 0.5) * w, means, w, yerr=errs, capsize=3,
               label=("Datos crudos (raw)" if sc == "raw" else "Estandarizados (std)"),
               color=("#4c72b0" if sc == "raw" else "#c44e52"), alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels(METHODS, rotation=15)
    ax.set_ylabel("Precisión de orientación")
    ax.set_title("Precisión de orientación: crudo vs. estandarizado", fontsize=10)
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIGURES_DIR, "fig_orientation.png"), dpi=150)
    plt.close(fig)

    # (Fig. 2) SHD: crudo vs estandarizado.
    g = df.groupby(["scale", "method"])["shd"].agg(["mean", "std"])
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    for i, sc in enumerate(["raw", "std"]):
        means = [g.loc[(sc, m), "mean"] if (sc, m) in g.index else 0 for m in METHODS]
        errs = [g.loc[(sc, m), "std"] if (sc, m) in g.index else 0 for m in METHODS]
        ax.bar(x + (i - 0.5) * w, means, w, yerr=errs, capsize=3,
               label=("Datos crudos (raw)" if sc == "raw" else "Estandarizados (std)"),
               color=("#4c72b0" if sc == "raw" else "#c44e52"), alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels(METHODS, rotation=15)
    ax.set_ylabel("SHD (menor es mejor)")
    ax.set_title("Distancia de Hamming estructural (SHD)", fontsize=10)
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIGURES_DIR, "fig_shd.png"), dpi=150)
    plt.close(fig)

    # (Fig. 3) Orientación vs varsortability (dispersión).
    COLORS = {"sortnregress": "#9e9e9e", "NOTEARS": "#1f77b4", "DAG-GNN": "#d62728",
              "PC": "#2ca02c", "GES": "#ff7f0e"}
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    for m in METHODS:
        sub = df[df.method == m]
        ax.scatter(sub.varsortability + np.random.uniform(-0.01, 0.01, len(sub)),
                   sub.orientation_accuracy, s=28, alpha=0.7, color=COLORS[m],
                   label=m, edgecolors="k", linewidths=0.3)
    ax.axhline(0.5, ls="--", c="gray", lw=1, label="azar (0.5)")
    ax.set_xlabel("Varsortability de los datos"); ax.set_ylabel("Precisión de orientación")
    ax.set_title("Orientación vs. varsortability", fontsize=10)
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIGURES_DIR, "fig_varsort_scatter.png"), dpi=150)
    plt.close(fig)

    # (Fig. 4) Esqueleto vs orientación en datos estandarizados.
    gs = df[df.scale == "std"].groupby("method")[["skeleton_f1", "orientation_accuracy"]].mean()
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    ax.bar(x - w/2, [gs.loc[m, "skeleton_f1"] for m in METHODS], w,
           label="F1 de esqueleto", color="#55a868")
    ax.bar(x + w/2, [gs.loc[m, "orientation_accuracy"] for m in METHODS], w,
           label="Precisión de orientación", color="#c44e52")
    ax.axhline(0.5, ls="--", c="gray", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(METHODS, rotation=15)
    ax.set_ylabel("Métrica"); ax.set_ylim(0, 1.05)
    ax.set_title("Datos estandarizados: esqueleto vs. orientación", fontsize=10)
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIGURES_DIR, "fig_skeleton_vs_orientation.png"), dpi=150)
    plt.close(fig)

    # (Fig. 5) Curvas de entrenamiento de DAG-GNN.
    if os.path.exists(os.path.join(RESULTS_DIR, f"curves_{config_name}.json")):
        curves = json.load(open(os.path.join(RESULTS_DIR, f"curves_{config_name}.json")))
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 3.0))
        for key in ["increasing_raw_seed0", "increasing_std_seed0"]:
            if key not in curves:
                continue
            lab = "raw" if "raw" in key else "std"
            a1.plot(curves[key]["recon"], label=lab, lw=1.2)
            a2.plot(np.abs(curves[key]["h"]) + 1e-12, label=lab, lw=1.2)
        a1.set_title("Reconstrucción (DAG-GNN)", fontsize=9)
        a1.set_xlabel("iteración"); a1.set_ylabel("MSE"); a1.legend(fontsize=8); a1.grid(alpha=0.3)
        a2.set_title("Restricción de aciclicidad h(A)", fontsize=9)
        a2.set_xlabel("iteración"); a2.set_ylabel("|h(A)|"); a2.set_yscale("log")
        a2.legend(fontsize=8); a2.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(FIGURES_DIR, "fig_training_curves.png"), dpi=150)
        plt.close(fig)

    print(f"Figuras guardadas en: {FIGURES_DIR}")


# ===========================================================================
# 9. PUNTO DE ENTRADA
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(
        description="Reproduce de principio a fin el estudio DAG-GNN vs baselines.")
    ap.add_argument("--config", default="full", choices=list(CONFIGS),
                    help="'full' = corrida del paper; 'quick' = demo rápida.")
    ap.add_argument("--no-figuras", action="store_true",
                    help="No generar las figuras (solo resultados numéricos).")
    args = ap.parse_args()

    print(f"== Reproduciendo estudio (config='{args.config}') ==")
    print(f"   torch={torch.__version__}  cuda_disponible={torch.cuda.is_available()}")
    run_experiment(args.config)
    if not args.no_figuras:
        make_figures(args.config)
    print("\nListo. Revisa las carpetas 'results/' y 'figures/'.")


if __name__ == "__main__":
    main()
