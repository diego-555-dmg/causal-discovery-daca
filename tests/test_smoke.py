"""Test de humo: verifica que todo el pipeline corre de principio a fin.

Uso:
    python tests/test_smoke.py

Corre en ~1-2 min con torch instalado. Si torch o causal-learn no están
disponibles, los tests correspondientes se marcan como SKIP y el resto se
ejecuta igual.
"""
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np

PASS, SKIP, FAIL = [], [], []


def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print(f"  [PASS] {name}")
    except ImportError as e:
        SKIP.append(name)
        print(f"  [SKIP] {name}: {e}")
    except Exception:
        FAIL.append(name)
        print(f"  [FAIL] {name}")
        traceback.print_exc()


def t_datos():
    from utils import set_seed, standardize, is_dag
    from data_gen import simulate_dag, simulate_linear_sem
    set_seed(0)
    rng = np.random.default_rng(0)
    B = simulate_dag(8, 8, "ER", rng=rng)
    assert is_dag(B), "el grafo generado debe ser un DAG"
    X, W = simulate_linear_sem(B, 300, noise_scale_policy="increasing", rng=rng)
    assert X.shape == (300, 8)
    Xs = standardize(X)
    assert np.allclose(Xs.std(axis=0), 1.0, atol=1e-6)


def t_varsortability():
    from utils import standardize
    from data_gen import simulate_dag, simulate_linear_sem
    from varsortability import varsortability
    rng = np.random.default_rng(0)
    B = simulate_dag(8, 8, "ER", rng=rng)
    X, W = simulate_linear_sem(B, 500, noise_scale_policy="increasing", rng=rng)
    assert varsortability(X, W) > 0.9, "increasing/raw debe tener varsortability alta"
    assert abs(varsortability(standardize(X), W) - 0.5) < 0.15, \
        "estandarizar debe llevar la varsortability cerca de 0.5"


def t_sortnregress_fix():
    """La corrección de la fuga numérica: en datos estandarizados el orden
    usado por sortnregress no debe reconstruir el orden causal."""
    from utils import standardize
    from data_gen import simulate_dag, simulate_linear_sem
    from varsortability import sortnregress
    from metrics import evaluate
    rng = np.random.default_rng(0)
    B = simulate_dag(10, 12, "ER", rng=rng)
    X, _ = simulate_linear_sem(B, 800, noise_scale_policy="increasing", rng=rng)
    Wsr = sortnregress(standardize(X), rng=np.random.default_rng(7))
    Bsr = (np.abs(Wsr) > 1e-6).astype(int)
    np.fill_diagonal(Bsr, 0)
    m = evaluate(B, Bsr)
    assert m["orientation_accuracy"] < 0.95, \
        "sin señal de escala, sortnregress no debería orientar casi perfecto"


def t_metricas():
    from metrics import evaluate
    B = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    perfecto = evaluate(B, B)
    assert perfecto["shd"] == 0 and perfecto["skeleton_f1"] == 1.0
    invertido = B.T
    m = evaluate(B, invertido)
    assert m["orientation_accuracy"] == 0.0 and m["skeleton_f1"] == 1.0


def t_notears():
    from baselines import run_notears
    from data_gen import simulate_dag, simulate_linear_sem
    from metrics import evaluate
    rng = np.random.default_rng(0)
    B = simulate_dag(6, 6, "ER", rng=rng)
    X, _ = simulate_linear_sem(B, 400, noise_scale_policy="increasing", rng=rng)
    Bnt, _ = run_notears(X)
    m = evaluate(B, Bnt)
    assert m["skeleton_f1"] > 0.7, f"NOTEARS raw debería recuperar el esqueleto (F1={m['skeleton_f1']:.2f})"


def t_sachs_data():
    from sachs_data import load_sachs, ground_truth_dag
    X, B, names = load_sachs()  # datos empaquetados (offline)
    assert X.shape[1] == 11 and len(names) == 11
    assert X.shape[0] > 100, "debe haber suficientes observaciones reales"
    assert int(B.sum()) == 17, "la red de consenso tiene 17 aristas"
    assert not np.isnan(X).any()


def t_pc_ges():
    from baselines import run_pc, run_ges  # requiere causal-learn
    from data_gen import simulate_dag, simulate_linear_sem
    from metrics import evaluate
    rng = np.random.default_rng(0)
    B = simulate_dag(6, 6, "ER", rng=rng)
    X, _ = simulate_linear_sem(B, 400, noise_scale_policy="increasing", rng=rng)
    for fn in (run_pc, run_ges):
        m = evaluate(B, fn(X))
        assert m["skeleton_f1"] > 0.5


def t_dag_gnn():
    import torch  # noqa: F401  (requiere torch)
    from dag_gnn import train_dag_gnn
    from data_gen import simulate_dag, simulate_linear_sem
    rng = np.random.default_rng(0)
    B = simulate_dag(5, 5, "ER", rng=rng)
    X, _ = simulate_linear_sem(B, 200, noise_scale_policy="increasing", rng=rng)
    Bg, W, hist = train_dag_gnn(X, nonlinear=False, epochs=30, max_outer=2, seed=0)
    assert Bg.shape == (5, 5)
    assert abs(hist["h"][-1]) < abs(hist["h"][0]) + 1e-6, "h(A) debería no crecer"


if __name__ == "__main__":
    print("Test de humo del pipeline:")
    check("generación de datos (DAG + SEM lineal)", t_datos)
    check("varsortability (alta en raw, ~0.5 en std)", t_varsortability)
    check("sortnregress sin fuga numérica en std", t_sortnregress_fix)
    check("métricas (SHD, esqueleto, orientación)", t_metricas)
    check("NOTEARS", t_notears)
    check("datos reales de Sachs (loader + ground truth)", t_sachs_data)
    check("PC y GES (causal-learn)", t_pc_ges)
    check("DAG-GNN (torch)", t_dag_gnn)
    print(f"\nResumen: {len(PASS)} PASS, {len(SKIP)} SKIP, {len(FAIL)} FAIL")
    sys.exit(1 if FAIL else 0)
