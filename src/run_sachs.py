"""Experimento complementario sobre datos REALES: red de proteínas de Sachs (2005).

Aplica los mismos cinco métodos del estudio (sortnregress, NOTEARS, PC, GES y
DAG-GNN) a la matriz observacional real de Sachs, en datos crudos (raw) y
estandarizados (std), y evalúa contra la red de consenso de 17 aristas.

Objetivo: comprobar en datos reales la tesis del paper sobre varsortability.
En datos sintéticos con alta varsortability, los métodos dependientes de escala
"ganan" de forma instrumental. En datos reales biológicos NO hay razón para que
la varianza marginal codifique el orden causal, de modo que se espera:
  (i)  varsortability cercana a azar (no ~1),
  (ii) poca diferencia entre raw y std para los métodos dependientes de escala,
  (iii) PC y GES (invariantes a escala) idénticos en raw y std.

Uso:
    python run_sachs.py                 # usa datos empaquetados (offline)
    python run_sachs.py --source hf     # descarga desde HuggingFace Hub
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

from utils import set_seed, standardize
from varsortability import varsortability, sortnregress
from metrics import evaluate
from baselines import run_pc, run_ges, run_notears
from sachs_data import load_sachs

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
os.makedirs(RESULTS, exist_ok=True)

FIELDS = ["scale", "method", "varsortability", "time_s", "shd",
          "skeleton_precision", "skeleton_recall", "skeleton_f1",
          "orientation_accuracy", "n_shared_edges"]


def run(source="bundled", seed=0, use_dag_gnn=True):
    set_seed(seed)
    X_raw, B_true, names = load_sachs(source=source)
    n, d = X_raw.shape
    print(f"Sachs: n={n}, d={d}, aristas ground truth={int(B_true.sum())}, fuente={source}")

    rows = []
    for scale in ["raw", "std"]:
        X = X_raw if scale == "raw" else standardize(X_raw)
        vs = varsortability(X, B_true.astype(float))

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

        if use_dag_gnn:
            try:
                from dag_gnn import train_dag_gnn
                t = time.time()
                Bg, _, _ = train_dag_gnn(X, nonlinear=False, epochs=250,
                                         max_outer=8, seed=seed)
                methods["DAG-GNN"] = (Bg, time.time() - t)
            except Exception as e:
                print("DAG-GNN omitido (¿falta torch?):", str(e)[:80])

        for name, (B_est, dt) in methods.items():
            m = evaluate(B_true, B_est)
            rows.append(dict(scale=scale, method=name, varsortability=round(vs, 3),
                             time_s=round(dt, 2), **m))
            print(f"[{scale:>3}] vs={vs:.2f} {name:>13}: SHD={m['shd']:>2} "
                  f"skF1={m['skeleton_f1']:.2f} orient={m['orientation_accuracy']:.2f}")

    df = pd.DataFrame(rows)[FIELDS]
    out = os.path.join(RESULTS, "results_sachs.csv")
    df.to_csv(out, index=False)
    print("\nGuardado:", out)
    # Resumen compacto
    piv = df.pivot(index="method", columns="scale",
                   values=["shd", "skeleton_f1", "orientation_accuracy"])
    print(piv.round(3).to_string())
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="bundled", choices=["bundled", "hf"])
    ap.add_argument("--no-dag-gnn", action="store_true")
    args = ap.parse_args()
    run(source=args.source, use_dag_gnn=not args.no_dag_gnn)
