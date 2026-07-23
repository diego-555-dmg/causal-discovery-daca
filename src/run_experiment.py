"""Driver de experimentos: DAG-GNN vs NOTEARS/PC/GES/sortnregress.

Compara métodos sobre datos ANM lineales sintéticos, en datos crudos (raw) y
estandarizados (std), para varias políticas de escala de ruido que controlan la
*varsortability*. El objetivo es responder: ¿cuándo la ventaja del método
neuronal (DAG-GNN) es real y cuándo es instrumental (artefacto de escala)?

El driver es *resumible*: guarda cada resultado en results/results_<cfg>.csv en
cuanto termina una combinación (policy, scale, seed) y, al reejecutarse, salta
las combinaciones ya calculadas. Esto permite correrlo por tramos.

Uso:
    python run_experiment.py --config quick     # demo rápida
    python run_experiment.py --config full      # corrida del paper
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from utils import set_seed, standardize
from data_gen import simulate_dag, simulate_linear_sem
from varsortability import varsortability, sortnregress
from metrics import evaluate
from baselines import run_pc, run_ges, run_notears
from dag_gnn import train_dag_gnn

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
os.makedirs(RESULTS, exist_ok=True)

CONFIGS = {
    "quick": dict(d=8, s0=8, n=500, noise="gauss",
                  policies=["increasing", "equal"], seeds=[0, 1],
                  scales=["raw", "std"], gnn_epochs=150, gnn_outer=6),
    "full": dict(d=12, s0=15, n=1000, noise="gauss",
                 policies=["increasing", "equal", "random"], seeds=[0, 1, 2],
                 scales=["raw", "std"], gnn_epochs=250, gnn_outer=8),
}

FIELDS = ["policy", "scale", "seed", "method", "varsortability", "time_s",
          "shd", "skeleton_precision", "skeleton_recall", "skeleton_f1",
          "orientation_accuracy", "n_shared_edges"]


def _load_done(csv_path):
    if not os.path.exists(csv_path):
        return set()
    df = pd.read_csv(csv_path)
    return set(zip(df.policy, df.scale, df.seed))


def _append_rows(csv_path, rows):
    df = pd.DataFrame(rows)[FIELDS]
    header = not os.path.exists(csv_path)
    df.to_csv(csv_path, mode="a", header=header, index=False)


def run(config_name="quick"):
    cfg = CONFIGS[config_name]
    csv_path = os.path.join(RESULTS, f"results_{config_name}.csv")
    curves_path = os.path.join(RESULTS, f"curves_{config_name}.json")
    done = _load_done(csv_path)
    curves = json.load(open(curves_path)) if os.path.exists(curves_path) else {}

    for policy in cfg["policies"]:
        for seed in cfg["seeds"]:
            for scale in cfg["scales"]:
                if (policy, scale, seed) in done:
                    continue
                set_seed(seed)
                rng = np.random.default_rng(seed)
                B_true = simulate_dag(cfg["d"], cfg["s0"], "ER", rng=rng)
                X_raw, W_true = simulate_linear_sem(
                    B_true, cfg["n"], noise=cfg["noise"],
                    noise_scale_policy=policy, rng=rng)
                X = X_raw if scale == "raw" else standardize(X_raw)
                vs = varsortability(X, W_true)

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
                    print("PC failed:", e)
                try:
                    t = time.time(); methods["GES"] = (run_ges(X), time.time() - t)
                except Exception as e:
                    print("GES failed:", e)
                t = time.time()
                Bg, _, hist = train_dag_gnn(
                    X, nonlinear=False, epochs=cfg["gnn_epochs"],
                    max_outer=cfg["gnn_outer"], seed=seed)
                methods["DAG-GNN"] = (Bg, time.time() - t)
                curves[f"{policy}_{scale}_seed{seed}"] = {
                    "recon": hist["recon"], "h": hist["h"]}

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

    # Resumen agregado
    df = pd.read_csv(csv_path)
    agg = (df.groupby(["policy", "scale", "method"])
             [["varsortability", "shd", "skeleton_f1", "orientation_accuracy"]]
             .mean().round(3).reset_index())
    agg.to_csv(os.path.join(RESULTS, f"summary_{config_name}.csv"), index=False)
    n_conditions = len(cfg["policies"]) * len(cfg["seeds"]) * len(cfg["scales"])
    print(f"\nCompletado {len(_load_done(csv_path))}/{n_conditions} condiciones.")
    print(agg.to_string(index=False))
    return df, agg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="quick", choices=list(CONFIGS))
    args = ap.parse_args()
    run(args.config)
