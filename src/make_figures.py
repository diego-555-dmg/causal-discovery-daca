"""Genera las figuras del paper a partir de results/results_full.csv."""
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "figures")
os.makedirs(FIG, exist_ok=True)

df = pd.read_csv(os.path.join(RES, "results_full.csv"))
METHODS = ["sortnregress", "NOTEARS", "DAG-GNN", "PC", "GES"]
COLORS = {"sortnregress": "#9e9e9e", "NOTEARS": "#1f77b4",
          "DAG-GNN": "#d62728", "PC": "#2ca02c", "GES": "#ff7f0e"}


def _agg(metric):
    g = df.groupby(["scale", "method"])[metric].agg(["mean", "std"])
    return g


def bar_raw_vs_std(metric, ylabel, fname, title):
    g = _agg(metric)
    x = np.arange(len(METHODS)); w = 0.38
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    for i, sc in enumerate(["raw", "std"]):
        means = [g.loc[(sc, m), "mean"] if (sc, m) in g.index else 0 for m in METHODS]
        errs = [g.loc[(sc, m), "std"] if (sc, m) in g.index else 0 for m in METHODS]
        ax.bar(x + (i - 0.5) * w, means, w, yerr=errs, capsize=3,
               label=("Datos crudos (raw)" if sc == "raw" else "Estandarizados (std)"),
               color=("#4c72b0" if sc == "raw" else "#c44e52"), alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels(METHODS, rotation=15)
    ax.set_ylabel(ylabel); ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, framealpha=0.9); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, fname), dpi=150)
    plt.close(fig)


def scatter_vs_orientation(fname):
    """Precisión de orientación vs varsortability, por método."""
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    for m in METHODS:
        sub = df[df.method == m]
        ax.scatter(sub.varsortability + np.random.uniform(-0.01, 0.01, len(sub)),
                   sub.orientation_accuracy, s=28, alpha=0.7,
                   color=COLORS[m], label=m, edgecolors="k", linewidths=0.3)
    ax.axhline(0.5, ls="--", c="gray", lw=1, label="azar (0.5)")
    ax.set_xlabel("Varsortability de los datos")
    ax.set_ylabel("Precisión de orientación")
    ax.set_title("Orientación vs. varsortability", fontsize=10)
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, fname), dpi=150)
    plt.close(fig)


def skeleton_vs_orientation(fname):
    """Separa esqueleto de orientación: barras agrupadas en std."""
    g = df[df.scale == "std"].groupby("method")[["skeleton_f1", "orientation_accuracy"]].mean()
    x = np.arange(len(METHODS)); w = 0.38
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    ax.bar(x - w/2, [g.loc[m, "skeleton_f1"] for m in METHODS], w,
           label="F1 de esqueleto", color="#55a868")
    ax.bar(x + w/2, [g.loc[m, "orientation_accuracy"] for m in METHODS], w,
           label="Precisión de orientación", color="#c44e52")
    ax.axhline(0.5, ls="--", c="gray", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(METHODS, rotation=15)
    ax.set_ylabel("Métrica"); ax.set_ylim(0, 1.05)
    ax.set_title("Datos estandarizados: esqueleto vs. orientación", fontsize=10)
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, fname), dpi=150)
    plt.close(fig)


def training_curves(fname):
    path = os.path.join(RES, "curves_full.json")
    if not os.path.exists(path):
        return
    curves = json.load(open(path))
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
    fig.tight_layout(); fig.savefig(os.path.join(FIG, fname), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    np.random.seed(0)
    bar_raw_vs_std("orientation_accuracy", "Precisión de orientación",
                   "fig_orientation.png",
                   "Precisión de orientación: crudo vs. estandarizado")
    bar_raw_vs_std("shd", "SHD (menor es mejor)", "fig_shd.png",
                   "Distancia de Hamming estructural (SHD)")
    scatter_vs_orientation("fig_varsort_scatter.png")
    skeleton_vs_orientation("fig_skeleton_vs_orientation.png")
    training_curves("fig_training_curves.png")
    print("Figuras generadas en figures/")
    print(os.listdir(FIG))
