"""Datos reales de Sachs et al. (2005): red de señalización de proteínas.

Sachs, K., Perez, O., Pe'er, D., Lauffenburger, D. A., Nolan, G. P. (2005).
"Causal Protein-Signaling Networks Derived from Multiparameter Single-Cell
Data." Science 308:523-529.

Se usa la condición observacional (cd3cd28), d = 11 proteínas/fosfolípidos.
El *ground truth* es la red de consenso de 17 aristas dirigidas ampliamente
usada como benchmark (idéntica a la red discreta `bnlearn/sachs`; se extrajo de
esa red y coincide con la usada por NOTEARS y DAG-GNN).

Fuentes de datos (en orden de preferencia):
  1. Archivo empaquetado en el repo: data/sachs_cd3cd28.tsv  (reproducible offline)
  2. HuggingFace Hub: dataset `pgmpy/example_datasets`
     (real/sachs/data/sachs.2005.continuous.txt) -> requiere red

Nota sobre HuggingFace: el requisito del curso pide un dataset de HuggingFace;
este dataset real está publicado allí como `pgmpy/example_datasets`.
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLED = os.path.join(HERE, "..", "data", "sachs_cd3cd28.tsv")

# Orden de columnas del dataset:
VAR_NAMES = ["raf", "mek", "plc", "pip2", "pip3", "erk", "akt", "pka", "pkc", "p38", "jnk"]
_IDX = {n: i for i, n in enumerate(VAR_NAMES)}

# Red de consenso (17 aristas dirigidas padre -> hijo), en nombres de nodo Sachs.
# raf=Raf, mek=Mek, plc=Plcg, pip2=PIP2, pip3=PIP3, erk=Erk, akt=Akt,
# pka=PKA, pkc=PKC, p38=P38, jnk=Jnk.
CONSENSUS_EDGES = [
    ("raf", "mek"), ("mek", "erk"), ("erk", "akt"),
    ("plc", "pip2"), ("plc", "pip3"), ("pip3", "pip2"),
    ("pka", "raf"), ("pka", "mek"), ("pka", "erk"), ("pka", "akt"),
    ("pka", "jnk"), ("pka", "p38"),
    ("pkc", "raf"), ("pkc", "mek"), ("pkc", "pka"), ("pkc", "jnk"), ("pkc", "p38"),
]


def ground_truth_dag() -> np.ndarray:
    """Matriz de adyacencia binaria 11x11 (i->j) de la red de consenso."""
    d = len(VAR_NAMES)
    B = np.zeros((d, d), dtype=int)
    for a, b in CONSENSUS_EDGES:
        B[_IDX[a], _IDX[b]] = 1
    return B


def _load_from_huggingface() -> pd.DataFrame:
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(repo_id="pgmpy/example_datasets", repo_type="dataset",
                        filename="real/sachs/data/sachs.2005.continuous.txt")
    return pd.read_csv(p, sep="\t")


def load_sachs(source: str = "bundled"):
    """Carga los datos de Sachs.

    source:
      - 'bundled' (por defecto): archivo del repo, reproducible sin red.
      - 'hf': descarga el archivo continuo completo desde HuggingFace Hub
              (pgmpy/example_datasets). Requiere conexión.

    Devuelve (X, B_true, var_names) con X de forma (n, 11).
    """
    if source == "hf":
        df = _load_from_huggingface()
        df = df[VAR_NAMES]
    else:
        df = pd.read_csv(BUNDLED, sep="\t")
    X = df[VAR_NAMES].to_numpy(dtype=float)
    return X, ground_truth_dag(), list(VAR_NAMES)


if __name__ == "__main__":
    X, B, names = load_sachs()
    print("X:", X.shape, "| aristas ground truth:", int(B.sum()))
    print("variables:", names)
