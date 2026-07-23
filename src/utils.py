"""Utilidades generales: semillas, aciclicidad, estandarización.

Adaptado y ampliado a partir del repositorio de Reisach et al. (2021),
"Beware of the Simulated DAG!" (https://github.com/Scriddie/Varsortability).
"""
import numpy as np


def set_seed(seed: int) -> None:
    """Fija semillas de numpy y (si está disponible) torch para reproducibilidad."""
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def is_dag(W: np.ndarray) -> bool:
    """Comprueba aciclicidad vía traza de potencias de la matriz de adyacencia."""
    import scipy.linalg as slin
    B = (np.abs(W) > 0).astype(float)
    return np.isclose(np.trace(slin.expm(B)), B.shape[0])


def standardize(X: np.ndarray) -> np.ndarray:
    """Estandariza columnas a media 0 y varianza 1 (elimina la señal de escala)."""
    return (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-12)


def threshold_W(W: np.ndarray, thr: float = 0.3) -> np.ndarray:
    """Umbraliza una matriz de pesos para obtener una adyacencia binaria dirigida."""
    B = (np.abs(W) > thr).astype(int)
    np.fill_diagonal(B, 0)
    return B
