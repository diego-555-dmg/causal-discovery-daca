"""Métricas de evaluación de estructura causal.

Separa explícitamente la calidad del *esqueleto* (aristas no dirigidas) de la
calidad de *orientación*, como pide la Propuesta_DACA, además del SHD estándar.
Soporta salidas dirigidas (DAG-GNN, NOTEARS, sortnregress) y CPDAG (PC, GES),
donde una arista no dirigida cuenta como 1/2 acierto de orientación.
"""
import numpy as np


def _skeleton(B):
    S = ((B + B.T) != 0).astype(int)
    np.fill_diagonal(S, 0)
    return S


def structural_hamming_distance(B_true, B_est):
    """SHD dirigido: aristas faltantes + extra + invertidas."""
    d = B_true.shape[0]
    shd = 0
    for i in range(d):
        for j in range(i + 1, d):
            t = (B_true[i, j], B_true[j, i])
            e = (B_est[i, j], B_est[j, i])
            if t == e:
                continue
            # sin arista vs con arista, o direcciones distintas
            if (t[0] or t[1]) and (e[0] or e[1]):
                shd += 1  # arista presente en ambos pero orientada distinto
            else:
                shd += 1
    return shd


def skeleton_prf(B_true, B_est):
    """Precisión, recall y F1 del esqueleto (presencia de arista no dirigida)."""
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
    """Fracción de aristas correctamente orientadas *entre las bien detectadas*.

    Solo se consideran los pares (i,j) que son arista verdadera y detectada en
    el esqueleto estimado. Arista no dirigida (i-j en ambos sentidos) => 0.5.
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
            # dirección verdadera
            if B_true[i, j] and not B_true[j, i]:
                td = "ij"
            elif B_true[j, i] and not B_true[i, j]:
                td = "ji"
            else:
                td = "both"
            if B_est[i, j] and B_est[j, i]:
                correct += 0.5  # no dirigida en CPDAG
            elif B_est[i, j] and not B_est[j, i]:
                correct += 1.0 if td == "ij" else 0.0
            elif B_est[j, i] and not B_est[i, j]:
                correct += 1.0 if td == "ji" else 0.0
    return dict(orientation_accuracy=(correct / total if total else 0.0),
                n_shared_edges=total)


def evaluate(B_true, B_est):
    out = {"shd": structural_hamming_distance(B_true, B_est)}
    out.update(skeleton_prf(B_true, B_est))
    out.update(orientation_accuracy(B_true, B_est))
    return out
