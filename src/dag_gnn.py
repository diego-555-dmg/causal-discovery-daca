"""DAG-GNN: descubrimiento causal por optimización continua con una GNN/VAE.

Reimplementación compacta y estable de Yu et al. (2019, ICML), "DAG-GNN: DAG
Structure Learning with Graph Neural Networks", adaptada a partir del marco de
Reisach et al. (2021).

Modelo generativo:  X se genera de un latente Z ~ N(0, I) a través de la
adyacencia ponderada aprendible A (d x d):

    Codificador (encoder):  Z = (I - A^T) · g_enc(X)
    Decodificador (decoder): X_hat = g_dec( (I - A^T)^{-1} Z )

donde g_enc, g_dec son MLPs *por nodo* (comparten pesos entre nodos), de modo
que la única vía de mezcla de información entre variables es la matriz A. En
modo lineal g_enc = g_dec = identidad. El objetivo es una cota variacional
(ELBO): NLL de reconstrucción + KL(q(Z|X) || N(0,I)); para el decodificador
lineal la señal de estructura proviene de empujar los residuos (I - A^T)X hacia
ruido independiente de mínima varianza.

Restricción de aciclicidad diferenciable (DAG-GNN):
    h(A) = tr[(I + (1/d) A∘A)^d] - d = 0,
optimizada con Lagrangiano aumentado (multiplicador alpha, penalización rho).

Las MLPs no lineales aportan la "flexibilidad" cuyo beneficio real evaluamos.
"""
import numpy as np
import torch
import torch.nn as nn


class NodeMLP(nn.Module):
    """MLP compartido aplicado independientemente a la característica de cada nodo."""

    def __init__(self, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):  # x: (n, d)
        n, d = x.shape
        return self.net(x.reshape(-1, 1)).reshape(n, d)


class DAGGNN(nn.Module):
    def __init__(self, d, hidden=16, nonlinear=True):
        super().__init__()
        self.d = d
        self.nonlinear = nonlinear
        self.A = nn.Parameter(torch.zeros(d, d))
        self.z_logvar = nn.Parameter(torch.zeros(d))
        if nonlinear:
            self.enc = NodeMLP(hidden)
            self.dec = NodeMLP(hidden)

    def _Am(self):
        return self.A * (1.0 - torch.eye(self.d, device=self.A.device))

    def h_acyclic(self):
        A = self._Am()
        M = torch.eye(self.d, device=A.device) + (1.0 / self.d) * (A * A)
        return torch.matrix_power(M, self.d).trace() - self.d

    def forward(self, X):
        d = self.d
        Am = self._Am()
        ImAt = torch.eye(d, device=Am.device) - Am.t()
        H = self.enc(X) if self.nonlinear else X
        z_mean = H @ ImAt.t()                       # (n,d) residuos = codificación
        var = torch.exp(self.z_logvar)
        z = z_mean + torch.sqrt(var) * torch.randn_like(z_mean)
        ImAt_inv = torch.inverse(ImAt)
        Hdec = z @ ImAt_inv.t()
        X_hat = self.dec(Hdec) if self.nonlinear else Hdec
        return X_hat, z_mean, var


def train_dag_gnn(X, hidden=16, nonlinear=False, epochs=300, lr=1e-2,
                  lambda_l1=0.02, w_threshold=0.3, max_outer=8, seed=0,
                  h_tol=1e-8, verbose=False):
    """Entrena DAG-GNN con Lagrangiano aumentado. Devuelve (B, W, history)."""
    torch.manual_seed(seed)
    n, d = X.shape
    Xt = torch.tensor(X, dtype=torch.float32)
    model = DAGGNN(d, hidden=hidden, nonlinear=nonlinear)

    rho, alpha, h_val = 1.0, 0.0, np.inf
    rho_max = 1e18
    history = {"recon": [], "h": [], "elbo": []}

    for outer in range(max_outer):
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for ep in range(epochs):
            opt.zero_grad()
            X_hat, z_mean, var = model(Xt)
            recon = ((X_hat - Xt) ** 2).mean()
            # KL(q(Z|X)=N(z_mean,var) || N(0,I)) por dimensión
            kl = 0.5 * (z_mean ** 2 + var - torch.log(var + 1e-12) - 1).mean()
            h = model.h_acyclic()
            l1 = model.A.abs().sum()
            loss = recon + kl + lambda_l1 * l1 + alpha * h + 0.5 * rho * h * h
            loss.backward()
            opt.step()
            history["recon"].append(float(recon.item()))
            history["h"].append(float(h.item()))
            history["elbo"].append(float((recon + kl).item()))
        with torch.no_grad():
            h_new = float(model.h_acyclic().item())
        if verbose:
            print(f"  outer {outer}: recon={recon.item():.4f} "
                  f"h={h_new:.2e} rho={rho:.1e}")
        alpha += rho * h_new
        if h_new > 0.25 * h_val and rho < rho_max:
            rho *= 10
        h_val = h_new
        if h_val <= h_tol:
            break

    W = model._Am().detach().numpy()
    B = (np.abs(W) > w_threshold).astype(int)
    np.fill_diagonal(B, 0)
    return B, W, history
