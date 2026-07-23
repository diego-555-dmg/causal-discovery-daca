# Reproducibilidad

Este documento consolida todo lo necesario para reproducir los resultados del
paper de forma exacta: semillas, hiperparámetros, hardware y versiones de
librerías.

## 1. Semillas (aleatoriedad controlada)

Toda la aleatoriedad del pipeline está sembrada; no hay fuentes no controladas.

| Componente | Semilla | Dónde se fija |
|---|---|---|
| Grafo (ER) y SEM lineal | `seed ∈ {0, 1, 2}` por condición | `np.random.default_rng(seed)` en `run_experiment.py` |
| NumPy / PyTorch globales | `seed` de la condición | `utils.set_seed(seed)` |
| Desempates de sortnregress | `1000 + seed` | `np.random.default_rng(1000+seed)` (evita la fuga numérica en datos estandarizados) |
| Entrenamiento DAG-GNN | `seed` de la condición | `torch.manual_seed(seed)` dentro de `train_dag_gnn` |
| Grafos SF (si se usan) | derivada del RNG de la condición | `barabasi_albert_graph(..., seed=int(rng.integers(1e6)))` |
| Figuras (jitter del scatter) | `0` | `np.random.seed(0)` en `make_figures.py` |

PC y GES son deterministas dados los datos (no requieren semilla).

La corrida del paper usa 18 condiciones: 3 políticas de ruido
(`increasing`, `equal`, `random`) × 2 escalas (`raw`, `std`) × 3 semillas.

## 2. Hiperparámetros

### Generación de datos (config `full`)

| Parámetro | Valor |
|---|---|
| Nodos `d` | 12 |
| Aristas esperadas `s0` | 15 (grafo Erdős–Rényi) |
| Muestras `n` | 1000 (partición conceptual 60/20/20) |
| Ruido | Gaussiano; escalas según política |
| Pesos del SEM | \|w_ij\| ~ U(0.5, 2.0), signo aleatorio |

### Métodos

| Método | Hiperparámetros |
|---|---|
| DAG-GNN | Adam, `lr = 1e-2`; 250 iteraciones × 8 pasos de Lagrangiano aumentado; `λ_ℓ1 = 0.02`; umbral 0.3; modo lineal; `h_tol = 1e-8` |
| NOTEARS | `λ1 = 0.1`; umbral 0.3; L-BFGS-B; `h_tol = 1e-8` |
| PC | test Fisher-z, `α = 0.05` (causal-learn) |
| GES | score BIC local (causal-learn) |
| sortnregress | selección Lasso-BIC; varianzas redondeadas a 8 decimales y empates rotos al azar |

La config `quick` (demo) usa `d=8, s0=8, n=500`, 2 políticas, 2 semillas,
150 iteraciones × 6 pasos para DAG-GNN.

## 3. Hardware

| Corrida | Hardware | Tiempos aproximados |
|---|---|---|
| Corrida del paper (autor) | CPU x86-64 de 2 núcleos, sin GPU | sortnregress 0.06 s; GES 0.34 s; PC ~0.2–2.6 s; NOTEARS ~1–3 s; DAG-GNN ~5.8 s por condición |
| Verificación independiente (jul 2026) | Linux x86-64 (sandbox limpio), CPU, sin GPU | resultados idénticos fila a fila para sortnregress, NOTEARS, PC y GES |

No se requiere GPU; la corrida completa toma ~30–45 min en CPU.

## 4. Versiones de librerías

### Entorno del autor (corrida del paper)

| Librería | Versión |
|---|---|
| Python | 3.10 |
| PyTorch | 1.13.1 |
| NumPy | 1.26 |
| causal-learn | ≥ 0.1.3 |
| scikit-learn, scipy, pandas, networkx, matplotlib | ver `requirements.txt` |

### Entorno de verificación independiente (jul 2026)

| Librería | Versión |
|---|---|
| Python | 3.10.12 |
| NumPy | 2.2.6 |
| SciPy | 1.15.3 |
| pandas | 2.3.3 |
| networkx | 3.4.2 |
| scikit-learn | 1.7.2 |
| matplotlib | 3.10.9 |
| causal-learn | 0.1.4.8 |

Los métodos no neuronales reprodujeron resultados idénticos en ambos entornos.

## 5. Datos reales de Sachs (experimento complementario)

- **Fuente:** red de señalización de proteínas de Sachs et al. (2005), *Science*
  308:523–529. Condición observacional (cd3cd28), 11 variables.
- **Publicado en HuggingFace:** dataset `pgmpy/example_datasets`
  (`real/sachs/data/sachs.2005.continuous.txt`). El loader
  `src/sachs_data.py` puede descargarlo con `--source hf`.
- **Empaquetado para reproducibilidad offline:** `data/sachs_cd3cd28.tsv`
  (n = 544 observaciones reales; medias verificadas, p. ej. `pka` ≈ 597,
  `raf` ≈ 63, coherentes con el dataset original). Es el que usa
  `run_sachs.py` por defecto, de modo que los resultados del repo se
  reproducen sin red.
- **Ground truth:** red de consenso de 17 aristas dirigidas, extraída de la red
  bayesiana `bnlearn/sachs` (idéntica a la usada por NOTEARS y DAG-GNN);
  hard-codeada en `sachs_data.py` (`CONSENSUS_EDGES`).
- **Semilla:** `seed=0` para desempates de sortnregress y entrenamiento de
  DAG-GNN; PC/GES/NOTEARS son deterministas dados los datos.

## 6. Receta de reproducción exacta

```bash
pip install -r requirements.txt
python tests/test_smoke.py            # sanity check (~1-2 min)
cd src
python run_experiment.py --config full   # ~30-45 min CPU; resumible
python make_figures.py
python run_sachs.py                       # datos reales de Sachs (~1 min)
```

`results/results_full.csv` debe coincidir fila a fila con el incluido en este
repositorio (mismas semillas ⇒ mismos números para sortnregress, NOTEARS, PC y
GES; DAG-GNN puede variar en decimales entre versiones de PyTorch por el orden
de reducción en coma flotante).
