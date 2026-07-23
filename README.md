# ¿Descubrimiento causal neuronal a prueba: ¿Ventajas reales del DAG-GNN frente a baselines no neuronales?

Evaluación crítica del descubrimiento causal por optimización continua y redes
neuronales (**DAG-GNN**) frente a baselines no neuronales (**NOTEARS, PC, GES**) y
al baseline trivial **sortnregress**, con foco en la *varsortability* (Reisach et
al., NeurIPS 2021). El código parte del repositorio
[`Scriddie/Varsortability`](https://github.com/Scriddie/Varsortability) y lo
mejora y amplía para responder: *¿bajo qué condiciones de los datos la
flexibilidad de DAG-GNN produce mejoras reales sobre baselines no neuronales?*

Proyecto Final — *Introducción a Deep Learning* (Doctorado).
Autor: **Diego Alonso Córdova Ayala** · `diego.cordova@dmg-pe.com`

---

## Resultado principal

En datos crudos con alta *varsortability* (≈1), NOTEARS, DAG-GNN y hasta el
baseline trivial sortnregress orientan casi perfectamente. Al **estandarizar**
(varsortability = 0.5), su orientación cae a azar (≈0.5), mientras **PC y GES,
invariantes a escala, mantienen su desempeño y pasan a ser los mejores**. La
ventaja neuronal es, en estos *benchmarks*, en gran parte **instrumental**
(dependiente de la escala), no una mejor recuperación de la estructura causal.

| Escala | Método | vs | SHD ↓ | F1 esqueleto | Orientación |
|---|---|:--:|:--:|:--:|:--:|
| crudo | sortnregress | 1.0 | **0.22** | 0.99 | **1.00** |
| crudo | NOTEARS | 1.0 | 0.44 | 0.99 | 0.99 |
| crudo | DAG-GNN | 1.0 | 5.33 | 0.78 | 0.98 |
| crudo | PC | 1.0 | 6.89 | 0.97 | 0.76 |
| crudo | GES | 1.0 | 7.78 | 0.93 | 0.73 |
| estand. | sortnregress | 0.5 | 17.56 | 0.73 | 0.53 |
| estand. | NOTEARS | 0.5 | 11.11 | 0.75 | 0.50 |
| estand. | DAG-GNN | 0.5 | 13.00 | 0.20 | 0.80* |
| estand. | **PC** | 0.5 | **6.89** | **0.97** | **0.76** |
| estand. | GES | 0.5 | 7.78 | 0.93 | 0.73 |

\* calculada sobre las pocas aristas que DAG-GNN conserva; no es indicativa.

### Validación en datos reales (Sachs et al., 2005)

Como control externo, los mismos métodos se aplican a la red de señalización de
proteínas de **Sachs** (datos reales, condición observacional, `d=11`, *ground
truth* de consenso de 17 aristas). Resultado clave: la **varsortability real es
0.59** (no ≈1 como en los sintéticos), así que la ventaja instrumental
desaparece y PC/GES son idénticos en crudo y estandarizado.

| Escala | Método | SHD ↓ | F1 esqueleto | Orientación |
|---|---|:--:|:--:|:--:|
| crudo | sortnregress | 14 | 0.56 | 0.57 |
| crudo | NOTEARS | 19 | 0.67 | 0.54 |
| crudo | PC | 17 | 0.62 | 0.38 |
| crudo | GES | 17 | 0.58 | 0.36 |
| estand. | NOTEARS | 15 | 0.38 | 0.50 |
| estand. | **PC** | **17** | **0.62** | **0.38** |
| estand. | GES | 17 | 0.58 | 0.36 |

Ningún método recupera bien la red (SHD 14–19 sobre 17 aristas): la superioridad
de los métodos continuos/neuronales en datos simulados **no se transfiere** a
datos reales. Reproducir con `python src/run_sachs.py` (DAG-GNN se incluye si hay
PyTorch instalado).

---

## Estructura del repositorio

```
.
├── src/
│   ├── utils.py            # semillas, aciclicidad, estandarización, umbral
│   ├── data_gen.py         # DAGs ER/SF y SEM lineal ANM con varsortability controlada
│   ├── varsortability.py   # varsortability + sortnregress (con corrección de fuga numérica)
│   ├── baselines.py        # NOTEARS (numpy/scipy) + PC y GES (causal-learn)
│   ├── dag_gnn.py          # DAG-GNN (VAE en grafo + Lagrangiano aumentado, PyTorch)
│   ├── metrics.py          # SHD, F1 de esqueleto, precisión de orientación
│   ├── run_experiment.py   # driver resumible (checkpoints por condición)
│   ├── sachs_data.py       # datos REALES de Sachs (2005) + ground truth de consenso
│   ├── run_sachs.py        # experimento complementario sobre datos reales
│   └── make_figures.py     # figuras del paper
├── data/
│   └── sachs_cd3cd28.tsv   # datos reales de Sachs (observacional), empaquetados
├── tests/
│   └── test_smoke.py       # test de humo del pipeline completo
├── reproducir_paper.py     # script único y autocontenido (todo el pipeline)
├── results/                # CSV de resultados y curvas de entrenamiento
├── figures/                # figuras PNG del paper
├── paper/                  # paper.tex y paper.pdf
├── docs/                   # documento de parafraseo y paper final en Word
├── requirements.txt
└── README.md
```

---

## Instalación

Probado con **Python 3.10**. Para CPU basta:

```bash
python -m venv .venv
# Windows:            .venv\Scripts\activate
# Linux/macOS:        source .venv/bin/activate
pip install -r requirements.txt
```

(Para GPU, instalar la rueda de `torch` correspondiente a tu versión de CUDA.)

---

## Instrucciones de ejecución

**1. Test de humo (~1–2 min).** Verifica que todo el pipeline funciona:

```bash
python tests/test_smoke.py
```

**2. Demo rápida (~5 min, CPU):**

```bash
cd src
python run_experiment.py --config quick
```

**3. Corrida completa del paper (~30–45 min, CPU):**

```bash
cd src
python run_experiment.py --config full
python make_figures.py
```

**4. Experimento con datos reales de Sachs (~1 min):**

```bash
cd src
python run_sachs.py            # datos empaquetados (offline, reproducible)
python run_sachs.py --source hf # opcional: descarga desde HuggingFace Hub
```

Los resultados se guardan en `results/`. El driver del experimento sintético es
**resumible**: si se interrumpe, al reejecutarlo salta las condiciones ya
calculadas.

> Nota: si cambias el código de un método, borra sus filas de
> `results/results_full.csv` (o el CSV completo) antes de reejecutar; de lo
> contrario el driver conservará los valores antiguos.

**Alternativa — archivo único autocontenido** (no requiere el paquete `src/`):

```bash
python reproducir_paper.py --config quick   # demo
python reproducir_paper.py                  # corrida completa
```

**Compilar el paper:**

```bash
cd paper
pdflatex paper.tex && pdflatex paper.tex
```

---

## Reproducibilidad

Semillas fijas, hiperparámetros completos, hardware y versiones exactas de
librerías están documentados en **[REPRODUCIBILIDAD.md](REPRODUCIBILIDAD.md)**.
En resumen: 18 condiciones sintéticas (3 políticas × 2 escalas × semillas 0–2) +
experimento real de Sachs, toda la aleatoriedad sembrada, corrida en CPU x86-64
de 2 núcleos sin GPU, Python 3.10 / PyTorch 1.13.1 / NumPy 1.26 / causal-learn.

---

## Verificación realizada (jul 2026)

- Los promedios de `results/results_full.csv` reproducen la tabla del paper.
- NOTEARS, PC, GES y sortnregress fueron reejecutados semilla por semilla y
  coinciden con las filas almacenadas.
- Se corrigieron 3 filas históricas de sortnregress (`increasing/std`) previas a
  la corrección de la fuga numérica; tabla y cifras del paper actualizadas
  (SHD 11.89 → 17.56, F1 0.82 → 0.73, orientación 0.68 → 0.53).
- Experimento real de Sachs añadido y verificado (varsortability 0.59;
  PC/GES idénticos raw/std).
- `tests/test_smoke.py`: 7 PASS + 1 SKIP (DAG-GNN requiere torch, presente en tu
  equipo).

## Licencia

Ver [LICENSE](LICENSE).

## Referencias

- Reisach, A. G., Seiler, C., Weichwald, S. (2021). *Beware of the Simulated
  DAG!* NeurIPS.
- Zheng, X. et al. (2018). *DAGs with NO TEARS.* NeurIPS.
- Yu, Y. et al. (2019). *DAG-GNN.* ICML.
- Sachs, K. et al. (2005). *Causal protein-signaling networks derived from
  multiparameter single-cell data.* Science 308:523–529.
