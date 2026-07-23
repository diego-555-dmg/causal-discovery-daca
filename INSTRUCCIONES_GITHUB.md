# Publicar este repositorio en GitHub

El repositorio git ya está inicializado y con su commit. Solo falta crear el
repo remoto y hacer push desde tu máquina.

## Opción A — con GitHub CLI (`gh`)

```bash
cd "C:\Users\gdias\OneDrive\Escritorio\Jorge Guevara\Jorge Guevara\causal-discovery-daca\entrega-github"
gh auth login                      # solo la primera vez
gh repo create causal-discovery-daca --public --source . --push
```

(Usa `--private` en lugar de `--public` si lo prefieres privado.)

## Opción B — manual

1. En github.com crea un repositorio vacío llamado `causal-discovery-daca`
   (sin README ni .gitignore iniciales).
2. Luego:

```bash
cd "C:\Users\gdias\OneDrive\Escritorio\Jorge Guevara\Jorge Guevara\causal-discovery-daca\entrega-github"
git remote add origin https://github.com/TU_USUARIO/causal-discovery-daca.git
git branch -M main
git push -u origin main
```

Si te pide credenciales, usa tu usuario y un **Personal Access Token**
(github.com → Settings → Developer settings → Tokens) como contraseña.

## Verificación local antes del push (recomendado)

```bash
pip install -r requirements.txt
python tests/test_smoke.py
```

Los 8 tests deben terminar en PASS con torch instalado (en la verificación en un
entorno Linux limpio fueron 7 PASS + 1 SKIP porque faltaba torch).

> Nota: usa el ZIP `causal-discovery-daca.zip` como copia de respaldo del
> proyecto completo; descomprímelo donde quieras si esta carpeta se pierde por
> la sincronización de OneDrive.
