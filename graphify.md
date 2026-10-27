# Graphify Guide (refreshed 2026-10-27)

This repository is primarily a Python machine-learning project with a React frontend. It is not a JVM project, so Java/Kotlin/Scala/Groovy knowledge-graph tooling does not apply.

## Project areas

- Python backend: `backend/`
- Python models: `models/`
- Python evaluation code: `evaluation/`
- Local Python package: `efficient_kan/`
- React frontend: `frontend/src/`
- Jupyter notebooks: `notebooks/`
- Trained checkpoints: `weights/`
- Generated adversarial/restored images: `adversarial_outputs/`

## Current generated graphs

The latest source-level DOT files are available at the repository root:

- `python-dependencies.dot` — local Python imports across `backend/`, `models/`, `evaluation/`, and `efficient_kan/`
- `frontend-dependencies.dot` — local React imports under `frontend/src/`

They intentionally exclude `.venv/`, `.venv-gpu/`, `frontend/node_modules/`, `frontend/dist/`, datasets, and model weights.

Graphviz can render them when installed:

```powershell
dot -Tsvg python-dependencies.dot -o python-dependencies.svg
dot -Tsvg frontend-dependencies.dot -o frontend-dependencies.svg
```

## Reproducible Python dependency graph

From the repository root (`D:\mini phase2`), install the Python graph tool:

```powershell
python -m pip install pydeps
```

Generate and display a dependency graph:

```powershell
pydeps backend models evaluation efficient_kan --show-deps --max-bacon 3
```

Generate a Graphviz DOT file:

```powershell
pydeps backend models evaluation efficient_kan `
  --show-deps `
  --max-bacon 3 `
  --dot `
  --output python-dependencies.dot
```

The `--max-bacon 3` option limits the graph depth so the result remains readable.

If `pydeps` is unavailable, the checked-in `python-dependencies.dot` is a dependency
graph generated from Python AST imports and is safe to render directly.

## React dependency graph

Generate a DOT graph for the frontend:

```powershell
npx madge --dot frontend/src > frontend-dependencies.dot
```

To generate an image directly:

```powershell
npx madge --image frontend-dependencies.svg frontend/src
```

The frontend entry points are `frontend/src/main.jsx` and `frontend/src/App.jsx`.

## Rendering DOT files

Graphviz is required to convert DOT files to SVG or PNG. Verify whether it is installed:

```powershell
dot -V
```

If available, render the Python graph with:

```powershell
dot -Tsvg python-dependencies.dot -o python-dependencies.svg
dot -Tpng python-dependencies.dot -o python-dependencies.png
```

Render the React graph with:

```powershell
dot -Tsvg frontend-dependencies.dot -o frontend-dependencies.svg
dot -Tpng frontend-dependencies.dot -o frontend-dependencies.png
```

If `dot` is not recognized, install Graphviz before running the rendering commands.

## Expected high-level structure

```text
frontend
   |
   v
backend/main.py
   |
   +--> six MNIST classifiers
   |      +--> ANN
   |      +--> CNN
   |      +--> LSTM / BiLSTM
   |      +--> KAN (B-spline classifier)
   |      +--> ViT
   |
   +--> backend/restoration.py
   |      +--> cubic B-spline projection
   |      +--> Mexican Hat wavelet
   |      +--> Morlet wavelet
   |      +--> learned residual denoiser
   |
   +--> backend/wgan_restoration.py
   |      +--> WGAN-GP restoration generator
   |      +--> restoration critic
   |
   +--> backend/train_wgan_restorer.py
   |      +--> GPU WGAN-GP training
   |      +--> weights/wgan_restorer.pth
   |
   +--> adversarial attacks
   |      +--> universal_fgsm.py
   |      +--> universal_pgd.py
   |      +--> gan_attack.py
   |
   +--> evaluation/
   |      +--> adversarial_pipeline.py (CIFAR-10)
   |      +--> generate_final_graphs.py
   |
   +--> backend/train_restorer.py
   |      +--> multi-classifier adaptive training
   |
   +--> backend/evaluate_restorer.py
   |      +--> per-model accuracy / ASR / clean-drop table
   |
   +--> evaluation/comparison_chart.py
          +--> B-spline / WGAN / combined CSV and PNG comparison
```

## Notes

- Keep generated graph files outside source directories when possible.
- Do not point destructive graph-generation scripts at the repository root or an artifacts directory unless the tool is known to be non-destructive.
- The root `package-lock.json` does not describe the frontend dependencies; use `frontend/package.json` when working with the React application.
- A Java knowledge graph should be skipped for this repository because it contains no Java, Kotlin, Scala, or Groovy source files.
