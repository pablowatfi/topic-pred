# topic-pred

A minimal Python starter project for topic prediction (placeholder implementation).

What is included
- `requirements.txt` – runtime dependencies to install (minimal)
- `src/topic_pred` – package with a `predict_topic` function and a small CLI
- `tests/` – a tiny pytest test to validate the package import and behavior

Quick start

1. Create a virtual environment and install dependencies (python <=3.12):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```


Notes
- The prediction function is a placeholder. Replace it with real model code when ready.

Quick prediction (using saved notebook artifacts)

If you ran the notebook and saved artifacts (e.g. `topic_predictor_model.pkl` or `cluster_artifacts.pkl`), you can run a quick prediction:

```bash
# from project root
PYTHONPATH=src python bin/predict_text.py "Some example text to predict"

# or specify artifact explicitly
PYTHONPATH=src python bin/predict_text.py "Some text" --artifact topic_predictor_model.pkl
```

If artifacts are not present the package will fall back to a tiny placeholder predictor.

Compatibility note
------------------

Some binary packages (notably PyTorch) are compiled against specific NumPy ABI versions. If you see errors like

```
RuntimeError: Numpy is not available
```

or messages about a module compiled with NumPy 1.x not running with NumPy 2.x, pin NumPy to a 1.x release before installing other packages:

```bash
# from project root
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "numpy<2,>=1.21"
python -m pip install -r requirements.txt
```

This ensures binary compatibility between PyTorch and NumPy. The `requirements.txt` in this repository already pins NumPy to `<2`.
