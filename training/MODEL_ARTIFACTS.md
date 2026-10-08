# Training

The ordered labels define the output indexes for both training paths. Dataset archives and extraction steps are in [`../datasets.md`](../datasets.md).

## Install

Use Python 3.10. From the repository root:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r training/requirements.txt
```

For Windows DirectML, install `training/requirements-directml.txt` instead.

## Fine-tune

```powershell
python training/fine_tune.py
```

After extracting the archives as described in `datasets.md`, the script uses the included checkpoint and dataset folders by default. Override paths and training options with `--help`. Each run writes `best.h5`, `labels.json`, `manifest.json`, and `history.json` under `training/models/runs/<run-id>/`.

## Train from scratch

```powershell
python training/train.py
```

This initializes the architecture with random weights and uses the same ordered labels. Set `--labels` to a new ordered label file when adding classes.

## Export

```powershell
python training/export.py training/models/runs/<run-id>
```

The exporter writes a TF.js Layers model and ordered `labels.json` into the run's `tfjs/` directory. Copy those files to `model/layers/` to deploy. Load the model with `tf.loadLayersModel` and map its output index through `labels.json`.
