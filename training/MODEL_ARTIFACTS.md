# Training

The ordered labels define the output indexes for both training paths. Dataset archives and extraction steps are in [`../datasets.md`](../datasets.md).

## Install

Use Python 3.10. From the repository root in PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r training/requirements.txt
```

<details>
<summary>Linux / macOS equivalent</summary>

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r training/requirements.txt
```

</details>

The tested Python 3.10 setup is TensorFlow 2.10.0 + NumPy 1.23.5. NumPy is pinned below 1.24 because `tensorflowjs==3.18.0` still uses `np.object`, which NumPy 1.24 removed. The standard requirements install a single TensorFlow distribution (`tensorflow==2.10.0`) and include the converter needed by `export.py`.

For Windows DirectML, install `training/requirements-directml.txt` instead. The DirectML plugin requires `tensorflow-cpu==2.10.0`, so that environment omits `tensorflowjs` to avoid installing a second TensorFlow package. Use a separate standard environment to run `export.py`.

## Predict

```powershell
python training/predict.py path\to\image.jpg
```

Prints the predicted class name and score. Full-scene spawn images are center-cropped the same way as the runtime before resizing; already-framed images at or below 128 px are left uncropped. See `python training/predict.py --help` for `--checkpoint` and `--labels` overrides.

## Fine-tune

```powershell
python training/fine_tune.py
```

After extracting the archives as described in `datasets.md`, the script uses the included checkpoint and dataset folders by default. Override paths and training options with `--help`. The defaults match the published run: 12 epochs, batch size 128, learning rate `1e-5` reduced by `ReduceLROnPlateau` (factor `0.3`, patience `4`, minimum `1e-6`), label smoothing `0.1`, no augmentation, and no oversampling. Only the final separable convolution and the hidden Dense layer are trainable; the output Dense and BatchNormalization layers stay frozen. Pass `--augmentation` or `--oversampling` to opt in. New runs write `best.h5`, `labels.json`, `manifest.json`, and `history.json` under `training/models/runs/<run-id>/`; the published run ships the first three.

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
