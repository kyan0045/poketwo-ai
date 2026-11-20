# Training

Both training scripts use the ordered labels to map model outputs to class names. See [`../datasets.md`](../datasets.md) for the dataset archives and extraction steps.

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

On Windows with DirectML, install `training/requirements-directml.txt` instead. The plugin requires `tensorflow-cpu==2.10.0`; to avoid installing a second TensorFlow package, that requirements file leaves out `tensorflowjs`. Use a separate standard environment for `export.py`.

## Predict

```powershell
python training/predict.py path\to\image.jpg
```

The script prints the predicted class name and score. It center-crops full-scene spawn images before resizing, matching the runtime preprocessing. Images with a shortest side of 128 px or less are left uncropped. Run `python training/predict.py --help` to see how to set `--checkpoint` and `--labels`.

## Fine-tune

```powershell
python training/fine_tune.py
```

After extracting the archives as described in `datasets.md`, the script builds the compact `GAP -> Dropout -> Dense(1303)` model (382,679 parameters), copies the convolutional trunk from the parent run, and fine-tunes every layer end to end. The published compact run used a two-phase schedule: three head-only epochs at `1e-3` with the trunk frozen, then eight end-to-end epochs at `2e-4` with `ReduceLROnPlateau` (factor `0.3`, patience `2`, minimum `1e-6`). Batch size is 128 and label smoothing is `0.1`. Augmentation is available with `--augmentation` but is off for the published run; the earlier augmented run scored lower on validation. Oversampling is off because the v2 split is balanced at 100 or more images per class. Override paths and training options with `--help`, including `--augmentation`, `--head-epochs`, and `--learning-rate`.

New runs write `best.h5`, `labels.json`, `manifest.json`, and `history.json` under `training/models/runs/<run-id>/`. The manifest records the architecture name, parameter count, parent checkpoint hash, and trainable layers.

## Train from scratch

```powershell
python training/train.py
```

This initializes the compact architecture with random weights and uses the same ordered labels. Set `--labels` to a new ordered label file when adding classes. Fine-tuning from the published checkpoint is faster and reproduces the published accuracy.

## Export

```powershell
python training/export.py training/models/runs/<run-id> --quantize-float16
```

The exporter writes a TF.js Layers model and ordered `labels.json` to the run's `tfjs/` directory. `--quantize-float16` stores weights as float16, which keeps the compact shard at about 0.75 MB. Without a quantization flag the exporter fails if any exported weight differs from the checkpoint, and it always checks that the output count matches the label count. The published `model/layers` files were produced with `--quantize-float16`. To deploy an export, copy the model files and `labels.json` to `model/layers/`. Load the model with `tf.loadLayersModel`, then use `labels.json` to map its output index to a class name.
