# Poketwo-AI

An open model and training toolkit for identifying Pokémon in Pokétwo spawn images. The model is designed for use in different bots and autocatchers.

## Production model

| Property | Value |
| :--- | :--- |
| Input | 64x64 RGB image, values scaled to 0-1 |
| Output | 1,279 classes |
| Parameters | 1,621,951 |
| Validation accuracy | 99.451% |
| Validation top-5 accuracy | 99.870% |

The published model files are:

* Keras checkpoint: [`model/h5/best_model.h5`](model/h5/best_model.h5)
* TensorFlow.js model: [`model/layers/`](model/layers/)
* Ordered output labels: [`model/layers/labels.json`](model/layers/labels.json)

Use the predicted class index to look up the name in the ordered `classes` array:

```javascript
const labels = require("./model/layers/labels.json").classes;
const predictedName = labels[predictedIndex];
```

Resize input images to 64x64 RGB and scale pixel values to 0-1 before inference. See [Known limitations](#known-limitations) for the runtime framing requirements and [`technical_notes.md`](technical_notes.md) for the model architecture and TensorFlow.js details.

## Try it on one image

From the repository root:

```powershell
python training/predict.py path\to\image.jpg
```

<details>
<summary>Linux / macOS equivalent</summary>

```bash
python training/predict.py path/to/image.jpg
```

</details>

This loads `model/h5/best_model.h5`, prints the predicted name and its score, for example `Pikachu (0.9821)`. Full-scene images are center-cropped automatically; images at or below 128 px are used as-is. Use `--checkpoint` and `--labels` to point at a different run. For use inside a bot, load the TF.js model from `model/layers/` instead.

## Train or update the model

Versioned training and validation archives are documented in [`datasets.md`](datasets.md). After extracting the datasets and installing the Python dependencies, you can train a new model or fine-tune the published checkpoint:

```powershell
python training/train.py
python training/fine_tune.py
```

See [`training/MODEL_ARTIFACTS.md`](training/MODEL_ARTIFACTS.md) for setup, options, and TensorFlow.js export instructions.

## Repository contents

* `model/`: production checkpoints, TensorFlow.js files, and ordered labels
* `datasets/`: versioned training and validation archives
* `training/`: model definition, data loading, training, fine-tuning, and export code
* [`technical_notes.md`](technical_notes.md): architecture choices and inference implementation details

## Known limitations

* **Unreliable classes.** None of `Type`, `Type  Null`, `Galarian Articuno`, `Galarian Zapdos`, `MissingNo`, `Regieleki`, `Scolipede`, or `Xurkitree` appears in the validation split, so their accuracy is unmeasured. Training coverage is thin: 1 image each for Galarian Articuno, Galarian Zapdos, Regieleki, Scolipede, and Xurkitree; 7 for `Type  Null`; 30 for `MissingNo`; none for `Type`. Treat a top-1 result in this set as unknown and fall back to a hint.
* **Type and Type Null.** The run-time missing-class lists show no `Type  Null` folder and a non-empty `Type` folder, while the published archives store the Type: Null images under `Type  Null` and contain no `Type` folder. The original folder layout is not preserved, and the archives hold 73,885 training images against 73,878 used for this run. Both output indexes are unreliable; do not merge them without retraining and re-exporting the model and labels as a set.
* **Training coverage.** The published archives contain no `Type` images; `Type  Null` has 7 training and 1 validation image. Several classes have no validation images, so their accuracy is not measured.
* **Preprocessing is part of the contract.** The model expects a 64x64 RGB squash with values divided by 255, and no normalization or saturation boost. For full-scene spawn screenshots, center-crop to 0.8x the shorter side (skip images at or below 128 px) before resizing. Without the crop, confidence drops sharply.
* **Label order is fixed.** Always pair a checkpoint with the `labels.json` from the same run. Loaders should reject a model whose output count does not match the label count.
* **Validation metrics are not a live guarantee.** 99.451% top-1 and 99.870% top-5 are from the run-time validation snapshot. Re-evaluating the published checkpoint on the published 17,659-image validation archive gives 99.445% top-1 and 99.870% top-5; the archive is not byte-identical to the run-time data. Live accuracy depends on framing, compression, and fallback behavior.

## Use note

Pokemon names and images belong to their owners. This repo is for research and for connecting a classifier to your own bot code. Autocatchers can break Discord or Poketwo rules, so use this at your own risk.

## License

See [LICENSE](LICENSE) for the code and model weights. Same terms as CatchTwo and Spammer. The dataset images are third-party content and are not covered by that license; see [Dataset license and usage](datasets.md#license-and-usage).
