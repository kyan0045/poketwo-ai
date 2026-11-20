# Poketwo-AI

An open model and training toolkit for identifying Pokémon in Pokétwo spawn images. The model is designed for use in different bots and autocatchers.

## Production model

| Property | Value |
| :--- | :--- |
| Input | 64x64 RGB image, values scaled to 0-1 |
| Output | 1,303 classes |
| Parameters | 382,679 |
| Validation accuracy (26,060 archive) | 99.7813% |
| Validation top-5 accuracy | 100% |
| Validation accuracy, mapped catch names | 99.9271% |

Validation figures use the published 26,060-image archive. The previous release scored 99.8810% exact and 99.9463% mapped on the same archive. Mapped figures collapse Arceus, Silvally, Alcremie, Toxtricity, Dudunsparce, Maushold and Unown forms to their base catch names. See [`BENCHMARK_COMPACT.md`](BENCHMARK_COMPACT.md) for the full comparison.

Model files:

* Keras checkpoint: [`model/h5/best_model.h5`](model/h5/best_model.h5)
* TensorFlow.js model: [`model/layers/`](model/layers/)
* Ordered output labels: [`model/layers/labels.json`](model/layers/labels.json)

The previous 1,279-class release, including its checkpoint, TensorFlow.js files, and ordered labels, is on the `20260723T122145Z-sepconv-1279-real-ft` branch.

Use the predicted class index to find the matching name in the ordered `classes` array:

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

This loads `model/h5/best_model.h5`, prints the predicted name and its score, for example `Pikachu (0.9821)`. Full-scene images are center-cropped automatically, and images at or below 128 px are used as-is. Use `--checkpoint` and `--labels` to point at a different run. For use inside a bot, load the TF.js model from `model/layers/` instead.

## Train or update the model

Training and validation archives are documented in [`datasets.md`](datasets.md). After extracting the datasets and installing the Python dependencies, you can train a new model or fine-tune the published checkpoint:

```powershell
python training/train.py
python training/fine_tune.py
```

See [`training/MODEL_ARTIFACTS.md`](training/MODEL_ARTIFACTS.md) for setup, options, and TensorFlow.js export instructions.

## Repository contents

* `model/`: checkpoints, TensorFlow.js files, and ordered labels
* `datasets/`: versioned training and validation archives
* `training/`: model definition, data-loading, training, fine-tuning, and export code
* [`BENCHMARK_COMPACT.md`](BENCHMARK_COMPACT.md): compact model accuracy, calibration, and runtime comparison
* [`technical_notes.md`](technical_notes.md): architecture and inference details

## Known limitations

* **The class set changed from the 1,279-class model.** This release adds 26 classes (13 Halloween 2026 forms, 8 Minior cores, and 5 alternate forms) and removes `Type` and `Beta Sentret`. Output indexes differ from the previous model, so use this repository's `labels.json` with this checkpoint.
* **Form names can be reduced to the base species.** Pokétwo accepts `Arceus`, `Silvally`, `Alcremie`, `Toxtricity`, `Dudunsparce`, `Maushold` and every `Unown` letter under their base name, so a consumer can map those form predictions to the base name and still catch the spawn. `Unown` needs a prefix match because `"Unown A"` does not end with `"Unown"`. Confirm base-name acceptance on a live spawn before relying on it.
* **Finizen and Palafin are hard to separate.** The zero form and Finizen differ mainly by a small chest mark that is mostly lost at 64x64, and the pair remains the largest group of raw top-1 errors. Skip or defer the catch when these two are the top predictions rather than guessing.
* **Preprocessing matters.** The model expects a 64x64 RGB image with pixel values divided by 255. Don't normalize the image or boost its saturation. For full-scene spawn screenshots, first center-crop to 0.8 times the shorter side, then resize; skip the crop for images with a shortest side of 128 px or less. Without this crop, confidence drops sharply.
* **Keep labels paired with their checkpoint.** Always use the `labels.json` from the same run as the model. The number of model outputs should match the number of labels.
* **Validation scores aren't a guarantee of live accuracy.** The shipped compact run reports 99.7813% top-1 and 100% top-5 on the 26,060-image archive. With the extended catch-name mapping, top-1 is 99.9271%. The previous release scored 99.8810% exact and 99.9463% mapped. Real-world results depend on framing, compression, and fallback behavior.

## Use note

Pokémon names and images belong to their respective owners. This repository is for research and for connecting a classifier to your own bot. Autocatchers may violate Discord's or Pokétwo's rules; use this project at your own risk.

## License

See [LICENSE](LICENSE) for the code and model weights. The dataset images are third-party content and are not covered by that license; see [Dataset license and usage](datasets.md#license-and-usage).
