# Poketwo-AI

An open model and training toolkit for identifying Pokémon in Pokétwo spawn images. The model is designed for use in different bots and autocatchers.

## Production model

| Property | Value |
| :--- | :--- |
| Input | 64x64 RGB image, values scaled to 0-1 |
| Output | 1,303 classes |
| Parameters | 1,646,551 |
| Validation accuracy (cleaned audit) | 99.8847% |
| Validation top-3 accuracy (cleaned audit) | 100% |
| Validation top-5 accuracy (cleaned audit) | 100% |

Validation figures come from the checkpoint audit on the cleaned 26,014-image split. Top-5 is 100% because top-3 is already 100%; top-2 was 99.9885%. The original 26,060-image split scored 99.7199% top-1 and 99.9156% top-5.

Model files:

* Keras checkpoint: [`model/h5/best_model.h5`](model/h5/best_model.h5)
* TensorFlow.js model: [`model/layers/`](model/layers/)
* Ordered output labels: [`model/layers/labels.json`](model/layers/labels.json)
* Training run: [`training/models/runs/20261102T141243Z-sepconv-accuracy-ft`](training/models/runs/20261102T141243Z-sepconv-accuracy-ft)

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
* [`technical_notes.md`](technical_notes.md): architecture and inference details

## Known limitations

* **The class set changed from the 1,279-class model.** This release adds 26 classes (13 Halloween 2026 forms, 8 Minior cores, and 5 alternate forms) and removes `Type` and `Beta Sentret`. Output indexes differ from the previous model, so use this repository's `labels.json` with this checkpoint.
* **Form names can be reduced to the base species.** Pokétwo accepts `Arceus` and `Silvally` without their form prefix, so a consumer can map those form predictions to the base name and still catch the spawn.
* **Finizen and Palafin are hard to separate.** The zero form and Finizen differ mainly by a small chest mark that is mostly lost at 64x64. 13 of the 30 remaining top-1 audit errors are this pair, at about 0.51 confidence. Skip or defer the catch when these two are the top predictions rather than guessing.
* **Preprocessing matters.** The model expects a 64x64 RGB image with pixel values divided by 255. Don't normalize the image or boost its saturation. For full-scene spawn screenshots, first center-crop to 0.8 times the shorter side, then resize; skip the crop for images with a shortest side of 128 px or less. Without this crop, confidence drops sharply.
* **Keep labels paired with their checkpoint.** Always use the `labels.json` from the same run as the model. The number of model outputs should match the number of labels.
* **Validation scores aren't a guarantee of live accuracy.** The shipped run reports 99.7199% top-1 and 99.9156% top-5 accuracy on the original 26,060-image validation split. A later manual cleanup removed 46 mislabeled images; on the cleaned 26,014-image set the checkpoint scored 99.8847% top-1, 99.9885% top-2, and 100% top-3. The cleaned set is held out from training, but it is a subset of the validation split used to select the shipped epoch, and the removals were guided by model failures, so the audit can still be optimistic. The published validation archive contains fresh renders in place of the removed images. Real-world results depend on framing, compression, and fallback behavior.

## Use note

Pokémon names and images belong to their respective owners. This repository is for research and for connecting a classifier to your own bot. Autocatchers may violate Discord's or Pokétwo's rules; use this project at your own risk.

## License

See [LICENSE](LICENSE) for the code and model weights. Same terms as [CatchTwo](https://github.com/kyan0045/catchtwo) and [Spammer](https://github.com/kyan0045/spammer). The dataset images are third-party content and are not covered by that license; see [Dataset license and usage](datasets.md#license-and-usage).
