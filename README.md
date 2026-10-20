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

Model files:

* Keras checkpoint: [`model/h5/best_model.h5`](model/h5/best_model.h5)
* TensorFlow.js model: [`model/layers/`](model/layers/)
* Ordered output labels: [`model/layers/labels.json`](model/layers/labels.json)

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

Versioned training and validation archives are documented in [`datasets.md`](datasets.md). After extracting the datasets and installing the Python dependencies, you can train a new model or fine-tune the published checkpoint:

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

* **Some classes are untested or poorly represented.** The validation split has no images for `Type`, `Type  Null`, `Galarian Articuno`, `Galarian Zapdos`, `MissingNo`, `Regieleki`, `Scolipede`, or `Xurkitree`, so accuracy for these labels is unknown. Training data is also sparse: there is 1 image each for Galarian Articuno, Galarian Zapdos, Regieleki, Scolipede, and Xurkitree; 7 for `Type  Null`; 30 for `MissingNo`; and none for `Type`. Treat predictions for these classes as uncertain and use a hint as a fallback.
* **The `Type` and `Type  Null` labels are especially unreliable.** The run-time data had a non-empty `Type` folder and no `Type  Null` folder. The published archives instead put seven images in `Type  Null` and have no `Type` folder. The original folder layout was not preserved, and the archives contain 73,885 training images rather than the 73,878 used for the published run. Don't combine these labels without retraining and exporting the model and labels together.
* **Preprocessing matters.** The model expects a 64x64 RGB image with pixel values divided by 255. Don't normalize the image or boost its saturation. For full-scene spawn screenshots, first center-crop to 0.8 times the shorter side, then resize; skip the crop for images with a shortest side of 128 px or less. Without this crop, confidence drops sharply.
* **Keep labels paired with their checkpoint.** Always use the `labels.json` from the same run as the model. The number of model outputs should match the number of labels.
* **Validation scores aren't a guarantee of live accuracy.** The run-time validation snapshot reports 99.451% top-1 and 99.870% top-5 accuracy. On the published 17,659-image validation archive, the checkpoint gets 99.445% top-1 and 99.870% top-5. That archive isn't an exact copy of the data used for the run, and real-world results depend on framing, compression, and fallback behavior.

## Use note

Pokémon names and images belong to their respective owners. This repository is for research and for connecting a classifier to your own bot. Autocatchers may violate Discord's or Pokétwo's rules; use this project at your own risk.

## License

See [LICENSE](LICENSE) for the code and model weights. Same terms as [CatchTwo](https://github.com/kyan0045/catchtwo) and [Spammer](https://github.com/kyan0045/spammer). The dataset images are third-party content and are not covered by that license; see [Dataset license and usage](datasets.md#license-and-usage).
