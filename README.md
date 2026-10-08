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

Resize input images to 64x64 RGB and scale pixel values to 0-1 before inference. See [`technical_notes.md`](technical_notes.md) for the model architecture and TensorFlow.js runtime details.

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
