# Technical notes

This document describes the image classifier and the TensorFlow.js inference code used to identify Pokémon spawns in Discord autocatchers.

Dataset archives are documented in [`datasets.md`](datasets.md). Training, fine-tuning, and export instructions are in [`training/MODEL_ARTIFACTS.md`](training/MODEL_ARTIFACTS.md).

## Design and architecture

### Target platforms and constraints
The runtime is intended for two main environments:
* **Desktop and servers**: Windows, Linux, and remote VPS servers.
* **Mobile**: Android devices running via Termux.

#### Running on Android / Termux
Termux doesn't support CUDA or DirectML, and prebuilt `@tensorflow/tfjs-node` binaries aren't available there. Inference runs on the CPU or through TensorFlow.js and WebAssembly (WASM) instead.

Larger classifiers such as ResNet and standard MobileNet can be slow and memory-hungry on mobile CPUs. This model keeps its parameter count and number of operations low, with a target of under 100 ms per image on low-power devices. The repository doesn't include device benchmarks.

### Model structure

```
[Raw Image]
   │
   ▼ (Sharp: center crop to 0.8x shorter side, resize to 64x64, strip alpha)
[64x64x3 Tensor, values scaled to 0-1]
   │
   ▼ (Conv2D: extract color and edge features)
[Initial Feature Maps]
   │
   ▼ (3 SeparableConv2D blocks)
[Aggregated Vectors]
   │
   ▼ (GlobalAveragePooling2D + Dense + Dropout)
[1279 Class Probabilities]
```

#### Input size ($64 \times 64 \times 3$)
A $224 \times 224 \times 3$ image has 150,528 input values; a $64 \times 64 \times 3$ image has 12,288, about 91.8% fewer. In testing, 64x64 retained enough color and silhouette detail for this classification task.

#### Separable Convolutions
After the first layer, the model uses depthwise separable convolutions (`SeparableConv2D`) instead of standard convolutions. This reduces the number of parameters and operations.

* **Standard Conv2D**: $3 \times 3 \times 32 \times 64 = 18,432$ parameters.
* **Separable Conv2D**: $(3 \times 3 \times 32) + (32 \times 64) = 2,336$ parameters.

This reduces parameters by 87.3% for this layer, lowering the model file size and the operations required per forward pass.

#### Label Smoothing
Discord spawn images can have compression artifacts or low contrast, and some Pokémon have subtle regional differences. With ordinary one-hot targets, a model can become overconfident on its training data.

The model uses label smoothing of 0.1: the target is 0.9 for the correct class, with the remaining 0.1 distributed across the other classes. This discourages overconfident predictions; its effect depends on the data.

## Model details

The production model takes 64x64 RGB images and predicts 1,279 classes. Its reported validation accuracy is 99.451%, with 99.870% top-5 accuracy. See the [known limitations](README.md#known-limitations) for gaps in class coverage. The ordered class list is in [`model/layers/labels.json`](model/layers/labels.json); each array position corresponds to one model output.

The network has 1,621,951 (1.6M) parameters. It uses max pooling to reduce spatial dimensions, then global average pooling and dense layers to produce class predictions.

| Layer Type | Configuration | Output Shape | Feature Maps / Params |
| :--- | :--- | :--- | :--- |
| **Input** | RGB Image Array | `(None, 64, 64, 3)` | Base spatial dimensions |
| **Conv2D** | 32 filters, 3x3 kernel, No Bias | `(None, 64, 64, 32)` | Raw edge/color extraction |
| **Batch Normalization** | Axis=3 | `(None, 64, 64, 32)` | Activation scaling |
| **Activation** | ReLU | `(None, 64, 64, 32)` | Introduce non-linearity |
| **MaxPooling2D** | 2x2 window, Stride 2 | `(None, 32, 32, 32)` | Spatial downsampling (50% reduction) |
| **SeparableConv2D** | 64 filters, 3x3 kernel, No Bias | `(None, 32, 32, 64)` | Low-FLOP depthwise/pointwise block |
| **Batch Normalization** | Axis=3 | `(None, 32, 32, 64)` | Activation scaling |
| **Activation** | ReLU | `(None, 32, 32, 64)` | Non-linearity |
| **MaxPooling2D** | 2x2 window, Stride 2 | `(None, 16, 16, 64)` | Spatial downsampling (50% reduction) |
| **SeparableConv2D** | 128 filters, 3x3 kernel, No Bias | `(None, 16, 16, 128)`| Medium-depth feature extraction |
| **Batch Normalization** | Axis=3 | `(None, 16, 16, 128)`| Activation scaling |
| **Activation** | ReLU | `(None, 16, 16, 128)`| Non-linearity |
| **MaxPooling2D** | 2x2 window, Stride 2 | `(None, 8, 8, 128)` | Spatial downsampling (50% reduction) |
| **SeparableConv2D** | 256 filters, 3x3 kernel, No Bias | `(None, 8, 8, 256)` | High-depth spatial features |
| **Batch Normalization** | Axis=3 | `(None, 8, 8, 256)` | Activation scaling |
| **Activation** | ReLU | `(None, 8, 8, 256)` | Non-linearity |
| **MaxPooling2D** | 2x2 window, Stride 2 | `(None, 4, 4, 256)` | Final spatial resolution |
| **GlobalAveragePooling2D**| None | `(None, 256)` | Flatten feature maps to a 256-D vector |
| **Dense** | 1024 units, Relu, With Bias | `(None, 1024)` | High-level classification mapping |
| **Dropout** | Rate: 0.5 | `(None, 1024)` | Mitigates memorization during training |
| **Dense (Output)** | 1279 units, Softmax | `(None, 1279)` | Output class probabilities |

## TensorFlow.js inference

The Node.js inference pipeline in [CatchTwo](https://github.com/kyan0045/catchtwo) uses the following techniques to run TensorFlow.js outside the browser.

### Lazy loading
`@tensorflow/tfjs` and `sharp` are loaded the first time a prediction is requested, rather than at startup:

```javascript
async function loadAiDependencies() {
  if (!tfLib) {
    tfLib = require("@tensorflow/tfjs");
    tfLib.enableProdMode();
  }
  if (!sharpLib) {
    sharpLib = require("sharp");
  }
  return { tf: tfLib, sharp: sharpLib };
}
```

### Loading `SeparableConv2D` layers
The Keras-to-TFJS converter writes `kernel_initializer`, `kernel_regularizer`, and `kernel_constraint` properties that the JavaScript `SeparableConv2D` implementation rejects. Removing those keys from the in-memory topology lets the model load without changing the files on disk:

```javascript
function scrubSeparableConv2D(config) {
  if (Array.isArray(config)) config.forEach(scrubSeparableConv2D);
  else if (config && typeof config === "object") {
    if (config.class_name === "SeparableConv2D" && config.config) {
      delete config.config.kernel_initializer;
      delete config.config.kernel_regularizer;
      delete config.config.kernel_constraint;
    }
    Object.values(config).forEach(scrubSeparableConv2D);
  }
}
```

### Loading weight shards
The loader reads the weight shards into memory buffers before loading the model:

```javascript
async function loadLayersModelFromDisk(modelDir) {
  const { tf } = await loadAiDependencies();
  const modelJsonPath = path.join(modelDir, "model.json");
  const modelJson = JSON.parse(await fs.promises.readFile(modelJsonPath, "utf8"));

  scrubSeparableConv2D(modelJson.modelTopology);

  const allWeightSpecs = [];
  const shardBuffers = [];

  for (const group of modelJson.weightsManifest || []) {
    if (Array.isArray(group.weights)) {
      allWeightSpecs.push(...group.weights);
    }
    for (const relPath of group.paths || []) {
      const absPath = path.join(modelDir, relPath);
      const shard = await fs.promises.readFile(absPath);
      shardBuffers.push(shard);
    }
  }

  const weightDataBuffer = Buffer.concat(shardBuffers);
  const weightData = weightDataBuffer.buffer.slice(
    weightDataBuffer.byteOffset,
    weightDataBuffer.byteOffset + weightDataBuffer.byteLength
  );

  return tf.loadLayersModel({
    load: async () => ({
      modelTopology: modelJson.modelTopology,
      format: modelJson.format,
      generatedBy: modelJson.generatedBy,
      convertedBy: modelJson.convertedBy,
      weightsManifest: modelJson.weightsManifest,
      weightSpecs: allWeightSpecs,
      weightData,
    }),
  });
}
```

### Image preprocessing
Training resizes images to 64x64 and divides pixel values by 255 (`training/data_loader.py`). The production fine-tune used no augmentation, cropping, normalization, or saturation adjustment. Live Pokétwo spawns are full-scene images, so the runtime center-crops them before resizing. Images with a shortest side of 128 px or less are left uncropped.

Do not add `.normalize()` or saturation changes. The production fine-tune disabled augmentation, and those transforms move live images away from the training distribution.

```javascript
const CENTER_CROP_MIN_SIDE = 128;
const CENTER_CROP_RATIO = 0.8;

function computeCenterCrop(width, height) {
  if (!width || !height) return null;
  const minSide = Math.min(width, height);
  if (minSide <= CENTER_CROP_MIN_SIDE) return null;

  const size = Math.round(minSide * CENTER_CROP_RATIO);
  return {
    left: Math.round((width - size) / 2),
    top: Math.round((height - size) / 2),
    width: size,
    height: size,
  };
}

async function preprocessImage(url) {
  const { tf, sharp } = await loadAiDependencies();
  const response = await axios({
    url,
    responseType: "arraybuffer",
    timeout: 8000,
    maxContentLength: 5 * 1024 * 1024,
    maxBodyLength: 5 * 1024 * 1024,
    validateStatus: (status) => status >= 200 && status < 300,
  });

  const contentType = response.headers?.["content-type"];
  if (contentType && !contentType.startsWith("image/")) {
    throw new Error(`Unexpected content-type: ${contentType}`);
  }

  let pipeline = sharp(Buffer.from(response.data));
  const metadata = await pipeline.metadata();
  const crop = computeCenterCrop(metadata.width, metadata.height);
  if (crop) pipeline = pipeline.extract(crop);

  const imageBuffer = await pipeline
    .resize(64, 64, { fit: "fill" })
    .removeAlpha()
    .raw()
    .toBuffer();

  return tf.tidy(() => {
    const tensor = tf.tensor3d(new Uint8Array(imageBuffer), [64, 64, 3], "int32");
    const scaled = tf.cast(tensor, "float32").div(tf.scalar(255.0));
    return tf.expandDims(scaled, 0); // [1, 64, 64, 3]
  });
}
```

The 0.8 crop ratio was chosen by testing on live full-scene spawns. The measurement logs aren't included in this repository. Without the crop, resizing squashes the subject and confidence drops sharply. Apply the crop when passing full-scene screenshots to the model.

### Memory management
To avoid memory leaks during continuous bot operation, intermediate tensor operations run inside `tf.tidy()`, and the input tensor is manually freed with `.dispose()` after inference:

```javascript
async function predict(url) {
  const { tf } = await loadAiDependencies();
  const activeModel = await getModel();

  const imageTensor = await preprocessImage(url);

  const predictedIndex = tf.tidy(() => {
    const prediction = activeModel.predict(imageTensor);
    const argMaxTensor = prediction.argMax(1);
    return argMaxTensor.dataSync()[0];
  });

  // Release the input tensor
  imageTensor.dispose();

  const labels = require("../data/labels.json").classes;
  const name = labels[predictedIndex];

  return name;
}
```
