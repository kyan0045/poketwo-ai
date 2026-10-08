# Poketwo-AI Technical Notes

An image classifier and runtime pipeline for identifying Pokémon spawns in Discord autocatchers.

Dataset archives are documented in [`datasets.md`](datasets.md). Training, fine-tuning, and export instructions are in [`training/MODEL_ARTIFACTS.md`](training/MODEL_ARTIFACTS.md).

## Part 1: Design Goals and Architecture Choices

### 1. Target Platforms and Constraints
The runtime is designed to run across two main environments:
* **Desktop and servers**: Windows, Linux, and remote VPS servers.
* **Mobile**: Android devices running via Termux.

#### Running on Android / Termux
Termux does not support GPU acceleration (CUDA or DirectML) or prebuilt `@tensorflow/tfjs-node` binaries. Inference therefore relies on CPU or WebAssembly (WASM) via TensorFlow.js.

Standard classification models like ResNet or standard MobileNet take 1.5 to 4 seconds per image on mobile CPUs and consume too much RAM. To achieve sub-100ms inference on low-power devices, the model minimizes parameter count, memory footprint, and operation count.

---

### 2. Architecture Choices

```
[Raw Image]
   │
   ▼ (Preprocessed via Sharp: Normalized, Saturation boosted, Stripped Alpha)
[64x64x3 Tensor]
   │
   ▼ (Standard Conv2D: Capture raw color and edge features)
[Initial Feature Maps]
   │
   ▼ (3x SeparableConv2D Blocks: High feature depth, low parameter count)
[Aggregated Vectors]
   │
   ▼ (GlobalAveragePooling2D + Dense + Dropout: Classify with low memory footprint)
[1279 Class Probabilities]
```

#### Input Dimensions ($64 \times 64 \times 3$)
A standard $224 \times 224 \times 3$ image has 150,528 input values. Reducing inputs to $64 \times 64 \times 3$ (12,288 values) gives a 91.8% reduction in data size at the network entrance. Testing confirmed that 64x64 preserves enough color and silhouette detail to distinguish Pokémon species reliably.

#### Separable Convolutions
To reduce CPU load, the network replaces standard convolutions with depthwise separable convolutions (`SeparableConv2D`) after the first layer.

* **Standard Conv2D**: $3 \times 3 \times 32 \times 64 = 18,432$ parameters.
* **Separable Conv2D**: $(3 \times 3 \times 32) + (32 \times 64) = 2,336$ parameters.

This reduces parameters by 87.3% for this layer, lowering the model file size and the operations required per forward pass.

#### Label Smoothing
Discord spawn images often have compression artifacts, low contrast, or subtle regional differences (such as Alolan forms). Standard one-hot cross-entropy can lead to overconfidence on training data.

The model uses label smoothing (0.1). Instead of targeting 1.0 for the true class and 0.0 for others, it targets 0.9 for the correct class and distributes 0.1 across the remaining classes. This regularizes predictions and improves accuracy on real Discord images.

---

## Part 2: Model Specifications

The production model accepts 64x64 RGB images and predicts 1,279 classes. Validation accuracy is 99.451%, with 99.870% top-5 accuracy. The ordered class list is provided in [`model/layers/labels.json`](model/layers/labels.json). Each array index matches the corresponding model output index.

The network uses localized pooling and global average pooling to transition from spatial data to class predictions. It has 1,621,951 parameters.

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

---

## Part 3: Client Inference Runtime

In [CatchTwo](https://github.com/kyan0045/catchtwo), the Node.js inference pipeline uses several practical optimizations to run TensorFlow.js efficiently outside the browser.

### 1. Lazy Loading AI Dependencies
To avoid slow startup and keep memory low while idle, `@tensorflow/tfjs` and `sharp` load only when the first prediction is requested:

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

---

### 2. SeparableConv2D Deserialization Fix
The Keras-to-TFJS converter includes layer properties (`kernel_initializer`, `kernel_regularizer`, and `kernel_constraint`) that the JavaScript `SeparableConv2D` implementation rejects. Cleaning these keys from the topology in memory allows the model to load without modifying the underlying files:

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

---

### 3. In-Memory Weight Shard Loading
Reading individual weight shards directly into memory buffers avoids file locking and latency spikes on slower filesystems:

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

---

### 4. Image Preprocessing with Sharp
Using `sharp` avoids browser canvas emulation and keeps image operations fast:

* **Remove alpha channels**: `.removeAlpha()` fills transparent backgrounds with solid pixels to prevent input distortions.
* **Normalize and saturate**: `.normalize().modulate({ saturation: 1.2 })` boosts contrast and saturation to highlight subtle differences between Pokémon variants.

```javascript
async function preprocessImage(url, optimizeImage = true) {
  const { tf, sharp } = await loadAiDependencies();
  const response = await axios({
    url,
    responseType: "arraybuffer",
  });

  let sharpInstance = sharp(Buffer.from(response.data));
  if (optimizeImage) {
    sharpInstance = sharpInstance.normalize().modulate({ saturation: 1.2 });
  }

  const imageBuffer = await sharpInstance
    .resize(64, 64)
    .removeAlpha()
    .raw()
    .toBuffer();

  return tf.tidy(() => {
    const tensor = tf.tensor3d(new Uint8Array(imageBuffer), [64, 64, 3], "int32");
    const normalizedImage = tf.cast(tensor, "float32").div(tf.scalar(255.0));
    return tf.expandDims(normalizedImage, 0); // [1, 64, 64, 3]
  });
}
```

---

### 5. Memory Management
To avoid memory leaks during continuous bot operation, intermediate tensor operations run inside `tf.tidy()`, and the input tensor is manually freed with `.dispose()` after inference:

```javascript
async function predict(url) {
  const { tf } = await loadAiDependencies();
  const activeModel = await getModel();

  const startTime = Date.now();
  const imageTensor = await preprocessImage(url, false);

  const predictedIndex = tf.tidy(() => {
    const prediction = activeModel.predict(imageTensor);
    const argMaxTensor = prediction.argMax(1);
    return argMaxTensor.dataSync()[0];
  });

  // Explicit garbage collection of input tensor
  imageTensor.dispose();

  const labels = require("../model/layers/labels.json").classes;
  const name = labels[predictedIndex];

  return name;
}
```