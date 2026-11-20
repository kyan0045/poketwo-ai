# Compact head benchmark

Comparison of the shipped 1,646,551-parameter model and the compact `GAP -> Dropout -> Dense(1303)` variant on the published 26,060-image validation archive (`catchtwo-dataset-validation-1303.zip`).

## Summary

| Metric | 1.65M baseline | Compact | Change |
| :--- | ---: | ---: | ---: |
| Parameters | 1,646,551 | 382,679 | -76.8% |
| float16 shard | ~3.2 MB | 0.75 MB | -4.2x |
| Compute per image | ~11.9M MACs | ~10.7M MACs | -10% |
| Exact top-1 | 99.8810% | 99.7813% | -0.0997 pp |
| Exact top-5 | 100% | 100% | 0 |
| Mapped top-1, current bases | 99.9309% | 99.8734% | -0.0575 pp |
| Mapped top-1, extended bases | 99.9463% | 99.9271% | -0.0192 pp |
| WASM inference | 2.317 ms | 2.096 ms | +10.5% throughput |
| WASM steady RSS | 101.0 MiB | 94.2 MiB | -6.8 MiB |
| WASM peak RSS | 111.2 MiB | 95.3 MiB | -15.9 MiB |

## Training

The trunk is warm-started from `training/models/runs/20261102T141243Z-sepconv-accuracy-ft/best.h5`. The new head is trained first with the trunk frozen, then the whole network is fine-tuned end to end.

| Stage | Epochs | Learning rate | Trunk | Augmentation |
| :--- | ---: | ---: | :--- | :--- |
| Head warm-up | 3 | 1e-3 | frozen | off |
| End-to-end | 8 | 2e-4 | trainable | off |

Batch size 128, label smoothing 0.1, Adam, `ReduceLROnPlateau` with factor 0.3 and patience 2. The best epoch was epoch 7 by validation loss. An earlier run with augmentation enabled scored lower and was rejected; augmentation is available with `--augmentation`.

Reproduce:

```bash
python training/fine_tune.py \
  --train-dir datasets/catchtwo-dataset-1303 \
  --validation-dir datasets/catchtwo-dataset-validation-1303 \
  --no-augmentation --head-epochs 3 --head-learning-rate 1e-3 \
  --epochs 8 --learning-rate 2e-4
```

## Export

```bash
python training/export.py training/models/runs/<run-id> --quantize-float16
```

The export is a single float16 shard of 765,358 bytes with 1,303 outputs. `training/export.py` strips the `SeparableConv2D` initializer keys that TensorFlow.js 4.22 rejects. The canonical `model/h5/best_model.h5` and `model/layers/` were replaced with this run. The canonical H5 is saved without optimizer state and is 1.59 MB.

Float16 quantization changed **0 top-1 decisions** in the simulated weights for both models.

## Accuracy

Exact class accuracy:

| Model | Top-1 | Top-5 |
| :--- | ---: | ---: |
| Baseline, Keras fp32 | 99.8810% | 100% |
| Baseline, Keras float16 | 99.8810% | 100% |
| Compact, Keras fp32 | 99.7813% | 100% |
| Compact, Keras float16 | 99.7813% | 100% |

Mapped catch-name accuracy:

| Mapping | Baseline top-1 | Compact top-1 | Baseline top-5 | Compact top-5 |
| :--- | ---: | ---: | ---: | ---: |
| Current (`Arceus`, `Silvally`, `Alcremie`) | 99.9309% | 99.8734% | 100% | 100% |
| Extended (adds `Toxtricity`, `Dudunsparce`, `Maushold`, prefix `Unown`) | 99.9463% | 99.9271% | 100% | 100% |


## Coverage and calibration

Coverage is the share of images whose top probability reaches the floor. Accepted accuracy is the top-1 accuracy of that share.

| Floor | Baseline coverage | Baseline accepted | Compact coverage | Compact accepted |
| ---: | ---: | ---: | ---: | ---: |
| 0.50 | 99.8196% | 99.9231% | 97.9240% | 99.9765% |
| 0.33 | 99.9847% | 99.8849% | 99.5012% | 99.8573% |
| 0.25 | 99.9847% | 99.8849% | 99.8312% | 99.8001% |
| 0.15 | 99.9923% | 99.8849% | 99.9540% | 99.7889% |
| 0.12 | 99.9923% | 99.8849% | 99.9731% | 99.7889% |

The compact head is less confident at a 0.5 floor: 97.92% coverage instead of 99.82%, with higher accepted accuracy. Lowering the floor to 0.25 restores 99.83% coverage.

Temperature scaling fitted on a 5,000-image training holdout:

| Model | NLL-optimal T | T for 99.81% coverage | Coverage at 0.5 | Accepted accuracy |
| :--- | ---: | ---: | ---: | ---: |
| Baseline | 0.650 | 1.016 | 99.8196% | 99.9231% |
| Compact | 0.548 | 0.538 | 99.8081% | 99.8347% |

At matched coverage the compact model is 0.0884 pp behind on accepted accuracy.

**Recommended deployment:** scale the compact logits by `T = 0.538` and keep the existing `0.5` floor. That gives 99.81% coverage at 99.83% accepted accuracy, matching the baseline coverage. If temperature scaling is not possible, lower the floor to `0.25` for 99.83% coverage at 99.80% accepted accuracy. Keeping `0.5` without scaling is the high-precision option: 97.92% coverage at 99.98% accepted accuracy.

## Runtime

TensorFlow.js 4.22, WASM backend, single thread, one pinned P-core, batch 1, 300 iterations, median of three runs:

| Model | ms/image | Images/sec | RPM | Steady RSS | Peak RSS |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Baseline float16 | 2.317 | 431.7 | 25,895 | 101.0 MiB | 111.2 MiB |
| Compact float16 | 2.096 | 477.2 | 28,626 | 94.2 MiB | 95.3 MiB |

These figures cover the model process only. Image download and decode dominate the end-to-end path.

## Operational errors

At matched coverage (compact floor 0.2527), the compact model makes one additional error that the baseline gets right, after the extended mapping and excluding predictions CatchTwo abstains on:

| True | Compact predicts | Note |
| :--- | :--- | :--- |
| Pokopia Ditto | Vulpix | Genuine lookalike error |

## Limitations

- Measurements are on one x86-64 machine. ARM and Termux absolute numbers will differ, though the model-to-model ratio should hold.
- The runtime table excludes the image preprocessing pipeline and network time.
