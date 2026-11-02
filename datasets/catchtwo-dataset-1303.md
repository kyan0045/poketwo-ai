# 1,303-class dataset

Training and validation data for the shipped 1,303-class model. The model was trained from scratch on this split, then fine-tuned with validation accuracy as the checkpoint criterion.

## Contents

| Archive | Contents | Images | Classes | Size |
| :--- | :--- | ---: | ---: | ---: |
| `catchtwo-dataset-1303-part1.zip` | Training, part 1 of 3 | 56,701 | 567 | 93.1 MB |
| `catchtwo-dataset-1303-part2.zip` | Training, part 2 of 3 | 56,667 | 566 | 92.8 MB |
| `catchtwo-dataset-1303-part3.zip` | Training, part 3 of 3 | 17,006 | 170 | 27.4 MB |
| `catchtwo-dataset-validation-1303.zip` | Validation | 26,060 | 1,303 | 43.1 MB |

- Training total: 130,374 images across 1,303 classes.
- Validation total: 26,060 images, exactly 20 per class.
- Every class has at least 100 training images. Six classes have more.
- Every image is a 64x64 RGB JPEG at quality 80.
- The validation archive was rebuilt after a manual cleanup. The original split
  used to select the checkpoint contained 46 mislabeled images. Those were
  removed and replaced with fresh renders, so the archive still has exactly 20
  images per class but is not byte-identical to the split the shipped run
  evaluated.
- The shipped run's manifest records the directory names used at training time
  (`catchtwo-dataset-v2` and `catchtwo-dataset-validation-v2`). These archives
  contain the same images and extract to `catchtwo-dataset-1303` and
  `catchtwo-dataset-validation-1303`.

The ordered class list is in [`../training/labels-1303.json`](../training/labels-1303.json).
It matches the labels in the shipped run, and the training scripts use that
run's labels by default.

## Differences from the 1,279-class archives

- 1,303 classes instead of 1,279.
- Removed `Type` and `Beta Sentret`.
- Added 26 classes: 13 Halloween 2026 forms, 8 Minior cores, and 5 alternate
  forms (`Origin Giratina`, `Sky Shaymin`, `Origin Palkia`, `Origin Dialga`,
  `Therian Enamorus`).
- Training images increased from 73,878 to 130,374.
- Validation images increased from 17,659 to 26,060 and now cover every class.
- Removed 658 byte-identical duplicate copies from the earlier training
  archives.
- Removed the long tail. The earlier archives had several classes with 1 image
  and one with 174.
- The new images were rendered from the live image API with the same 0.8
  center crop and Lanczos resize used at inference time. Older images are
  retained, so each class blends saved Discord captures with current renders.

## Extract

The three training archives are separate pieces of one folder and must all be
extracted into the same directory.

PowerShell, from the repository root:

```powershell
New-Item -ItemType Directory -Force datasets | Out-Null
Expand-Archive datasets/catchtwo-dataset-1303-part1.zip -DestinationPath datasets -Force
Expand-Archive datasets/catchtwo-dataset-1303-part2.zip -DestinationPath datasets -Force
Expand-Archive datasets/catchtwo-dataset-1303-part3.zip -DestinationPath datasets -Force
Expand-Archive datasets/catchtwo-dataset-validation-1303.zip -DestinationPath datasets -Force
```

Linux or macOS:

```bash
mkdir -p datasets
unzip -o datasets/catchtwo-dataset-1303-part1.zip -d datasets
unzip -o datasets/catchtwo-dataset-1303-part2.zip -d datasets
unzip -o datasets/catchtwo-dataset-1303-part3.zip -d datasets
unzip -o datasets/catchtwo-dataset-validation-1303.zip -d datasets
```

This creates:

```text
datasets/
  catchtwo-dataset-1303/
  catchtwo-dataset-validation-1303/
```

## Train

From scratch with the new labels:

```bash
python training/train.py \
  --labels training/labels-1303.json \
  --train-dir datasets/catchtwo-dataset-1303 \
  --validation-dir datasets/catchtwo-dataset-validation-1303
```

The defaults use 100 epochs, oversampling to 300 images per class, and
augmentation. To reproduce the shipped checkpoint, train from scratch first,
then fine-tune that run:

```bash
python training/fine_tune.py \
  --parent-run training/models/runs/20261102T123826Z-sepconv-train
```

The fine-tune selected epoch 3 of 5 by validation accuracy.

## Checksums

```text
af1a9e5627e3fefa40923499464e2fb21c2a3865cde4d3a4fcbfc09055cb28a2  catchtwo-dataset-1303-part1.zip
fbde9e71ea3b15fd4529b614fe0432df90663bf4afd70f26eeaccea128b4e9c8  catchtwo-dataset-1303-part2.zip
159d3cc7783525951e15684536cffedb34d27eddff841cfd766ca7709ccb575b  catchtwo-dataset-1303-part3.zip
a3fbd66f6b93d02fbb1be50f730877dee08458d0e8153f3685b8caf65df9812b  catchtwo-dataset-validation-1303.zip
```

## License and usage

Same terms as the 1,279-class archives. See [Dataset license and usage](../datasets.md#license-and-usage).
The images are Pokétwo and Discord content, included for non-commercial
research. The class labels and folder structure are CC BY 4.0.
