# Dataset archives

This repository includes version 1 archives of the current training and validation image splits. Keep the two splits separate: train on the training set and use validation data to evaluate the model. These archives are provided for future training runs and are not a byte-for-byte snapshot of the data used for the published model metrics.

## Downloads

| Archive | Contents | Images | Size |
| :--- | :--- | ---: | ---: |
| [`catchtwo-dataset-100-part1-v1.zip`](datasets/catchtwo-dataset-100-part1-v1.zip) | Training split, part 1 of 2 | 36,880 | 56.7 MiB |
| [`catchtwo-dataset-100-part2-v1.zip`](datasets/catchtwo-dataset-100-part2-v1.zip) | Training split, part 2 of 2 | 37,005 | 56.8 MiB |
| [`catchtwo-dataset-validation-v1.zip`](datasets/catchtwo-dataset-validation-v1.zip) | Validation split | 17,659 | 27.5 MiB |

The training split is divided into two archives to keep each file below GitHub's 100 MiB per-file limit. Both parts are required. They contain separate class folders and can be extracted into the same `datasets` directory.

## Extract

From the repository root in PowerShell:

```powershell
New-Item -ItemType Directory -Force datasets | Out-Null
Expand-Archive datasets/catchtwo-dataset-100-part1-v1.zip -DestinationPath datasets -Force
Expand-Archive datasets/catchtwo-dataset-100-part2-v1.zip -DestinationPath datasets -Force
Expand-Archive datasets/catchtwo-dataset-validation-v1.zip -DestinationPath datasets -Force
```

This creates:

```text
datasets/
  catchtwo-dataset-100/
  catchtwo-dataset-validation/
```

Pass these two directories to the training or fine-tuning script as its training and validation inputs.

## Dataset details

Images are organized into folders by class and stored as JPEG files. Folder names that are invalid or ambiguous on Windows were normalized to the model's class names when creating the archives:

* `Mime Jr.` is named `Mime Jr`.
* `MissingNo.` is named `MissingNo`.

The training and validation splits currently have no images for `Type`. The validation split also has no images for Galarian Articuno, Galarian Zapdos, MissingNo, Regieleki, Scolipede, or Xurkitree. The model's ordered class list is available at [`model/layers/labels.json`](model/layers/labels.json).

## Checksums

SHA-256 checksums for verifying downloaded archives:

```text
6b1e6752e2e5a3c5f4648c4462cef2802238e35ddaa8832e6590b1af955c94ea  catchtwo-dataset-100-part1-v1.zip
c323624e6499ef2665cfd555f39beb82a506245bf39c9d84ccd354ed735af834  catchtwo-dataset-100-part2-v1.zip
9146732aebb0dfff7d889eb0bf04dc154ddbdb1831eb2bdd957f5b7a4a637a6c  catchtwo-dataset-validation-v1.zip
```
