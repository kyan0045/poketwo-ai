# Dataset archives

The current archives are the 1,303-class training and validation splits. The training archives produced the shipped model; the validation archive was rebuilt after a cleanup of mislabeled images. The archives for the earlier 1,279-class model are in the `20260723T122145Z-sepconv-1279-real-ft` archive branch.

Keep training and validation splits separate: use the training split to fit a model and the validation split to evaluate it.

## 1,303-class archives

| Archive | Contents | Images | Classes | Size |
| :--- | :--- | ---: | ---: | ---: |
| [`catchtwo-dataset-1303-part1.zip`](datasets/catchtwo-dataset-1303-part1.zip) | Training, part 1 of 3 | 56,701 | 567 | 93.1 MB |
| [`catchtwo-dataset-1303-part2.zip`](datasets/catchtwo-dataset-1303-part2.zip) | Training, part 2 of 3 | 56,667 | 566 | 92.8 MB |
| [`catchtwo-dataset-1303-part3.zip`](datasets/catchtwo-dataset-1303-part3.zip) | Training, part 3 of 3 | 17,006 | 170 | 27.4 MB |
| [`catchtwo-dataset-validation-1303.zip`](datasets/catchtwo-dataset-validation-1303.zip) | Validation | 26,060 | 1,303 | 43.1 MB |

- Training total: 130,374 images across 1,303 classes.
- Validation total: 26,060 images, exactly 20 per class.
- The validation archive was rebuilt after a manual cleanup. The 46 mislabeled images removed from the original split were replaced with fresh renders, so the archive is not byte-identical to the split used to select the checkpoint.
- Every class has at least 100 training images. Six classes have more.
- Every image is a 64x64 RGB JPEG at quality 80.

The ordered class list is in [`training/labels-1303.json`](training/labels-1303.json). It matches the labels in [`model/layers/labels.json`](model/layers/labels.json).

See [`datasets/catchtwo-dataset-1303.md`](datasets/catchtwo-dataset-1303.md) for class-list differences, training details, and notes on the validation data.

### Extract

The three training archives are separate pieces of one folder and must all be extracted into the same directory.

PowerShell, from the repository root:

```powershell
New-Item -ItemType Directory -Force datasets | Out-Null
Expand-Archive datasets/catchtwo-dataset-1303-part1.zip -DestinationPath datasets -Force
Expand-Archive datasets/catchtwo-dataset-1303-part2.zip -DestinationPath datasets -Force
Expand-Archive datasets/catchtwo-dataset-1303-part3.zip -DestinationPath datasets -Force
Expand-Archive datasets/catchtwo-dataset-validation-1303.zip -DestinationPath datasets -Force
```

<details>
<summary>Linux / macOS equivalent</summary>

```bash
mkdir -p datasets
unzip -o datasets/catchtwo-dataset-1303-part1.zip -d datasets
unzip -o datasets/catchtwo-dataset-1303-part2.zip -d datasets
unzip -o datasets/catchtwo-dataset-1303-part3.zip -d datasets
unzip -o datasets/catchtwo-dataset-validation-1303.zip -d datasets
```

</details>

This creates:

```text
datasets/
  catchtwo-dataset-1303/
  catchtwo-dataset-validation-1303/
```

The training scripts use these directories by default.

### Checksums

```text
af1a9e5627e3fefa40923499464e2fb21c2a3865cde4d3a4fcbfc09055cb28a2  catchtwo-dataset-1303-part1.zip
fbde9e71ea3b15fd4529b614fe0432df90663bf4afd70f26eeaccea128b4e9c8  catchtwo-dataset-1303-part2.zip
159d3cc7783525951e15684536cffedb34d27eddff841cfd766ca7709ccb575b  catchtwo-dataset-1303-part3.zip
a3fbd66f6b93d02fbb1be50f730877dee08458d0e8153f3685b8caf65df9812b  catchtwo-dataset-validation-1303.zip
```

## 1,279-class archives (earlier runs)

The archives for the earlier 1,279-class model are in the `20260723T122145Z-sepconv-1279-real-ft` archive branch, together with that model's checkpoint, TensorFlow.js files, and ordered labels.

## License and usage

The archives contain the original Pokémon spawn images as saved from Discord. The Pokémon characters and artwork are owned by Nintendo and other rights holders; the images originate from Pokétwo/Discord content. This repository does not own the images and cannot grant a license over them. The images are included for non-commercial research and educational use only. You are responsible for complying with the rights holders' terms and with Discord's and Pokétwo's terms of service.

The class labels, folder structure, and other curation metadata are released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). This license does not grant any rights to the underlying images.

Further redistribution of the images requires permission from the rights holders. For takedown requests, contact [@kyan0045](https://github.com/kyan0045) or open an issue in this repository.
