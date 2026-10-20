# Dataset archives

This repository includes version 1 archives of the training and validation splits. Keep them separate: use the training split to fit a model and the validation split to evaluate it. These archives are meant for future runs; they are not an exact copy of the data used to calculate the published metrics.

## Downloads

| Archive | Contents | Images | Size |
| :--- | :--- | ---: | ---: |
| [`catchtwo-dataset-100-part1-v1.zip`](datasets/catchtwo-dataset-100-part1-v1.zip) | Training split, part 1 of 2 | 36,880 | 56.7 MiB |
| [`catchtwo-dataset-100-part2-v1.zip`](datasets/catchtwo-dataset-100-part2-v1.zip) | Training split, part 2 of 2 | 37,005 | 56.7 MiB |
| [`catchtwo-dataset-validation-v1.zip`](datasets/catchtwo-dataset-validation-v1.zip) | Validation split | 17,659 | 27.5 MiB |

The training split is split across two archives so each file stays below GitHub's 100 MiB limit. You need both parts. Each contains separate class folders; extract both into the same `datasets` directory.

## Extract

From the repository root in PowerShell:

```powershell
New-Item -ItemType Directory -Force datasets | Out-Null
Expand-Archive datasets/catchtwo-dataset-100-part1-v1.zip -DestinationPath datasets -Force
Expand-Archive datasets/catchtwo-dataset-100-part2-v1.zip -DestinationPath datasets -Force
Expand-Archive datasets/catchtwo-dataset-validation-v1.zip -DestinationPath datasets -Force
```

<details>
<summary>Linux / macOS equivalent</summary>

```bash
mkdir -p datasets
unzip -o datasets/catchtwo-dataset-100-part1-v1.zip -d datasets
unzip -o datasets/catchtwo-dataset-100-part2-v1.zip -d datasets
unzip -o datasets/catchtwo-dataset-validation-v1.zip -d datasets
```

</details>

This creates:

```text
datasets/
  catchtwo-dataset-100/
  catchtwo-dataset-validation/
```

Pass these two directories to the training or fine-tuning script as its training and validation inputs.

## Dataset details

Images are JPEGs, organized into folders by class. When preparing the archives, folder names that Windows can't handle reliably were changed to match the model's labels:

* `Mime Jr.` is named `Mime Jr`.
* `MissingNo.` is named `MissingNo`.

Neither split contains images for `Type`. The validation split also has no images for Galarian Articuno, Galarian Zapdos, MissingNo, Regieleki, Scolipede, or Xurkitree. This archive snapshot contains 7 training images and 1 validation image for `Type  Null`. In the original run-time folders, `Type  Null` had no folder of its own and `Type` was non-empty. The original folder layout wasn't preserved, so the archives don't exactly reproduce that run: they contain 73,885 training images, compared with 73,878 in the run manifest. See [`model/layers/labels.json`](model/layers/labels.json) for the ordered class list.

## License and usage

The archives contain the original Pokémon spawn images as saved from Discord. The Pokémon characters and artwork are owned by Nintendo and other rights holders; the images originate from Pokétwo/Discord content. This repository does not own the images and cannot grant a license over them. The images are included for non-commercial research and educational use only. You are responsible for complying with the rights holders' terms and with Discord's and Pokétwo's terms of service.

The class labels, folder structure, and other curation metadata are released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). This license does not grant any rights to the underlying images.

Further redistribution of the images requires permission from the rights holders. For takedown requests, contact [@kyan0045](https://github.com/kyan0045) or open an issue in this repository.

## Checksums

Use these SHA-256 checksums to verify the downloaded archives:

```text
6b1e6752e2e5a3c5f4648c4462cef2802238e35ddaa8832e6590b1af955c94ea  catchtwo-dataset-100-part1-v1.zip
c323624e6499ef2665cfd555f39beb82a506245bf39c9d84ccd354ed735af834  catchtwo-dataset-100-part2-v1.zip
9146732aebb0dfff7d889eb0bf04dc154ddbdb1831eb2bdd957f5b7a4a637a6c  catchtwo-dataset-validation-v1.zip
```
