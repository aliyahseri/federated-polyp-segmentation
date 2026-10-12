# Federated Polyp Segmentation with Frequency-Space Domain Generalization

Federated training of a TransNetR-style polyp segmentation network across six hospital centres. Each client also trains on frequency-space augmented images so the model can generalise to unseen centres. The centres never share raw images, only model weights.

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?logo=pytorch&logoColor=white)
![Federated Learning](https://img.shields.io/badge/Federated-FedAvg-4B8BBE)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## Overview

Colonoscopy images look very different from hospital to hospital: devices, lighting and colour processing all vary. A segmentation model trained at one centre often degrades at another, and privacy rules make it hard to pool the data. This project combines:

- **Federated averaging (FedAvg)** over the 6 centres of PolypGen2021 (`training.py`). Each client trains locally and the server averages the weights equally.
- **Frequency-space domain generalization**, adapted from FedDG (`freq_space_interpolation.py`). For each local image, the low-frequency part of its Fourier amplitude spectrum (a centred window of size `L = 0.1`) is interpolated with the amplitude spectrum of an image from another centre, with a random ratio λ ∈ [0, 1]. The phase is kept. The result looks like the local image with another centre's "style". In this implementation, a pool of up to 50 amplitude spectra per other centre is precomputed on each client at the start of the round. Small target amplitudes are hard-thresholded (`threshold.py`).
- **A TransNetR-style segmentation network** (`model.py`). The ResNet-50 encoder (ImageNet weights) is used up to `layer3`. 1×1 convolutions reduce each skip connection to 64 channels, and the decoder uses two residual transformer blocks and a residual block. It outputs a 1-channel 256×256 logit map.
- **Out-of-distribution evaluation** on Kvasir-SEG (`test_transnetr_fdg.py`), a dataset the model never sees during training. It reports Dice, IoU, precision, recall, F2 and a Hausdorff distance, and saves predicted masks and overlays.
- **An FFT visualization demo** (`main.py`) that saves a source image, a target image and the frequency-interpolated result.

### Training procedure (as implemented)

| Setting | Value |
|---|---|
| Clients | 6 (PolypGen centres C1–C6) |
| Rounds | 6 |
| Aggregation | FedAvg with equal weights (1/6) |
| Local work per round | 1 epoch on frequency-augmented images, then 1 epoch on real images |
| Optimizer / loss | Adam (lr 1e-4, re-created each round) / `BCEWithLogitsLoss` |
| Batch size | 1 |
| Input size | 256×256, scaled to [0, 1] |
| Device | hardcoded to `"cpu"` in `training.py` (change it to `"cuda"` to use a GPU) |

## Tech stack

- **PyTorch / torchvision**: model, training loop, `transforms.v2` (needs torchvision ≥ 0.16), `torchvision.io` image reading.
- **`torch.fft`**: amplitude/phase decomposition for the frequency-space augmentation.
- **ResNet-50** encoder from a vendored copy of torchvision's `resnet.py`, with ImageNet weights downloaded through `torch.hub`.
- **`nn.TransformerEncoder`** inside the decoder blocks, following TransNetR's residual transformer design.
- **SciPy** (`directed_hausdorff`) for the Hausdorff metric, and **Pillow / NumPy** for saving predictions.
- No federated-learning framework is used. Clients are simulated sequentially in one process, which keeps the FedAvg logic explicit and easy to read (`federated_aggregate`).

## Demo / Screenshots

<!-- TODO: add a screenshot at docs/screenshot.png -->
![Screenshot](docs/screenshot.png)

*To do for the owner: run `python main.py` (or the test script) and combine `visualize_output/source_new.jpg`, `target_new.jpg` and `result_new.jpg` (or a few images from `predictions/overlays/`) into one figure. Save it as `docs/screenshot.png`.*

## Dataset

**No datasets or model weights are included in this repository.** Download them yourself and place them in the repository root, as shown below.

### Training: PolypGen2021 (multi-centre)

- **What it is**: colonoscopy frames with pixel-level polyp masks, collected at 6 centres in different countries. It was built to test how well models generalise across centres.
- **Source**: S. Ali et al., *A multi-centre polyp detection and segmentation dataset for generalisability assessment*, Scientific Data 10, 75 (2023). It is distributed through Synapse. Check the dataset page for access and license terms. [TODO: confirm: download link and license used]
- **Format expected by `data_loader.PolyGen`**: JPEG images, with masks named `<image name>_mask.jpg`. Brackets and parentheses are stripped from the image name when building the mask name.

```
PolypGen2021_MultiCenterData_v3/
├── data_C1/
│   ├── images_C1/   <name>.jpg
│   └── masks_C1/    <name>_mask.jpg
├── data_C2/
│   ├── images_C2/
│   └── masks_C2/
...
└── data_C6/
    ├── images_C6/
    └── masks_C6/
```

### Testing: Kvasir-SEG

- **What it is**: 1,000 polyp images from colonoscopy, each with a segmentation mask.
- **Source**: D. Jha et al., *Kvasir-SEG: A Segmented Polyp Dataset*, MultiMedia Modeling (MMM) 2020. Available at <https://datasets.simula.no/kvasir-seg/>; see that page for terms of use.
- **Format expected by `test_transnetr_fdg.py`**: images and masks are paired by **sorted file order**, so the file names must match.

```
Kvasir-SEG/
├── images/   *.jpg
└── masks/    *.jpg
```

### Model weights

`training.py` writes the global model after every round to `saved_models/global_round_N.pth`, and the final one to `saved_models/global_final.pth` (plain `state_dict`s). `test_transnetr_fdg.py` loads `saved_models/global_final.pth`. None of these are included; they come from running training. Training also writes a few synthetic (frequency-augmented) previews per client and round to `synthetic_vis/`.

## Installation

```bash
git clone https://github.com/aliyahseri/federated-polyp-segmentation.git
cd federated-polyp-segmentation

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

For GPU support, install a CUDA build of PyTorch from <https://pytorch.org/get-started/locally/>. Then change `device = "cpu"` in `training.py` (inside `if __name__ == "__main__":`). The test script already picks CUDA when it's available.

The first run needs internet access to download the ImageNet ResNet-50 weights through `torch.hub`.

## Usage

Run from the repository root, with the datasets placed as shown above.

```bash
# 1. Federated training on PolypGen C1–C6 (6 rounds) -> saved_models/, synthetic_vis/
python training.py

# 2. Out-of-distribution evaluation on Kvasir-SEG -> prints metrics, writes predictions/masks and predictions/overlays
python test_transnetr_fdg.py

# 3. Frequency-space interpolation demo (PolypGen C1 image styled with C3) -> visualize_output/
python main.py

# Optional: shape check of the network (prints torch.Size([4, 1, 256, 256]))
python model.py
```

None of the scripts take command-line arguments. Paths, centres, rounds and the device are set at the bottom of each script.

## Results

**No results have been recorded yet.** Earlier project notes quoted an "expected" Dice score with no supporting run or log, so that figure has been dropped. [TODO: confirm results]

To fill in after running `python test_transnetr_fdg.py`. The script prints these metrics. Don't report Hausdorff until the bug listed below is fixed.

| Train data | Test data | Dice | IoU | Precision | Recall | F2 |
|---|---|---|---|---|---|---|
| PolypGen C1–C6 (FedAvg + freq. aug.) | Kvasir-SEG | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |
| PolypGen C1–C6 (FedAvg, no freq. aug.) — baseline | Kvasir-SEG | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |

The baseline row needs a small code change (skip the synthetic epoch) and is suggested as an ablation.

## Project structure

```
federated-polyp-segmentation/
├── training.py                  # FedAvg loop + local training with frequency augmentation
├── freq_space_interpolation.py  # FFT amplitude-spectrum interpolation (FedDG-style)
├── threshold.py                 # hard / soft thresholding helpers
├── model.py                     # TransNetR-style segmentation network
├── resnet.py                    # vendored torchvision ResNet (BSD-3-Clause)
├── data_loader.py               # PolypGen (PolyGen) dataset class; unused Kvasir class
├── test_transnetr_fdg.py        # Kvasir-SEG evaluation, metrics, saved predictions
├── main.py                      # FFT interpolation visualization demo
├── requirements.txt
└── LICENSE
```

## Challenges & what I learned

- **Simulating federated learning**: FedAvg comes down to averaging `state_dict`s. Each client has to start from a fresh copy of the global weights (`copy.deepcopy`) so that clients don't leak updates into each other.
- **Domain shift in medical imaging**: applying FedDG's idea that low-frequency amplitude carries centre-specific "style" while phase carries structure. Mixing amplitudes across centres lets a client see other centres' appearance without seeing their images.
- **Working with the Fourier transform in PyTorch**: `fftshift` to centre the low frequencies, choosing the window size `L`, and recombining amplitude and phase with `torch.exp(1j * phase)`.
- **Messy real-world data**: PolypGen file names contain brackets that don't appear in the mask names, so the loader rebuilds mask names explicitly.
- **Evaluating on an unseen dataset** (Kvasir-SEG) to measure generalisation rather than in-distribution fit. [TODO: confirm]

## Known issues & limitations

- **Hausdorff distance is wrong** (`test_transnetr_fdg.py`, `hausdorff_distance`): masks are flattened, so the "points" are 1-D pixel indices rather than (row, col) coordinates. It also returns `0.0` (a perfect score) when the prediction is empty.
- **Training masks are not binarized** (`data_loader.PolyGen` + `training.py`): JPEG masks are resized bilinearly together with the image and used as soft targets. Only the Kvasir test loader thresholds its masks.
- **Transformer reshape only works at 256×256** (`model.py`, `residual_transformer_block`): `d_model` is fixed to the patch count at that size (64 and 256). Because `batch_first` isn't set, attention runs over the **batch** axis. With batch size 1, each block attends over a single token.
- **`layer4` is built but never used** (`model.py`, `Model.__init__`). The encoder stops at `layer3`, but `layer4` weights are still allocated and saved.
- **Pretrained weights reloaded on every `Model()` call** (`resnet.py`, `resnet50(pretrained=True)`): training creates a new `Model()` per client per round (37 in a default run), and each one reloads ImageNet weights from the torch.hub cache. The first call needs network access. The test script also loads them before overwriting with the checkpoint.
- **Device hardcoded to CPU** (`training.py`, line ~173): `device = "cpu"` overrides CUDA auto-detection.
- **Unused `Kvasir` class with an absolute default path** (`data_loader.py`, `root='/dataset/Kvasir-SEG/'`). The test script uses its own `KvasirSEG` class instead.
- **`soft_threshold` is incorrect and unused** (`threshold.py`): `copysign(|x| - T, x)` doesn't clamp at zero, so values below `T` flip sign instead of becoming 0.
- **No seeds and no validation split**: runs aren't reproducible, there's no model selection, and the final round is simply saved.
- **Batch size 1 with BatchNorm**: BatchNorm statistics are computed per image during training.
- **Synthetic images aren't clipped** to [0, 1] after the inverse FFT (`freq_space_interpolation.py`).
- **Hardcoded paths and no CLI**: dataset roots, checkpoint path and output folders are set in code. Prediction files are saved as `<name>.jpg.png`.
- **Unused import**: `resnet34` in `model.py`.

## Future improvements

- Fix the Hausdorff computation (2-D coordinates; handle empty masks explicitly) and binarize the training masks.
- Add argparse for paths, rounds, device and the augmentation ratio, plus fixed seeds.
- Hold out one PolypGen centre for leave-one-centre-out evaluation, as in the FedDG/PolypGen protocols.
- Run the ablation without frequency augmentation and fill in the results table.
- Use `batch_first=True` and patch embeddings that work at any input size. Drop the unused `layer4`.
- Load the ImageNet weights once and reuse them across clients. Support GPU and batched local training.
- Weight FedAvg by client dataset size and add FedProx or episodic meta-learning (the full FedDG-ELCFS method).

## Credits / Acknowledgements

- **TransNetR**: D. Jha, N. K. Tomar, V. Sharma, U. Bagci. *TransNetR: Transformer-based Residual Network for Polyp Segmentation with Multi-Center Out-of-Distribution Testing*, 2023. Code: <https://github.com/DebeshJha/TransNetR>. The model architecture in `model.py` is based on it. [TODO: confirm upstream license]
- **FedDG**: Q. Liu, C. Chen, J. Qin, Q. Dou, P.-A. Heng. *FedDG: Federated Domain Generalization on Medical Image Segmentation via Episodic Learning in Continuous Frequency Space*, CVPR 2021. Code: <https://github.com/liuquande/FedDG-ELCFS>. The frequency-space amplitude interpolation in `freq_space_interpolation.py` is adapted from it. [TODO: confirm upstream license]
- **torchvision**: `resnet.py` is adapted from `torchvision/models/resnet.py`, BSD 3-Clause License, Copyright (c) Soumith Chintala 2016. Its original notice applies to that file.
- **PolypGen**: S. Ali et al., *A multi-centre polyp detection and segmentation dataset for generalisability assessment*, Scientific Data, 2023.
- **Kvasir-SEG**: D. Jha, P. H. Smedsrud, M. A. Riegler, P. Halvorsen, T. de Lange, D. Johansen, H. D. Johansen. *Kvasir-SEG: A Segmented Polyp Dataset*, MMM 2020.
- **FedAvg**: B. McMahan et al., *Communication-Efficient Learning of Deep Networks from Decentralized Data*, AISTATS 2017.

## License

MIT, see [LICENSE](LICENSE). `resnet.py` remains under torchvision's BSD 3-Clause License. Datasets keep their own terms.

## Author

**Aliyah Seri**

- GitHub: [`aliyahseri`](https://github.com/aliyahseri)
