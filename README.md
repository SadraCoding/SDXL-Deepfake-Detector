# Spatial–Frequency Hybrid Detector for Synthetic Face Images

![Research banner](media/border.png)

**Live demo:** [sdxldd.ir](https://sdxldd.ir)

**A research implementation for binary classification of face images as `human` or `artificial`.** The project fine-tunes the Swin image-classification model [`Organika/sdxl-detector`](https://huggingface.co/Organika/sdxl-detector), combines its spatial representation with a learned representation of the input image's centered two-dimensional Fourier log-magnitude, and includes an optional confidence-based early-exit path.

> **Scope and responsible use.** This is an experimental image classifier, not an authenticity verifier. Its output is a model prediction conditioned on its training data and preprocessing. It does not establish provenance, identity, consent, or whether an image was manipulated. Do not use it as the sole basis for high-impact decisions.

## Research summary

| Item | Description |
| --- | --- |
| Task | Binary face-image classification (`artificial` / `human`) |
| Spatial backbone | Swin model from `Organika/sdxl-detector` |
| Additional representation | Centered 2D FFT log-magnitude, encoded by a compact convolutional branch |
| Fusion | Concatenated spatial and frequency embeddings; batch normalization, ReLU, dropout, and classification layer |
| Optional inference mode | Auxiliary Swin-feature classifier with a configurable confidence threshold |
| Dataset cited by the project | [xhlulu/140k-real-and-fake-faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces), downloaded separately |
| Included result artifact | [`model/results/metrics.json`](model/results/metrics.json), one run with incomplete provenance |

The research question is whether an explicit frequency representation adds useful evidence beyond spatial features for this dataset and evaluation protocol. The repository does not establish that it does: a controlled Swin-only baseline, documented split provenance, repeated runs, and external-generator tests are needed. See the [research report](REVISED_PROJECT_REPORT.md) and [protocol](docs/research_protocol.md).

## Repository map

| Path | Contents |
| --- | --- |
| [`model/scripts/custom_model.py`](model/scripts/custom_model.py) | Hybrid architecture, output/config classes, and image diagnostics |
| [`model/scripts/train.py`](model/scripts/train.py) | Fine-tuning entry point and chunked preprocessing |
| [`model/scripts/evaluate.py`](model/scripts/evaluate.py) | Clean and synthetic-corruption evaluation |
| [`model/scripts/predict.py`](model/scripts/predict.py) | Single-image inference |
| [`model/scripts/requirements.txt`](model/scripts/requirements.txt) | Python dependencies |
| `model/SDXL-Deepfake-Detector/` | Fine-tuned model and image-processor files (checkpoint stored with Git LFS) |
| `model/results/metrics.json` | Machine-readable results snapshot |
| `REVISED_PROJECT_REPORT.md` | Research and implementation report, results, limitations, and review notes |
| `docs/research_protocol.md` | Recommended protocol for new experiments and publication |

The dataset is not bundled. Create `model/dataset/` locally after reviewing the source's access and reuse terms. Local data and optional example images are excluded from Git by `.gitignore`.

## Method

### Spatial branch

The starting point is the Hugging Face image-classification checkpoint `Organika/sdxl-detector`. Its model card describes an SDXL-versus-Wikimedia training task and cautions that performance can be lower for generators other than SDXL. It is therefore a transfer-learning initialization, not evidence of general deepfake performance. The checkpoint's processor resizes inputs to 224 × 224 and applies ImageNet channel normalization; see `model/SDXL-Deepfake-Detector/preprocessor_config.json`.

### Frequency branch and fusion

The input `pixel_values` are transformed with a 2D FFT over spatial dimensions, shifted so the zero-frequency component is centered, and converted to log magnitude. A convolutional encoder maps this representation to a 128-dimensional feature vector. The pooled Swin feature and frequency vector are concatenated and passed through the fusion classifier. During training, cross-entropy is also applied to an auxiliary classifier on the Swin representation, weighted by 0.25 in the model configuration.

This architecture is motivated by prior work reporting generator-specific frequency artifacts, but the existence of such artifacts does not guarantee transfer to unseen image generators or post-processing. See the [frequency-analysis paper](https://arxiv.org/abs/2003.08685) and the [Swin Transformer paper](https://doi.org/10.1109/ICCV48922.2021.00986). The fusion design is a hypothesis to test, not an established improvement.

For adaptive inference context, Byun et al. study token reduction and early exit for medical vision transformers, using dataset-specific profiling to select inference strategies. Their reported results apply to their medical datasets and experimental setup; they are not directly comparable to this project. This repository implements a simpler confidence-threshold early exit and does not implement token reduction or the paper's dynamic strategy predictor. See [Byun et al. (2026)](https://proceedings.mlr.press/v315/byun26b.html).

### Adaptive inference

When `--adaptive` is enabled, the auxiliary head first classifies from pooled Swin features. Samples whose maximum softmax probability meets the configured threshold (default 0.90) exit early. Uncertain samples use the FFT branch and fusion classifier. Thresholds must be selected on held-out validation data and frozen before testing. Confidence is not a calibrated probability unless calibration has been separately assessed.

## Data and provenance

The project references the Kaggle [140k Real and Fake Faces dataset](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces). The Kaggle collection is commonly described as roughly 70,000 real face images and 70,000 StyleGAN-generated face images, resized to 256 × 256; verify the exact release, metadata, and terms on the dataset page before use. This is a face-only benchmark for a particular GAN domain. It does not by itself support claims about SDXL, face swaps, video, current text-to-image systems, or images in the wild.

Download separately using the Kaggle website or CLI (after configuring Kaggle credentials):

```bash
kaggle datasets download \
  -d xhlulu/140k-real-and-fake-faces \
  --unzip \
  -p ./dataset-download
```

Arrange images into the class-folder layout below. The scripts do not create splits or convert Kaggle archive layouts automatically. Keep the original files and split manifest outside Git, record how each item was assigned, and check for identity overlap, duplicate images, and source-specific artifacts across partitions.

```text
model/dataset/
├── train/{fake,real}/
├── val/{fake,real}/
└── test/{fake,real}/
```

The folder labels map as `fake → artificial` and `real → human`; IDs are read from checkpoint metadata. For cross-generator testing, keep each generator's test set separate and report its metrics separately. Dataset access terms and image rights are separate from the repository's code license.

## Installation

Use Python 3.12 or newer. Install a PyTorch/torchvision build appropriate for the host from the [official PyTorch installer](https://pytorch.org/get-started/locally/), then install the listed packages:

```bash
python -m pip install -r model/scripts/requirements.txt
```

The requirements file pins minimum versions rather than a fully locked environment. For a reproducible study, record the exact resolved versions, Python version, CUDA/runtime version, and hardware. Git LFS is required to retrieve the fine-tuned checkpoint when cloning from a remote that stores it as an LFS object. New training initialization also needs access to the Hugging Face Hub for the base model unless its files are already cached locally.

## Quick start

Run the scripts from `model/` so their default relative paths work. Replace the example path with an image you have permission to process.

```bash
cd model
python scripts/predict.py --image /path/to/image.jpg
python scripts/evaluate.py \
  --test_dir ./dataset/test \
  --output_json ./results/evaluation.json
```

The scripts select CUDA when available and otherwise use CPU. Evaluation defaults to batch size 16. `predict.py` reports the predicted label, class confidence, adaptive-path status, measured model-call latency, and descriptive image diagnostics.

## Training

From the `model/` directory:

```bash
python scripts/train.py \
  --train_dir ./dataset/train \
  --val_dir ./dataset/val
```

Defaults in `train.py` are 10 epochs, batch size 32, learning rate `2e-5`, preprocessing chunk size 20,000, seed 42, weight decay 0.01, and confidence threshold 0.90. Training augmentation consists of random horizontal flip (0.5), rotation up to 15 degrees, and brightness/contrast jitter (0.1). Validation preprocessing is deterministic. The Trainer evaluates and saves each epoch, selects by validation accuracy, keeps up to two checkpoints, and enables FP16 when CUDA is available. The final model is written to `./SDXL-Deepfake-Detector`.

Available command-line controls include `--epochs`, `--batch_size`, `--lr`, `--chunk_size`, `--seed`, and `--confidence_threshold`. The training script clears `./temp_chunks/` before preparing data, writes intermediate chunk caches there, and removes the cache after loading the processed data. Do not place unrelated valuable files in that temporary directory.

These settings describe the implementation, not a tuned or statistically validated protocol. Save the exact command, code revision, data manifest, package versions, seed, and hardware for each run. A fixed seed does not by itself guarantee bitwise reproducibility across devices and software stacks.

## Evaluation

`evaluate.py` evaluates a class-folder test set in full and adaptive modes. It records accuracy, weighted precision/recall/F1, confusion matrix, per-class classification report, mean model-call latency per image, and early-exit rate. It applies these deterministic corruption procedures to each test image:

| Variant | Procedure |
| --- | --- |
| Clean | Original decoded RGB image |
| JPEG q=50 | Encode and decode JPEG at quality 50 |
| Gaussian blur | 5 × 5 kernel, OpenCV default sigma |
| Low resolution | Resize to half width/height, then bilinear resize to original size |
| Gaussian noise | Add seeded Gaussian noise with σ=15 in 8-bit RGB values and clip to [0, 255] |

The same test samples are evaluated across variants. Noise uses a fixed NumPy seed (2026) for each variant evaluation. These are controlled synthetic degradations, not a comprehensive robustness assessment. Latency covers the model call after preprocessing and excludes image loading and processor execution; the evaluation batches images, so the reported per-image mean is batch-dependent. The early-exit rate is a compute proxy. The implementation does not estimate FLOPs or throughput.

```bash
cd model
python scripts/evaluate.py --test_dir ./dataset/test
python scripts/evaluate.py \
  --test_dir ./dataset/test \
  --batch_size 16 \
  --confidence_threshold 0.95 \
  --output_json ./results/robustness.json
python scripts/evaluate.py \
  --test_dir ./cross_generator/sdxl/test \
  --output_json ./results/sdxl.json
```

The default threshold in evaluation is read from the checkpoint; `--confidence_threshold` overrides it. Do not tune a threshold against test results. The blur, noise, and JPEG quality diagnostics printed or serialized by the project are descriptive image measurements, not learned quality or authenticity scores.

## Recorded result snapshot

The table below summarizes the checked-in [`metrics.json`](model/results/metrics.json). It records 10,905 images, `device: cuda`, and a test directory string of `dataset/test`. **The artifact does not record the dataset version, split construction, class-folder support breakdown by source, GPU model, software versions, checkpoint hash, or threshold-selection procedure.** Consequently, these figures are a historical run record, not a reproducible benchmark or a generalization claim.

| Corruption | Accuracy (full / adaptive) | Weighted precision | Weighted recall | Weighted F1 | Latency (full / adaptive, ms/image) | Adaptive early-exit rate | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Clean | 93.27% / 93.27% | 93.58% | 93.27% | 93.26% | 6.60 / 6.24 | 99.06% | 1.058× |
| JPEG q=50 | 91.30% / 91.30% | 92.07% | 91.30% | 91.25% | 6.63 / 6.27 | 98.73% | 1.058× |
| Gaussian blur 5 × 5 | 92.86% / 92.86% | 92.89% | 92.86% | 92.85% | 6.66 / 6.30 | 98.56% | 1.057× |
| Half resolution, upscaled | 92.11% / 92.11% | 92.14% | 92.11% | 92.11% | 6.65 / 6.28 | 98.59% | 1.057× |
| Gaussian noise, σ=15 | 91.24% / 91.24% | 91.44% | 91.24% | 91.23% | 6.59 / 6.22 | 98.42% | 1.060× |

In this artifact, adaptive and full modes report the same predicted labels for every listed variant; adaptive latency is approximately 5.5–5.8% lower. The recorded threshold is not explicitly stored in the metrics file, and no uncertainty intervals or repeated-run estimates are provided. See the raw JSON for weighted metrics, confusion matrices, class reports, and image-quality summaries. Do not compare these values with published results unless the dataset, split, preprocessing, and metric definitions match.

## Limitations and research recommendations

- **Narrow task and domain.** The named dataset focuses on face images and StyleGAN fakes. It is not equivalent to manipulation detection, video deepfake detection, or broad AI-image detection.
- **Potential shortcut learning.** Collection, compression, resizing, or source-specific artifacts can correlate with labels. Identity-aware splitting and external datasets are essential.
- **No demonstrated fusion gain.** Compare against the same fine-tuned Swin model without the frequency branch, using identical splits, preprocessing, seeds, and compute reporting.
- **Adaptive-head risk.** A high softmax score can still be wrong or miscalibrated. Report validation-based calibration, accuracy, exit coverage, and latency together.
- **No universal detector claim.** Distribution shift, unseen generators, post-processing, demographic variation, and image provenance are not resolved by this implementation.
- **Responsible handling.** Faces are sensitive personal data. Follow dataset terms, institutional review requirements, privacy rules, and local law; avoid publishing identifiable examples without authorization.

## Licensing and attribution

The project [`LICENSE`](LICENSE) is MIT for original code. The Hugging Face page labels [`Organika/sdxl-detector`](https://huggingface.co/Organika/sdxl-detector) **CC BY-NC 3.0** and describes non-commercial use; review the full upstream terms before using or redistributing the derived checkpoint. The Kaggle dataset has its own terms and upstream image-rights considerations. Neither the MIT license nor this documentation grants rights to third-party weights, dataset images, or user-provided images.

Suggested dataset citation: xhlulu, “[140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces),” Kaggle dataset, accessed YYYY-MM-DD. Cite the base model as Organika, “[SDXL Detector](https://huggingface.co/Organika/sdxl-detector),” Hugging Face model repository, accessed YYYY-MM-DD. Include model revision hashes and access dates in formal work.

## References

1. Liu, Z. et al. (2021). “Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.” *ICCV*. [DOI](https://doi.org/10.1109/ICCV48922.2021.00986).
2. Frank, J. et al. (2020). “Leveraging Frequency Analysis for Deep Fake Image Recognition.” [arXiv:2003.08685](https://arxiv.org/abs/2003.08685).
3. Byun, J. Y. et al. (2026). “Adaptive Inference for Medical Vision Transformers: Token Reduction or Early Exit?” *Proceedings of the 9th International Conference on Medical Imaging with Deep Learning*, PMLR 315:2171–2191. [Proceedings page](https://proceedings.mlr.press/v315/byun26b.html).
4. Organika. “[SDXL Detector](https://huggingface.co/Organika/sdxl-detector).” Hugging Face model card.
5. xhlulu. “[140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces).” Kaggle dataset.
