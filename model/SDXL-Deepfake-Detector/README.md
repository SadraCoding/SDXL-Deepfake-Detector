---
license: cc-by-nc-3.0
tags:
  - image-classification
  - deepfake-detection
  - synthetic-image-detection
  - swin-transformer
  - frequency-analysis
library_name: transformers
base_model: Organika/sdxl-detector
---

# Spatial–Frequency Hybrid Detector for Synthetic Face Images

This repository contains a fine-tuned checkpoint for binary image classification. It predicts whether an input image belongs to the `artificial` or `human` class. The project combines a Swin image-classification backbone with a learned representation of the image's centered two-dimensional Fourier log-magnitude spectrum.

> **Experimental research model.** This is a classifier, not an authenticity or provenance verifier. Its output does not establish who created an image, whether a face is real, whether an image was manipulated, or whether a person consented to its use. Do not use it as the sole basis for consequential decisions.

## Project links

- [GitHub source, training and evaluation scripts](https://github.com/SadraCoding/SDXL-Deepfake-Detector)
- [Technical report](https://github.com/SadraCoding/SDXL-Deepfake-Detector/blob/main/REVISED_PROJECT_REPORT.md)
- [Research protocol](https://github.com/SadraCoding/SDXL-Deepfake-Detector/blob/main/docs/research_protocol.md)
- [Live demo](https://sdxldd.ir)
- [Base checkpoint: Organika/sdxl-detector](https://huggingface.co/Organika/sdxl-detector)
- [Dataset cited by the project: 140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)

## Model details

The model uses pooled spatial features from the Swin backbone and a 128-dimensional feature vector learned from the centered FFT log-magnitude representation. It concatenates these features and classifies them with a fusion head. An auxiliary classifier on the Swin features is trained with an additional loss. The checkpoint's class mapping is:

| Class ID | Label |
| --- | --- |
| 0 | `artificial` |
| 1 | `human` |

The image processor resizes images to 224 × 224, rescales pixel values, and applies ImageNet channel normalization. The checkpoint also includes an optional confidence-based early-exit path: with adaptive inference enabled, predictions at or above the configured confidence threshold (default 0.90) use the auxiliary Swin head; less confident samples use the full spatial–frequency path. A softmax score is not a calibrated probability unless calibration has been measured separately.

## Intended use and limitations

The project references the Kaggle 140k Real and Fake Faces collection, commonly described as real face images and StyleGAN-generated faces. This is a narrow face-image domain. It does not establish performance on face swaps, video, SDXL or other unseen generators, or arbitrary images in the wild. The checkpoint does not include enough dataset version, split construction, and run provenance to reproduce or independently validate its reported result.

The checked-in [GitHub metrics artifact](https://github.com/SadraCoding/SDXL-Deepfake-Detector/blob/main/model/results/metrics.json) records one evaluation on 10,905 images (`device: cuda`, test directory `dataset/test`). There are 5,492 `artificial` examples (ID 0) and 5,413 `human` examples (ID 1). Percentages below are rounded to two decimals. Full and adaptive inference have identical aggregate and per-class classification metrics and confusion matrices for every listed variant, so classification metrics appear once per variant.

| Variant | Accuracy | Weighted P / R / F1 | Macro P / R / F1 | Artificial P / R / F1 | Human P / R / F1 | Confusion matrix (true rows, predicted columns; IDs 0, 1) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Clean | 93.27% | 93.58 / 93.27 / 93.26% | 93.60 / 93.24 / 93.25% | 89.97 / 97.51 / 93.59% | 97.23 / 88.97 / 92.92% | `[[5355, 137], [597, 4816]]` |
| JPEG quality 50 | 91.30% | 92.07 / 91.30 / 91.25% | 92.11 / 91.25 / 91.25% | 86.44 / 98.11 / 91.91% | 97.77 / 84.39 / 90.59% | `[[5388, 104], [845, 4568]]` |
| Gaussian blur 5 × 5 | 92.86% | 92.89 / 92.86 / 92.85% | 92.89 / 92.85 / 92.85% | 91.80 / 94.23 / 93.00% | 93.98 / 91.46 / 92.71% | `[[5175, 317], [462, 4951]]` |
| Half resolution, upscaled | 92.11% | 92.14 / 92.11 / 92.11% | 92.14 / 92.10 / 92.11% | 91.21 / 93.34 / 92.26% | 93.07 / 90.87 / 91.96% | `[[5126, 366], [494, 4919]]` |
| Gaussian noise, σ = 15 | 91.24% | 91.44 / 91.24 / 91.23% | 91.46 / 91.22 / 91.23% | 88.64 / 94.76 / 91.60% | 94.28 / 87.68 / 90.86% | `[[5204, 288], [667, 4746]]` |

P / R / F1 means precision / recall / F1. Confusion matrices use rows for true labels and columns for predicted labels, ordered `artificial` then `human`.

| Variant | Full latency (ms/image) | Adaptive latency (ms/image) | Early-exit rate (full / adaptive) | Speedup (full / adaptive) |
| --- | ---: | ---: | ---: | ---: |
| Clean | 6.60 | 6.24 | 0.00% / 99.06% | 1.058× |
| JPEG quality 50 | 6.63 | 6.27 | 0.00% / 98.73% | 1.058× |
| Gaussian blur 5 × 5 | 6.66 | 6.30 | 0.00% / 98.56% | 1.057× |
| Half resolution, upscaled | 6.65 | 6.28 | 0.00% / 98.59% | 1.057× |
| Gaussian noise, σ = 15 | 6.59 | 6.22 | 0.00% / 98.42% | 1.060× |

For clean inputs, the artifact reports mean blur score **628.65**, mean noise score **6.07**, and mean JPEG-50 MAE **3.57**. It does not include these quality diagnostics for the corrupted variants. They are descriptive measurements, not learned quality or authenticity scores.

These controlled degradations are not a comprehensive robustness test. The artifact omits dataset revision, split construction, per-source support, GPU model, software versions, checkpoint hash, and threshold-selection procedure. The project Hugging Face page has also displayed a self-reported 97% accuracy, but available evaluation details do not establish comparability with this artifact; do not combine the figures. The repository has not demonstrated that frequency fusion improves performance over a matched Swin-only baseline.

Potential dataset shortcuts, identity or duplicate overlap, source-specific artifacts, compression, preprocessing, demographic coverage, and distribution shift can affect predictions. The project is not a universal deepfake detector. See the linked [technical report](https://github.com/SadraCoding/SDXL-Deepfake-Detector/blob/main/REVISED_PROJECT_REPORT.md) and [research protocol](https://github.com/SadraCoding/SDXL-Deepfake-Detector/blob/main/docs/research_protocol.md) for detailed caveats and recommendations for controlled evaluation.

## Using the checkpoint

This checkpoint uses the project's custom `SwinFrequencyHybridDetector` architecture; it cannot be loaded as a standard `AutoModelForImageClassification` without the custom model code. Clone the GitHub repository and install a PyTorch/torchvision build that matches your platform using the [official PyTorch selector](https://pytorch.org/get-started/locally/). Then install the project dependencies and run inference from the `model/` directory:

```bash
git clone https://github.com/SadraCoding/SDXL-Deepfake-Detector.git
cd SDXL-Deepfake-Detector
python -m pip install -r model/scripts/requirements.txt
cd model
python scripts/predict.py --image /path/to/image.jpg
```

The script loads the bundled fine-tuned checkpoint from `model/SDXL-Deepfake-Detector/`, selects CUDA when available (otherwise CPU), and reports the predicted label, confidence, adaptive-path status, model-call latency, and descriptive image-quality measurements. Adaptive inference is off by default. Enable it with `--adaptive`; optionally set `--confidence_threshold` to a value in `(0, 1]`. The threshold should be selected on validation data and frozen before test evaluation.

To evaluate a class-folder test set, use `python scripts/evaluate.py --test_dir ./dataset/test` from `model/`. The evaluator reports full and adaptive results and applies the synthetic corruption procedures described above. For training, arrange data in `dataset/train/{fake,real}/` and `dataset/val/{fake,real}/`, then run `python scripts/train.py --train_dir ./dataset/train --val_dir ./dataset/val`. The scripts do not download or arrange the dataset, validate split integrity, or establish that a split is leakage-free. Consult the GitHub README for complete options and defaults.

The dataset is not included. The GitHub project describes obtaining it separately from Kaggle and arranging it into `train/{fake,real}`, `val/{fake,real}`, and `test/{fake,real}` directories (`fake → artificial`, `real → human`). Review the current dataset terms, image rights, and privacy requirements before use. Keep credentials, restricted data, and identifiable examples out of public repositories.

## Training and evaluation context

The reported training hardware is one NVIDIA RTX 3060 with 12 GB VRAM. The repository's training defaults are 10 epochs, batch size 32, learning rate `2e-5`, weight decay 0.01, preprocessing chunk size 20,000, seed 42, and confidence threshold 0.90. Training uses horizontal flips, rotations up to 15 degrees, and brightness/contrast jitter; validation preprocessing is deterministic. These are implementation defaults, not tuned or statistically validated settings. The requirements specify minimum package versions rather than a locked environment.

For research use, preserve the exact code revision, checkpoint and processor hashes, dataset revision and split manifest, commands, resolved software versions, hardware, seeds, and threshold-selection procedure. Prefer identity/source-aware splits, check for duplicates and leakage, keep test data untouched during model and threshold selection, compare against a matched Swin-only baseline, and report uncertainty and per-class metrics. The [research protocol](https://github.com/SadraCoding/SDXL-Deepfake-Detector/blob/main/docs/research_protocol.md) provides a fuller checklist.

## License and attribution

The project declares MIT for its original code. That license does not grant rights to third-party model weights or dataset images. The upstream [`Organika/sdxl-detector`](https://huggingface.co/Organika/sdxl-detector) model card identifies a CC BY-NC 3.0 license and non-commercial restrictions; review its current terms before using or redistributing this derived checkpoint. The Kaggle dataset has separate terms and upstream image-rights considerations. Face data may also be subject to privacy, consent, and institutional requirements.

Suggested attribution:

- SadraCoding, [SDXL-Deepfake-Detector](https://github.com/SadraCoding/SDXL-Deepfake-Detector), source code and research artifacts.
- Organika, [SDXL Detector](https://huggingface.co/Organika/sdxl-detector), upstream checkpoint.
- xhlulu, [140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces), Kaggle dataset.

For formal work, record immutable model revisions and access dates, and cite the relevant dataset and model documentation.

## References

1. Liu, Z. et al. (2021). “Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.” *ICCV*. [DOI](https://doi.org/10.1109/ICCV48922.2021.00986).
2. Frank, J. et al. (2020). “Leveraging Frequency Analysis for Deep Fake Image Recognition.” [arXiv:2003.08685](https://arxiv.org/abs/2003.08685).
3. Byun, J. Y. et al. (2026). “Adaptive Inference for Medical Vision Transformers: Token Reduction or Early Exit?” *Proceedings of the 9th International Conference on Medical Imaging with Deep Learning*, PMLR 315:2171–2191. [Proceedings](https://proceedings.mlr.press/v315/byun26b.html). This provides methodological context; its medical-dataset findings are not directly comparable to this project.
