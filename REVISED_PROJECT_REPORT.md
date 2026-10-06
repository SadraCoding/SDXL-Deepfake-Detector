# Research and Technical Report

## Spatial–Frequency Fusion for Binary Classification of Synthetic Face Images

**Project:** SDXL Deepfake Detector

**Artifact type:** Research code and a preliminary evaluation snapshot

**Task as implemented:** Classify an input face image into `artificial` or `human`

**Base checkpoint:** [`Organika/sdxl-detector`](https://huggingface.co/Organika/sdxl-detector)

**Dataset cited by the project:** [140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)

**Code license:** MIT; model and dataset rights are separate

**Training hardware reported for this model:** One NVIDIA RTX 3060 with 12 GB VRAM

**Quick links:** [Project README](README.md) · [Live demo](https://sdxldd.ir) · [Hugging Face model](https://huggingface.co/SadraCoding/SDXL-Deepfake-Detector)

## Abstract

This project implements a two-branch image classifier that combines a pretrained Swin Transformer representation with learned features from a centered two-dimensional Fourier log-magnitude spectrum. An auxiliary classifier supports optional confidence-based early exit. The training and evaluation scripts provide a reproducible entry-point structure, class-folder inputs, chunked preprocessing, synthetic corruption measurements, and machine-readable metrics.

The checked-in result artifact reports 93.27% clean accuracy on 10,905 images and similar classification metrics for full and adaptive inference. These values are **not sufficient evidence of generalization or a controlled benchmark**: the result JSON omits dataset version and split construction, GPU model, software versions, checkpoint revision, and threshold-selection procedure. The project should therefore be treated as an experimental implementation. The key scientific test—whether frequency fusion improves over a matched Swin-only baseline—remains open.

## 1. Scope and research question

### 1.1 Task definition

The scripts accept face-image files organized in folders named `fake` and `real`. Those folder names map to the checkpoint labels `artificial` and `human`. This is a binary image-classification task over the represented domains. It is not a video task, face-swap detector, provenance system, identity tool, or general-purpose detector for every form of generated content.

The name of the base model refers to SDXL, but the cited Kaggle benchmark concerns real and StyleGAN-generated face images. The distinction matters: a result on this benchmark does not by itself measure performance on SDXL, diffusion models, face swaps, or in-the-wild media.

### 1.2 Research question

> Does adding a learned representation of the image's Fourier magnitude spectrum to a fine-tuned Swin classifier improve classification performance or efficiency under a controlled and independently evaluated protocol?

This implementation specifies a method for testing that question. The metrics snapshot does not answer it because it lacks a matched spatial-only ablation and sufficient experimental provenance.

## 2. Related work and rationale

Swin Transformer uses hierarchical visual representations with shifted-window attention and is designed as a general-purpose vision backbone ([Liu et al., 2021](https://doi.org/10.1109/ICCV48922.2021.00986)). Organika's [`sdxl-detector`](https://huggingface.co/Organika/sdxl-detector) is an image-classification model; its model card says it was fine-tuned from the `umm-maybe AI art detector` using paired SDXL and Wikimedia images. Its model card reports validation metrics for that upstream task and cautions that performance may be lower on generators other than SDXL. Those upstream metrics must not be attributed to this project.

Frequency-domain evidence has been studied for generated-image analysis. For example, Frank et al. investigate frequency analysis for deep-fake image recognition ([2020](https://arxiv.org/abs/2003.08685)). This literature motivates an experimental frequency branch; it does not imply that a Fourier branch will be robust to new generators, compression, resampling, or data-source shifts. The benefit must be measured through ablations and external testing.

Adaptive inference has also been studied for medical vision transformers. Byun et al. (2026) combine token reduction and early exiting, using dataset-specific profiling and a lightweight predictor to select strategies. They report results across five medical datasets, including an INSIGHT cataract dataset. This is related work on sample-adaptive computation, but its medical-image results do not transfer directly to this face-image classifier. The current project implements a confidence-threshold early exit only; it has neither token reduction nor the paper's strategy-selection predictor.

## 3. Model specification

### 3.1 Spatial representation

The model loads a Swin image-classification backbone from the configured base-model identifier. The Swin classifier layer is replaced by an identity mapping so the pooled spatial embedding can be used by both the auxiliary head and the fusion classifier. The processor metadata in the local checkpoint specifies 224 × 224 inputs, RGB channel normalization, and rescaling from 8-bit values.

### 3.2 Frequency representation

Given image tensor \(x\), the implementation computes a 2D FFT over width and height, applies `fftshift`, takes the stabilized log magnitude, and feeds the resulting three-channel spectrum to a convolutional encoder. The encoder uses two convolutional blocks with batch normalization, ReLU, and max pooling/adaptive pooling, followed by a dense projection to 128 dimensions.

### 3.3 Fusion and optimization objective

The pooled Swin representation and 128-dimensional frequency representation are concatenated and passed through a dense classifier with batch normalization, ReLU, dropout (0.3), and a final two-class projection. The training loss is the fusion-head cross-entropy plus an auxiliary Swin-head cross-entropy weighted by 0.25. The architecture and objective are defined in `model/scripts/custom_model.py`.

### 3.4 Adaptive inference

At inference, an auxiliary linear head produces preliminary logits from the Swin features. With early exit enabled, a sample is eligible to exit when its largest softmax probability is at least the configured confidence threshold (default 0.90). If every sample in a batch is eligible, the frequency branch is skipped for the batch; in mixed batches it is applied only to ineligible items. With adaptive mode disabled, all items use the full fusion path.

The confidence value is not automatically calibrated. Select the threshold on validation data, report the calibration method and exit coverage, and freeze it before test evaluation. Adaptive inference has its own classification behavior because it may return auxiliary-head predictions instead of fusion-head predictions.

## 4. Data and preprocessing

### 4.1 Referenced dataset

The project identifies the Kaggle dataset [140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces). The dataset is commonly described as approximately 70,000 real face images sourced from Flickr and 70,000 StyleGAN-generated faces, with images resized to 256 × 256. Researchers should verify the current Kaggle page, exact downloaded release, archive contents, licensing, and provenance before use or publication. The dataset is a narrow benchmark and should not be described as representative of all generated imagery.

The repository does not supply an automated downloader, split generator, duplicate detector, or archive-layout converter. The training and evaluation scripts expect folders in the following form:

```text
model/dataset/
├── train/{fake,real}/
├── val/{fake,real}/
└── test/{fake,real}/
```

Training applies random horizontal flip (probability 0.5), random rotation up to 15 degrees, and brightness/contrast jitter (0.1). Validation inputs are processed deterministically. The image processor resizes and normalizes the RGB images.

### 4.2 Leakage and split policy

Chunk-cache isolation in `train.py` prevents stale processed chunks from being reused within a run. It does not prevent leakage across train, validation, and test partitions. Near duplicates, source identity, photographer/source collection, or generation provenance may be shared across random image-level splits. Before claiming independent generalization, split by the most appropriate source unit, inspect exact and perceptual duplicates, keep the test set untouched during model and threshold selection, and publish a split manifest or immutable dataset version.

## 5. Training implementation

The training entry point is `model/scripts/train.py`. Its configured defaults are:

| Setting | Default |
| --- | ---: |
| Epochs | 10 |
| Per-device train/evaluation batch size | 32 |
| Learning rate | `2e-5` |
| Weight decay | 0.01 |
| Preprocessing chunk size | 20,000 images |
| Random seed and data seed | 42 |
| Early-exit threshold stored in config | 0.90 |
| Checkpoint selection | Best validation accuracy |
| Save/evaluate strategy | Each epoch; retain up to two Trainer checkpoints |
| FP16 | Enabled when CUDA is available |

The model was trained on one NVIDIA RTX 3060 with 12 GB VRAM. The checked-in metrics JSON records the evaluation device only as `cuda`, so it does not independently identify the hardware used for its timing measurements.

The script clears and recreates `./temp_chunks/`, preprocesses training and validation data in chunks, loads the processed datasets, then removes the temporary cache. It sets Python, NumPy, and PyTorch seeds, including CUDA seeds when available. Deterministic kernels are not explicitly enabled; results need not be bitwise identical across hardware or library versions. The model is saved to `./SDXL-Deepfake-Detector` relative to the working directory.

Training command, run from `model/`:

```bash
python scripts/train.py \
  --train_dir ./dataset/train \
  --val_dir ./dataset/val \
  --epochs 10 \
  --batch_size 32 \
  --lr 2e-5 \
  --seed 42
```

The scripts rely on packages listed in `model/scripts/requirements.txt`. PyTorch and torchvision wheels should match the compute environment; the requirements file uses lower bounds, not exact pins. Record resolved package versions, Python and CUDA versions, GPU/CPU details, and the exact command for each reported run.

## 6. Evaluation implementation

`model/scripts/evaluate.py` discovers files recursively in the `fake` and `real` test folders. It runs full and adaptive inference on the same samples for five variants:

| Variant | Implemented transformation |
| --- | --- |
| Clean | Decoded RGB image, unchanged |
| JPEG q=50 | JPEG encode/decode at quality 50 |
| Gaussian blur | 5 × 5 kernel with OpenCV's default sigma |
| Low resolution | Half-width/half-height resize, then bilinear resize to original dimensions |
| Gaussian noise | NumPy Gaussian noise, σ=15 in 8-bit RGB units, clipped to [0, 255] |

The noise generator is initialized with seed 2026 separately for each variant evaluation. Reported outputs include image count, accuracy, weighted precision/recall/F1, confusion matrix, class-wise report, mean model-call milliseconds per image, and adaptive exit rate. The latency timer surrounds model inference only; image opening, corruption, feature processing, and processor execution are outside the timed interval. Batch size affects the reported per-image latency. The script does not estimate FLOPs or throughput.

For clean images, the evaluator also aggregates native resolution, Laplacian-variance blur score, high-pass residual-noise estimate, and mean absolute error after JPEG q=50 reconstruction. These are descriptive statistics, not calibrated image-quality or authenticity scores.

## 7. Results currently recorded in the repository

[`model/results/metrics.json`](model/results/metrics.json) records one run over 10,905 images. Its metadata gives `test_dir: dataset/test` and `device: cuda`, but it does not identify the dataset release, how the split was formed, the checkpoint hash, GPU model, software versions, threshold-selection procedure, or uncertainty estimates. Therefore these results are reported for transparency, not as a reproducible benchmark or broad performance claim.

The [Hugging Face model page](https://huggingface.co/SadraCoding/SDXL-Deepfake-Detector) currently shows a self-reported 97% accuracy. The local metrics snapshot reports 93.27% clean accuracy. The available metadata does not establish that the two values use the same checkpoint, split, or evaluation protocol, so cite them separately rather than comparing them directly.

| Condition | Accuracy (full / adaptive) | Weighted precision | Weighted recall | Weighted F1 | Latency (full / adaptive, ms/image) | Adaptive exit rate | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Clean | 93.27% / 93.27% | 93.58% | 93.27% | 93.26% | 6.60 / 6.24 | 99.06% | 1.058× |
| JPEG q=50 | 91.30% / 91.30% | 92.07% | 91.30% | 91.25% | 6.63 / 6.27 | 98.73% | 1.058× |
| Gaussian blur 5 × 5 | 92.86% / 92.86% | 92.89% | 92.86% | 92.85% | 6.66 / 6.30 | 98.56% | 1.057× |
| Half resolution, then upscale | 92.11% / 92.11% | 92.14% | 92.11% | 92.11% | 6.65 / 6.28 | 98.59% | 1.057× |
| Gaussian noise, σ=15 | 91.24% / 91.24% | 91.44% | 91.24% | 91.23% | 6.59 / 6.22 | 98.42% | 1.060× |

Values are transcribed from the metrics artifact; accuracy and latency do not capture uncertainty or independent generalization. The adaptive and full paths have the same reported aggregate classification metrics for the listed variants in this run. Adaptive latency is about 5.5–5.8% lower under the recorded environment, while the metric file does not identify that environment in sufficient detail to reproduce the timing. The clean confusion matrix is `[[5355, 137], [597, 4816]]` in label-ID order `[artificial, human]`.

## 8. Review feedback and contribution boundaries

Earlier project review notes raised concerns about model identification, data leakage, novelty, benchmark comparison, and real-time claims. The repository addresses these concerns as follows:

- **Model identification:** The spatial backbone is the Swin classifier in `Organika/sdxl-detector`; the detector is not a ViT-Base architecture implemented from scratch.
- **Data leakage:** Training/validation cache paths are isolated and stale cache contents are removed. This is an engineering safeguard, not a substitute for identity-aware dataset partitioning or duplicate checks.
- **Algorithmic contribution:** Spatial-frequency fusion and confidence-gated early exit are implemented hypotheses. Their value must be established through ablation and threshold-controlled tests.
- **Engineering contributions:** Chunked preprocessing, epoch-level evaluation/checkpointing, mixed precision when CUDA is available, and runnable train/evaluate/predict interfaces are implementation and resource-management contributions, not new learning theory.
- **Benchmark comparisons:** A prior project draft mentioned a 2025 EfficientNetV2-B2 result of 99.88% accuracy. The cited bibliographic details and comparable protocol are not present in the repository, so that figure is unverified and is not used as evidence or a direct comparison here. Any future comparison must cite the primary study and match dataset version, split, preprocessing, and metric definitions.
- **Real-time claims:** The current JSON contains measured latency for one unspecified CUDA device and batch configuration. It does not support a hardware-independent real-time claim.

## 9. Limitations and responsible research

1. **Dataset scope:** A face-only, StyleGAN-oriented benchmark does not cover the breadth of image synthesis or manipulation methods.
2. **Source and identity confounding:** Random splits may make collection artifacts or identity overlap predictive. The current result artifact does not disclose its split policy.
3. **No controlled ablation:** The snapshot does not compare a Swin-only model with the hybrid under matched conditions.
4. **Single-run reporting:** There are no repeated seeds, confidence intervals, calibration curves, or significance analyses in the snapshot.
5. **Threshold provenance:** The exact threshold used for the recorded adaptive run is not written into the JSON artifact.
6. **External validity:** No independent cross-dataset or cross-generator result is included in the snapshot.
7. **Sensitive data:** Face images can be personal data. Researchers should follow dataset terms, institutional review and ethics processes where applicable, privacy regulations, and careful access controls.
8. **Licenses:** The project code is MIT. Hugging Face labels the base model CC BY-NC 3.0. Dataset terms and underlying source-image rights are independent. Review all applicable terms before use, publication, or redistribution.

Detector outputs should be framed as uncertain model scores, not proof of image authenticity. Do not deploy as the sole basis for consequential decisions or make claims about an individual's identity or intent.

## 10. Reproducibility checklist

Before treating results as research evidence, preserve:

- code commit SHA and working-tree status;
- exact checkpoint identifier, revision, and file hash;
- dataset URL, access date, release/archive checksum, terms, and class counts;
- split manifest and duplicate/identity leakage assessment;
- full training command, seed, augmentation, and selected checkpoint epoch;
- resolved software versions, operating system, accelerator, memory, and CUDA/runtime details;
- evaluation command, batch size, corruption parameters, threshold-selection method, and raw JSON;
- results for matched spatial-only and hybrid models, multiple seeds, and independent generators;
- per-class results, confusion matrices, calibration/coverage statistics, and uncertainty estimates where appropriate.

The companion [`docs/research_protocol.md`](docs/research_protocol.md) expands this checklist into an experiment workflow.

## References

1. Liu, Z. et al. (2021). “Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.” *Proceedings of ICCV*. [https://doi.org/10.1109/ICCV48922.2021.00986](https://doi.org/10.1109/ICCV48922.2021.00986).
2. Frank, J. et al. (2020). “Leveraging Frequency Analysis for Deep Fake Image Recognition.” [https://arxiv.org/abs/2003.08685](https://arxiv.org/abs/2003.08685).
3. Byun, J. Y. et al. (2026). “Adaptive Inference for Medical Vision Transformers: Token Reduction or Early Exit?” *Proceedings of the 9th International Conference on Medical Imaging with Deep Learning*, PMLR 315:2171–2191. [https://proceedings.mlr.press/v315/byun26b.html](https://proceedings.mlr.press/v315/byun26b.html).
4. SadraCoding. “SDXL-Deepfake-Detector.” Hugging Face model repository. [https://huggingface.co/SadraCoding/SDXL-Deepfake-Detector](https://huggingface.co/SadraCoding/SDXL-Deepfake-Detector).
5. Organika. “SDXL Detector.” Hugging Face model card. [https://huggingface.co/Organika/sdxl-detector](https://huggingface.co/Organika/sdxl-detector).
6. xhlulu. “140k Real and Fake Faces.” Kaggle dataset. [https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces).
