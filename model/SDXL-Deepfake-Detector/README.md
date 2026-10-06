---
license: mit
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

This repository contains a fine-tuned checkpoint for binary image classification. Given an input image, the model predicts one of two labels: `artificial` or `human`.

> **Experimental research model.** This classifier is not an authenticity or provenance verifier. Its predictions reflect its training data and preprocessing; they do not establish who created an image, whether a face is real, or whether an image was manipulated. Do not use it as the sole basis for high-impact decisions.

## Links

- **GitHub source, scripts, report, and protocol:** [SadraCoding/SDXL-Deepfake-Detector](https://github.com/SadraCoding/SDXL-Deepfake-Detector)
- **Live demo:** [sdxldd.ir](https://sdxldd.ir)
- **Base checkpoint:** [Organika/sdxl-detector](https://huggingface.co/Organika/sdxl-detector)
- **Dataset referenced by the project:** [140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)

## Model details

The model starts from the Swin image-classification checkpoint `Organika/sdxl-detector`. It combines pooled spatial features from Swin with a learned encoding of the input's centered 2D Fourier log-magnitude. A fusion classifier predicts the final label, and an auxiliary classifier is trained on the Swin features.

The image processor resizes inputs to 224 × 224, rescales pixel values, and applies ImageNet channel normalization. The class mapping stored with this checkpoint is:

| ID | Label |
| --- | --- |
| 0 | `artificial` |
| 1 | `human` |

An optional adaptive inference path predicts first from the Swin features and runs the frequency branch for samples below a confidence threshold. The checkpoint's default threshold is 0.90. Softmax confidence should not be interpreted as calibrated probability unless calibration has been evaluated separately.

## Intended use and limitations

The project references the Kaggle 140k Real and Fake Faces collection, a face-image dataset commonly described as containing real and StyleGAN-generated images. This is a narrow benchmark domain. It does not establish performance on unseen generators, face swaps, video, current text-to-image systems, or arbitrary images in the wild. Dataset version, split construction, and complete run provenance are not included with the checkpoint.

The project's checked-in metrics artifact reports 93.27% clean accuracy on one run of 10,905 images. Split provenance and other details needed to reproduce or independently interpret this result are incomplete; treat it as a historical result, not a generalization claim. The Hugging Face model page's previously reported 97% accuracy has different or undocumented evaluation details and should not be compared directly with that artifact.

Potential dataset shortcuts, distribution shift, image compression, preprocessing, identity overlap, and generator-specific artifacts can affect predictions. The project has not established that the frequency branch improves performance over a matched Swin-only baseline. See the GitHub [research report](https://github.com/SadraCoding/SDXL-Deepfake-Detector/blob/main/REVISED_PROJECT_REPORT.md) and [research protocol](https://github.com/SadraCoding/SDXL-Deepfake-Detector/blob/main/docs/research_protocol.md) for details.

## Using the checkpoint

This checkpoint uses a custom architecture (`SwinFrequencyHybridDetector`), so the model implementation and inference scripts are provided in the GitHub repository. Follow its [installation and quick-start instructions](https://github.com/SadraCoding/SDXL-Deepfake-Detector#installation). In brief, clone the source, install the requirements and a compatible PyTorch build, then run the prediction script from `model/`:

```bash
python scripts/predict.py --image /path/to/image.jpg
```

The script selects CUDA when available and otherwise uses CPU. It reports the predicted class, confidence, adaptive-path status, measured model-call latency, and descriptive image diagnostics. Only process images you have permission to use.

## Training data

The dataset is not included in this repository. The source project describes downloading it separately from [Kaggle](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) and arranging it into `train/{fake,real}`, `val/{fake,real}`, and `test/{fake,real}` directories. Review the dataset's current access terms and image rights before use. Dataset terms are separate from the code license.

## License and attribution

The project declares MIT for its original code. The initialized upstream model, derived checkpoint, and dataset may have separate terms. The upstream `Organika/sdxl-detector` card identifies a CC BY-NC 3.0 license and non-commercial restrictions; review its current terms before using or redistributing this checkpoint. The MIT license does not grant rights to third-party weights or dataset images.

Please cite the [source project](https://github.com/SadraCoding/SDXL-Deepfake-Detector), the [base model](https://huggingface.co/Organika/sdxl-detector), and the [dataset](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) as relevant to your work.
