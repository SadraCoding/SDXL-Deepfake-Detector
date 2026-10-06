# Research Protocol

This protocol is intended for university projects, theses, and papers using this repository. It separates the implemented software behavior from the evidence a defensible study should collect. A completed checklist does not itself establish validity; study design and data provenance determine what can be concluded.

## 1. Define the claim before running experiments

Write the research question and primary endpoint before examining test results. A suitable question for this codebase is:

> Under a fixed, leakage-aware split and matched training conditions, does adding the FFT branch improve held-out classification over the same fine-tuned Swin backbone?

Specify the target population and domain. The `140k Real and Fake Faces` collection is face-only and is commonly described as containing real Flickr faces and StyleGAN fakes. Results on that benchmark should not be generalized to face swaps, video, SDXL, other generators, or real-world images without independent evidence.

Predefine:

- primary metric and decision threshold;
- intended comparison (hybrid versus Swin-only, and optionally adaptive versus full inference);
- data partitions and independent unit (identity, source image, generator/prompt, or another justified unit);
- number of random seeds or repeated runs;
- inclusion/exclusion rules and how corruptions are generated;
- subgroup analyses, if any, and their rationale;
- confidence interval or uncertainty method;
- rules for selecting a checkpoint and calibrating the adaptive threshold.

Do not use the test set to choose hyperparameters, epochs, thresholds, corruption parameters, or which result to report.

## 2. Data governance and provenance

Before downloading or using images:

1. Read and archive the current Kaggle dataset page, version/revision information, and applicable terms.
2. Trace upstream sources and record licenses, privacy constraints, consent statements, and any institutional review requirements.
3. Determine whether research, redistribution, public display, and model release are separately permitted. Public availability does not automatically grant every downstream right.
4. Keep downloaded data, credentials, and identifiable examples out of public Git commits. Restrict access to data according to the applicable terms and institutional policy.
5. Record the exact archive name, access date, checksum, extraction procedure, and any excluded/corrupt files.

For every split, report counts by class and source where available. Save a manifest with a stable relative identifier, label, split, source/generator, and exclusion reason. Avoid publishing personal data in the manifest; use stable hashes or pseudonymous IDs where appropriate.

### Split integrity

- Prefer partitioning by identity or source provenance rather than randomly by image when those relations exist.
- Keep near-duplicate and transformed copies in a single partition. Run exact-hash and perceptual duplicate checks across partitions.
- Avoid class-correlated differences in image size, compression, naming, metadata, or collection pipeline unless those are part of the research question.
- Preserve an untouched test set. If selecting a confidence threshold, use validation data only.
- For cross-generator tests, hold out an entire generator where possible and report each generator separately.
- Publish enough information to reconstruct the split without republishing restricted images.

The training script's isolated temporary chunk caches prevent stale cache reuse; they do not perform or validate any of the split checks above.

## 3. Freeze the software and model artifacts

For each run, record:

- repository URL, commit SHA, local modifications, and script commands;
- base model identifier and immutable revision; fine-tuned checkpoint path and cryptographic hash;
- processor configuration and image resize/normalization behavior;
- Python, PyTorch, torchvision, Transformers, NumPy, Pillow, OpenCV, scikit-learn, and datasets versions;
- operating system, CPU, GPU model and memory, CUDA/driver versions, and relevant precision settings;
- all random seeds and whether deterministic algorithms were enabled;
- training/validation/test sample counts and class balance;
- model-selection epoch and validation metric;
- all preprocessing, augmentations, and corruption parameters.

The project requirements file specifies minimum dependency versions, not a fully locked environment. Export a resolved environment (for example, a package lock or explicit version inventory) for publication-grade reproducibility.

## 4. Train matched models

Train at least these conditions using identical partitions, training schedule, augmentations, selection criteria, and seeds:

1. **Swin-only baseline:** fine-tune the spatial backbone and classification head without FFT features.
2. **Swin–frequency hybrid:** use the implementation in this repository.
3. **Adaptive inference:** evaluate the trained hybrid's auxiliary early-exit behavior separately; do not compare it as a distinct training architecture unless that is part of the design.

The provided training defaults are 10 epochs, batch size 32, learning rate `2e-5`, weight decay 0.01, preprocessing chunk size 20,000, seed 42, horizontal flip probability 0.5, rotation up to 15 degrees, and brightness/contrast jitter 0.1. They are starting defaults, not a validated optimum. Report any changed values and how they were selected. Run multiple seeds where compute allows and report variability rather than only the best seed.

The script evaluates at each epoch and selects by validation accuracy. If the study's primary endpoint differs, make checkpoint selection consistent with the predeclared protocol and disclose code changes.

## 5. Evaluate without leakage

### Core classification results

On the untouched test partition, report:

- number of examples and class support;
- accuracy and balanced accuracy;
- per-class precision, recall, and F1;
- macro- and weighted-average metrics, with averaging conventions stated;
- confusion matrix with explicit label order;
- confidence intervals or another justified uncertainty estimate;
- threshold and any validation-only calibration procedure.

Do not rely on a single aggregate accuracy, especially when the test set is imbalanced. Report results separately for each dataset and generator. If reporting a combined score, define the aggregation rule and also show the constituent results.

### Corruption and robustness

The supplied evaluator applies JPEG quality 50, a 5 × 5 Gaussian blur, half-resolution bilinear downsampling/upscaling, and additive Gaussian noise with standard deviation 15 in 8-bit RGB values. State that these are synthetic image degradations and specify the evaluator's fixed noise seed (2026). They do not represent every real-world transformation or adversarial attack.

Apply each corruption to the same held-out images when paired comparisons are intended. Preserve clean/corrupted sample correspondence. Report the exact parameters, random seed, and whether corruption is applied before or after decoding/resizing. Add realistic in-domain transformations only when they match the deployment or scientific question.

### Adaptive inference and calibration

Choose the confidence threshold using validation data only. On test data, report the frozen threshold, full-path and adaptive metrics, early-exit coverage/rate, and latency together. Include calibration diagnostics (for example, reliability plots or expected calibration error with stated binning) if confidence is interpreted probabilistically. A high softmax score alone does not demonstrate correct or calibrated prediction.

Recent adaptive-inference research for medical vision transformers combines token reduction and early exit using dataset-specific profiling and a learned strategy selector ([Byun et al., 2026](https://proceedings.mlr.press/v315/byun26b.html)). The implementation in this repository uses confidence-threshold early exit only. Treat the paper as methodological context: its medical-dataset results and FLOPs reductions are not directly comparable to this project's face-image results or latency measurements.

The evaluator measures the model-call interval after processor execution and reports mean milliseconds per image for a batch. State hardware, batch size, warm-up policy, number of repetitions, synchronization policy, timing scope, and whether the statistic is mean, median, or a distribution. Report throughput separately if measured. Early-exit rate is not a FLOPs measurement.

## 6. Statistical analysis and ablation

- Compare matched models on the same examples. Use paired uncertainty estimates or tests appropriate to the metric and sampling unit.
- Account for clustering by identity/source if multiple samples are related; ordinary independent-image intervals may be too narrow.
- Report seed-to-seed variation separately from test-sample uncertainty.
- Define primary and secondary outcomes to avoid selective reporting.
- Include an ablation of the frequency branch and, when relevant, the auxiliary head/early-exit path.
- Make clear whether augmentation, thresholding, and preprocessing are identical between models.
- Avoid interpreting small differences as meaningful without uncertainty and a prespecified analysis.

## 7. Required run record

Store machine-readable metadata with every evaluation. The following is a template; replace placeholders rather than publishing them unchanged:

```json
{
  "run_id": "YYYY-MM-DD-description",
  "code_revision": "git commit SHA",
  "working_tree_clean": true,
  "base_model": {
    "id": "Organika/sdxl-detector",
    "revision": "immutable revision"
  },
  "checkpoint_sha256": "...",
  "dataset": {
    "name": "140k Real and Fake Faces",
    "source_url": "https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces",
    "revision_or_access_date": "...",
    "archive_sha256": "...",
    "split_manifest_sha256": "...",
    "counts": {
      "train": {"fake": 0, "real": 0},
      "validation": {"fake": 0, "real": 0},
      "test": {"fake": 0, "real": 0}
    }
  },
  "training": {
    "command": "...",
    "seed": 42,
    "selected_epoch": 0,
    "threshold_selection": "validation-only procedure"
  },
  "software": {
    "python": "...",
    "torch": "...",
    "torchvision": "...",
    "transformers": "...",
    "numpy": "...",
    "opencv": "..."
  },
  "hardware": {
    "device": "...",
    "accelerator": "...",
    "memory_gb": 0
  },
  "evaluation": {
    "batch_size": 0,
    "threshold": 0.0,
    "corruptions": [],
    "metrics": {}
  }
}
```

The current `model/results/metrics.json` does not contain all of this metadata. Treat it as an incomplete historical record and do not infer missing provenance from filenames or README prose.

## 8. Reporting and interpretation

Describe predictions as classifier outputs for a defined dataset and protocol. Discuss domain shift, possible shortcut learning, class balance, source identity, demographic coverage, and the limits of any subgroup analysis. Do not present the detector as proof of authenticity or use it as the sole basis for consequential decisions.

For face data, follow institutional policies and applicable privacy/data-protection rules. Avoid publishing identifiable images unless rights and authorization are clear. Document model/data licenses independently: this repository's MIT license covers its code only; Hugging Face lists the base model as CC BY-NC 3.0, and the Kaggle dataset has separate terms and upstream rights.

## References and project materials

- Project implementation and current results: [`README.md`](../README.md), [`model/scripts/`](../model/scripts/), and [`model/results/metrics.json`](../model/results/metrics.json).
- Base model card: [Organika/sdxl-detector](https://huggingface.co/Organika/sdxl-detector).
- Dataset source: [xhlulu/140k-real-and-fake-faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces).
- Swin Transformer: Liu, Z. et al. (2021), [DOI: 10.1109/ICCV48922.2021.00986](https://doi.org/10.1109/ICCV48922.2021.00986).
- Frequency analysis for generated-image recognition: Frank, J. et al. (2020), [arXiv:2003.08685](https://arxiv.org/abs/2003.08685).
- Adaptive inference for medical vision transformers: Byun, J. Y. et al. (2026), *PMLR* 315:2171–2191, [proceedings page](https://proceedings.mlr.press/v315/byun26b.html).
