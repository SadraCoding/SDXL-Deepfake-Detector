"""Evaluate clean and synthetically corrupted class-folder test data."""

import argparse
import io
import json
import time
from pathlib import Path

import numpy as np
import cv2
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from transformers import AutoImageProcessor
from custom_model import ImageQualityAnalyzer, SwinFrequencyHybridDetector

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def corrupt_image(image, attack, rng):
    """Apply synthetic corruption to simulate real-world degradation."""
    if attack == "jpeg_q50":
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=50)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB").copy()
    if attack == "gaussian_blur_5x5":
        return Image.fromarray(cv2.GaussianBlur(np.asarray(image), (5, 5), 0))
    if attack == "low_resolution_50pct":
        small = image.resize((max(1, image.width // 2), max(1, image.height // 2)), Image.Resampling.BILINEAR)
        return small.resize(image.size, Image.Resampling.BILINEAR)
    if attack == "gaussian_noise_std15":
        arr = np.asarray(image, dtype=np.float32)
        noisy = np.clip(arr + rng.normal(0, 15, arr.shape), 0, 255).astype(np.uint8)
        return Image.fromarray(noisy)
    return image


def _synchronize(device):
    """Ensure accurate timing on CUDA devices by waiting for kernel completion."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _evaluate_variant(model, processor, samples, device, batch_size, adaptive, variant):
    """Run evaluation on a specific corruption variant."""
    rng = np.random.default_rng(2026)
    y_true, y_pred, exits = [], [], []
    elapsed_total = 0.0
    quality = []
    
    for start in range(0, len(samples), batch_size):
        batch = samples[start:start + batch_size]
        images = []
        for path, _ in batch:
            with Image.open(path) as source:
                image = source.convert("RGB")
            image = corrupt_image(image, variant, rng)
            
            # Only collect quality metrics for the clean variant to avoid skewing
            if variant == "clean":
                quality.append(ImageQualityAnalyzer.get_image_metrics(np.asarray(image)))
            images.append(image)
            
        inputs = processor(images=images, return_tensors="pt").to(device)
        _synchronize(device)
        started = time.perf_counter()
        
        with torch.inference_mode():
            output = model(**inputs, enable_early_exit=adaptive)
            
        _synchronize(device)
        elapsed_total += time.perf_counter() - started
        
        exits.extend(model.last_early_exited.cpu().tolist())
        y_true.extend(label for _, label in batch)
        y_pred.extend(output.logits.argmax(dim=-1).cpu().tolist())
        
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    results = {
        "num_images": len(y_true), 
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision), 
        "recall_weighted": float(recall), 
        "f1_weighted": float(f1),
        "mean_latency_ms_per_image": float(elapsed_total * 1000 / len(y_true)), 
        "early_exit_rate": float(np.mean(exits)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    }
    
    if quality:
        results["mean_image_quality"] = {
            "blur_score": float(np.mean([x["blur_score"] for x in quality])),
            "noise_score": float(np.mean([x["noise_score"] for x in quality])),
            "jpeg50_mae": float(np.mean([x["jpeg50_mae"] for x in quality])),
        }
    return results


def evaluate(test_dir_path, model_dir_path, batch_size=16, output_path=None, confidence_threshold=None):
    test_dir, model_dir = Path(test_dir_path), Path(model_dir_path)
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
        
    model = SwinFrequencyHybridDetector.from_pretrained(model_dir)
    if confidence_threshold is not None:
        if not 0 < confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be in (0, 1]")
        model.confidence_threshold = confidence_threshold
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    processor = AutoImageProcessor.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    aliases = {"fake": "artificial", "real": "human"}
    label_to_id = {}
    for folder_name, model_label in aliases.items():
        if model_label not in model.label2id:
            raise ValueError(f"Model label {model_label!r} not found; available labels: {model.label2id}")
        label_to_id[folder_name] = model.label2id[model_label]
    unknown = sorted(p.name for p in test_dir.iterdir()
                     if p.is_dir() and p.name.lower() not in label_to_id)
    if unknown:
        raise ValueError(f"Unknown class folders in {test_dir}: {unknown}; expected {sorted(label_to_id)}")
    samples = []
    for folder in sorted(p for p in test_dir.iterdir() if p.is_dir()):
        label = label_to_id[folder.name.lower()]
        samples.extend((str(path), label) for path in sorted(folder.rglob("*"))
                       if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS)
    if not samples:
        raise ValueError(f"No supported images found under {test_dir}")

    variants = ("clean", "jpeg_q50", "gaussian_blur_5x5", "low_resolution_50pct", "gaussian_noise_std15")
    results = {
        "model": str(model_dir),
        "test_dir": str(test_dir),
        "device": str(device),
        "full_inference": {variant: _evaluate_variant(model, processor, samples, device,
                                                       batch_size, False, variant)
                           for variant in variants},
    }
    results["adaptive_inference"] = {
        variant: _evaluate_variant(model, processor, samples, device, batch_size, True, variant)
        for variant in variants
    }
    results["adaptive_speedup"] = {
        variant: (results["full_inference"][variant]["mean_latency_ms_per_image"] /
                  results["adaptive_inference"][variant]["mean_latency_ms_per_image"])
        for variant in variants
    }
    if output_path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate clean and corrupted image detection")
    parser.add_argument("--test_dir", default="./dataset/test")
    parser.add_argument("--model_dir", default="./SDXL-Deepfake-Detector")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--confidence_threshold", type=float, default=None)
    parser.add_argument("--output_json", "--output", dest="output_path", default=None)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch_size must be positive")
    results = evaluate(args.test_dir, args.model_dir, args.batch_size,
                       args.output_path, args.confidence_threshold)
    for variant, metrics in results["full_inference"].items():
        adaptive = results["adaptive_inference"][variant]
        print(f"{variant}: full accuracy={metrics['accuracy']:.4f}, "
              f"adaptive accuracy={adaptive['accuracy']:.4f}, "
              f"early-exit rate={adaptive['early_exit_rate']:.3f}, "
              f"speedup={results['adaptive_speedup'][variant]:.3f}x")
    if args.output_path:
        print(f"Detailed results saved to {args.output_path}")


if __name__ == "__main__":
    main()
