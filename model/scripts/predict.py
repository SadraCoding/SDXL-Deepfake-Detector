"""Run single-image inference with a fine-tuned detector."""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor
from custom_model import ImageQualityAnalyzer, SwinFrequencyHybridDetector


def main():
    parser = argparse.ArgumentParser(description="Classify an image with the trained detector")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--model_dir", default="./SDXL-Deepfake-Detector")
    parser.add_argument("--adaptive", action="store_true", help="Enable confidence-calibrated early exit")
    parser.add_argument("--confidence_threshold", type=float, default=None)
    args = parser.parse_args()
    image_path = Path(args.image)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    model = SwinFrequencyHybridDetector.from_pretrained(args.model_dir)
    if args.confidence_threshold is not None:
        if not 0 < args.confidence_threshold <= 1:
            parser.error("confidence_threshold must be in (0, 1]")
        model.confidence_threshold = args.confidence_threshold
    processor = AutoImageProcessor.from_pretrained(args.model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    quality = ImageQualityAnalyzer.get_image_metrics(np.asarray(image))
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.inference_mode():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        output = model(**inputs, enable_early_exit=args.adaptive)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        latency_ms = (time.perf_counter() - started) * 1000
        logits = output.logits
        probabilities = torch.softmax(logits, dim=-1)[0]
    index = int(probabilities.argmax().item())
    label = model.id2label.get(index, str(index))
    print(f"Image: {image_path}\nLabel: {label}\nConfidence: {probabilities[index].item() * 100:.2f}%"
          f"\nAdaptive inference: {'enabled' if args.adaptive else 'disabled'}"
          f"\nEarly exit: {bool(model.last_early_exited[0].item())}"
          f"\nLatency: {latency_ms:.2f} ms\nQuality: {quality}")


if __name__ == "__main__":
    main()
