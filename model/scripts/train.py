"""Fine-tune Organika/sdxl-detector (a Swin image classifier)."""

import argparse
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, Image as HFImage, concatenate_datasets, load_from_disk
from PIL import Image
from sklearn.metrics import accuracy_score
from torchvision import transforms
from transformers import (
    AutoImageProcessor,
    DefaultDataCollator,
    Trainer,
    TrainingArguments,
)
from custom_model import SwinFrequencyHybridDetector

MODEL_DIR = "Organika/sdxl-detector"
OUTPUT_DIR = "./SDXL-Deepfake-Detector"
TEMP_CACHE_BASE = "./temp_chunks"
VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

train_transforms = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
    ]
)


def list_image_paths_and_labels(root_dir, label_to_id):
    """Read class-folder images, mapping folder names to the model's class IDs."""
    root = Path(root_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Directory not found: {root}")
    unknown = sorted(p.name for p in root.iterdir() if p.is_dir() and p.name not in label_to_id)
    if unknown:
        raise ValueError(f"Unrecognized class folders in {root}: {unknown}; expected {sorted(label_to_id)}")

    images, labels = [], []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS:
                images.append(str(path))
                labels.append(label_to_id[folder.name])
    if not images:
        raise ValueError(f"No supported images found under {root}")
    return images, labels


def main():
    parser = argparse.ArgumentParser(description="Fine-tune the Swin SDXL deepfake detector")
    parser.add_argument("--train_dir", default="./dataset/train")
    parser.add_argument("--val_dir", default="./dataset/val")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--chunk_size", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--confidence_threshold", type=float, default=0.90,
                        help="Inference confidence for early exit (calibrate on held-out validation data)")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.chunk_size < 1:
        parser.error("epochs, batch_size, and chunk_size must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    processor = AutoImageProcessor.from_pretrained(MODEL_DIR)
    model = SwinFrequencyHybridDetector(model_name=MODEL_DIR, num_classes=2,
                                        confidence_threshold=args.confidence_threshold)
    print(f"Base architecture: {model.swin_backbone.__class__.__name__} with FFT frequency branch and auxiliary early-exit head ({MODEL_DIR})")
    model_labels = {str(name).lower(): int(idx) for idx, name in model.config.id2label.items()}
    # Common dataset names are translated explicitly to the detector's semantic labels.
    aliases = {"fake": "artificial", "real": "human"}
    label_to_id = {}
    for source_name in ("fake", "real"):
        model_name = aliases[source_name]
        if model_name not in model_labels:
            raise ValueError(f"Model label {model_name!r} not found. Available labels: {model_labels}")
        label_to_id[source_name] = model_labels[model_name]

    train_paths, train_labels = list_image_paths_and_labels(args.train_dir, label_to_id)
    val_paths, val_labels = list_image_paths_and_labels(args.val_dir, label_to_id)
    train_dataset = Dataset.from_dict({"image": train_paths, "label": train_labels}).cast_column("image", HFImage())
    val_dataset = Dataset.from_dict({"image": val_paths, "label": val_labels}).cast_column("image", HFImage())

    def preprocess(examples, augment=False):
        imgs = []
        for item in examples["image"]:
            img = item if isinstance(item, Image.Image) else Image.open(item)
            img = img.convert("RGB")
            imgs.append(train_transforms(img) if augment else img)
        features = processor(images=imgs, return_tensors="np")
        features["labels"] = examples["label"]
        return features

    def process_and_save_chunk(dataset, index, chunk_size, target_dir, augment):
        start = index * chunk_size
        chunk = dataset.select(range(start, min(start + chunk_size, len(dataset))))
        chunk = chunk.map(
            lambda batch: preprocess(batch, augment=augment), batched=True,
            batch_size=32, num_proc=1, load_from_cache_file=False,
        )
        cache_path = Path(target_dir) / f"chunk_{index}"
        chunk.save_to_disk(str(cache_path))

    def load_chunks(cache_dir, count):
        if count == 0:
            raise ValueError("Cannot concatenate an empty dataset")
        return concatenate_datasets([load_from_disk(str(Path(cache_dir) / f"chunk_{i}")) for i in range(count)])

    train_cache_dir = Path(TEMP_CACHE_BASE) / "train"
    val_cache_dir = Path(TEMP_CACHE_BASE) / "val"
    # Clear stale caches to prevent old chunks from entering this run.
    if Path(TEMP_CACHE_BASE).exists():
        shutil.rmtree(TEMP_CACHE_BASE)
    train_cache_dir.mkdir(parents=True)
    val_cache_dir.mkdir(parents=True)
    n_train = (len(train_dataset) + args.chunk_size - 1) // args.chunk_size
    n_val = (len(val_dataset) + args.chunk_size - 1) // args.chunk_size
    try:
        for i in range(n_train):
            process_and_save_chunk(train_dataset, i, args.chunk_size, train_cache_dir, True)
        for i in range(n_val):
            process_and_save_chunk(val_dataset, i, args.chunk_size, val_cache_dir, False)
        train_processed = load_chunks(train_cache_dir, n_train)
        val_processed = load_chunks(val_cache_dir, n_val)
    finally:
        shutil.rmtree(TEMP_CACHE_BASE, ignore_errors=True)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=torch.cuda.is_available(),
        seed=args.seed,
        data_seed=args.seed,
        logging_steps=50,
        report_to=[],
        save_total_limit=2,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        return {"accuracy": accuracy_score(labels, np.argmax(logits, axis=-1))}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_processed,
        eval_dataset=val_processed,
        data_collator=DefaultDataCollator(),
        compute_metrics=compute_metrics,
        processing_class=processor,
    )
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    print(f"Fine-tuned Swin-frequency hybrid model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
