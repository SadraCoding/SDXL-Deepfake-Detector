"""Swin plus Fourier-spectrum classifier for generated-image detection."""

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForImageClassification, PretrainedConfig, PreTrainedModel
from transformers.utils import ModelOutput


@dataclass
class HybridDetectorOutput(ModelOutput):
    """Trainer-compatible output carrying only the optional loss and logits."""

    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None


class ImageQualityAnalyzer:
    """Simple, input-image quality measurements; not a learned quality score."""

    @staticmethod
    def estimate_blur(image_np):
        gray = cv2.cvtColor(np.asarray(image_np, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def estimate_noise(image_np):
        gray = cv2.cvtColor(np.asarray(image_np, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
        smooth = cv2.GaussianBlur(gray, (5, 5), 0)
        return float(np.std(gray.astype(np.float32) - smooth.astype(np.float32)))

    @staticmethod
    def get_image_metrics(image_np):
        image_np = np.asarray(image_np, dtype=np.uint8)
        h, w = image_np.shape[:2]
        # Mean absolute reconstruction error gives an interpretable JPEG-50 degradation measure.
        ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR),
                                   [cv2.IMWRITE_JPEG_QUALITY, 50])
        jpeg_error = 0.0
        if ok:
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
            jpeg_error = float(np.mean(np.abs(image_np.astype(np.float32) - decoded.astype(np.float32))))
        return {"resolution": f"{w}x{h}", "blur_score": round(ImageQualityAnalyzer.estimate_blur(image_np), 2),
                "noise_score": round(ImageQualityAnalyzer.estimate_noise(image_np), 2),
                "jpeg50_mae": round(jpeg_error, 2)}


class FrequencyAnalysisModule(nn.Module):
    """Encode the centered log-magnitude 2D Fourier spectrum of an image."""

    def __init__(self, in_channels=3, freq_dim=128):
        super().__init__()
        self.freq_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(), nn.Linear(64 * 8 * 8, freq_dim), nn.ReLU(),
        )

    def forward(self, x):
        spectrum = torch.fft.fft2(x.float(), dim=(-2, -1), norm="ortho")
        spectrum = torch.fft.fftshift(spectrum, dim=(-2, -1))
        log_magnitude = torch.log(torch.abs(spectrum).clamp_min(1e-8))
        return self.freq_conv(log_magnitude.to(dtype=self.freq_conv[0].weight.dtype))


class SwinFrequencyHybridConfig(PretrainedConfig):
    """Configuration for the complete Swin and frequency hybrid detector."""

    model_type = "swin-frequency-hybrid"

    def __init__(self, base_model_name="Organika/sdxl-detector", num_classes=2,
                 confidence_threshold=0.90, early_exit_loss_weight=0.25,
                 id2label=None, label2id=None, **kwargs):
        num_classes = int(num_classes)
        if id2label is None:
            id2label = {0: "artificial", 1: "human"} if num_classes == 2 else None
        if id2label is not None:
            id2label = {int(key): value for key, value in id2label.items()}
        if label2id is None and id2label is not None:
            label2id = {str(value): int(key) for key, value in id2label.items()}
        kwargs.pop("num_labels", None)
        super().__init__(num_labels=num_classes, id2label=id2label, label2id=label2id, **kwargs)
        self.base_model_name = str(base_model_name)
        self.num_classes = num_classes
        self.confidence_threshold = float(confidence_threshold)
        self.early_exit_loss_weight = float(early_exit_loss_weight)


class SwinFrequencyHybridDetector(PreTrainedModel):
    """Fuse pooled Swin features with learned features from the 2D FFT spectrum."""

    config_class = SwinFrequencyHybridConfig
    base_model_prefix = "swin_frequency_hybrid"
    main_input_name = "pixel_values"

    def __init__(self, config=None, model_name=None, num_classes=2, confidence_threshold=0.90,
                 early_exit_loss_weight=0.25):
        initialize_from_pretrained_backbone = config is None
        if config is None:
            base_model_name = model_name or "Organika/sdxl-detector"
            backbone_config = AutoConfig.from_pretrained(base_model_name)
            id2label = getattr(backbone_config, "id2label", None)
            config = SwinFrequencyHybridConfig(
                base_model_name=base_model_name,
                num_classes=num_classes,
                confidence_threshold=confidence_threshold,
                early_exit_loss_weight=early_exit_loss_weight,
                id2label=id2label,
            )
        elif not isinstance(config, SwinFrequencyHybridConfig):
            raise TypeError("config must be a SwinFrequencyHybridConfig")
        else:
            backbone_config = AutoConfig.from_pretrained(config.base_model_name)
        super().__init__(config)
        # Transformers 5.x expects this instance attribute while materializing
        # parameters missing from a checkpoint. This model has no tied weights.
        self.all_tied_weights_keys = {}
        self.base_model_name = config.base_model_name
        self.num_classes = config.num_classes
        self.confidence_threshold = config.confidence_threshold
        self.early_exit_loss_weight = config.early_exit_loss_weight
        if initialize_from_pretrained_backbone:
            # A fresh training instance needs the pretrained Swin weights.
            # When this hybrid model is itself being restored, Transformers may
            # construct it inside an empty/meta-device context. A nested
            # from_pretrained() there is invalid; build from config and let the
            # outer from_pretrained() restore the complete hybrid state dict.
            self.swin_backbone = AutoModelForImageClassification.from_pretrained(
                self.base_model_name, config=backbone_config, ignore_mismatched_sizes=True
            )
        else:
            self.swin_backbone = AutoModelForImageClassification.from_config(backbone_config)
        classifier = getattr(self.swin_backbone, "classifier", None)
        if classifier is None or not hasattr(classifier, "in_features"):
            raise ValueError("Expected a classifier layer with in_features on the Swin model")
        swin_feature_dim = classifier.in_features
        self.swin_backbone.classifier = nn.Identity()
        self.early_exit_head = nn.Linear(swin_feature_dim, self.num_classes)
        self.freq_module = FrequencyAnalysisModule(in_channels=3, freq_dim=128)
        self.fusion_layer = nn.Sequential(
            nn.Linear(swin_feature_dim + 128, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(256, self.num_classes),
        )
        self.id2label = {int(k): v for k, v in self.config.id2label.items()}
        self.label2id = {str(v): int(k) for k, v in self.id2label.items()}

    def forward(self, pixel_values=None, labels=None, enable_early_exit=False, **kwargs):
        if pixel_values is None:
            pixel_values = kwargs.get("pixel_values")
        if pixel_values is None:
            raise ValueError("pixel_values is required")
        start = time.perf_counter()
        outputs = self.swin_backbone.swin(pixel_values=pixel_values)
        spatial_features = outputs.pooler_output
        early_logits = self.early_exit_head(spatial_features)
        eligible = (F.softmax(early_logits.float(), dim=-1).amax(dim=-1) >= self.confidence_threshold)
        use_exit = bool(enable_early_exit and not self.training)
        if use_exit and eligible.all():
            logits = early_logits
            early_exited = torch.ones(logits.shape[0], dtype=torch.bool, device=logits.device)
        elif use_exit and eligible.any():
            # Run the FFT/fusion path only on uncertain members of a mixed batch.
            indexes = (~eligible).nonzero(as_tuple=True)[0]
            freq_features = self.freq_module(pixel_values.index_select(0, indexes))
            fused = self.fusion_layer(torch.cat((spatial_features.index_select(0, indexes), freq_features), dim=1))
            logits = early_logits.clone().index_copy(0, indexes, fused)
            early_exited = eligible
        else:
            freq_features = self.freq_module(pixel_values)
            logits = self.fusion_layer(torch.cat((spatial_features, freq_features), dim=1))
            early_exited = torch.zeros(logits.shape[0], dtype=torch.bool, device=logits.device)
        loss = None
        if labels is not None:
            labels = labels.view(-1)
            loss = F.cross_entropy(logits, labels)
            if self.training:
                loss = loss + self.early_exit_loss_weight * F.cross_entropy(early_logits, labels)
        latency_ms = (time.perf_counter() - start) * 1000
        # Trainer/Accelerate recursively pads every value in ModelOutput. Keep
        # Python diagnostics out of that structure; inference utilities can read
        # them from the model after the forward call.
        self.last_early_exited = early_exited.detach()
        self.last_latency_ms = latency_ms
        return HybridDetectorOutput(loss=loss, logits=logits)

    @classmethod
    def from_pretrained(cls, model_name_or_path, **kwargs):
        """Load standard Transformers checkpoints and supported legacy checkpoints."""
        path = Path(model_name_or_path)
        if not path.is_dir():
            # Preserve the previous convenience behavior for initializing from
            # a Hub backbone identifier rather than a saved hybrid directory.
            return cls(model_name=model_name_or_path, **kwargs)
        if path.is_dir():
            # The first custom format used a backbone config plus hybrid_config.json.
            legacy_metadata = path / "hybrid_config.json"
            legacy_weights = path / "pytorch_model.bin"
            if legacy_metadata.is_file() and legacy_weights.is_file():
                import json

                metadata = json.loads(legacy_metadata.read_text(encoding="utf-8"))
                config = SwinFrequencyHybridConfig(**metadata)
                model = cls(config)
                state = torch.load(legacy_weights, map_location="cpu", weights_only=True)
                incompatible = model.load_state_dict(state, strict=False)
                allowed_missing = {"early_exit_head.weight", "early_exit_head.bias"}
                if (set(incompatible.missing_keys) - allowed_missing) or incompatible.unexpected_keys:
                    raise RuntimeError(f"Checkpoint/model mismatch: {incompatible}")
                return model

            # Earlier Trainer saves had backbone-only config.json plus hybrid weights.
            # Keep this narrow compatibility path for those checkpoints.
            config_path = path / "config.json"
            import json

            raw_config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
            is_hybrid_config = raw_config.get("model_type") == cls.config_class.model_type
            if not is_hybrid_config and (path / "model.safetensors").is_file():
                model = cls(**kwargs)
                from safetensors.torch import load_file
                state = load_file(str(path / "model.safetensors"), device="cpu")
                incompatible = model.load_state_dict(state, strict=False)
                allowed_missing = {"early_exit_head.weight", "early_exit_head.bias"}
                if (set(incompatible.missing_keys) - allowed_missing) or incompatible.unexpected_keys:
                    raise RuntimeError(f"Checkpoint/model mismatch: {incompatible}")
                return model

        return super().from_pretrained(model_name_or_path, **kwargs)
