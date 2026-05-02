from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


CLASS_ORDER = ["halfling", "human", "lizard", "orc", "wingedrat"]
HOSTILE_CLASSES = {"orc", "lizard", "wingedrat"}
BRIBABLE_CLASSES = {"human", "halfling"}


@dataclass
class EntityPrediction:
    label: str
    confidence: float
    is_hostile: bool
    is_bribable: bool


class Task1EntityClassifier:
    """Lightweight adapter around Task1 models for on-policy entity hints."""

    def __init__(
        self,
        model_type: str,
        model_path: Path,
        image_size: int = 80,
        device: str = "cpu",
    ) -> None:
        self.model_type = model_type
        self.model_path = model_path
        self.image_size = image_size
        self.device = device

        self._svm_model = None
        self._resnet_model = None
        self._resnet_preprocess = None
        self.class_names = CLASS_ORDER.copy()

        if model_type == "hog_svm" and model_path.exists():
            self._load_hog_svm()
        elif model_type == "resnet18" and model_path.exists():
            self._load_resnet18()

    def _load_hog_svm(self) -> None:
        import joblib

        payload = joblib.load(self.model_path)
        self._svm_model = payload.get("model")
        self.class_names = payload.get("class_names", CLASS_ORDER)

    def _load_resnet18(self) -> None:
        import torch
        from torchvision import models, transforms

        payload = torch.load(self.model_path, map_location=self.device)
        class_names = payload.get("class_names", CLASS_ORDER)
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
        model.load_state_dict(payload["model"])
        model.to(self.device)
        model.eval()

        self.class_names = class_names
        self._resnet_model = model
        self._resnet_preprocess = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @staticmethod
    def _to_rgb(camera_view: np.ndarray) -> np.ndarray:
        arr = np.asarray(camera_view, dtype=np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        return arr

    def _predict_with_rule(self, camera_view: np.ndarray) -> EntityPrediction:
        mean_intensity = float(np.asarray(camera_view).mean())
        if mean_intensity <= 5:
            label = "orc"
            confidence = 0.6
        elif mean_intensity >= 245:
            label = "human"
            confidence = 0.6
        else:
            label = "halfling"
            confidence = 0.5
        return EntityPrediction(
            label=label,
            confidence=confidence,
            is_hostile=label in HOSTILE_CLASSES,
            is_bribable=label in BRIBABLE_CLASSES,
        )

    def predict(self, camera_view: np.ndarray) -> EntityPrediction:
        if self._svm_model is not None:
            from PIL import Image
            from skimage.feature import hog

            img = Image.fromarray(np.asarray(camera_view, dtype=np.uint8)).convert("L").resize((self.image_size, self.image_size))
            arr = np.asarray(img, dtype=np.float32) / 255.0
            feat = hog(
                arr,
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                block_norm="L2-Hys",
                feature_vector=True,
            )
            feat = np.asarray(feat, dtype=np.float32).reshape(1, -1)
            pred = str(self._svm_model.predict(feat)[0])
            confidence = 0.5
            if hasattr(self._svm_model, "decision_function"):
                margin = np.asarray(self._svm_model.decision_function(feat)).reshape(-1)
                confidence = float(1.0 / (1.0 + np.exp(-np.max(margin))))
            return EntityPrediction(
                label=pred,
                confidence=confidence,
                is_hostile=pred in HOSTILE_CLASSES,
                is_bribable=pred in BRIBABLE_CLASSES,
            )

        if self._resnet_model is not None and self._resnet_preprocess is not None:
            import torch

            rgb = self._to_rgb(camera_view)
            x = self._resnet_preprocess(rgb).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self._resnet_model(x)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            idx = int(np.argmax(probs))
            pred = str(self.class_names[idx])
            return EntityPrediction(
                label=pred,
                confidence=float(probs[idx]),
                is_hostile=pred in HOSTILE_CLASSES,
                is_bribable=pred in BRIBABLE_CLASSES,
            )

        return self._predict_with_rule(camera_view)


def resolve_task1_model_path(task3_root: Path, model_type: str, seed: int) -> Path:
    model_dir = task3_root.parent / "task1_supervised" / "outputs" / "models"
    if model_type == "hog_svm":
        return model_dir / f"hog_svm_seed{seed}.joblib"
    if model_type == "resnet18":
        return model_dir / f"resnet18_seed{seed}.pt"
    return Path("missing.model")
