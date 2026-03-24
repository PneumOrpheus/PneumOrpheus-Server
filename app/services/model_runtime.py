from __future__ import annotations

from pathlib import Path


class ModelRuntime:
    """Hook point for your real DL inference code.

    Replace `run` with model loading/inference (PyTorch, MONAI, nnU-Net, etc.).
    Keep the return structure aligned with `app/schemas.py` so `pneumorpheus-app`
    can consume responses without changes.
    """

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path

    def run(self, file_bytes: bytes, modality: str) -> dict:
        # Placeholder implementation. Replace with actual segmentation/classification.
        _ = (file_bytes, modality)
        return {
            "predicted_type": "Adenocarcinoma",
            "confidence": 0.86,
            "tnm": "T1N0M0",
        }
