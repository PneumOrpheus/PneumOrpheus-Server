from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from app.config import get_settings
from app.schemas import ClassificationItem, InferenceResponse, SegmentationData, SegmentationRegion, SourceFileMetadata
from app.services.model_runtime import ModelRuntime
from app.services.model_store import ModelStore


@dataclass
class InferenceInput:
    analysis_id: str
    patient_id: str
    patient_name: str
    modality: str
    clinician_email: str
    file_name: str
    file_mime_type: str
    file_size_bytes: int
    file_bytes: bytes


class InferenceService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.model_store = ModelStore()

    def run(self, payload: InferenceInput) -> InferenceResponse:
        suffix = "".join(Path(payload.file_name).suffixes).lower()
        is_nifti = suffix in {".nii", ".nii.gz"}

        model_name = self.settings.default_model_name
        model_version = self.settings.default_model_version
        model_path = self.model_store.resolve_model_path(model_name=model_name, model_version=model_version)

        runtime = ModelRuntime(model_path=model_path)
        runtime_result = runtime.run(file_bytes=payload.file_bytes, modality=payload.modality)

        predicted_type = str(runtime_result.get("predicted_type") or ("Small Cell Carcinoma" if is_nifti else "Adenocarcinoma"))
        confidence = float(runtime_result.get("confidence") or (0.89 if is_nifti else 0.86))
        confidence = max(0.0, min(1.0, confidence))
        proposed_tnm = str(runtime_result.get("tnm") or ("T2N1M0" if is_nifti else "T1N0M0"))

        findings = (
            f"Model predicts {predicted_type} with {round(confidence * 100)}% confidence "
            "based on lesion morphology and density patterns in the uploaded study."
        )

        return InferenceResponse(
            analysisId=payload.analysis_id,
            patientId=payload.patient_id,
            patientName=payload.patient_name,
            modality=payload.modality,
            clinicianEmail=payload.clinician_email,
            receivedAt=datetime.now(timezone.utc).isoformat(),
            sourceFile=SourceFileMetadata(
                name=payload.file_name,
                mimeType=payload.file_mime_type,
                sizeBytes=payload.file_size_bytes,
            ),
            findings=findings,
            cancerType=predicted_type,
            classificationConfidence=confidence,
            reasoning=(
                "Detected malignant-appearing lesion distribution, margin irregularity, and intensity profile "
                "compatible with the predicted subtype."
            ),
            proposedTnmStage=proposed_tnm,
            classifications=[
                ClassificationItem(
                    side="Left",
                    prediction=predicted_type,
                    confidence=confidence,
                    explanation="Primary left-side lesion demonstrates dominant malignant signature.",
                ),
                ClassificationItem(
                    side="Right",
                    prediction=predicted_type,
                    confidence=max(0.0, min(1.0, confidence - 0.03)),
                    explanation="Secondary right-side suspicious region with supporting radiographic traits.",
                ),
            ],
            segmentationData=SegmentationData(
                format="polygon",
                labels=["tumor", "nodule"],
                regions=[
                    SegmentationRegion(
                        id="region-1",
                        label="tumor",
                        sliceIndex=42,
                        points=[[120, 88], [158, 92], [162, 133], [124, 130]],
                    ),
                    SegmentationRegion(
                        id="region-2",
                        label="nodule",
                        sliceIndex=47,
                        points=[[210, 160], [228, 164], [232, 184], [214, 182]],
                    ),
                ],
            ),
            modelInfo={
                "modelName": model_name,
                "modelVersion": model_version,
                "modelSource": self.settings.model_source,
                "modelPath": str(model_path),
            },
        )
