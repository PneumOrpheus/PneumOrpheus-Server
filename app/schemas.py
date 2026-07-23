from typing import Any

from pydantic import BaseModel, Field


class SourceFileMetadata(BaseModel):
    name: str
    mimeType: str
    sizeBytes: int


class ClassificationItem(BaseModel):
    side: str
    prediction: str
    confidence: float
    explanation: str


class InferenceResponse(BaseModel):
    analysisId: str
    patientId: str
    patientName: str
    modality: str
    clinicianEmail: str
    receivedAt: str
    sourceFile: SourceFileMetadata
    findings: str
    cancerType: str
    classificationConfidence: float = Field(ge=0, le=1)
    reasoning: str
    proposedTnmStage: str
    classifications: list[ClassificationItem]
    segmentationData: dict[str, Any] | None = None
    modelInfo: dict[str, Any] = Field(default_factory=dict)


class DiagnosisResponse(BaseModel):
    status: str


class InferenceJobAccepted(BaseModel):
    analysisId: str
    status: str = "processing"
