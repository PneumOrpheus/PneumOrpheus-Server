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


class SegmentationRegion(BaseModel):
    id: str
    label: str
    sliceIndex: int
    points: list[list[int]]


class SegmentationData(BaseModel):
    format: str = "polygon"
    labels: list[str]
    regions: list[SegmentationRegion]


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
    segmentationData: SegmentationData
    modelInfo: dict[str, Any] = Field(default_factory=dict)


class DiagnosisResponse(BaseModel):
    status: str
