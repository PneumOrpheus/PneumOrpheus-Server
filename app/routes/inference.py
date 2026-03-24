from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.schemas import InferenceResponse
from app.security import verify_api_key
from app.services.inference_service import InferenceInput, InferenceService, InferenceServiceError

router = APIRouter(tags=["inference"])
service = InferenceService()


@router.post("/infer", response_model=InferenceResponse, dependencies=[Depends(verify_api_key)])
@router.post("/v1/infer", response_model=InferenceResponse, dependencies=[Depends(verify_api_key)])
async def infer(
    analysisId: str = Form(...),
    patientId: str = Form(...),
    patientName: str = Form(...),
    modality: str = Form(...),
    clinicianEmail: str = Form(...),
    studyFile: UploadFile = File(...),
) -> InferenceResponse:
    file_bytes = await studyFile.read()

    payload = InferenceInput(
        analysis_id=analysisId,
        patient_id=patientId,
        patient_name=patientName,
        modality=modality,
        clinician_email=clinicianEmail,
        file_name=studyFile.filename or "study-file",
        file_mime_type=studyFile.content_type or "application/octet-stream",
        file_size_bytes=len(file_bytes),
        file_bytes=file_bytes,
    )

    try:
        return service.run(payload)
    except InferenceServiceError as error:
        raise HTTPException(status_code=503, detail=f"Inference service unavailable: {error}") from error
