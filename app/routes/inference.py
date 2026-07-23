import asyncio

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.schemas import InferenceJobAccepted, InferenceResponse
from app.security import verify_api_key
from app.services import job_store
from app.services.inference_service import InferenceInput, InferenceService, InferenceServiceError

router = APIRouter(tags=["inference"])
service = InferenceService()

# Keep references to spawned background tasks
_background_tasks: set[asyncio.Task] = set()


async def _run_job(analysis_id: str, payload: InferenceInput) -> None:
    try:
        result = await asyncio.to_thread(service.run, payload)
        job_store.complete_job(analysis_id, result)
    except InferenceServiceError as error:
        job_store.fail_job(analysis_id, str(error))
    except Exception as error:
        job_store.fail_job(analysis_id, str(error))


@router.post("/infer", response_model=InferenceJobAccepted, dependencies=[Depends(verify_api_key)])
@router.post("/v1/infer", response_model=InferenceJobAccepted, dependencies=[Depends(verify_api_key)])
async def infer(
    analysisId: str = Form(...),
    patientId: str = Form(...),
    patientName: str = Form(...),
    modality: str = Form(...),
    clinicianEmail: str = Form(...),
    studyFile: UploadFile = File(...),
) -> InferenceJobAccepted:
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

    job_store.start_job(analysisId)
    task = asyncio.create_task(_run_job(analysisId, payload))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

    return InferenceJobAccepted(analysisId=analysisId)


@router.get("/infer/{analysisId}", dependencies=[Depends(verify_api_key)])
@router.get("/v1/infer/{analysisId}", dependencies=[Depends(verify_api_key)])
async def infer_status(analysisId: str) -> dict:
    job = job_store.take_job(analysisId)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown or expired analysis job.")

    if job["status"] == "processing":
        return {"status": "processing"}

    if job["status"] == "failed":
        return {"status": "failed", "error": job["error"]}

    result: InferenceResponse = job["result"]
    return {"status": "completed", **result.model_dump()}
