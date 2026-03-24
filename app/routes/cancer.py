from fastapi import APIRouter

from app.schemas import DiagnosisResponse

router = APIRouter(tags=["cancer"])


@router.get("/cancer", response_model=DiagnosisResponse)
async def cancer() -> DiagnosisResponse:
    return DiagnosisResponse(status="ok")
