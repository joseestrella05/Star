from fastapi import APIRouter, Request

from src.reports.engine import generate_report
from src.reports.schemas import ModelInfo, ReportRequest, ReportResponse

router = APIRouter(prefix="/api/v1/reports", tags=["reports"])


@router.post("/generate", response_model=ReportResponse)
def generate(req: ReportRequest, request: Request):  # sync def: runs in a worker thread
    star = request.app.state.star
    info = ModelInfo(checkpoint_sha256=star["checkpoint_sha256"],
                     params=star["model"].num_params(non_embedding=False))
    return generate_report(req, star["complete"], info)
