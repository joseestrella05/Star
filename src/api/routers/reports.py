from fastapi import APIRouter, HTTPException, Request, Response

from src.reports.engine import generate_report
from src.reports.export import to_pdf
from src.reports.schemas import ExportRequest, ModelInfo, ReportRequest, ReportResponse

router = APIRouter(prefix="/api/v1/reports", tags=["reports"])


@router.post("/generate", response_model=ReportResponse)
def generate(req: ReportRequest, request: Request):  # sync def: runs in a worker thread
    star = request.app.state.star
    info = ModelInfo(checkpoint_sha256=star["checkpoint_sha256"],
                     params=star["model"].num_params(non_embedding=False))
    return generate_report(req, star["complete"], info)


@router.post("/export", response_class=Response,
             responses={200: {"content": {"application/pdf": {}, "text/markdown": {}}}})
def export(req: ExportRequest):
    if req.format == "md":
        content, media_type = req.markdown.encode("utf-8"), "text/markdown; charset=utf-8"
    else:
        try:
            content = to_pdf(req.markdown, req.footer)
        except OSError as e:  # missing Pango/GObject system libraries
            raise HTTPException(503, f"PDF export unavailable on this host: {e}") from e
        media_type = "application/pdf"
    return Response(content, media_type=media_type,
                    headers={"Content-Disposition": f'attachment; filename="star-report.{req.format}"'})
