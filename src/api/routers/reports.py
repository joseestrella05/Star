from fastapi import APIRouter, HTTPException, Request, Response

from src.reports.engine import generate_report
from src.reports.export import to_pdf
from src.reports.ingest import MAX_BYTES, IngestError, parse_upload, template_csv
from src.reports.schemas import AnyReportRequest, ExportRequest, ModelInfo, ReportResponse

router = APIRouter(prefix="/api/v1/reports", tags=["reports"])


@router.post("/generate", response_model=ReportResponse)
def generate(req: AnyReportRequest, request: Request):  # sync def: runs in a worker thread
    star = request.app.state.star
    info = ModelInfo(checkpoint_sha256=star["checkpoint_sha256"],
                     params=star["model"].num_params(non_embedding=False))
    return generate_report(req, star["complete"], info, star["currency_label"])


@router.get("/template.csv", response_class=Response)
def template():
    return Response(template_csv(), media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": 'attachment; filename="star-metrics-template.csv"'})


@router.post("/ingest", openapi_extra={"requestBody": {"content": {
    "text/csv": {"schema": {"type": "string", "format": "binary"}},
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": {
        "schema": {"type": "string", "format": "binary"}}}}})
async def ingest(request: Request):
    """Raw file body (CSV or XLSX, detected by content) filled from /template.csv."""
    if int(request.headers.get("content-length") or 0) > MAX_BYTES:
        raise HTTPException(413, f"file larger than {MAX_BYTES // 1000} KB")
    try:
        result = parse_upload(await request.body())
    except IngestError as e:
        raise HTTPException(422, e.errors) from e
    return {"fields": result["fields"], "metrics": result["metrics"].model_dump(mode="json")}


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
