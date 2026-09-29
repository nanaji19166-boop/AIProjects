from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from pathlib import Path
import uuid

from translator import analyze_pdf, translate_pdf

BASE = Path(__file__).parent
DATA = BASE / "data"
DATA.mkdir(exist_ok=True)

app = FastAPI(title="TeluguSetu", version="1.0.0")

MAX_UPLOAD_BYTES = 50 * 1024 * 1024

@app.get("/health")
def health():
    return {"status": "ok", "service": "TeluguSetu"}

@app.get("/", response_class=HTMLResponse)
def home():
    return (BASE / "templates" / "index.html").read_text(encoding="utf-8")

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        return JSONResponse({"error": "Please upload a PDF."}, status_code=400)
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        return JSONResponse({"error": "PDF is larger than 50 MB."}, status_code=413)
    job = uuid.uuid4().hex
    src = DATA / f"{job}.pdf"
    src.write_bytes(data)
    try:
        result = analyze_pdf(src)
        result["job_id"] = job
        return result
    except Exception as exc:
        src.unlink(missing_ok=True)
        return JSONResponse({"error": f"PDF analysis failed: {exc}"}, status_code=500)

@app.post("/translate")
async def translate(job_id: str = Form(...)):
    src = DATA / f"{job_id}.pdf"
    if not src.exists():
        return JSONResponse({"error": "Job not found."}, status_code=404)
    try:
        return translate_pdf(src, DATA / job_id)
    except Exception as exc:
        return JSONResponse({"error": f"Translation failed: {exc}"}, status_code=500)

@app.get("/download/{job_id}/{kind}")
def download(job_id: str, kind: str):
    if kind not in {"pdf", "docx"}:
        return JSONResponse({"error": "Invalid format."}, status_code=400)
    p = DATA / f"{job_id}_telugu.{kind}"
    if not p.exists():
        return JSONResponse({"error": "Output is not ready."}, status_code=404)
    media = "application/pdf" if kind == "pdf" else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    return FileResponse(p, filename=p.name, media_type=media)
