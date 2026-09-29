from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading, uuid, traceback
from translator import analyze_pdf, translate_pdf
BASE=Path(__file__).parent; DATA=BASE/"data"; DATA.mkdir(exist_ok=True)
app=FastAPI(title="TeluguSetu",version="1.1.0"); MAX_UPLOAD_BYTES=50*1024*1024
executor=ThreadPoolExecutor(max_workers=1); jobs={}; jobs_lock=threading.Lock()
@app.get("/health")
def health(): return {"status":"ok","service":"TeluguSetu","version":app.version}
@app.get("/",response_class=HTMLResponse)
def home(): return (BASE/"templates"/"index.html").read_text(encoding="utf-8")
@app.post("/analyze")
async def analyze(file:UploadFile=File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"): return JSONResponse({"error":"Please upload a PDF."},status_code=400)
    data=await file.read()
    if len(data)>MAX_UPLOAD_BYTES: return JSONResponse({"error":"PDF is larger than 50 MB."},status_code=413)
    job=uuid.uuid4().hex; src=DATA/f"{job}.pdf"; src.write_bytes(data)
    try:
        result=analyze_pdf(src); result["job_id"]=job
        with jobs_lock: jobs[job]={"state":"analyzed",**result}
        return result
    except Exception as exc:
        src.unlink(missing_ok=True); return JSONResponse({"error":f"PDF analysis failed: {exc}"},status_code=500)
def _set_progress(job_id,progress):
    with jobs_lock:
        if job_id in jobs: jobs[job_id]["progress"]=progress
def _run_translation(job_id):
    try:
        with jobs_lock: jobs[job_id]={"state":"translating","progress":0}
        result=translate_pdf(DATA/f"{job_id}.pdf",DATA/job_id,progress_callback=lambda p:_set_progress(job_id,p))
        with jobs_lock: jobs[job_id]={"state":"complete",**result}
    except Exception as exc:
        traceback.print_exc()
        with jobs_lock: jobs[job_id]={"state":"error","error":f"Translation failed: {exc}"}
@app.post("/translate")
async def translate(job_id:str=Form(...)):
    if not (DATA/f"{job_id}.pdf").exists(): return JSONResponse({"error":"Job not found."},status_code=404)
    with jobs_lock:
        current=jobs.get(job_id,{})
        if current.get("state") in {"queued","translating"}: return {"job_id":job_id,"state":current["state"],"progress":current.get("progress",0)}
        jobs[job_id]={"state":"queued","progress":0}
    executor.submit(_run_translation,job_id); return {"job_id":job_id,"state":"queued","progress":0}
@app.get("/status/{job_id}")
def status(job_id:str):
    with jobs_lock: state=jobs.get(job_id)
    if not state:
        # A free Render instance can restart during a memory-heavy job. Give the
        # browser a useful retry message instead of a generic 404/translation failure.
        if (DATA/f"{job_id}.pdf").exists():
            return {"state":"error","error":"The translation worker restarted before finishing. Please tap Translate to Telugu again."}
        return JSONResponse({"error":"Job not found. Please analyze the PDF again."},status_code=404)
    return state
@app.get("/download/{job_id}/{kind}")
def download(job_id:str,kind:str):
    if kind not in {"pdf","docx"}: return JSONResponse({"error":"Invalid format."},status_code=400)
    p=DATA/f"{job_id}_telugu.{kind}"
    if not p.exists(): return JSONResponse({"error":"Output is not ready."},status_code=404)
    media="application/pdf" if kind=="pdf" else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    return FileResponse(p,filename=p.name,media_type=media)
