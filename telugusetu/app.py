from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from pathlib import Path
import uuid
from translator import analyze_pdf, translate_pdf

BASE=Path(__file__).parent
DATA=BASE/'data'; DATA.mkdir(exist_ok=True)
app=FastAPI(title='TeluguSetu')

@app.get('/',response_class=HTMLResponse)
def home(): return (BASE/'templates'/'index.html').read_text(encoding='utf-8')

@app.post('/analyze')
async def analyze(file:UploadFile=File(...)):
    if not file.filename.lower().endswith('.pdf'): return JSONResponse({'error':'Please upload a PDF.'},status_code=400)
    job=uuid.uuid4().hex; src=DATA/f'{job}.pdf'; src.write_bytes(await file.read())
    result=analyze_pdf(src); result['job_id']=job; return result

@app.post('/translate')
async def translate(job_id:str):
    src=DATA/f'{job_id}.pdf'
    if not src.exists(): return JSONResponse({'error':'Job not found.'},status_code=404)
    try: return translate_pdf(src,DATA/job_id)
    except Exception as e: return JSONResponse({'error':str(e)},status_code=500)

@app.get('/download/{job_id}/{kind}')
def download(job_id:str,kind:str):
    if kind not in {'pdf','docx'}: return JSONResponse({'error':'Invalid format.'},status_code=400)
    p=DATA/f'{job_id}_telugu.{kind}'
    if not p.exists(): return JSONResponse({'error':'Output is not ready.'},status_code=404)
    return FileResponse(p,filename=p.name)
