import re
from pathlib import Path
import fitz
from docx import Document
from docx.shared import Pt
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,PageBreak
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

GLOSSARY={'New START':'New START','NATO':'NATO','UN':'UN','WHO':'WHO','GDP':'GDP','EPCIS':'EPCIS','UPSC':'UPSC','MCQ':'MCQ'}

def protect(text):
    protected={}
    for i,(src,val) in enumerate(GLOSSARY.items()):
        token=f'__TS_{i}__'
        if src in text: text=text.replace(src,token); protected[token]=val
    return text,protected

def restore(text,p):
    for k,v in p.items(): text=text.replace(k,v)
    return text

class LocalTranslator:
    def __init__(self): self.model=None
    def load(self):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        from IndicTransToolkit import IndicProcessor
        self.ip=IndicProcessor(inference=True)
        self.tokenizer=AutoTokenizer.from_pretrained('ai4bharat/indictrans2-en-indic-dist-200M',trust_remote_code=True)
        self.model=AutoModelForSeq2SeqLM.from_pretrained('ai4bharat/indictrans2-en-indic-dist-200M',trust_remote_code=True)
        self.model.eval()
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
    def translate(self,text):
        if not text.strip(): return text
        if self.model is None: self.load()
        import torch
        safe,p=protect(text)
        batch=self.ip.preprocess_batch([safe],src_lang='eng_Latn',tgt_lang='tel_Telu',visualize=False)
        inputs=self.tokenizer(batch,padding='longest',truncation=True,max_length=256,return_tensors='pt').to(self.device)
        with torch.inference_mode():
            out=self.model.generate(**inputs,num_beams=5,num_return_sequences=1,max_length=256)
        decoded=self.tokenizer.batch_decode(out,skip_special_tokens=True)
        return restore(decoded[0] if decoded else safe,p)

translator=LocalTranslator()

def extract_pages(pdf):
    doc=fitz.open(pdf); out=[]
    for n,page in enumerate(doc,1):
        blocks=sorted(page.get_text('blocks'),key=lambda b:(b[1],b[0]))
        out.append({'page':n,'text':'\n'.join(b[4].strip() for b in blocks if b[4].strip())})
    return out

def analyze_pdf(pdf):
    pages=extract_pages(pdf); full='\n'.join(x['text'] for x in pages)
    q=re.findall(r'(?m)^\s*(?:Q(?:uestion)?\s*)?(\d{1,4})[).:\-\s]',full,re.I)
    opts=re.findall(r'(?m)^\s*[A-D][).:\-\s]',full)
    return {'pages':len(pages),'questions_detected':len(set(q)),'options_detected':len(opts),'nonempty_pages':sum(bool(x['text']) for x in pages),'answer_markers':len(re.findall(r'(?im)^\s*(answer|ans\.?|correct answer|explanation)\s*[:\-]',full))}

def translate_pdf(pdf,out_prefix):
    pages=extract_pages(pdf); translated=[]
    for p in pages: translated.append({'page':p['page'],'text':translator.translate(p['text'])})
    out_pdf=Path(f'{out_prefix}_telugu.pdf'); out_docx=Path(f'{out_prefix}_telugu.docx')
    font=Path('fonts/NotoSansTelugu-Regular.ttf')
    if font.exists(): pdfmetrics.registerFont(TTFont('NotoTelugu',str(font))); fname='NotoTelugu'
    else: fname='Helvetica'
    style=ParagraphStyle('body',fontName=fname,fontSize=10.5,leading=15)
    doc=SimpleDocTemplate(str(out_pdf),rightMargin=36,leftMargin=36,topMargin=36,bottomMargin=36)
    story=[]
    for p in translated:
        for line in p['text'].split('\n'):
            if line.strip(): story += [Paragraph(line.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;'),style),Spacer(1,5)]
        story.append(PageBreak())
    doc.build(story)
    d=Document()
    for i,p in enumerate(translated):
        if i: d.add_page_break()
        for line in p['text'].split('\n'):
            if line.strip():
                r=d.add_paragraph(line); r.style.font.name='Noto Sans Telugu'; r.style.font.size=Pt(10.5)
    d.save(out_docx)
    return {'status':'complete','pages':len(pages),'pdf':f'/download/{Path(out_prefix).name}/pdf','docx':f'/download/{Path(out_prefix).name}/docx'}
