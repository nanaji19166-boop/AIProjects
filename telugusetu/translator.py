import html,os,re,urllib.request
from pathlib import Path
import fitz,numpy as np
from docx import Document
from docx.shared import Pt
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,PageBreak
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
MODEL_ID=os.getenv("MODEL_ID","hari31416/indictrans2-en-indic-dist-200M-ONNX-int8"); SRC_LANG="eng_Latn"; TGT_LANG="tel_Telu"
FONT_URL="https://raw.githubusercontent.com/notofonts/noto-fonts/main/hinted/ttf/NotoSansTelugu/NotoSansTelugu-Regular.ttf"
GLOSSARY={"New START":"New START","NATO":"NATO","UN":"UN","WHO":"WHO","GDP":"GDP","EPCIS":"EPCIS","UPSC":"UPSC","MCQ":"MCQ","GS":"GS","IAS":"IAS","IPS":"IPS","PDF":"PDF"}
MARKER_RE=re.compile(r"^(\s*(?:(?:Q(?:uestion)?\s*)?\d{1,4}[\).:\-]|[A-D][\).:\-]|(?:Correct\s+Answer|Answer|Ans\.?|Explanation)\s*[:\-]))\s*(.*)$",re.I)
class IndicTransONNX:
    def __init__(self,model_path):
        import json,onnxruntime as ort
        from tokenizers import Tokenizer
        from huggingface_hub import snapshot_download
        if "/" in model_path and not Path(model_path).exists(): model_path=snapshot_download(repo_id=model_path)
        snap=Path(model_path); self.src_tok=Tokenizer.from_file(str(snap/"tokenizer_src.json")); self.tgt_tok=Tokenizer.from_file(str(snap/"tokenizer_tgt.json")); self.meta=json.loads((snap/"tokenizer_meta.json").read_text(encoding="utf-8"))
        cfg=json.loads((snap/"generation_config.json").read_text(encoding="utf-8")) if (snap/"generation_config.json").exists() else {}; self.start_id=int(cfg.get("decoder_start_token_id",2)); self.eos_id=int(cfg.get("eos_token_id",2))
        opts=ort.SessionOptions(); opts.intra_op_num_threads=min(2,os.cpu_count() or 2); opts.inter_op_num_threads=1
        self.enc=ort.InferenceSession(str(snap/"encoder_model.onnx"),sess_options=opts,providers=["CPUExecutionProvider"]); self.dec=ort.InferenceSession(str(snap/"decoder_model.onnx"),sess_options=opts,providers=["CPUExecutionProvider"]); self.dec_past=ort.InferenceSession(str(snap/"decoder_with_past_model.onnx"),sess_options=opts,providers=["CPUExecutionProvider"]); self.num_layers=(len(self.dec.get_outputs())-1)//4
    def _past_feed(self,past):
        f={}
        for i in range(self.num_layers):
            b=i*4; f[f"past_key_values.{i}.decoder.key"]=past[b]; f[f"past_key_values.{i}.decoder.value"]=past[b+1]; f[f"past_key_values.{i}.encoder.key"]=past[b+2]; f[f"past_key_values.{i}.encoder.value"]=past[b+3]
        return f
    def translate(self,text,max_new_tokens=160):
        e=self.src_tok.encode(f"{SRC_LANG} {TGT_LANG} {text}"); ids=np.array([[i if i<self.meta["src_dict_size"] else self.meta["unk_id"] for i in e.ids]],dtype=np.int64); mask=np.array([e.attention_mask],dtype=np.int64); enc=self.enc.run(["last_hidden_state"],{"input_ids":ids,"attention_mask":mask})[0]; dec_ids=np.array([[self.start_id]],dtype=np.int64); out_ids=[self.start_id]; past=None
        for step in range(max_new_tokens):
            out=self.dec.run(None,{"input_ids":dec_ids,"encoder_hidden_states":enc,"encoder_attention_mask":mask}) if step==0 else self.dec_past.run(None,{"input_ids":dec_ids,"encoder_attention_mask":mask,**self._past_feed(past)}); past=list(out[1:]); nxt=int(np.argmax(out[0][0,-1,:])); out_ids.append(nxt)
            if nxt==self.eos_id: break
            dec_ids=np.array([[nxt]],dtype=np.int64)
        return self.tgt_tok.decode([i if i<self.meta["tgt_dict_size"] else self.meta["unk_id"] for i in out_ids],skip_special_tokens=True)
class LocalTranslator:
    def __init__(self): self.model=None
    def load(self):
        if self.model is None: self.model=IndicTransONNX(MODEL_ID)
    def translate_batch(self,texts):
        self.load(); out=[]
        for text in texts:
            protected={}; i=0
            for src,val in sorted(GLOSSARY.items(),key=lambda x:-len(x[0])):
                if src in text: token=f"ZXQTERM{i}ZXQ"; text=text.replace(src,token); protected[token]=val; i+=1
            for value in re.findall(r"https?://\S+|\b[A-Z][A-Z0-9./_-]{2,}\b",text):
                if value not in protected: token=f"ZXQTERM{i}ZXQ"; text=text.replace(value,token); protected[token]=value; i+=1
            v=self.model.translate(text)
            for token,val in protected.items(): v=v.replace(token,val)
            out.append(v)
        return out
translator=LocalTranslator()
def extract_pages(pdf):
    doc=fitz.open(pdf); pages=[]
    try:
        for n,page in enumerate(doc,1):
            blocks=sorted(page.get_text("blocks"),key=lambda b:(b[1],b[0])); pages.append({"page":n,"text":"\n".join(b[4].strip() for b in blocks if b[4].strip())})
    finally: doc.close()
    return pages
def analyze_pdf(pdf):
    pages=extract_pages(pdf); full="\n".join(x["text"] for x in pages); q=re.findall(r"(?m)^\s*(?:Q(?:uestion)?\s*)?(\d{1,4})[).:\-\s]",full,re.I); opts=re.findall(r"(?m)^\s*[A-D][).:\-\s]",full); answers=re.findall(r"(?im)^\s*(answer|ans\.?|correct answer|explanation)\s*[:\-]",full)
    return {"pages":len(pages),"questions_detected":len(set(q)),"options_detected":len(opts),"nonempty_pages":sum(bool(x["text"]) for x in pages),"answer_markers":len(answers)}
def _split_marker(line):
    m=MARKER_RE.match(line); return (m.group(1),m.group(2)) if m else ("",line)
def translate_pages(pages,progress_callback=None):
    translated=[]; total=len(pages)
    for pi,page in enumerate(pages,1):
        lines=page["text"].splitlines(); result=[]; pending=[]; indexes=[]
        def flush():
            nonlocal pending,indexes
            if not pending:return
            vals=translator.translate_batch(pending)
            for idx,val in zip(indexes,vals): result[idx]=val
            pending=[]; indexes=[]
        for line in lines:
            if not line.strip(): result.append(""); continue
            marker,body=_split_marker(line)
            if not body.strip(): result.append(marker); continue
            if re.fullmatch(r"[\d\s|/_.:-]+",body.strip()): result.append(marker+body); continue
            result.append(marker); pending.append(body); indexes.append(len(result)-1)
            if len(pending)>=8: flush()
        flush(); translated.append({"page":page["page"],"lines":result})
        if progress_callback: progress_callback(int(pi*70/max(total,1)))
    return translated
def _pdf_font():
    for font in [Path("/usr/share/fonts/truetype/noto/NotoSansTelugu-Regular.ttf"),Path("fonts/NotoSansTelugu-Regular.ttf"),Path("data/NotoSansTelugu-Regular.ttf")]:
        if font.exists():
            try: pdfmetrics.registerFont(TTFont("NotoTelugu",str(font))); return "NotoTelugu"
            except Exception: pass
    target=Path("data/NotoSansTelugu-Regular.ttf"); target.parent.mkdir(exist_ok=True); urllib.request.urlretrieve(FONT_URL,target); pdfmetrics.registerFont(TTFont("NotoTelugu",str(target))); return "NotoTelugu"
def translate_pdf(pdf,out_prefix,progress_callback=None):
    pages=translate_pages(extract_pages(pdf),progress_callback); out_pdf=Path(f"{out_prefix}_telugu.pdf"); out_docx=Path(f"{out_prefix}_telugu.docx"); style=ParagraphStyle("body",fontName=_pdf_font(),fontSize=10.2,leading=14,spaceAfter=4); doc=SimpleDocTemplate(str(out_pdf),rightMargin=36,leftMargin=36,topMargin=36,bottomMargin=36,title="TeluguSetu Translation"); story=[]
    for p in pages:
        for line in p["lines"]:
            if line.strip(): story.extend([Paragraph(html.escape(line),style),Spacer(1,2)])
        story.append(PageBreak())
    if story and isinstance(story[-1],PageBreak): story.pop()
    doc.build(story); d=Document()
    for pi,page in enumerate(pages):
        if pi:d.add_page_break()
        for line in page["lines"]:
            if line.strip():
                para=d.add_paragraph(line); para.paragraph_format.space_after=Pt(3)
                for run in para.runs: run.font.name="Noto Sans Telugu"; run.font.size=Pt(10.5)
    d.save(out_docx)
    if progress_callback: progress_callback(100)
    return {"status":"complete","pages":len(pages),"pdf":f"/download/{Path(out_prefix).name}/pdf","docx":f"/download/{Path(out_prefix).name}/docx"}
