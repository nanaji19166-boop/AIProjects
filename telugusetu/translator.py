import html
import os
import re
from pathlib import Path
import fitz
from docx import Document
from docx.shared import Pt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

MODEL_ID = os.getenv("MODEL_ID", "naklitechie/indictrans2-en-indic-dist-200M")
SRC_LANG = "eng_Latn"
TGT_LANG = "tel_Telu"

GLOSSARY = {
    "New START": "New START", "NATO": "NATO", "UN": "UN", "WHO": "WHO",
    "GDP": "GDP", "EPCIS": "EPCIS", "UPSC": "UPSC", "MCQ": "MCQ",
    "GS": "GS", "IAS": "IAS", "IPS": "IPS", "PDF": "PDF",
}

MARKER_RE = re.compile(
    r"^(\s*(?:(?:Q(?:uestion)?\s*)?\d{1,4}[\).:\-]|[A-D][\).:\-]|"
    r"(?:Correct\s+Answer|Answer|Ans\.?|Explanation)\s*[:\-]))\s*(.*)$",
    re.I,
)

class LocalTranslator:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.ip = None
        self.device = None

    def load(self):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        from IndicTransToolkit.processor import IndicProcessor

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_ID, trust_remote_code=True, low_cpu_mem_usage=True
        )
        self.model.to(self.device)
        self.model.eval()
        self.ip = IndicProcessor(inference=True)

    def _protect(self, text):
        protected = {}
        i = 0
        for src, val in sorted(GLOSSARY.items(), key=lambda x: -len(x[0])):
            if src in text:
                token = f" ZXQTERM{i}ZXQ "
                text = text.replace(src, token)
                protected[token.strip()] = val
                i += 1
        # Keep URLs, emails and long all-caps tokens stable.
        for value in re.findall(r"https?://\S+|\b[A-Z][A-Z0-9./_-]{2,}\b", text):
            if value not in protected:
                token = f" ZXQTERM{i}ZXQ "
                text = text.replace(value, token)
                protected[token.strip()] = value
                i += 1
        return text, protected

    def _restore(self, text, protected):
        for token, value in protected.items():
            text = text.replace(token, value)
        return text

    def translate_batch(self, texts):
        if self.model is None:
            self.load()
        if not texts:
            return []
        safe = []
        maps = []
        for t in texts:
            s, p = self._protect(t)
            safe.append(s)
            maps.append(p)

        import torch
        out = []
        batch_size = 4
        for i in range(0, len(safe), batch_size):
            chunk = safe[i:i+batch_size]
            pre = self.ip.preprocess_batch(chunk, src_lang=SRC_LANG, tgt_lang=TGT_LANG)
            inputs = self.tokenizer(
                pre, truncation=True, padding="longest",
                return_tensors="pt", return_attention_mask=True
            ).to(self.device)
            with torch.inference_mode():
                generated = self.model.generate(
                    **inputs, use_cache=True, min_length=0,
                    max_length=256, num_beams=5, num_return_sequences=1
                )
            decoded = self.tokenizer.batch_decode(
                generated, skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            decoded = self.ip.postprocess_batch(decoded, lang=TGT_LANG)
            for j, value in enumerate(decoded):
                out.append(self._restore(value, maps[i+j]))
            del inputs, generated
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return out

translator = LocalTranslator()

def extract_pages(pdf):
    doc = fitz.open(pdf)
    pages = []
    try:
        for n, page in enumerate(doc, 1):
            blocks = sorted(page.get_text("blocks"), key=lambda b: (b[1], b[0]))
            text = "\n".join(b[4].strip() for b in blocks if b[4].strip())
            pages.append({"page": n, "text": text})
    finally:
        doc.close()
    return pages

def analyze_pdf(pdf):
    pages = extract_pages(pdf)
    full = "\n".join(x["text"] for x in pages)
    q = re.findall(r"(?m)^\s*(?:Q(?:uestion)?\s*)?(\d{1,4})[).:\-\s]", full, re.I)
    opts = re.findall(r"(?m)^\s*[A-D][).:\-\s]", full)
    answers = re.findall(r"(?im)^\s*(answer|ans\.?|correct answer|explanation)\s*[:\-]", full)
    return {
        "pages": len(pages),
        "questions_detected": len(set(q)),
        "options_detected": len(opts),
        "nonempty_pages": sum(bool(x["text"]) for x in pages),
        "answer_markers": len(answers),
    }

def _split_marker(line):
    m = MARKER_RE.match(line)
    if m:
        return m.group(1), m.group(2)
    return "", line

def translate_pages(pages):
    translated = []
    for page in pages:
        lines = page["text"].splitlines()
        result_lines = []
        pending = []
        pending_indexes = []

        def flush():
            nonlocal pending, pending_indexes
            if not pending:
                return
            vals = translator.translate_batch(pending)
            for idx, val in zip(pending_indexes, vals):
                result_lines[idx] = val
            pending, pending_indexes = [], []

        for line in lines:
            if not line.strip():
                result_lines.append("")
                continue
            marker, body = _split_marker(line)
            if not body.strip():
                result_lines.append(marker)
                continue
            # Keep short page headers/footers that are mostly identifiers readable.
            if re.fullmatch(r"[\d\s|/_.:-]+", body.strip()):
                result_lines.append(marker + body)
                continue
            result_lines.append(marker)
            pending.append(body)
            pending_indexes.append(len(result_lines) - 1)
            if len(pending) >= 8:
                flush()
        flush()
        translated.append({"page": page["page"], "lines": result_lines})
    return translated

def _pdf_font():
    candidates = [
        Path("/usr/share/fonts/truetype/noto/NotoSansTelugu-Regular.ttf"),
        Path("fonts/NotoSansTelugu-Regular.ttf"),
    ]
    for font in candidates:
        if font.exists():
            pdfmetrics.registerFont(TTFont("NotoTelugu", str(font)))
            return "NotoTelugu"
    raise RuntimeError("Noto Sans Telugu font is not installed.")

def translate_pdf(pdf, out_prefix):
    pages = translate_pages(extract_pages(pdf))
    out_pdf = Path(f"{out_prefix}_telugu.pdf")
    out_docx = Path(f"{out_prefix}_telugu.docx")

    fname = _pdf_font()
    style = ParagraphStyle(
        "body", fontName=fname, fontSize=10.2, leading=14,
        spaceAfter=4, allowWidows=1, allowOrphans=1
    )
    doc = SimpleDocTemplate(
        str(out_pdf), rightMargin=36, leftMargin=36,
        topMargin=36, bottomMargin=36, title="TeluguSetu Translation"
    )
    story = []
    for p in pages:
        for line in p["lines"]:
            if line.strip():
                safe = html.escape(line).replace("\n", "<br/>")
                story.append(Paragraph(safe, style))
                story.append(Spacer(1, 2))
        story.append(PageBreak())
    if story and isinstance(story[-1], PageBreak):
        story.pop()
    doc.build(story)

    d = Document()
    for pi, page in enumerate(pages):
        if pi:
            d.add_page_break()
        for line in page["lines"]:
            if line.strip():
                para = d.add_paragraph(line)
                para.paragraph_format.space_after = Pt(3)
                for run in para.runs:
                    run.font.name = "Noto Sans Telugu"
                    run.font.size = Pt(10.5)
    d.save(out_docx)

    return {
        "status": "complete",
        "pages": len(pages),
        "pdf": f"/download/{Path(out_prefix).name}/pdf",
        "docx": f"/download/{Path(out_prefix).name}/docx",
    }
