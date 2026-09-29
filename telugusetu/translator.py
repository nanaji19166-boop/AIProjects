import gc, html, json, os, re, urllib.request
from pathlib import Path

import fitz
import numpy as np
from docx import Document
from docx.shared import Pt
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak

MODEL_ID = os.getenv("MODEL_ID", "hari31416/indictrans2-en-indic-dist-200M-ONNX-int8")
SRC_LANG = "eng_Latn"
TGT_LANG = "tel_Telu"
FONT_URL = "https://raw.githubusercontent.com/notofonts/noto-fonts/main/hinted/ttf/NotoSansTelugu/NotoSansTelugu-Regular.ttf"

GLOSSARY = {
    "New START": "New START", "NATO": "NATO", "UN": "UN", "WHO": "WHO",
    "GDP": "GDP", "EPCIS": "EPCIS", "UPSC": "UPSC", "MCQ": "MCQ",
    "GS": "GS", "IAS": "IAS", "IPS": "IPS", "PDF": "PDF",
}
MARKER_RE = re.compile(
    r"^(\s*(?:(?:Q(?:uestion)?\s*)?\d{1,4}[\).:\-]|"
    r"[A-D][\).:\-]|(?:Correct\s+Answer|Answer|Ans\.?|Explanation)\s*[:\-]))"
    r"\s*(.*)$", re.I
)


class IndicTransONNX:
    def __init__(self, model_path):
        import onnxruntime as ort
        from huggingface_hub import snapshot_download
        from tokenizers import Tokenizer
        from IndicTransToolkit import IndicProcessor

        if "/" in str(model_path) and not Path(model_path).exists():
            model_path = snapshot_download(repo_id=model_path)
        snap = Path(model_path)
        self.ip = IndicProcessor(inference=True)
        self.src_tok = Tokenizer.from_file(str(snap / "tokenizer_src.json"))
        self.tgt_tok = Tokenizer.from_file(str(snap / "tokenizer_tgt.json"))
        self.meta = json.loads((snap / "tokenizer_meta.json").read_text(encoding="utf-8"))
        cfg_path = snap / "generation_config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
        self.start_id = int(cfg.get("decoder_start_token_id", 2))
        self.eos_id = int(cfg.get("eos_token_id", 2))

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        # Free Render has only 512 MB RAM. Avoid ONNX Runtime weight prepacking,
        # which can create large temporary copies of the shared decoder weights.
        opts.add_session_config_entry("session.disable_prepacking", "1")
        opts.enable_cpu_mem_arena = False
        opts.enable_mem_pattern = False
        opts.enable_mem_reuse = True
        opts.add_session_config_entry("session.prepack.enable_parallel", "0")
        providers = ["CPUExecutionProvider"]

        # Keep only encoder + one decoder graph in RAM. The previous
        # implementation loaded three graphs and then reloaded the first-step
        # decoder for every sentence, which made long PDFs impractically slow
        # and could restart a free Render instance.
        self.enc = ort.InferenceSession(
            str(snap / "encoder_model.onnx"), sess_options=opts, providers=providers
        )
        self.dec = ort.InferenceSession(
            str(snap / "decoder_model.onnx"), sess_options=opts, providers=providers
        )
        self.num_layers = (len(self.dec.get_outputs()) - 1) // 4

    def translate(self, text, max_new_tokens=64):
        if hasattr(self.ip, "_placeholder_entity_maps"):
            try:
                self.ip._placeholder_entity_maps.queue.clear()
            except Exception:
                pass

        prefixed = self.ip.preprocess_batch(
            [text], src_lang=SRC_LANG, tgt_lang=TGT_LANG
        )[0]
        encoded = self.src_tok.encode(prefixed)
        ids = np.array([[
            i if i < self.meta["src_dict_size"] else self.meta["unk_id"]
            for i in encoded.ids
        ]], dtype=np.int64)
        mask = np.array([encoded.attention_mask], dtype=np.int64)

        enc_out = self.enc.run(
            ["last_hidden_state"], {"input_ids": ids, "attention_mask": mask}
        )[0]

        generated = [self.start_id]
        for _ in range(max_new_tokens):
            dec_ids = np.array([generated], dtype=np.int64)
            out = self.dec.run(None, {
                "input_ids": dec_ids,
                "encoder_hidden_states": enc_out,
                "encoder_attention_mask": mask,
            })
            next_id = int(np.argmax(out[0][0, -1, :]))
            generated.append(next_id)
            if next_id == self.eos_id:
                break

        safe_ids = [
            i if i < self.meta["tgt_dict_size"] else self.meta["unk_id"]
            for i in generated
        ]
        raw = self.tgt_tok.decode(safe_ids, skip_special_tokens=True)
        return self.ip.postprocess_batch([raw], lang=TGT_LANG)[0]


class LocalTranslator:
    def __init__(self):
        self.model = None

    def load(self):
        if self.model is None:
            self.model = IndicTransONNX(MODEL_ID)

    def translate_batch(self, texts):
        self.load()
        return [self.model.translate(text) for text in texts]


translator = LocalTranslator()


def extract_pages(pdf):
    doc = fitz.open(pdf)
    pages = []
    try:
        for n, page in enumerate(doc, 1):
            blocks = sorted(page.get_text("blocks"), key=lambda b: (b[1], b[0]))
            pages.append({
                "page": n,
                "text": "\n".join(b[4].strip() for b in blocks if b[4].strip()),
            })
    finally:
        doc.close()
    return pages


def analyze_pdf(pdf):
    pages = extract_pages(pdf)
    full = "\n".join(x["text"] for x in pages)
    q = re.findall(r"(?m)^\s*(?:Q(?:uestion)?\s*)?(\d{1,4})[).:\-\s]", full, re.I)
    opts = re.findall(r"(?m)^\s*[A-D][).:\-\s]", full)
    answers = re.findall(
        r"(?im)^\s*(answer|ans\.?|correct answer|explanation)\s*[:\-]", full
    )
    return {
        "pages": len(pages),
        "questions_detected": len(set(q)),
        "options_detected": len(opts),
        "nonempty_pages": sum(bool(x["text"]) for x in pages),
        "answer_markers": len(answers),
    }


def _split_marker(line):
    m = MARKER_RE.match(line)
    return (m.group(1), m.group(2)) if m else ("", line)


def translate_pages(pages, progress_callback=None):
    translated = []
    total = len(pages)
    for pi, page in enumerate(pages, 1):
        lines = page["text"].splitlines()
        result, pending, indexes = [], [], []

        def flush():
            nonlocal pending, indexes
            if not pending:
                return
            vals = translator.translate_batch(pending)
            for idx, val in zip(indexes, vals):
                result[idx] = val
            pending, indexes = [], []

        for line in lines:
            if not line.strip():
                result.append("")
                continue
            marker, body = _split_marker(line)
            if not body.strip():
                result.append(marker)
                continue
            if re.fullmatch(r"[\d\s|/_.:-]+", body.strip()):
                result.append(marker + body)
                continue
            result.append(marker)
            pending.append(body)
            indexes.append(len(result) - 1)
            if len(pending) >= 4:
                flush()
        flush()
        translated.append({"page": page["page"], "lines": result})
        if progress_callback:
            progress_callback(int(pi * 70 / max(total, 1)))
    return translated


def _pdf_font():
    candidates = [
        Path("/usr/share/fonts/truetype/noto/NotoSansTelugu-Regular.ttf"),
        Path("fonts/NotoSansTelugu-Regular.ttf"),
        Path("data/NotoSansTelugu-Regular.ttf"),
    ]
    for font in candidates:
        if font.exists():
            try:
                pdfmetrics.registerFont(TTFont("NotoTelugu", str(font)))
                return "NotoTelugu"
            except Exception:
                pass
    target = Path("data/NotoSansTelugu-Regular.ttf")
    target.parent.mkdir(exist_ok=True)
    urllib.request.urlretrieve(FONT_URL, target)
    pdfmetrics.registerFont(TTFont("NotoTelugu", str(target)))
    return "NotoTelugu"


def translate_pdf(pdf, out_prefix, progress_callback=None):
    pages = translate_pages(extract_pages(pdf), progress_callback)
    out_pdf = Path(f"{out_prefix}_telugu.pdf")
    out_docx = Path(f"{out_prefix}_telugu.docx")
    style = ParagraphStyle(
        "body", fontName=_pdf_font(), fontSize=10.2, leading=14, spaceAfter=4
    )
    doc = SimpleDocTemplate(
        str(out_pdf), rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36,
        title="TeluguSetu Translation"
    )
    story = []
    for p in pages:
        for line in p["lines"]:
            if line.strip():
                story.extend([Paragraph(html.escape(line), style), Spacer(1, 2)])
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
    if progress_callback:
        progress_callback(100)
    return {
        "status": "complete",
        "pages": len(pages),
        "pdf": f"/download/{Path(out_prefix).name}/pdf",
        "docx": f"/download/{Path(out_prefix).name}/docx",
    }
