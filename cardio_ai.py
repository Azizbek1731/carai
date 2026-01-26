import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

from tqdm import tqdm
from rank_bm25 import BM25Okapi
from pypdf import PdfReader
from docx import Document

from google import genai  # pip install google-genai


# =========================
# SOZLAMALAR
# =========================
STANDARDS_DIR = Path("standardlar")   # 1-papka
PROTOCOLS_DIR = Path("protokollar")  # 2-papka

CHUNK_CHARS = 2200
CHUNK_OVERLAP = 300
TOP_K = 10

MODEL_NAME = "gemini-2.5-flash"
TEMPERATURE = 0.2


SYSTEM_PROMPT = """SENING ROLING: CardioAI — kardiologiya bo‘yicha "Standardlar" va "Protokollar" hujjatlariga tayangan klinik qarorlarni qo‘llab-quvvatlovchi (decision support) yordamchi.

QAT’IY QOIDALAR:
1) Faqat berilgan KONTEXT (hujjat bo‘laklari) asosida javob ber. Umumiy tibbiy bilimdan taxmin (hallucination) qilma.
2) Agar KONTEXTda yetarli dalil bo‘lmasa, "Hujjatlarda topilmadi" deb yoz va aynan qaysi ma’lumotlar yetishmayotganini ro‘yxat qil.
3) Javob tili: O‘zbek.
4) Har muhim da’vo uchun manba ber: FAYL_NOMI + SEGMENT.
5) Dori dozalari/aniq buyruqlar faqat KONTEXTda aniq ko‘rsatilgan bo‘lsa beriladi. Aks holda "Hujjatlarda aniq doza ko‘rsatilmagan" deb yoz.

JAVOB FORMATI (qat’iy):
1) Asosiy tashxis (AI tavsiyasi): <bitta tashxis nomi>
   - Mezoni: (bemor ma’lumoti ichidan mos kelgan 3–6 ta dalil)
   - Hujjat dalillari: [manbalar]
2) Differensial tashxislar (Top-3/5):
   - Har birida: Nega ko‘rib chiqiladi + ajratib beruvchi tekshiruv/mezoni + [manbalar]
3) Tavsiya etiladigan keyingi qadamlar:
   - Tekshiruvlar (EKG/EKO/lab/KT va h.k. — faqat KONTEXTga mos)
   - Shoshilinch "qizil bayroqlar" (agar KONTEXTda bo‘lsa)
4) Manbalar ro‘yxati:
   - [1] FAYL | SEGMENT
   - [2] ...
5) Eslatma:
   - Oxirgi qarorni qabul qilish faqat shifokor zimmasida. CardioAI faqat shifokorga yordam berish uchun.
"""


# =========================
# MATNNI O‘QISH / BO‘LAKLASH
# =========================
def normalize_text(s: str) -> str:
    s = (s or "").replace("\u00a0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def tokenize(s: str) -> List[str]:
    s = s.lower()
    # uz/ru/en aralash tokenlar
    s = re.sub(r"[^a-z0-9а-яёқғҳў\- ]+", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    return s.split()

def chunk_text(text: str, chunk_chars: int, overlap: int) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_chars, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks

def extract_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        t = normalize_text(page.extract_text() or "")
        if t:
            pages.append(f"[PAGE {i}]\n{t}")
    return "\n\n".join(pages)

def extract_docx(path: Path) -> str:
    doc = Document(str(path))
    paras = []
    for p in doc.paragraphs:
        t = normalize_text(p.text)
        if t:
            paras.append(t)
    return "\n".join(paras)

@dataclass
class Chunk:
    file_name: str
    source_folder: str   # "standardlar" yoki "protokollar"
    segment: str         # "segment 3"
    text: str


def iter_docs(d: Path) -> List[Path]:
    if not d.exists():
        return []
    out = []
    for p in d.rglob("*"):
        if p.is_file() and p.suffix.lower() in [".pdf", ".docx"]:
            out.append(p)
    return sorted(out)

def build_index(standards_dir: Path, protocols_dir: Path):
    chunks: List[Chunk] = []
    corpus_tokens: List[List[str]] = []

    files = [(standards_dir, "standardlar"), (protocols_dir, "protokollar")]
    all_paths = []
    for root, _tag in files:
        all_paths.extend(iter_docs(root))

    if not all_paths:
        raise FileNotFoundError("PDF/DOCX topilmadi. 'standardlar/' va 'protokollar/' ichini tekshiring.")

    for root, tag in files:
        paths = iter_docs(root)
        for f in tqdm(paths, desc=f"Indekslash: {tag}"):
            try:
                if f.suffix.lower() == ".pdf":
                    full = extract_pdf(f)
                else:
                    full = extract_docx(f)

                segs = chunk_text(full, CHUNK_CHARS, CHUNK_OVERLAP)
                for i, seg in enumerate(segs, start=1):
                    c = Chunk(
                        file_name=f.name,
                        source_folder=tag,
                        segment=f"segment {i}",
                        text=seg
                    )
                    chunks.append(c)
                    corpus_tokens.append(tokenize(seg))
            except Exception as e:
                print(f"[WARN] O‘qilmadi: {f} ({e})", file=sys.stderr)

    bm25 = BM25Okapi(corpus_tokens)
    return bm25, chunks

def retrieve(bm25: BM25Okapi, chunks: List[Chunk], query: str, top_k: int) -> List[Chunk]:
    q = tokenize(query)
    scores = bm25.get_scores(q)
    idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [chunks[i] for i in idxs if scores[i] > 0]

def build_user_prompt(patient_text: str, question: str, retrieved: List[Chunk]) -> str:
    if not retrieved:
        context = "KONTEXT YO‘Q."
    else:
        parts = []
        for i, ch in enumerate(retrieved, start=1):
            parts.append(
                f"[{i}] PAPKA: {ch.source_folder} | FAYL: {ch.file_name} | JOYI: {ch.segment}\n"
                f"MATN:\n{ch.text}\n"
            )
        context = "\n---\n".join(parts)

    return f"""BEMOR MA’LUMOTI:
{patient_text}

SAVOL:
{question}

KONTEXT (faqat shu hujjatlarga tayan):
{context}

Talab: System qoidalariga qat’iy amal qilib, faqat KONTEXTga asoslangan javob yoz.
"""

def call_gemini(system_prompt: str, user_prompt: str) -> str:
    # GEMINI_API_KEY env’da bo‘lishi kerak
    client = genai.Client()
    resp = client.models.generate_content(
        model=MODEL_NAME,
        contents=user_prompt,
        config={
            "system_instruction": system_prompt,
            "temperature": TEMPERATURE,
        },
    )
    return getattr(resp, "text", None) or str(resp)


def main():
    # Papkalar borligini tekshirish
    if not STANDARDS_DIR.exists():
        print(f"[XATO] Papka topilmadi: {STANDARDS_DIR.resolve()}")
        sys.exit(1)
    if not PROTOCOLS_DIR.exists():
        print(f"[XATO] Papka topilmadi: {PROTOCOLS_DIR.resolve()}")
        sys.exit(1)

    print("Index qurilmoqda (birinchi marta biroz vaqt oladi)...")
    bm25, chunks = build_index(STANDARDS_DIR, PROTOCOLS_DIR)
    print(f"Index tayyor. Chunklar soni: {len(chunks)}")

    print("\nInteraktiv rejim. Chiqish uchun: exit\n")
    while True:
        patient_text = input("Bemor ma’lumoti > ").strip()
        if patient_text.lower() in ["exit", "quit"]:
            break

        # Demo uchun savolni default qilib qo'ysangiz ham bo'ladi
        question = input("Savol (Enter bosilsa default) > ").strip()
        if question.lower() in ["exit", "quit"]:
            break
        if not question:
            question = "Bemor ma’lumotiga asosan asosiy tashxis (AI tavsiyasi), differensial tashxislar va keyingi qadamlar?"

        retrieval_query = patient_text + "\n" + question
        retrieved = retrieve(bm25, chunks, retrieval_query, TOP_K)

        user_prompt = build_user_prompt(patient_text, question, retrieved)
        answer = call_gemini(SYSTEM_PROMPT, user_prompt)

        print("\n" + "=" * 90)
        print(answer)
        print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
