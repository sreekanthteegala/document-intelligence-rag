from app.core.vectorstore import load_faiss
from app.core.loader import load_pdf
from app.config import UPLOAD_DIR
import re

# ---------------- MODELS ----------------

_summarizer = None
_qa_llm = None


def get_summarizer():
    global _summarizer
    if _summarizer is None:
        from transformers import pipeline
        _summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=-1
        )
    return _summarizer


def get_qa_llm():
    global _qa_llm
    if _qa_llm is None:
        from transformers import pipeline
        _qa_llm = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            device=-1
        )
    return _qa_llm


# ---------------- HELPERS ----------------

def is_summary_question(question: str) -> bool:
    q = question.lower()
    return any(
        phrase in q for phrase in [
            "what is this pdf about",
            "what is this document about",
            "what is this paper about",
            "summary",
            "summarize",
            "overview"
        ]
    )


def clean_text_light(text: str) -> str:
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def mask_sensitive_entities(text: str) -> str:
    # Mask full names (capitalized first + last names)
    text = re.sub(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', 'the candidate', text)

    # Mask IDs / codes
    text = re.sub(r'\b(ID|Id|id)[:\s]*\w+\b', 'an identifier', text)

    # Mask dates
    text = re.sub(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', 'a date', text)

    # Normalize second-person language
    text = re.sub(r'\byou\b', 'the recipient', text, flags=re.IGNORECASE)
    text = re.sub(r'\byour\b', 'their', text, flags=re.IGNORECASE)

    return text


def detect_document_type(text: str) -> str:
    t = text.lower()

    if any(k in t for k in ["abstract", "introduction", "methodology", "references"]):
        return "research_paper"

    if any(k in t for k in ["offer letter", "letter of intent", "employment", "joining"]):
        return "employment_document"

    if any(k in t for k in ["invoice", "amount due", "bill to"]):
        return "invoice"

    if any(k in t for k in ["dear", "regards", "sincerely"]):
        return "letter_or_email"

    return "generic"


def build_summary_prompt(doc_type: str, text: str) -> str:
    prompts = {
        "research_paper":
            "Summarize the main objective and contribution of this research paper:\n\n",

        "employment_document":
            "Summarize the purpose of this employment-related document:\n\n",

        "letter_or_email":
            "Summarize the key message of this letter or email:\n\n",

        "invoice":
            "Summarize what this billing document is about:\n\n",

        "generic":
            "Summarize the main topic and purpose of this document:\n\n"
    }

    return prompts.get(doc_type, prompts["generic"]) + text


# ---------------- MAIN ENTRY ----------------

def answer_question(question: str):
    """
    Summary → Intent-aware direct summarization (NO RAG)
    QA → RAG + QA model
    """

    # 🔹 SUMMARY MODE
    if is_summary_question(question):
        pdf_files = sorted(
            UPLOAD_DIR.glob("*.pdf"),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )

        if not pdf_files:
            return "No document uploaded.", []

        raw_text = load_pdf(str(pdf_files[0]))
        if not raw_text:
            return "Document is empty.", []

        # Take only document-intent region
        raw_text = raw_text[:900]
        raw_text = clean_text_light(raw_text)
        raw_text = mask_sensitive_entities(raw_text)

        doc_type = detect_document_type(raw_text)
        prompt = build_summary_prompt(doc_type, raw_text)

        summarizer = get_summarizer()
        summary = summarizer(
            prompt,
            max_length=140,
            min_length=50,
            do_sample=False
        )[0]["summary_text"]

        return summary.strip(), []

    # 🔹 QA MODE (RAG)
    db = load_faiss()
    docs = db.similarity_search(question, k=3)

    if not docs:
        return "I don't know", []

    context = clean_text_light(" ".join(d.page_content for d in docs))

    llm = get_qa_llm()
    prompt = f"""
Answer the question using ONLY the context below.
If the answer is not present, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
"""

    output = llm(
        prompt,
        max_new_tokens=150,
        temperature=0.2,
        do_sample=False
    )[0]["generated_text"]

    if "i don't know" in output.lower():
        return "I don't know", []

    sources = [doc.page_content[:200] for doc in docs]

    return output.strip(), sources
