from fastapi import APIRouter, UploadFile, File
from app.core.loader import load_pdf
from app.core.chunker import chunk_text
from app.core.vectorstore import save_to_faiss
from app.core.rag import answer_question
from app.schemas.request import QuestionRequest
from app.schemas.response import AnswerResponse
from app.config import UPLOAD_DIR

router = APIRouter()

@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    path = UPLOAD_DIR / file.filename
    with open(path, "wb") as f:
        f.write(await file.read())

    text = load_pdf(str(path))
    chunks = chunk_text(text)
    save_to_faiss(chunks)

    return {"message": "Document indexed successfully"}

@router.post("/ask", response_model=AnswerResponse)
def ask_question(req: QuestionRequest):
    answer, sources = answer_question(req.question)
    return AnswerResponse(answer=answer, sources=sources)
