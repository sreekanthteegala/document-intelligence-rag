from langchain_community.vectorstores import FAISS
from app.core.embeddings import get_embeddings
from app.config import FAISS_DIR

def save_to_faiss(chunks):
    embeddings = get_embeddings()
    db = FAISS.from_texts(chunks, embeddings)
    db.save_local(str(FAISS_DIR))

def load_faiss():
    embeddings = get_embeddings()
    return FAISS.load_local(
        str(FAISS_DIR),
        embeddings,
        allow_dangerous_deserialization=True
    )
