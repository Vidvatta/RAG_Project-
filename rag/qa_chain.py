from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL, TOP_K
from rag.retriever import retrieve
from rag.llm import call_llm

embed_model = SentenceTransformer(EMBEDDING_MODEL)

def answer_question(question, index, chunks):
    retrieved_chunks = retrieve(
        question, index, chunks, embed_model, TOP_K
    )

    context = "\n".join(retrieved_chunks)
    return call_llm(context, question)
