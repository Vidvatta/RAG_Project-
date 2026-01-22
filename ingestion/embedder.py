import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def create_faiss_index(chunks, model_name):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    return index, chunks
