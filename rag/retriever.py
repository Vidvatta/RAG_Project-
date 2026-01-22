import numpy as np

def retrieve(query, index, chunks, embed_model, top_k=4):
    query_embedding = embed_model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)

    return [chunks[i] for i in indices[0]]
