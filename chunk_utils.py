import numpy as np
import faiss
from sentence_transformers import util
import torch

def get_colored_chunks_text(chunks: list) -> str:
    ansi_colors = ["\033[31m", "\033[32m", "\033[33m", "\033[34m", "\033[35m", "\033[36m", "\033[37m"]
    reset_color = "\033[0m"
    output = ""
    for i, chunk in enumerate(chunks):
        color = ansi_colors[i % len(ansi_colors)]
        output += f"{color}Chunk {i+1}:\n{chunk}\n{reset_color}\n"
    return output


def print_colored_chunks(chunks: list):
    print(get_colored_chunks_text(chunks))

# =============================================================================
# Indexing function for chunks using FAISS
# =============================================================================

def index_chunks(chunks, model):
    chunk_embeddings = model.encode(chunks, convert_to_tensor=False)
    index = faiss.IndexFlatL2(chunk_embeddings.shape[1])
    index.add(np.array(chunk_embeddings))
    return index, chunk_embeddings

def retrieve_chunks_with_filtering(question, index, chunks, model, top_k=5, top_filtered_k=3):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    question_embedding = model.encode([question], convert_to_tensor=True).to(dev)
    distances, indices = index.search(question_embedding.cpu().numpy(), top_k)
    candidate_chunks = [chunks[i] for i in indices[0] if i != -1]
    candidate_embeddings = model.encode(candidate_chunks, convert_to_tensor=True).to(dev)
    scores = util.pytorch_cos_sim(question_embedding, candidate_embeddings).squeeze(0)
    ranked_indices = torch.argsort(scores, descending=True)
    filtered_chunks = [candidate_chunks[i] for i in ranked_indices[:top_filtered_k]]
    return filtered_chunks
