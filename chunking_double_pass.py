from nltk.tokenize import sent_tokenize
import torch
from sentence_transformers import SentenceTransformer

def double_pass_chunking(text: str, initial_threshold: float, appending_threshold: float,
                         merging_threshold: float, max_chunk_length: int,
                         model: SentenceTransformer) -> list:
    sentences = sent_tokenize(text)
    if not sentences:
        return []
    model_device = next(model.parameters()).device
    # Precompute embeddings for all sentences on GPU
    embeddings = model.encode(sentences, convert_to_tensor=True).to(model_device)
    chunks = []
    i = 0
    while i < len(sentences):
        current_chunk = [sentences[i]]
        # First pass: group sentences with cosine similarity >= initial_threshold
        while i + 1 < len(sentences):
            cs = torch.cosine_similarity(embeddings[i].unsqueeze(0), embeddings[i+1].unsqueeze(0)).item()
            if cs >= initial_threshold and len(current_chunk) < max_chunk_length:
                current_chunk.append(sentences[i+1])
                i += 1
            else:
                break
        # Second pass: try to append additional sentences based on the combined embedding
        while i + 1 < len(sentences):
            combined_chunk = " ".join(current_chunk)
            combined_embedding = model.encode(combined_chunk, convert_to_tensor=True).to(model_device)
            cs = torch.cosine_similarity(combined_embedding.unsqueeze(0), embeddings[i+1].unsqueeze(0)).item()
            if cs >= appending_threshold and len(current_chunk) < max_chunk_length:
                current_chunk.append(sentences[i+1])
                i += 1
            else:
                break
        chunks.append(" ".join(current_chunk))
        i += 1
    # Merge adjacent chunks if their similarity is above merging_threshold
    merged_chunks = []
    i = 0
    while i < len(chunks):
        current_chunk = chunks[i]
        current_embedding = model.encode(current_chunk, convert_to_tensor=True).to(model_device)
        if i + 1 < len(chunks):
            next_embedding = model.encode(chunks[i+1], convert_to_tensor=True).to(model_device)
            cs_next = torch.cosine_similarity(current_embedding.unsqueeze(0), next_embedding.unsqueeze(0)).item()
            if cs_next >= merging_threshold:
                current_chunk += " " + chunks[i+1]
                i += 1
        merged_chunks.append(current_chunk)
        i += 1
    return merged_chunks
