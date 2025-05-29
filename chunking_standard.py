from nltk.tokenize import sent_tokenize
from sentence_transformers import util
from sentence_transformers import SentenceTransformer

def calculate_cosine_distances(embeddings) -> list:
    distances = []
    for i in range(len(embeddings) - 1):
        sim = util.pytorch_cos_sim(embeddings[i], embeddings[i+1]).item()
        distances.append(1 - sim)
    return distances


def standard_chunking(text: str, model: SentenceTransformer, threshold: float = 0.3, min_chunk_size: int = 2) -> list:
    sentences = sent_tokenize(text)
    if not sentences:
        return []
    # Ensure embeddings are computed on GPU
    model_device = next(model.parameters()).device
    embeddings = model.encode(sentences, convert_to_tensor=True).to(model_device)
    distances = calculate_cosine_distances(embeddings)
    chunks = []
    current_chunk = []
    for i, sentence in enumerate(sentences):
        current_chunk.append(sentence)
        if i < len(distances) and distances[i] > threshold and len(current_chunk) >= min_chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks