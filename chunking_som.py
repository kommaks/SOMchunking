import torch
import numpy as np
import pickle
from nltk.tokenize import sent_tokenize
from sentence_transformers import util
from collections import Counter

class TorchSOM:
    def __init__(self, m, n, dim, n_iterations=100, learning_rate=0.5, sigma=None, device="cuda"):
        self.m = m
        self.n = n
        self.dim = dim
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.sigma = sigma if sigma is not None else max(m, n) / 2.0
        self.device = device
        self.weights = torch.rand(m, n, dim, device=self.device)
        x_coords = torch.arange(m, device=self.device).float()
        y_coords = torch.arange(n, device=self.device).float()
        self.neuron_locations = torch.stack(torch.meshgrid(x_coords, y_coords, indexing="ij"), dim=-1)

    def _get_bmu(self, x):
        x_expanded = x.view(1, 1, self.dim)
        distances = torch.norm(self.weights - x_expanded, dim=-1)
        bmu_idx = torch.argmin(distances)
        bmu_location = torch.tensor([bmu_idx // self.n, bmu_idx % self.n], device=self.device)
        return bmu_location, distances

    def winner(self, x):
        return self._get_bmu(x)[0]

    def update(self, x, bmu_location, iteration):
        lr = self.learning_rate * (1 - iteration / self.n_iterations)
        sigma = self.sigma * (1 - iteration / self.n_iterations)
        bmu_location = bmu_location.view(1, 1, 2)
        d = torch.sum((self.neuron_locations - bmu_location) ** 2, dim=-1)
        h = torch.exp(-d / (2 * (sigma ** 2)))
        h = h.unsqueeze(-1)
        x_expanded = x.view(1, 1, self.dim)
        self.weights = self.weights + lr * h * (x_expanded - self.weights)

    def train(self, data):
        N = data.shape[0]
        for it in range(self.n_iterations):
            idx = torch.randint(0, N, (1,)).item()
            x = data[idx]
            bmu_location, _ = self._get_bmu(x)
            self.update(x, bmu_location, it)
        return self.weights


class TorchSOMTrainer:
    def __init__(self, embedding_model, map_size=(10, 10), sigma=1.0, lr=0.5, iterations=100, device="cuda"):
        self.map_size = map_size
        self.sigma = sigma
        self.lr = lr
        self.iterations = iterations
        self.embedding_model = embedding_model
        self.device = device
        self.som = None
        self.corpus_distances = None  # numpy array of shape (N, 1)
        self.corpus_sentences = None  # list of sentences

    def train(self, corpus):
        all_distances = []
        all_sentences = []
        for doc in corpus:
            sentences = sent_tokenize(doc)
            if len(sentences) < 2:
                continue
            all_sentences.extend(sentences)
            emb = self.embedding_model.encode(sentences, convert_to_tensor=False, show_progress_bar=False)
            emb = np.array(emb)
            for i in range(len(emb) - 1):
                vec1 = torch.tensor(emb[i], device=self.device, dtype=torch.float32)
                vec2 = torch.tensor(emb[i+1], device=self.device, dtype=torch.float32)
                sim = util.pytorch_cos_sim(vec1, vec2).item()
                distance = 1 - sim
                all_distances.append([distance])
        if len(all_distances) == 0:
            raise ValueError("No sentence pairs available for SOM training.")
        self.corpus_distances = np.array(all_distances, dtype=np.float32)
        self.corpus_sentences = all_sentences

        data_min = self.corpus_distances.min()
        data_max = self.corpus_distances.max()
        data_norm = (self.corpus_distances - data_min) / (data_max - data_min + 1e-8)
        data_tensor = torch.tensor(data_norm, device=self.device, dtype=torch.float32)

        self.som = TorchSOM(m=self.map_size[0], n=self.map_size[1], dim=1,
                             n_iterations=self.iterations, learning_rate=self.lr,
                             sigma=self.sigma, device=self.device)
        print("Starting TorchSOM training on GPU:")
        self.som.train(data_tensor)
        print("Training complete.")
        return self.som

    def get_cluster_counts(self):
        if self.som is None or self.corpus_distances is None:
            raise ValueError("SOM not trained yet!")
        data_min = self.corpus_distances.min()
        data_max = self.corpus_distances.max()
        data_norm = (self.corpus_distances - data_min) / (data_max - data_min + 1e-8)
        data_tensor = torch.tensor(data_norm, device=self.device, dtype=torch.float32)
        bmu_coords = [tuple(map(int, self.som._get_bmu(data_tensor[i])[0].cpu().numpy()))
                      for i in range(data_tensor.shape[0])]
        return Counter(bmu_coords)
    
    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump({
                "som_weights": self.som.weights.cpu().numpy(),
                "corpus_distances": self.corpus_distances,
                "corpus_sentences": self.corpus_sentences
            }, f)
        print(f"\nSOM model saved to {filename}")

    def load(self, filename):
        with open(filename, "rb") as f:
            data = pickle.load(f)
            weights = torch.tensor(data["som_weights"], device=self.device, dtype=torch.float32)
        self.som = TorchSOM(m=self.map_size[0], n=self.map_size[1], dim=1,
                             n_iterations=self.iterations, learning_rate=self.lr,
                             sigma=self.sigma, device=self.device)
        self.som.weights = weights
        self.corpus_distances = data["corpus_distances"]
        self.corpus_sentences = data["corpus_sentences"]
        print(f"SOM model loaded from {filename}")
        return self.som


def infer_document_clusters(document: str, som_trainer: TorchSOMTrainer, rare_clusters, top_n: int, visualize: bool = False) -> dict:
    sentences = sent_tokenize(document)
    if len(sentences) < 2:
        return {"bmu_coords": [], "rare_hits": [], "top_rare_hits": [], "fraction_rare": 0, "sentences": sentences}
    model_device = next(som_trainer.embedding_model.parameters()).device
    embeddings = som_trainer.embedding_model.encode(sentences, convert_to_tensor=False, show_progress_bar=False)
    embeddings = np.array(embeddings)
    raw_distances = []
    for i in range(len(embeddings) - 1):
        sim = util.pytorch_cos_sim(
            torch.tensor(embeddings[i], device=som_trainer.device, dtype=torch.float32),
            torch.tensor(embeddings[i+1], device=som_trainer.device, dtype=torch.float32)
        ).item()
        raw_distances.append([1 - sim])
    raw_distances = np.array(raw_distances)
    train_min = som_trainer.corpus_distances.min()
    train_max = som_trainer.corpus_distances.max()
    distances_norm = (raw_distances - train_min) / (train_max - train_min + 1e-8)
    bmu_coords = [tuple(map(int, som_trainer.som._get_bmu(torch.tensor(d, device=som_trainer.device, dtype=torch.float32))[0].cpu().numpy()))
                  for d in distances_norm]
    rare_clusters_coords = [cluster for cluster, count in rare_clusters]
    rare_hits = []
    for idx, bmu in enumerate(bmu_coords):
        if bmu in rare_clusters_coords:
            rare_hits.append((idx, distances_norm[idx][0]))
    top_rare_hits = sorted(rare_hits, key=lambda x: -x[1])[:top_n]
    return {
        "bmu_coords": bmu_coords,
        "rare_hits": rare_hits,
        "top_rare_hits": top_rare_hits,
        "sentences": sentences
    }


def split_document_by_anomalies(document: str, top_rare_hits) -> list:
    sentences = sent_tokenize(document)
    boundaries = sorted([hit[0] for hit in top_rare_hits])
    chunks = []
    prev = 0
    for b in boundaries:
        chunks.append(" ".join(sentences[prev:b+1]))
        prev = b + 1
    chunks.append(" ".join(sentences[prev:]))
    return chunks

# =============================================================================
# Colored chunk output
# =============================================================================
