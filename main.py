import sys
import time
import numpy as np
import pickle
import random
import math
import torch
import os
from pathlib import Path

from typing import List, Any, Callable, Optional, Sequence
from datasets import load_dataset

from llama_index.core.base.embeddings.base import BaseEmbedding, SimilarityMode, similarity
from llama_index.core.evaluation.base import BaseEvaluator, EvaluationResult
from llama_index.core.prompts.mixin import PromptDictType
from llama_index.core.settings import Settings

from chunking_som import TorchSOMTrainer
from model_utils import initialize_models
from constants import *
from experiment_runner import run_experiment, get_rare_clusters

import nltk
nltk.download('punkt_tab')

import yaml
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "experiments_config.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
    cfg = yaml.safe_load(fh)

DATASET_NAMES = cfg["datasets"]
EXPERIMENTS   = cfg["experiments"]


# Set fixed seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    # Р”РѕРїРѕР»РЅРёС‚РµР»СЊРЅР°СЏ РЅР°СЃС‚СЂРѕР№РєР° GPU: РІРєР»СЋС‡РµРЅРёРµ РѕРїС‚РёРјРёР·Р°С†РёРё cuDNN РґР»СЏ СѓСЃРєРѕСЂРµРЅРёСЏ РІС‹С‡РёСЃР»РµРЅРёР№
    torch.backends.cudnn.benchmark = True

from datasets import load_dataset     # уже есть, просто напоминание

def predownload_datasets(ds_names, cache_dir=None, max_retries=3, delay=5):
    print("=" * 60)
    print("Prefetching datasets so we can work fully offline later …")
    for name in ds_names:
        for attempt in range(1, max_retries + 1):
            try:
                _ = load_dataset(
                    "rungalileo/ragbench",
                    name,
                    cache_dir=cache_dir,
                    download_mode="reuse_dataset_if_exists",
                )
                print(f"  ✔  {name} cached")
                break
            except Exception as e:
                print(f"  ✖  {name} attempt {attempt}/{max_retries} failed: {e}")
                if attempt == max_retries:
                    print("     will retry later in the main loop")
                else:
                    time.sleep(delay)

    # с этого момента интернет не нужен для HF datasets
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    print("=" * 60)

# --- вверху файла ----
import argparse
# ...

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num_samples", type=int, default=500)
    p.add_argument("--m", type=int, default=20)
    p.add_argument("--n", type=int, default=35)
    p.add_argument("--dataset", type=str, help="run only this dataset")
    return p.parse_args()

def main():
    args = parse_args()
    max_samples = args.num_samples
    m, n       = args.m, args.n

    # список датасетов берём из YAML, но можем сузить:
    datasets_to_run = DATASET_NAMES
    if args.dataset:
        datasets_to_run = [args.dataset]
    print(f"Using max_samples = {max_samples}, m = {m}, n = {n}")
    print(f"Running on datasets: {datasets_to_run}")
    
    overall_results = {}
    overall_metrics = {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    embedding_model, generation_model, tokenizer, judge_model = initialize_models(device)
    
    # >>> caching upfront <<<
    predownload_datasets(datasets_to_run)
    
    # Takes 104 minutes(1,73 hours) for 1 for 500 samples
    if max_samples < 20:
        first_samples = max_samples
    else:
        first_samples = 20
        
    base_dir = Path(__file__).parent
    exp_start_time = time.perf_counter()
    for ds_name in datasets_to_run:
        print("\n" + "="*50)
        print(f"Loading dataset: {ds_name}")
        try:
            ds = load_dataset("rungalileo/ragbench", ds_name)
            # Initialize SOM trainer and load pre-trained SOM model
            som_trainer = TorchSOMTrainer(embedding_model, map_size=(m, n), sigma=1.0, lr=0.5, iterations=100, device=device)
            model_path = base_dir / 'som_models' / f'som_model_{m}x{n}_torch_{ds_name}.pkl'
            som_trainer.load(model_path)
            cluster_counts = som_trainer.get_cluster_counts()
            print("Cluster distribution:", cluster_counts)
        except Exception as e:
            print(f"Error loading dataset {ds_name}: {str(e)}")
            continue
        results_dict = {}
        metrics_dict = {}
        for exp in EXPERIMENTS:
            print(f"Running experiment: {exp} on dataset: {ds_name}")
            exp_key, res, met = run_experiment(exp, max_samples, som_trainer, threshold_standard, min_chunk_size, initial_threshold, appending_threshold, merging_threshold, max_chunk_length, visualize, DATASET_NAMES, cluster_counts, ds, embedding_model, generation_model, judge_model, tokenizer)
            results_dict[exp_key] = res
            metrics_dict[exp_key] = met
        overall_results[ds_name] = results_dict
        overall_metrics[ds_name] = metrics_dict
        
        print("\nAggregate metrics for dataset:", ds_name)
        for key in results_dict:
            conf_arr = np.array(metrics_dict[key]['confidence_scores'])
            sem_arr = np.array(metrics_dict[key]['semantic_similarity'])
            llm_acc_arr = np.array(metrics_dict[key]['llm_accuracy'])
            avg_conf = float(np.mean(conf_arr))
            avg_sem = float(np.mean(sem_arr))
            avg_llm = float(np.mean(llm_acc_arr))
            print(f"\nExperiment: {key}")
            print("Average Confidence score:", avg_conf)
            print("Average Semantic similarity:", avg_sem)
            print("Average LLM accuracy:", avg_llm)
            print("="*50)

        output_filename = f"experiments/qa_experiments_{max_samples}_samples_{m}x{n}_{ds_name}.txt"
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(f"Dataset: {ds_name}\n")
            f.write("Aggregate average metrics for each experiment:\n")
            for key in results_dict:
                relevance_arr = np.array(metrics_dict[key]['relevance_scores'])
                confidence_arr = np.array(metrics_dict[key]['confidence_scores'])
                semantic_arr = np.array(metrics_dict[key]['semantic_similarity'])
                llm_acc_arr = np.array(metrics_dict[key]['llm_accuracy'])
                avg_relevance = float(np.mean(relevance_arr))
                avg_confidence = float(np.mean(confidence_arr))
                avg_semantic = float(np.mean(semantic_arr))
                avg_llm_acc = float(np.mean(llm_acc_arr))
                f.write("\nExperiment: " + key + "\n")
                f.write("Average Relevance score: " + str(avg_relevance) + "\n")
                f.write("Average Confidence score: " + str(avg_confidence) + "\n")
                f.write("Average Semantic similarity: " + str(avg_semantic) + "\n")
                f.write("Average LLM accuracy: " + str(avg_llm_acc) + "\n")
                f.write("=" * 50 + "\n")
            f.write("\n" + "=" * 50 + "\n")
            f.write("First 20 sample outputs for each experiment:\n")
            for key in results_dict:
                f.write("\n" + "=" * 50 + "\n")
                f.write(f"Experiment: {key}\n")
                samples_to_print = results_dict[key][:first_samples]
                for i, sample in enumerate(samples_to_print):
                    f.write("\n" + "-" * 50 + "\n")
                    f.write(f"Sample #{i}\n")
                    f.write("Question: " + str(sample.get('question', 'N/A')) + "\n")
                    f.write("Expected answer: " + str(sample.get('expected_answer', 'N/A')) + "\n")
                    f.write("Predicted answer: " + str(sample.get('predicted_answer', 'N/A')) + "\n")
                    f.write("Colored chunks:\n")
                    f.write(sample.get('colored_chunks', '') + "\n")
                    f.write("-" * 50 + "\n")
            total_exp_time = time.perf_counter() - exp_start_time
            f.write(f"\nTotal experiment execution time: {total_exp_time:.2f} seconds\n")
        print(f"Metrics for dataset {ds_name} written to {output_filename}")
    
    total_exp_time = time.perf_counter() - exp_start_time
    print(f"\nTotal experiment execution time: {total_exp_time:.2f} seconds")

if __name__ == "__main__":
    main()
