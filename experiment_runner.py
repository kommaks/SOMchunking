import numpy as np
from sklearn.metrics import mean_squared_error, roc_auc_score, accuracy_score, f1_score
from scipy.special import softmax

from model_utils import generate_answer, evaluate_llm_accuracy
from chunking_double_pass import *
from chunking_standard import *
from chunking_som import *
from chunk_utils import *

from llama_index.core.base.embeddings.base import BaseEmbedding, SimilarityMode, similarity
from llama_index.core.evaluation.base import BaseEvaluator, EvaluationResult
from llama_index.core.prompts.mixin import PromptDictType
from llama_index.core.settings import Settings
from typing import List, Any, Callable, Optional, Sequence

class SemanticSimilarityEvaluator(BaseEvaluator):
    def __init__(
        self,
        embed_model: Optional[BaseEmbedding] = None,
        similarity_fn: Optional[Callable[..., float]] = None,
        similarity_mode: Optional[SimilarityMode] = None,
        similarity_threshold: float = 0.8,
    ) -> None:
        self._embed_model = embed_model or Settings.embed_model
        if similarity_fn is None:
            similarity_mode = similarity_mode or SimilarityMode.DEFAULT
            self._similarity_fn = lambda x, y: similarity(x, y, mode=similarity_mode)
        else:
            if similarity_mode is not None:
                raise ValueError("Cannot specify both similarity_fn and similarity_mode")
            self._similarity_fn = similarity_fn
        self._similarity_threshold = similarity_threshold

    def _get_prompts(self) -> PromptDictType:
        return {}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        pass

    def evaluate(
        self,
        response: Optional[str] = None,
        reference: Optional[str] = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        if response is None or reference is None:
            raise ValueError("Must specify both response and reference")
        response_embedding = self._embed_model.encode(response)
        reference_embedding = self._embed_model.encode(reference)
        similarity_score = self._similarity_fn(response_embedding, reference_embedding)
        passing = similarity_score >= self._similarity_threshold
        return EvaluationResult(
            score=similarity_score,
            passing=passing,
            feedback=f"Similarity score: {similarity_score}",
        )

    def aevaluate(
        self,
        response: Optional[str] = None,
        reference: Optional[str] = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        return self.evaluate(response=response, reference=reference, **kwargs)

def evaluate_rag_methods(method, dataset, som_trainer, embedding_model, generation_model, tokenizer,
                           threshold_standard, min_chunk_size,
                           initial_threshold, appending_threshold, merging_threshold,
                           max_chunk_length=3, max_samples=10, visualize=False, local_rare_clusters=None):
    results = {"standard": [], "double_pass": [], "som": []}
    metrics = {
        "standard": {'rmse': None, 'roc_auc': None, 'accuracy': None, 'f1_score': None,
                     'relevance_scores': [], 'confidence_scores': [], 'semantic_similarity': [], 'llm_accuracy': []},
        "double_pass": {'rmse': None, 'roc_auc': None, 'accuracy': None, 'f1_score': None,
                        'relevance_scores': [], 'confidence_scores': [], 'semantic_similarity': [], 'llm_accuracy': []},
        "som": {'rmse': None, 'roc_auc': None, 'accuracy': None, 'f1_score': None,
                'relevance_scores': [], 'confidence_scores': [], 'semantic_similarity': [], 'llm_accuracy': []}
    }
    try:
        samples = dataset.select(range(min(max_samples, len(dataset))))
    except Exception as e:
        print(f"Error selecting samples: {str(e)}")
        return results, metrics

    semantic_evaluator = SemanticSimilarityEvaluator(embed_model=embedding_model, similarity_threshold=0.8)

    for example in samples:
        try:
            question = example["question"]
            context = " ".join(example["documents"])
            expected_answer = example["response"]

            if method == 'standard':
                # Use improved standard chunking with cosine distance-based splitting
                chunks = standard_chunking(context, embedding_model, threshold_standard, min_chunk_size)
                colored_chunks = get_colored_chunks_text(chunks)
                if visualize:
                    print("Standard Chunking Output:")
                    print(colored_chunks)
                index, _ = index_chunks(chunks, embedding_model)
                relevant_chunks = retrieve_chunks_with_filtering(question, index, chunks, embedding_model)
                predicted_answer = generate_answer(question, relevant_chunks, generation_model, tokenizer)
                print(predicted_answer)
                q_emb = embedding_model.encode(question, convert_to_tensor=True)
                c_emb = embedding_model.encode(context, convert_to_tensor=True)
                relevance_score = util.pytorch_cos_sim(q_emb, c_emb).item()
                inputs = tokenizer(predicted_answer, return_tensors="pt", truncation=True)
                dev_gen = next(generation_model.parameters()).device
                inputs = {k: v.to(dev_gen) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = generation_model(**inputs)
                    logits = outputs.logits
                    probs = softmax(logits.cpu().numpy(), axis=-1)
                    confidence_score = np.mean(np.max(probs, axis=-1))
                eval_result = semantic_evaluator.evaluate(response=predicted_answer, reference=expected_answer)
                llm_acc, llm_eval_text = evaluate_llm_accuracy(question, predicted_answer, expected_answer, generation_model, tokenizer)
                metrics["standard"]['relevance_scores'].append(relevance_score)
                metrics["standard"]['confidence_scores'].append(confidence_score)
                metrics["standard"]['semantic_similarity'].append(eval_result.score)
                metrics["standard"]['llm_accuracy'].append(llm_acc)
                results["standard"].append({
                    "question": question,
                    "expected_answer": expected_answer,
                    "predicted_answer": predicted_answer,
                    "colored_chunks": colored_chunks
                })

            elif method == 'double_pass':
                # Use improved double-pass chunking
                chunks = double_pass_chunking(context, initial_threshold, appending_threshold, merging_threshold, max_chunk_length, embedding_model)
                colored_chunks = get_colored_chunks_text(chunks)
                if visualize:
                    print("Double-pass Chunking Output:")
                    print(colored_chunks)
                index_dp, _ = index_chunks(chunks, embedding_model)
                relevant_chunks = retrieve_chunks_with_filtering(question, index_dp, chunks, embedding_model)
                predicted_answer = generate_answer(question, relevant_chunks, generation_model, tokenizer)
                q_emb = embedding_model.encode(question, convert_to_tensor=True)
                c_emb = embedding_model.encode(context, convert_to_tensor=True)
                relevance_score = util.pytorch_cos_sim(q_emb, c_emb).item()
                inputs = tokenizer(predicted_answer, return_tensors="pt", truncation=True)
                dev_gen = next(generation_model.parameters()).device
                inputs = {k: v.to(dev_gen) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = generation_model(**inputs)
                    logits = outputs.logits
                    probs = softmax(logits.cpu().numpy(), axis=-1)
                    confidence_score = np.mean(np.max(probs, axis=-1))
                eval_result = semantic_evaluator.evaluate(response=predicted_answer, reference=expected_answer)
                llm_acc, llm_eval_text = evaluate_llm_accuracy(question, predicted_answer, expected_answer, generation_model, tokenizer)
                metrics["double_pass"]['relevance_scores'].append(relevance_score)
                metrics["double_pass"]['confidence_scores'].append(confidence_score)
                metrics["double_pass"]['semantic_similarity'].append(eval_result.score)
                metrics["double_pass"]['llm_accuracy'].append(llm_acc)
                results["double_pass"].append({
                    "question": question,
                    "expected_answer": expected_answer,
                    "predicted_answer": predicted_answer,
                    "colored_chunks": colored_chunks
                })

            elif method == 'som':
                inference_result = infer_document_clusters(context, som_trainer, local_rare_clusters, top_n=3, visualize=visualize)
                chunks = split_document_by_anomalies(context, inference_result["top_rare_hits"])
                colored_chunks = get_colored_chunks_text(chunks)
                if visualize:
                    print("SOM Chunking Output:")
                    print(colored_chunks)
                index_som, _ = index_chunks(chunks, embedding_model)
                relevant_chunks = retrieve_chunks_with_filtering(question, index_som, chunks, embedding_model)
                predicted_answer = generate_answer(question, relevant_chunks, generation_model, tokenizer)
                q_emb = embedding_model.encode(question, convert_to_tensor=True)
                c_emb = embedding_model.encode(context, convert_to_tensor=True)
                relevance_score = util.pytorch_cos_sim(q_emb, c_emb).item()
                inputs = tokenizer(predicted_answer, return_tensors="pt", truncation=True)
                dev_gen = next(generation_model.parameters()).device
                inputs = {k: v.to(dev_gen) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = generation_model(**inputs)
                    logits = outputs.logits
                    probs = softmax(logits.cpu().numpy(), axis=-1)
                    confidence_score = np.mean(np.max(probs, axis=-1))
                eval_result = semantic_evaluator.evaluate(response=predicted_answer, reference=expected_answer)
                llm_acc, llm_eval_text = evaluate_llm_accuracy(question, predicted_answer, expected_answer, generation_model, tokenizer)
                metrics["som"]['relevance_scores'].append(relevance_score)
                metrics["som"]['confidence_scores'].append(confidence_score)
                metrics["som"]['semantic_similarity'].append(eval_result.score)
                metrics["som"]['llm_accuracy'].append(llm_acc)
                results["som"].append({
                    "question": question,
                    "expected_answer": expected_answer,
                    "predicted_answer": predicted_answer,
                    "colored_chunks": colored_chunks
                })
            else:
                print("Invalid method provided. Choose among 'standard', 'double_pass', or 'som'.")
        except Exception as e:
            print(f"Error processing example: {str(e)}")
            results[method].append({
                "question": example.get("question", None),
                "expected_answer": example.get("response", None),
                "predicted_answer": None,
                "colored_chunks": ""
            })
            metrics[method]['relevance_scores'].append(0)
            metrics[method]['confidence_scores'].append(0)
            metrics[method]['semantic_similarity'].append(0)
            metrics[method]['llm_accuracy'].append(0)

    # (Aggregate metrics code remains unchanged)
    for key in results.keys():
        if results[key]:
            actual = np.array([1 if res["predicted_answer"] == res["expected_answer"] else 0 for res in results[key]])
            rel_scores = np.array(metrics[key]['relevance_scores'])
            conf_scores = np.array(metrics[key]['confidence_scores'])
            sem_scores = np.array(metrics[key]['semantic_similarity'])
            llm_acc_scores = np.array(metrics[key]['llm_accuracy'])
            if rel_scores.size > 0:
                metrics[key]['rmse'] = float(np.sqrt(mean_squared_error(actual, rel_scores)))
            if conf_scores.size > 0:
                try:
                    metrics[key]['roc_auc'] = float(roc_auc_score(actual, conf_scores))
                except Exception:
                    metrics[key]['roc_auc'] = None
            if sem_scores.size > 0:
                metrics[key]['mean_semantic_similarity'] = float(np.mean(sem_scores))
            if llm_acc_scores.size > 0:
                metrics[key]['mean_llm_accuracy'] = float(np.mean(llm_acc_scores))
            y_true = [1 if res["predicted_answer"] == res["expected_answer"] else 0 for res in results[key]]
            y_pred = [1 if res["predicted_answer"] is not None else 0 for res in results[key]]
            if len(y_true) > 0 and len(y_pred) > 0:
                metrics[key]['accuracy'] = accuracy_score(y_true, y_pred)
                metrics[key]['f1_score'] = f1_score(y_true, y_pred)
    return results, metrics


def get_rare_clusters(cluster_counts, rare_percent):
    clusters_sorted = sorted(cluster_counts.items(), key=lambda x: x[1])
    n_rare = int(len(clusters_sorted) * rare_percent)
    return clusters_sorted[:n_rare]


def run_experiment(exp_params, max_samples, som_trainer, threshold_standard, min_chunk_size, initial_threshold, appending_threshold, merging_threshold, max_chunk_length, visualize, dataset_names, cluster_counts, dataset, embedding_model, generation_model, tokenizer):
    method = exp_params["method"]
    
    if method == "double_pass":
        # Получаем значения параметров из словаря эксперимента.
        local_initial_threshold = exp_params.get("initial_threshold", initial_threshold)
        local_appending_threshold = exp_params.get("appending_threshold", appending_threshold)
        local_merging_threshold = exp_params.get("merging_threshold", merging_threshold)
        full_res, full_met = evaluate_rag_methods(
            method,
            dataset["test"],
            som_trainer,
            embedding_model,
            generation_model,
            tokenizer,
            threshold_standard,   # для double_pass этот параметр не используется
            min_chunk_size,
            local_initial_threshold,
            local_appending_threshold,
            local_merging_threshold,
            max_chunk_length,
            max_samples,
            visualize
        )
        res = full_res["double_pass"]
        met = full_met["double_pass"]
        exp_key = f"double_pass_inTh_{local_initial_threshold}_appTh_{local_appending_threshold}_mergeTh_{local_merging_threshold}"
        
    elif method == "standard":
        local_threshold_standard = exp_params.get("threshold_standard", threshold_standard)
        full_res, full_met = evaluate_rag_methods(
            method,
            dataset["test"],
            som_trainer,
            embedding_model,
            generation_model,
            tokenizer,
            local_threshold_standard,
            min_chunk_size,
            initial_threshold,     # не используется для standard
            appending_threshold,
            merging_threshold,
            max_chunk_length,
            max_samples,
            visualize
        )
        res = full_res["standard"]
        met = full_met["standard"]
        exp_key = f"standard_ts_{local_threshold_standard}"
        
    elif method == "som":
        rare_percent = exp_params["rare_percent"]
        local_rare_clusters = get_rare_clusters(cluster_counts, rare_percent)
        full_res, full_met = evaluate_rag_methods(
            method,
            dataset["test"],
            som_trainer,
            embedding_model,
            generation_model,
            tokenizer,
            threshold_standard,
            min_chunk_size,
            initial_threshold,
            appending_threshold,  # не используется для SOM метода
            merging_threshold,
            max_chunk_length,
            max_samples,
            visualize,
            local_rare_clusters
        )
        res = full_res["som"]
        met = full_met["som"]
        exp_key = f"som_rare_{rare_percent}"
        
    else:
        print("Invalid method provided. Choose among 'standard', 'double_pass', or 'som'.")
        return None, None, None
        
    return exp_key, res, met