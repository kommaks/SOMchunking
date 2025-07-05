import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

FIXED_TEMPERATURE = 0.7

def initialize_models(device: torch.device | str):
    """
    Возвращает (embedding_model, generation_model, tokenizer).
    Phi-2 грузим в FP16 без quantization – bnb-0.42.0 не нужен.
    """
    emb_model = SentenceTransformer("all-MiniLM-L6-v2").to(device)

    llm_name = "microsoft/phi-2"
    torch.cuda.empty_cache()

    gen_model = AutoModelForCausalLM.from_pretrained(
        llm_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,     # обязательно для phi-2
    )

    tok = AutoTokenizer.from_pretrained(llm_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    gen_model.config.pad_token_id = tok.pad_token_id
    
    from sentence_transformers import CrossEncoder        # new import

    JUDGE_MODEL_NAME = "cross-encoder/nli-deberta-v3-base"   # 78 M params, ~1 GB vRAM
    judge_model = CrossEncoder(JUDGE_MODEL_NAME)   # ← same CUDA device

    return emb_model, gen_model, tok, judge_model


def generate_answer(
    question: str,
    relevant_chunks: list[str],
    generation_model,
    tokenizer,
    max_new_tokens: int = 128,
):
    try:
        context = " ".join(relevant_chunks)[:4096]  # safety cut
        prompt = f"""
        You are an expert in data analysis and clear communication. Based on the context provided, answer the following question in a clear, definitive, and detailed manner. Your answer should:

        - Begin with a clear affirmation or negation (e.g., "Yes," "No," or a similarly definitive statement).
        - Include specific supporting details or findings from the context.
        - Explain any nuances or uncertainties if the context is not entirely conclusive.
        - Use formal, objective language.
        - Follow the style demonstrated in the examples below.

        Example 1:
        Question:
        Do Surface Porosity and Pore Size Influence Mechanical Properties and Cellular Response to PEEK?
        Expected answer:
        Yes, surface porosity and pore size do influence the mechanical properties and cellular response to PEEK. The micro-CT analysis showed that PEEK-SP with different pore sizes had varying mechanical properties, including tensile strength and interfacial shear strength. Additionally, PEEK-SP exhibited greater proliferation and cell-mediated mineralization compared to smooth PEEK and Ti6Al4V.
        
        Example 2:
        Question:
        Does prosthetic covering of nitinol stents alter healing characteristics or hemodynamics?
        Expected answer:
        Yes, prosthetic covering alters healing characteristics and hemodynamics. The covered stents showed increased neointimal thickness and a higher intima-to-media ratio compared to bare stents.
        
        Example 3:
        Question:
        Is diabetes mellitus a negative prognostic factor for the treatment of advanced non-small-cell lung cancer?
        Expected answer:
        No, based on the provided context, diabetes mellitus has not been consistently identified as a negative prognostic factor for the treatment of advanced non-small-cell lung cancer.
        
        Context: {context}
        Question: {question}
        Answer:
        """
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(generation_model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = generation_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,                 # ← включён sampling
                temperature=FIXED_TEMPERATURE,
                top_p=0.9,
                no_repeat_ngram_size=2,
                num_beams=1,                    # beam-search не нужен
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = text.split("Answer:")[-1].strip()
        return answer

    except Exception as e:
        print(f"[generate_answer] {e}")
        return "Sorry, I couldn't generate an answer due to an error."


def evaluate_llm_accuracy(
    question: str,
    predicted_answer: str,
    ground_truth_answer: str,
    judge_model,      # ← kept for call-site compatibility (ignored here)
    tokenizer,             # ← kept (ignored)
    max_new_tokens: int = 150,
):
    """
    Returns (score:int, explanation:str).

    We replace the old “LLM-as-judge” with a single NLI forward-pass.
    * entailment prob >= 0.5  → Decision TRUE
    * entailment prob < 0.5 → Decision FALSE
    """
    try:
        # (1) получаем вероятность энтэйлмента
        probs = judge_model.predict(
            [(predicted_answer, ground_truth_answer)],
            apply_softmax=True            # ← вернёт np.array shape (1, 3)
        )[0]                              # → [contr, neutral, entail]
        prob_entail = float(probs[2])     # индекс 2 — entailment

        # (2) превращаем в 0 / 1
        score = 1 if prob_entail >= 0.5 else 0
        explanation = f"Entailment prob = {prob_entail:.2f}"

        if score == 0:
            print("[evaluate_llm_accuracy] entailment < 0.5 → score=0")
        return score, explanation

    except Exception as e:
        print(f"[evaluate_llm_accuracy] {e} – returning score=0")
        return 0, ""
