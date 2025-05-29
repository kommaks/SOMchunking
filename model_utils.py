import torch
from scipy.special import softmax
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

FIXED_TEMPERATURE = 0.7

def initialize_models(device):
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embedding_model.to(device)
    generation_model_name = "Qwen/Qwen2.5-7B-Instruct"
    torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained(generation_model_name)
    generation_model = AutoModelForCausalLM.from_pretrained(
        generation_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer.pad_token = tokenizer.eos_token
    generation_model.config.pad_token_id = tokenizer.eos_token_id
    return embedding_model, generation_model, tokenizer

# =============================================================================
# Answer generation and evaluation functions (remain unchanged)
# =============================================================================


def generate_answer(question, relevant_chunks, generation_model, tokenizer, max_new_tokens=100):
    try:
        context = " ".join(relevant_chunks)
        context = " ".join(context.split()[:256])
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
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
        gen_device = next(generation_model.parameters()).device
        inputs = {key: value.to(gen_device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = generation_model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                num_beams=3,
                temperature=FIXED_TEMPERATURE,
                no_repeat_ngram_size=2,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                early_stopping=True,
            )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = generated_text.split("Answer:")[-1].strip()
        # Clear GPU cache if needed
        torch.cuda.empty_cache()
        return answer
    except Exception as e:
        print(f"Error in generate_answer: {str(e)}")
        return "Sorry, I couldn't generate an answer due to an error."

# =============================================================================
# Semantic similarity evaluator using embeddings (unchanged)
# =============================================================================

def evaluate_llm_accuracy(question, predicted_answer, ground_truth_answer, generation_model, tokenizer, max_new_tokens=150):
    prompt = f"""
        System: You are a helpful assistant.
        User: \"\"\"===Task===
        I need your help in evaluating an answer provided by an LLM against a ground truth answer. Your task is to determine if the ground truth answer is present in the LLM’s response. Please analyze the provided data and make a decision.
        ===Instructions===
        1. Carefully compare the "Predicted Answer" with the "Ground Truth Answer".
        2. Consider the substance of the answers – look for equivalent information or correct answers. Do not focus on exact wording unless the exact wording is crucial to the meaning.
        3. Your final decision should be based on whether the meaning and the vital facts of the "Ground Truth Answer" are present in the "Predicted Answer:"
        ===Input Data===
        - Question: {question}
        - Predicted Answer: {predicted_answer}
        - Ground Truth Answer: {ground_truth_answer}
        ===Output Format===
        Provide your final evaluation in the following format (without any extra words):
        "Explanation:" (How you made the decision?)
        "Decision:" ("TRUE" or "FALSE")

        Please proceed with the evaluation. \"\"\" 
        """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    gen_device = next(generation_model.parameters()).device
    inputs = {k: v.to(gen_device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = generation_model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=FIXED_TEMPERATURE)
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if " Explanation: " in full_output:
        generated_text = full_output.split(" Explanation: ", 1)[1].strip()
    else:
        raise ValueError("No 'Explanation:' found in the generated text.")
    if "Decision:" in generated_text:
        decision_line = generated_text.split("Decision:", 1)[1].strip()
        decision_word = decision_line.split()[0]
        decision = decision_word.upper()
    else:
        decision = "not given"
        print("No 'Decision:' found in the generated text.")
    score = 1 if decision == "TRUE" else 0
    return score, generated_text
