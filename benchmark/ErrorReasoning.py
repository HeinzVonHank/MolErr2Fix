import os
import openai
import json
import yaml
import glob
import argparse
from typing import List, Dict, Tuple
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu

# Assuming MolBenchFormater, get_llm_interfance, GPT4Interface, and GeminiInterface are defined elsewhere
# from dataloader import MolBenchFormater
# from llm_interface import get_llm_interfance, GPT4Interface, GeminiInterface

###############################################################################
# 1) Helper Functions
###############################################################################
def save_as_json(data, file_path: str):
    """Saves data to a JSON file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"Data successfully saved to {file_path}")
    except Exception as e:
        print(f"Error saving {file_path}: {e}")


def load_yaml_to_dict(yaml_file):
    """Loads a YAML file, checks for its existence, and returns the error type dictionary."""
    if not os.path.exists(yaml_file):
        print(f"Warning: YAML file '{yaml_file}' not found. Using empty error mapping.")
        return {}
    
    with open(yaml_file, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
    
    if "error_types" not in data:
        print(f"Warning: 'error_types' key not found in YAML file '{yaml_file}'.")
        return {}

    return {err["codename"]: err for err in data["error_types"]}


###############################################################################
# 2) LLM Interfaces
###############################################################################
# from llm_interface import GPT4Interface, GeminiInterface


###############################################################################
# 3) ReasonAnswerGenerator
###############################################################################
class ReasonAnswerGenerator:
    """
    Generates a reason from GPT based on molecule structure, description, and error span,
    and saves it to sample["inside_errors"][idx]["gpt_answer"].
    """
    def __init__(self, gpt_interface, error_dict: Dict):
        self.gpt_interface = gpt_interface

    def create_prompt_with_reason(self, smiles: str, description: str, error_span: str) -> str:
        prompt = f"""
You are a chemistry expert specialized in molecular structure understanding and error detection.

You are given a molecule and a faulty description of it. A specific fragment in the description is believed to be incorrect. Your task is to explain **why** that part is wrong based on the molecule's actual structure.

**Molecule Information:**
- SMILES: {smiles}

**Faulty Description:**
"{description}"

**Suspected Faulty Fragment:**
"{error_span}"

**Your Task:**
Explain briefly and clearly why the above fragment is incorrect. Limit your explanation to 1-2 concise sentences.
"""
        return prompt

    def generate_all_reasons(self, samples: List[Dict]) -> List[Dict]:
        for sample in tqdm(samples, desc="Generating GPT reasons"):
            for idx, err_info in sample["inside_errors"].items():
                try:
                    error_span = err_info.get("wrong_point", "")
                    if not error_span:
                        continue  # Skip entries with no error_span

                    prompt = self.create_prompt_with_reason(
                        sample["SMILES"],
                        sample["Generated_Description"],
                        error_span
                    )

                    gpt_answer = self.gpt_interface.inference_text_only(
                        query=prompt,
                        system_message="You are a helpful assistant.",
                        temperature=0.7,
                        max_tokens=200
                    )

                    print(f"llm_answer: {gpt_answer}")

                    if gpt_answer.startswith("Error:"):
                        print(f"[Warning] GPT call failed for sample {sample.get('index', 'N/A')}, error: {gpt_answer}")
                        err_info["gpt_answer"] = ""
                        continue

                    err_info["gpt_answer"] = gpt_answer

                except Exception as e:
                    print(f"[Exception] Generating reason failed. Sample index: {sample.get('index','N/A')}, error: {str(e)}")
                    err_info["gpt_answer"] = ""

        return samples

###############################################################################
# 4) ReasonAnswerEvaluator
###############################################################################
class ReasonAnswerEvaluator:
    """
    Compares the GPT-generated reason (gpt_answer) with the human-annotated one (wrong_reason)
    by calling GPT to determine if they convey the same meaning.
    """
    def __init__(self, gpt_interface):
        self.gpt_interface = gpt_interface

    def compare_two_sentences(self, sentA: str, sentB: str) -> bool:
        """
        Uses GPT to determine whether two sentences convey the same meaning.
        Returns True if GPT answers "Yes", otherwise False.
        """
        compare_prompt = f"""
            You are a strict evaluator. Please read the two statements below, which describe 
            the reason behind a specific chemical error. Determine if they convey the same meaning.

            If they do, respond with "Yes" (and nothing else).
            If they do not, respond with "No" (and nothing else).

            Statement A: "{sentA}"
            Statement B: "{sentB}"
        """
        response = self.gpt_interface.inference_text_only(
            query=compare_prompt,
            system_message="You are a strict and concise evaluator.",
            temperature=0.0,
            max_tokens=100
        )
        if response.startswith("Error:"):
            print(f"[Warning] compare_two_sentences GPT call failed: {response}")
            return False

        return "Yes" in response

    def evaluate_stored_reasons(self, samples: List[Dict]) -> Dict:
        """
        Reads gpt_answer and human_reason from samples and calls GPT to check for semantic consistency.
        If compare_two_sentences fails or returns an error, it skips the entry.

        Calculates overall and per-error-type match rates and returns the following structure:
        {
            "total_count": int,
            "match_count": int,
            "accuracy": float,
            "details": [...],
            "error_types": {
                "E1": {...},
                "E2": {...},
                ...
            }
        }
        """
        results = {
            "total_count": 0,
            "match_count": 0,
            "accuracy": 0.0,
            "details": [],
            "error_types": {}
        }

        for sample in tqdm(samples, desc="Evaluating GPT answers"):
            for idx, err_info in sample["inside_errors"].items():
                human_reason = err_info.get("wrong_reason", "")
                gpt_answer = err_info.get("gpt_answer", "")
                wrong_type = err_info.get("wrong_type", "")

                if not gpt_answer:
                    continue

                if wrong_type not in results["error_types"]:
                    results["error_types"][wrong_type] = {
                        "total_count": 0,
                        "match_count": 0,
                        "accuracy": 0.0
                    }

                try:
                    is_match = self.compare_two_sentences(human_reason, gpt_answer)
                except Exception as e:
                    print(f"[Exception] compare_two_sentences failed. Sample index: {sample.get('index','N/A')}, {str(e)}")
                    is_match = False

                results["total_count"] += 1
                results["error_types"][wrong_type]["total_count"] += 1

                if is_match:
                    results["match_count"] += 1
                    results["error_types"][wrong_type]["match_count"] += 1

                results["details"].append({
                    "sample_index": sample.get("CID", "N/A"),
                    "error_idx": idx,
                    "wrong_type": wrong_type,
                    "human_reason": human_reason,
                    "gpt_answer": gpt_answer,
                    "is_match": is_match
                })

        if results["total_count"] > 0:
            results["accuracy"] = results["match_count"] / results["total_count"]

        for wt, stat in results["error_types"].items():
            if stat["total_count"] > 0:
                stat["accuracy"] = stat["match_count"] / stat["total_count"]
            else:
                stat["accuracy"] = 0.0

        return results


###############################################################################
# 5) Main Process
###############################################################################
def main():
    parser = argparse.ArgumentParser(description="Error Explanation Evaluation (Closed Mode)")
    parser.add_argument('--tsv', type=str, default="benchmark/Datasets/labels_v3/LabeledAll.tsv",
                        help="Path to the dataset TSV file")
    parser.add_argument('--error_yaml', type=str, default="benchmark/error_type_new.yaml",
                        help="Path to the error definition YAML file")
    parser.add_argument('--model', type=str, default="gpt-4o",
                        help="Name of the LLM model for explanation generation")
    parser.add_argument('--model_eval', type=str, default="gpt-4o",
                        help="Name of the LLM model for explanation evaluation")
    parser.add_argument('--end', type=int, default=5,
                        help="End index for sample loading (-1 for all samples)")
    parser.add_argument('--start', type=int, default=0,
                        help="End index for sample loading (-1 for all samples)")
    args = parser.parse_args()

    # 1) Load error type definitions
    error_dict = load_yaml_to_dict(args.error_yaml)

    # 2) Load data
    molbench = MolBenchFormater(args.tsv)
    molbench.samples = molbench.load(start=args.start, end=args.end, return_json=True)

    # 3) Initialize LLM interface for generation
    eval_api = get_llm_interfance(args.model)

    # 4) Generation Phase
    reason_generator = ReasonAnswerGenerator(eval_api, error_dict)
    samples_with_gpt = reason_generator.generate_all_reasons(molbench.samples)

    # 5) Save generation results
    output_dir = os.path.join("benchmark", "Results", "Reasoning")
    os.makedirs(output_dir, exist_ok=True)
    json_answer_path = os.path.join(output_dir, f"generated_reasons_closed_{args.model}_{args.end}.json")
    save_as_json(samples_with_gpt, json_answer_path)

    # 6) Initialize LLM interface for evaluation
    gen_api = GPT4Interface(model="gpt-4o")
    # 7) Evaluation Phase
    evaluator = ReasonAnswerEvaluator(gen_api)
    eval_results = evaluator.evaluate_stored_reasons(samples_with_gpt)

    # 8) Save evaluation results
    json_eval_path = os.path.join(output_dir, f"evaluation_reason_results_closed_{args.model}_by_{args.model_eval}_{args.end}.json")
    save_as_json(eval_results, json_eval_path)

    # 9) Print summary results
    print("\n=== Explanation Evaluation Summary ===")
    print(f"Total:   {eval_results['total_count']}")
    print(f"Matched: {eval_results['match_count']}")
    print(f"Accuracy:{eval_results['accuracy']:.3f}\n")
    print("=== Per-Error-Type Results ===")
    for wt, stat in eval_results["error_types"].items():
        print(f" - {wt}: {stat['match_count']}/{stat['total_count']} matched ({stat['accuracy']:.3f})")


def summarize_explanation(results_dir: str):
    """
    Scans the reasoning generation and evaluation files under results_dir,
    calculates overall BLEU score by comparing human_reason with gpt_answer, and prints.
    """
    eval_pattern = os.path.join(results_dir, "evaluation_reason_results_*_*.json")
    results = []
    for gen_path in sorted(glob.glob(eval_pattern)):
        fname = os.path.basename(gen_path)
        model = fname.split("_")[4]
        with open(gen_path, "r", encoding="utf-8") as f:
            samples = json.load(f)

        refs, hyps = [], []
        for s in samples["details"]:
            hr = s.get("human_reason", "").strip()
            ga = s.get("gpt_answer", "").strip()
            if hr and ga:
                refs.append(hr.split())
                hyps.append(ga.split())

        bleu_scores = [sentence_bleu([r], h) for r, h in zip(refs, hyps)] if refs else []
        avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0

        match_rate = samples.get("accuracy", 0.0)
        results.append((model, avg_bleu, match_rate))

    for result in results:
        model, avg_bleu, match_rate = result
        print(f"{model:<15} BLEU: {avg_bleu:6.3f}    GPT-Match: {match_rate:6.3f}")

if __name__ == "__main__":
    # main()
    summarize_explanation("benchmark/Results/Reasoning")