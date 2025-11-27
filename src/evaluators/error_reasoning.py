from tqdm import tqdm


class ErrorReasoning:
    """Generate and evaluate error reasoning."""

    def __init__(self, llm_interface):
        self.llm_interface = llm_interface

    def create_prompt(self, smiles, description, error_span):
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

    def generate_all_reasons(self, samples):
        """Generate GPT reasons for all error spans."""
        for sample in tqdm(samples, desc="Generating reasons"):
            for idx, err_info in sample["inside_errors"].items():
                error_span = err_info.get("wrong_point", "")
                if not error_span:
                    continue

                prompt = self.create_prompt(
                    sample["SMILES"],
                    sample["Generated_Description"],
                    error_span
                )

                gpt_answer = self.llm_interface.inference_text_only(
                    query=prompt,
                    system_message="You are a helpful assistant.",
                    temperature=0.7,
                    max_tokens=200
                )

                err_info["gpt_answer"] = gpt_answer

        return samples


class ReasoningEvaluator:
    """Evaluate generated reasoning against human annotations."""

    def __init__(self, llm_interface):
        self.llm_interface = llm_interface

    def compare_two_sentences(self, sentA, sentB):
        """Use GPT to determine if two sentences convey the same meaning."""
        compare_prompt = f"""
            You are a strict evaluator. Please read the two statements below, which describe
            the reason behind a specific chemical error. Determine if they convey the same meaning.

            If they do, respond with "Yes" (and nothing else).
            If they do not, respond with "No" (and nothing else).

            Statement A: "{sentA}"
            Statement B: "{sentB}"
        """
        response = self.llm_interface.inference_text_only(
            query=compare_prompt,
            system_message="You are a strict and concise evaluator.",
            temperature=0.0,
            max_tokens=100
        )
        return "Yes" in response

    def evaluate_stored_reasons(self, samples):
        """Evaluate GPT answers against human reasons."""
        results = {
            "total_count": 0,
            "match_count": 0,
            "accuracy": 0.0,
            "details": [],
            "error_types": {}
        }

        for sample in tqdm(samples, desc="Evaluating reasons"):
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

                is_match = self.compare_two_sentences(human_reason, gpt_answer)

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

        return results
