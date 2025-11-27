import numpy as np
from collections import Counter
from string import Template
from tqdm import tqdm

from src.dataloader import CODE_TO_ERROR
from src.utils import calc_f1


class ErrorDetector:
    """Open-ended error type detection evaluator."""

    def __init__(self, interface, molbench_formatter, error_dict):
        self.interface = interface
        self.molbench_formatter = molbench_formatter
        self.error_dict = error_dict
        self._construct_prompt_template(error_dict)
        self.problems = []

    def _construct_prompt_template(self, error_dict):
        info_lines = []
        for _, info in error_dict.items():
            definition = " ".join(info["definition"].splitlines()).strip()
            info_lines.append(f"{info['codename']}: {definition}")

        error_types_info = "\n".join(f"- {line}" for line in info_lines)
        self.prompt_template = Template(Template("""
        You are an expert in molecular chemistry and error detection. Your task is to analyze a given molecule
        and determine which error types are present.

        **List of Possible Error Types:**
        $error_types_info

        **Molecule Information:**
        - **Molecule's structure (SMILES format)**: $smiles
        - **Molecule's Description**: $description

        **Task:**
        Based on the provided structure and description, carefully examine whether any of the error types listed above exist
        in this molecule.
        Please return ONLY a list of error codenames (e.g., ["E1","E2"]). DO NOT add anything else.
        **Response Format:**
        Your response must be a valid string that can be converted to list by the eval function in Python. If no error is detected, return an empty list `[]`.
        """).safe_substitute(error_types_info=error_types_info))

    def evaluate(self, smiles, description):
        """Evaluate a single molecule for errors."""
        prompt = self.prompt_template.substitute(smiles=smiles, description=description)
        response = self.interface.inference_text_only(prompt)

        predicted_errors = eval(response)
        return predicted_errors, response

    def evaluate_all(self, verbose=False):
        """Evaluate all molecules in the dataset."""
        results = {"TP": [], "FP": [], "FN": []}
        N = self.molbench_formatter.db.shape[0]

        for idx in tqdm(range(N)):
            row = self.molbench_formatter.db.iloc[idx]
            CID = row['CID']
            smiles = row['SMILES']
            description = row['Generated_Description']
            ground_truth = np.array([l[0] for l in row['Parsed Wrong Class/ Reasons']])

            predicted_errors, llm_response = self.evaluate(smiles, description)
            if predicted_errors == 'Error':
                continue

            predicted_errors = np.array(predicted_errors)
            TP = np.intersect1d(predicted_errors, ground_truth)
            FP = np.setdiff1d(predicted_errors, TP)
            FN = np.setdiff1d(ground_truth, TP)

            results["TP"] += TP.tolist()
            results["FP"] += FP.tolist()
            results["FN"] += FN.tolist()

        return results

    def compute_metrics(self, results):
        """Compute precision, recall, F1 for overall and per error type."""
        metrics = {"overall": {}, "per_error_type": {}}

        metrics["overall"]["precision"], metrics["overall"]["recall"], metrics["overall"]["f1_score"] = calc_f1(
            len(results["TP"]), len(results["FP"]), len(results["FN"]))

        results["TP"] = Counter(results["TP"])
        results["FP"] = Counter(results["FP"])
        results["FN"] = Counter(results["FN"])

        for error_type in CODE_TO_ERROR.keys():
            tp = results["TP"].get(error_type, 0)
            fp = results["FP"].get(error_type, 0)
            fn = results["FN"].get(error_type, 0)

            precision, recall, f1_score = calc_f1(tp, fp, fn)
            metrics["per_error_type"][error_type] = {
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score
            }

        return metrics
