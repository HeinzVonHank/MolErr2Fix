import tqdm
from string import Template
from collections import Counter
from typing import List, Dict, Tuple
import numpy as np

from dataloader import *
from llm_interface import get_llm_interfance
from utilities import load_yaml_to_dict, calc_f1

ERROR_DICT = load_yaml_to_dict('./error_type_new.yaml')

class OpenEndErrorEvaluator:
    def __init__(self, interface, molbench_formatter, error_dict=ERROR_DICT,):
        self.interface = interface
        self.molbench_formatter = molbench_formatter
        self.error_dict = error_dict
        self._contrusct_prompt_template(error_dict)
        self.problems = []

    def _contrusct_prompt_template(self, error_dict):
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

    def evaluate(self, smiles: str, description: str) -> Tuple[List[str], str]:

        prompt = self.prompt_template.substitute(
            smiles=smiles,
            description=description
        )

        response = self.interface.inference_text_only(prompt)
        
        try:
            predicted_errors = eval(response)
        except Exception as e:
            print(f"Error in eval: {e}")
            predicted_errors = 'Error'
            self.problems.append((smiles, description, response))
        
        return predicted_errors, response
    

    def evaluate_all(self, verbose=False) -> Dict:
        
        results = {"TP":[],"FP":[],"FN":[]}
        N = self.molbench_formatter.db.shape[0]
        for idx in tqdm.tqdm(range(N)):
            row = self.molbench_formatter.db.iloc[idx]
            CID = row['CID']
            smiles = row['SMILES']
            description = row['Initial Description']
            ground_truth = np.array([l[0] for l in row['Parsed Wrong Class/ Reasons']])
            predicted_errors, llm_response = self.evaluate(smiles, description)
            if verbose:
                print(f"CID: {CID}, ground truth:{ground_truth}, LLM response: {llm_response}")
            if predicted_errors == 'Error':
                print(f"Error processing{CID}, LLM return is: {llm_response}")
                continue

            predicted_errors = np.array(predicted_errors)
            # intersection for TP
            TP = np.intersect1d(predicted_errors, ground_truth)
            # A - B for FP
            FP = np.setdiff1d(predicted_errors, TP)
            # B - A for FN
            FN = np.setdiff1d(ground_truth, TP)
            if verbose:
                print(f"CID: {CID}, TP: {TP}, FP: {FP}, FN: {FN}") 
                
            results["TP"] += TP.tolist()
            results["FP"] += FP.tolist()
            results["FN"] += FN.tolist()

        return results

    def compute_metrics(self, results: Dict) -> Dict:

        metrics = {"overall": {}, "per_error_type": {}}

        metrics["overall"]["precision"], metrics["overall"]["recall"], metrics["overall"]["f1_score"] = calc_f1(
            len(results["TP"]), len(results["FP"]), len(results["FN"]) )

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


if __name__=='__main__':
    import argparse, json
    from pathlib import Path
    parser = argparse.ArgumentParser(description="Open-ended error detection")
    parser.add_argument('--read_from', type=str, default="ALL", help="Path to dataset csv")
    parser.add_argument('--error_yaml', type=str, default="./error_type_new.yaml", help="Path to error definition YAML")
    parser.add_argument('--model', type=str, default="gemini-2.0-flash", help="Model name for detection API")
    parser.add_argument('--index_range', type=str, default=None, help='index range in the form of "(start_index,end_index)"')
    args = parser.parse_args()

    # load definitions and data
    errdict = load_yaml_to_dict(args.error_yaml)
    if args.read_from == "ALL":
        loaded_data = ALL_LOADED
    else:
        loaded_data = MolBenchFormater(args.read_from)

    start, end = eval(args.index_range) if args.index_range is not None else (0, -1)
    loaded_data.load(start, end)

    llm_api = get_llm_interfance(args.model)

    evaluator = OpenEndErrorEvaluator(llm_api, loaded_data, errdict)

    results = evaluator.evaluate_all(verbose=True)
    print("Evaluation results:", results)
    metrics = evaluator.compute_metrics(results)

    metrics_to_save = json.dumps(metrics, indent=4, ensure_ascii=False)
    print(metrics_to_save)
    metrics["results"] = results
    save_to = Path("./test_results") / args.model
    save_to.mkdir(parents=True, exist_ok=True)

    with open(save_to/"evaluation_results_OpenEndError.json", 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    with open(save_to/"evaluation_results_OpenEndError_problem_record.json", 'w', encoding='utf-8') as f:
        json.dump(evaluator.problems, f, indent=4, ensure_ascii=False)