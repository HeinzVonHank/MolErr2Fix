import os
import argparse
import json
import glob
from typing import List, Dict
from tqdm import tqdm

from dataloader import MolBenchFormater, ALL_LOADED
from llm_interface import get_llm_interfance
from utilities import save_as_json, load_yaml_to_dict


def save_as_json(data, file_path: str):
    """Save data to a JSON file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"Data successfully saved to {file_path}")
    except Exception as e:
        print(f"Error saving {file_path}: {e}")


def tokenize(text: str) -> List[str]:
    return text.strip().split()

def compute_iou(pred: str, true: str) -> float:
    pred_tokens = set(tokenize(pred))
    true_tokens = set(tokenize(true))
    if not pred_tokens and not true_tokens:
        return 1.0
    intersection = pred_tokens.intersection(true_tokens)
    union = pred_tokens.union(true_tokens)
    return len(intersection) / len(union)

def exact_match(pred: str, true: str) -> bool:
    return pred.strip() == true.strip()


class UnifiedErrorLocalizaitonGenerator:
    """
    Unified error localization generator (open-ended mode only).
    Lists all error types and their definitions, and prompts the model to return a JSON array.
    Each element in the array contains "error_type" and "error_span" fields.
    """
    def __init__(self, gpt_interface, error_dict: Dict, mode: str = "openended"):
        self.gpt_interface = gpt_interface
        self.error_dict = error_dict
        self.mode = mode

    def create_prompt(self, smiles: str, description: str) -> str:
        # 1) Extract codename, name, and definition for each type from YAML
        info_lines = []
        for codename, info in self.error_dict.items():
            # Combine multi-line definitions into a single line
            definition = " ".join(info["definition"].splitlines()).strip()
            info_lines.append(f"{codename} ({info['name']}): {definition}")

        # Create a Markdown-style list
        error_types_info = "\n".join(f"- {line}" for line in info_lines)

        prompt = f"""
            You are an expert in molecular chemistry and error detection.
            Below are the seven error types with their definitions:
            {error_types_info}
            Given the following molecule information and its erroneous description, list all the errors present in the description.
            For each error, output the error type and the exact text segment (error span) from the description.
            Please output the result in JSON format as an array of objects, each object having the keys "error_type" and "error_span".
            Do not include any additional commentary.

            Molecule's structure (SMILES): {smiles}
            Erroneous Description: {description}
        """
        return prompt

    def error_localization(self, samples: List[Dict]) -> List[Dict]:
        for sample in tqdm(samples, desc="Generating open-ended error detections"):
            try:
                prompt = self.create_prompt(sample["SMILES"], sample["Initial Description"])
                gpt_output = self.gpt_interface.inference_text_only(
                    query=prompt,
                    system_message="You are a helpful assistant.",
                    temperature=0.7,
                    max_tokens=300
                )
                print(f"gpt_output: {gpt_output}")
                try:
                    # Remove potential Markdown code block markers
                    gpt_output = gpt_output.strip()
                    if gpt_output.startswith("```"):
                        lines = gpt_output.splitlines()
                        if lines[0].startswith("```"):
                            lines = lines[1:]
                        if lines and lines[-1].startswith("```"):
                            lines = lines[:-1]
                        gpt_output = "\n".join(lines)
                    errors = json.loads(gpt_output)
                    if not isinstance(errors, list):
                        errors = []
                except Exception as e:
                    print(f"[JSON Parse Error] Sample {sample.get('index', 'N/A')}: {e}")
                    errors = []
                sample["error_detections"] = errors
            except Exception as e:
                print(f"[Exception] Open-ended mode for sample {sample.get('index', 'N/A')}: {str(e)}")
                sample["error_detections"] = []
        return samples


def get_ground_truth_errors(sample: Dict) -> List[Dict]:
    """
    Extracts human-annotated errors from sample["inside_errors"]
    and returns a list where each element is {"error_type": <type>, "error_span": <error text>}.
    """
    errors = []
    for _, err in sample.get("inside_errors", {}).items():
        etype = err.get("wrong_type", "").strip()
        span = err.get("wrong_point", "").strip()
        if etype and span:
            errors.append({"error_type": etype, "error_span": span})
    return errors


class UnifiedErrorDetectionEvaluator:
    """
    Unified error detection evaluator (open-ended mode only).
    Extracts a list of human-annotated errors (each with "error_type" and "error_span")
    and a list of model-predicted errors (same format) from each sample.
    
    The evaluation is in two parts:
    1. Overall: Ignores error type and calculates the match between error spans,
       providing overall span_accuracy, exact_match_accuracy, precision, and F1 score.
    2. Per Error Type: Calculates the same metrics for each error type for fine-grained analysis.
    """
    def __init__(self, iou_threshold: float = 0.5, mode="opended"):
        self.iou_threshold = iou_threshold

    def evaluate(self, samples: List[Dict]) -> Dict:
        # Overall metric statistics
        overall_total_errors = 0   # Total count of ground truth errors across all samples
        overall_total_pred = 0     # Total count of predicted errors across all samples
        overall_span_match_count = 0
        overall_exact_match_count = 0
        overall_iou_sum = 0.0
        overall_details = {}       # Record match details per sample

        # Summarize metrics per error type
        per_error_type = {}

        # Iterate through each sample
        for sample in tqdm(samples, desc="Evaluating overall detections"):
            sample_index = sample.get("CID", "N/A")
            overall_details[sample_index] = {}
            # Extract ground truth and predicted error lists
            gt_errors = get_ground_truth_errors(sample)
            raw_preds = sample.get("error_detections", [])

            # Convert each prediction to a dict
            pred_errors = []
            for item in raw_preds:
                if isinstance(item, dict):
                    pred_errors.append(item)
                else:
                    try:
                        pred_errors.append(json.loads(item))
                    except json.JSONDecodeError:
                        # Skip if parsing fails
                        continue

            overall_details[sample_index]["gt_errors"] = gt_errors
            overall_details[sample_index]["pred_errors"] = pred_errors

            overall_total_errors += len(gt_errors)
            overall_total_pred += len(pred_errors)

            # Overall evaluation: Match based on error_span, ignoring error_type
            candidates = []
            for i, gt in enumerate(gt_errors):
                for j, pred in enumerate(pred_errors):
                    iou = compute_iou(pred["error_span"], gt["error_span"])
                    candidates.append((iou, i, j))
            candidates.sort(key=lambda x: x[0], reverse=True)
            used_gt = set()
            used_pred = set()
            matched_pairs = []
            for iou, i, j in candidates:
                if iou < self.iou_threshold:
                    continue
                if i not in used_gt and j not in used_pred:
                    used_gt.add(i)
                    used_pred.add(j)
                    matched_pairs.append((i, j, iou))
            sample_span_match = len(matched_pairs)
            sample_exact_match = sum(1 for (i, j, iou) in matched_pairs
                                     if exact_match(pred_errors[j]["error_span"], gt_errors[i]["error_span"]))
            sample_iou_sum = sum(iou for (_, _, iou) in matched_pairs)
            overall_span_match_count += sample_span_match
            overall_exact_match_count += sample_exact_match
            overall_iou_sum += sample_iou_sum
            overall_details[sample_index]["matched_pairs"] = matched_pairs

            # Per Error Type evaluation: Match within each error_type
            per_sample_type_details = {}
            error_types_in_sample = set(err["error_type"] for err in gt_errors)
            for etype in error_types_in_sample:
                gt_list = [err for err in gt_errors if err["error_type"] == etype]
                pred_list = [err for err in pred_errors if err["error_type"] == etype]
                candidates_type = []
                for i, gt in enumerate(gt_list):
                    for j, pred in enumerate(pred_list):
                        iou = compute_iou(pred["error_span"], gt["error_span"])
                        candidates_type.append((iou, i, j))
                candidates_type.sort(key=lambda x: x[0], reverse=True)
                used_gt_type = set()
                used_pred_type = set()
                matched_pairs_type = []
                for iou, i, j in candidates_type:
                    if iou < self.iou_threshold:
                        continue
                    if i not in used_gt_type and j not in used_pred_type:
                        used_gt_type.add(i)
                        used_pred_type.add(j)
                        matched_pairs_type.append((i, j, iou))
                span_match_type = len(matched_pairs_type)
                exact_match_type = sum(1 for (i, j, iou) in matched_pairs_type
                                        if exact_match(pred_list[j]["error_span"], gt_list[i]["error_span"]))
                iou_sum_type = sum(iou for (_, _, iou) in matched_pairs_type)
                if etype not in per_error_type:
                    per_error_type[etype] = {"total": 0, "span_match": 0, "exact_match": 0, "iou_sum": 0.0, "pred_total": 0}
                per_error_type[etype]["total"] += len(gt_list)
                per_error_type[etype]["span_match"] += span_match_type
                per_error_type[etype]["exact_match"] += exact_match_type
                per_error_type[etype]["iou_sum"] += iou_sum_type
                per_error_type[etype]["pred_total"] += len(pred_list)
                per_sample_type_details[etype] = {
                    "gt": gt_list,
                    "pred": pred_list,
                    "matched_pairs": matched_pairs_type,
                    "span_match": span_match_type,
                    "exact_match": exact_match_type,
                    "iou_sum": iou_sum_type
                }
            overall_details[sample_index]["per_error_type"] = per_sample_type_details

        # Calculate Overall metrics
        overall_span_accuracy = overall_span_match_count / overall_total_errors if overall_total_errors > 0 else 0.0
        overall_exact_match_accuracy = overall_exact_match_count / overall_total_errors if overall_total_errors > 0 else 0.0
        overall_average_iou = overall_iou_sum / overall_total_errors if overall_total_errors > 0 else 0.0
        overall_prediction_precision = overall_span_match_count / overall_total_pred if overall_total_pred > 0 else 0.0
        overall_f1_score = (2 * overall_span_accuracy * overall_prediction_precision /
                             (overall_span_accuracy + overall_prediction_precision)
                             if (overall_span_accuracy + overall_prediction_precision) > 0 else 0.0)

        # Calculate metrics for each error_type
        for etype, data in per_error_type.items():
            total = data["total"]
            data["span_accuracy"] = data["span_match"] / total if total > 0 else 0.0
            data["exact_match_accuracy"] = data["exact_match"] / total if total > 0 else 0.0
            data["average_iou"] = data["iou_sum"] / total if total > 0 else 0.0
            data["prediction_precision"] = data["span_match"] / data["pred_total"] if data["pred_total"] > 0 else 0.0
            data["f1_score"] = (2 * data["span_accuracy"] * data["prediction_precision"] /
                                 (data["span_accuracy"] + data["prediction_precision"])
                                 if (data["span_accuracy"] + data["prediction_precision"]) > 0 else 0.0)

        results = {
            "total_error_count": overall_total_errors,
            "total_predicted_count": overall_total_pred,
            "span_match_count": overall_span_match_count,
            "exact_match_count": overall_exact_match_count,
            "average_iou": overall_average_iou,
            "span_accuracy": overall_span_accuracy,
            "exact_match_accuracy": overall_exact_match_accuracy,
            "prediction_precision": overall_prediction_precision,
            "f1_score": overall_f1_score,
            "details": overall_details,
            "per_error_type": per_error_type
        }
        return results

def summarize_localization(results_dir: str = "test_results/Localization", yaml_path: str="benchmark/error_type_new.yaml",
                           iou_thresholds: List[float]=[0.5, 0.7],
                           end: int=-1):
    """
    For each model's localization JSON file under results_dir:
      error_localization_openended_{model}_{end}.json
    It loads samples (with gt & pred spans),
    recalculates metrics using the evaluator for each iou_threshold, and prints a table.
    """
    pattern = os.path.join(results_dir, f"error_localization_openended_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"[WARN] No files found: {pattern}")
        return

    results = []
    for path in files:
        fname = os.path.basename(path)
        # Extract model name
        parts = fname.split("_")
        model = parts[3]

        # Read the generated sample list (containing sample["error_detections"])
        with open(path, "r", encoding="utf-8") as f:
            samples = json.load(f)

        row = [f"{model:>12}"]
        avg_iou_across = []

        for t in iou_thresholds:
            evaluator = UnifiedErrorDetectionEvaluator(iou_threshold=t)
            metrics = evaluator.evaluate(samples)
            span_acc = metrics.get("span_accuracy", 0.0)
            avg_iou  = metrics.get("average_iou", 0.0)
            row.append(f"{span_acc:12.3f}")
            avg_iou_across.append(avg_iou)

        # Use the average of average_iou from two thresholds as the overall avg IoU
        overall_avg_iou = sum(avg_iou_across) / len(avg_iou_across)
        row.append(f"{overall_avg_iou:12.3f}")

        results.append(row)
    
    headers = ["Model"] + [f"IoU≥{t:.1f}" for t in iou_thresholds] + ["Avg IoU"]
    print("  ".join(f"{h:>12}" for h in headers))
    print("-" * (14 * len(headers)))
    for result in results:
        print("  ".join(result))


def main():
    parser = argparse.ArgumentParser(description="Open-ended Error Detection & Evaluation")
    parser.add_argument(
        '--tsv', type=str,
        default="benchmark/Datasets/labels_v3/LabeledAll.tsv",
        help="Path to the dataset TSV file"
    )
    parser.add_argument(
        '--error_yaml', type=str,
        default="error_type_new.yaml",
        help="Path to the error-definition YAML"
    )
    parser.add_argument(
        '--model', type=str,
        default="gemini-2.0-flash",
        help="Model name for error localization (e.g. gpt-4, gpt-4o, gemini-2.0-flash)"
    )
    parser.add_argument(
        '--end', type=int,
        default=-1,
        help="End index for sample loading (-1 for all samples)"
    )
    parser.add_argument(
        '--start', type=int,
        default=0,
        help="End index for sample loading (-1 for all samples)"
    )
    args = parser.parse_args()

    # 1) Load dataset
    molbench = ALL_LOADED
    samples  = molbench.load(start=args.start,end=args.end,return_json=True)

    # 2) Load error definitions
    error_dict = load_yaml_to_dict(args.error_yaml)

    # 3) Initialize LLM interface
    llm = get_llm_interfance(args.model)

    # 4) Generate error detections (Open-ended mode)
    mode = "openended"
    generator = UnifiedErrorLocalizaitonGenerator(llm, error_dict, mode=mode)
    samples_with_errors = generator.error_localization(samples)

    # 5) Save generation results
    output_dir = os.path.join("test_results", "Localization")
    os.makedirs(output_dir, exist_ok=True)
    json_gen = os.path.join(output_dir, f"error_localization_{mode}_{args.model}_{args.end}.json")
    save_as_json(samples_with_errors, json_gen)

    # 6) Evaluation phase
    evaluator = UnifiedErrorDetectionEvaluator(iou_threshold=0.5, mode=mode)
    eval_results = evaluator.evaluate(samples_with_errors)

    # 7) Save evaluation results
    json_eval = os.path.join(output_dir, f"evaluation_localization_{mode}_{args.model}_{args.end}.json")
    save_as_json(eval_results, json_eval)

    # 8) Print summary results
    print(f"\n=== {mode.capitalize()} Error Detection Evaluation Summary ===")
    print(f"Total error count: {eval_results['total_error_count']}")
    print(f"Span-based accuracy (IoU ≥ {evaluator.iou_threshold}): {eval_results['span_accuracy']:.3f}")
    print(f"Exact-match accuracy: {eval_results['exact_match_accuracy']:.3f}")
    print(f"Prediction Precision: {eval_results['prediction_precision']:.3f}")
    print(f"F1 Score: {eval_results['f1_score']:.3f}")
    print(f"Average IoU: {eval_results['average_iou']:.3f}")
    print("\n--- Per Error Type Details ---")
    for etype, stats in eval_results["per_error_type"].items():
        print(
            f"{etype}: Total={stats['total']}, "
            f"Span Acc={stats['span_accuracy']:.3f}, "
            f"Exact Acc={stats['exact_match_accuracy']:.3f}, "
            f"Avg IoU={stats['average_iou']:.3f}"
        )

    summarize_localization("test_results/Localization", "benchmark/error_type_new.yaml", [0.5, 0.7], end=args.end)

if __name__ == "__main__":
    main()