import json
from tqdm import tqdm


def tokenize(text):
    return text.strip().split()


def compute_iou(pred, true):
    pred_tokens = set(tokenize(pred))
    true_tokens = set(tokenize(true))
    if not pred_tokens and not true_tokens:
        return 1.0
    intersection = pred_tokens.intersection(true_tokens)
    union = pred_tokens.union(true_tokens)
    return len(intersection) / len(union)


def exact_match(pred, true):
    return pred.strip() == true.strip()


class ErrorLocalization:
    """Open-ended error localization generator."""

    def __init__(self, llm_interface, error_dict):
        self.llm_interface = llm_interface
        self.error_dict = error_dict

    def create_prompt(self, smiles, description):
        info_lines = []
        for codename, info in self.error_dict.items():
            definition = " ".join(info["definition"].splitlines()).strip()
            info_lines.append(f"{codename} ({info['name']}): {definition}")

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

    def error_localization(self, samples):
        """Generate error localizations for all samples."""
        for sample in tqdm(samples, desc="Generating error detections"):
            prompt = self.create_prompt(sample["SMILES"], sample["Generated_Description"])
            gpt_output = self.llm_interface.inference_text_only(
                query=prompt,
                system_message="You are a helpful assistant.",
                temperature=0.7,
                max_tokens=300
            )

            gpt_output = gpt_output.strip()
            if gpt_output.startswith("```"):
                lines = gpt_output.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                gpt_output = "\n".join(lines)

            errors = json.loads(gpt_output) if gpt_output else []
            if not isinstance(errors, list):
                errors = []
            sample["error_detections"] = errors

        return samples


def get_ground_truth_errors(sample):
    """Extract human-annotated errors from sample."""
    errors = []
    for _, err in sample.get("inside_errors", {}).items():
        etype = err.get("wrong_type", "").strip()
        span = err.get("wrong_point", "").strip()
        if etype and span:
            errors.append({"error_type": etype, "error_span": span})
    return errors


class LocalizationEvaluator:
    """Evaluate error localization results."""

    def __init__(self, iou_threshold=0.5):
        self.iou_threshold = iou_threshold

    def evaluate(self, samples):
        """Evaluate overall and per-error-type localization metrics."""
        overall_total_errors = 0
        overall_total_pred = 0
        overall_span_match_count = 0
        overall_exact_match_count = 0
        overall_iou_sum = 0.0
        overall_details = {}
        per_error_type = {}

        for sample in tqdm(samples, desc="Evaluating detections"):
            sample_index = sample.get("CID", "N/A")
            overall_details[sample_index] = {}
            gt_errors = get_ground_truth_errors(sample)
            raw_preds = sample.get("error_detections", [])

            pred_errors = []
            for item in raw_preds:
                if isinstance(item, dict):
                    pred_errors.append(item)

            overall_details[sample_index]["gt_errors"] = gt_errors
            overall_details[sample_index]["pred_errors"] = pred_errors

            overall_total_errors += len(gt_errors)
            overall_total_pred += len(pred_errors)

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

        overall_span_accuracy = overall_span_match_count / overall_total_errors if overall_total_errors > 0 else 0.0
        overall_exact_match_accuracy = overall_exact_match_count / overall_total_errors if overall_total_errors > 0 else 0.0
        overall_average_iou = overall_iou_sum / overall_total_errors if overall_total_errors > 0 else 0.0
        overall_prediction_precision = overall_span_match_count / overall_total_pred if overall_total_pred > 0 else 0.0
        overall_f1_score = (2 * overall_span_accuracy * overall_prediction_precision /
                             (overall_span_accuracy + overall_prediction_precision)
                             if (overall_span_accuracy + overall_prediction_precision) > 0 else 0.0)

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
