import json
import os
import yaml


def calc_f1(tp, fp, fn):
    """Calculate precision, recall, and F1 score."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1


def save_json(data, filepath):
    """Save data to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def load_yaml(filepath):
    """Load YAML file and return error type dictionary."""
    if not os.path.exists(filepath):
        return {}

    with open(filepath, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    if "error_types" not in data:
        return {}

    return {err["name"]: err for err in data["error_types"]}


def clean_json_response(response):
    """Remove markdown code blocks from JSON response."""
    response = response.replace("Corrected Description:", "").strip()
    if response.startswith("```"):
        lines = response.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        response = "\n".join(lines).strip()
    return response
