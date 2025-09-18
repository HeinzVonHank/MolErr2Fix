import json, os
import yaml
from typing import List, Dict, Tuple

def calc_f1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

def save_as_json(data, file_path: str):
    """Save data to a JSON file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"Data successfully saved to {file_path}")
    except Exception as e:
        print(f"Error saving {file_path}: {e}")
        

def load_yaml_to_dict(yaml_file):
    """Load a YAML file, check for its existence, and return the error type dictionary."""
    if not os.path.exists(yaml_file):
        print(f"Warning: YAML file '{yaml_file}' not found. Using empty error mapping.")
        return {}
    
    with open(yaml_file, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
    
    if "error_types" not in data:
        print(f"Warning: 'error_types' key not found in YAML file '{yaml_file}'.")
        return {}

    return {err["name"]: err for err in data["error_types"]}


def clean_json_response(response: str) -> str:
    response = response.replace("Corrected Description:", "").strip()
    if response.startswith("```"):
        lines = response.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        response = "\n".join(lines).strip()
    return response