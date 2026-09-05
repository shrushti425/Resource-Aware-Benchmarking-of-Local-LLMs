"""JSON Generation and Schema Compliance Evaluation Metrics."""

import json
import re
from typing import Any, Dict, List, Tuple


def extract_json_payload(text: str) -> str:
    """Extracts JSON substring from markdown backticks or raw response text."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match_raw = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if match_raw:
        return match_raw.group(1).strip()
    return text


def evaluate_json_output(prediction_text: str, expected_schema_keys: List[str], ground_truth_dict: Dict[str, Any] = None) -> Dict[str, float]:
    """Evaluates JSON validity, schema compliance ratio, and value correctness."""
    cleaned = extract_json_payload(prediction_text)

    is_valid = 0.0
    parsed_data = None
    try:
        parsed_data = json.loads(cleaned)
        if isinstance(parsed_data, (dict, list)):
            is_valid = 1.0
    except Exception:
        is_valid = 0.0

    if not is_valid or not isinstance(parsed_data, dict):
        return {
            "json_validity": 0.0,
            "schema_compliance": 0.0,
            "field_correctness": 0.0,
        }

    # Schema key coverage
    if expected_schema_keys:
        found_keys = sum(1 for k in expected_schema_keys if k in parsed_data)
        schema_comp = round(found_keys / len(expected_schema_keys), 4)
    else:
        schema_comp = 1.0

    # Field value correctness against ground truth dict
    field_corr = 0.0
    if ground_truth_dict and isinstance(ground_truth_dict, dict):
        correct_fields = 0
        total_fields = len(ground_truth_dict)
        for k, v in ground_truth_dict.items():
            if k in parsed_data and str(parsed_data[k]).strip().lower() == str(v).strip().lower():
                correct_fields += 1
        field_corr = round(correct_fields / total_fields, 4) if total_fields > 0 else 1.0
    else:
        field_corr = schema_comp

    return {
        "json_validity": is_valid,
        "schema_compliance": schema_comp,
        "field_correctness": field_corr,
    }
