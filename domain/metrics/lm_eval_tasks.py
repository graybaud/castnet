"""Downstream task evaluation via lm_eval — Pure domain logic.

These functions prepare models for lm_eval but do NOT import lm_eval themselves.
The infrastructure layer handles the actual lm_eval calls.
"""

from typing import Protocol, Any


class LMEvalModelProtocol(Protocol):
    """Protocol for models compatible with lm_eval."""
    pretrained: Any
    tokenizer: Any


TASK_REGISTRY = {
    "mmlu": {
        "tasks": [
            "mmlu_abstract_algebra", "mmlu_anatomy", "mmlu_astronomy",
            "mmlu_business_ethics", "mmlu_clinical_knowledge", "mmlu_college_biology",
            "mmlu_college_chemistry", "mmlu_college_computer_science", "mmlu_college_mathematics",
            "mmlu_college_medicine", "mmlu_college_physics", "mmlu_computer_security",
            "mmlu_conceptual_physics", "mmlu_econometrics", "mmlu_electrical_engineering",
            "mmlu_elementary_mathematics", "mmlu_formal_logic", "mmlu_global_facts",
            "mmlu_high_school_biology", "mmlu_high_school_chemistry",
            "mmlu_high_school_computer_science", "mmlu_high_school_european_history",
            "mmlu_high_school_geography", "mmlu_high_school_government_and_politics",
            "mmlu_high_school_macroeconomics", "mmlu_high_school_mathematics",
            "mmlu_high_school_microeconomics", "mmlu_high_school_physics",
            "mmlu_high_school_psychology", "mmlu_high_school_statistics",
            "mmlu_high_school_us_history", "mmlu_high_school_world_history",
            "mmlu_human_aging", "mmlu_human_sexuality", "mmlu_international_law",
            "mmlu_jurisprudence", "mmlu_logical_fallacies", "mmlu_machine_learning",
            "mmlu_management", "mmlu_marketing", "mmlu_medical_genetics",
            "mmlu_miscellaneous", "mmlu_moral_disputes", "mmlu_moral_scenarios",
            "mmlu_nutrition", "mmlu_philosophy", "mmlu_prehistory",
            "mmlu_professional_accounting", "mmlu_professional_law", "mmlu_professional_medicine",
            "mmlu_professional_psychology", "mmlu_public_relations", "mmlu_security_studies",
            "mmlu_sociology", "mmlu_us_foreign_policy", "mmlu_virology",
            "mmlu_world_religions",
        ],
        "num_fewshot": 5,
        "batch_size": 1,
    },
    "lambada": {
        "tasks": ["lambada_openai"],
        "num_fewshot": 0,
        "batch_size": 1,
    },
    "hellaswag": {
        "tasks": ["hellaswag"],
        "num_fewshot": 0,
        "batch_size": 1,
    },
    "nq_open": {
        "tasks": ["nq_open"],
        "num_fewshot": 0,
        "batch_size": 1,
    },
    "winogrande": {
        "tasks": ["winogrande"],
        "num_fewshot": 0,
        "batch_size": 1,
    },
    "piqa": {
        "tasks": ["piqa"],
        "num_fewshot": 0,
        "batch_size": 1,
    },
    "arc_easy": {
        "tasks": ["arc_easy"],
        "num_fewshot": 0,
        "batch_size": 1,
    },
    "arc_challenge": {
        "tasks": ["arc_challenge"],
        "num_fewshot": 0,
        "batch_size": 1,
    },
}


def get_task_config(task_name: str) -> dict | None:
    """Get lm_eval configuration for a task."""
    return TASK_REGISTRY.get(task_name)


def list_available_tasks() -> list[str]:
    """List all available downstream tasks."""
    return list(TASK_REGISTRY.keys())


def compute_mmlu_summary(results: dict) -> dict:
    """Compute MMLU average and std from per-category results.

    Args:
        results: Dict of category_name -> score

    Returns:
        dict with avg, std, n_tasks.
    """
    scores = list(results.values())
    if not scores:
        return {"avg": 0.0, "std": 0.0, "n_tasks": 0}

    import statistics
    return {
        "avg": round(statistics.mean(scores), 4),
        "std": round(statistics.stdev(scores), 4) if len(scores) > 1 else 0.0,
        "n_tasks": len(scores),
    }
