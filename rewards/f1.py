import verifiers as vf

from utils.metrics import (
    calculate_f1,
    calculate_precision,
    calculate_recall,
    get_file_sets,
)


def f1_reward(prompt, completion: vf.Messages, answer, state, task, info) -> float:
    """
    Calculate file-level F1 score.

    F1 = 2 * (precision * recall) / (precision + recall)

    Measures: Harmonic mean of precision and recall.

    Args:
        answer: Should contain the patch string
    """
    result_files, patch_files = get_file_sets(completion, answer)

    if result_files is None or patch_files is None:
        return 0.0

    precision = calculate_precision(result_files, patch_files)
    recall = calculate_recall(result_files, patch_files)

    return calculate_f1(precision, recall)
