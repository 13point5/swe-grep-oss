"""Utility functions for calculating result metrics."""

import verifiers as vf

from utils.get_result_tool_call import get_file_list_result
from utils.parse_patch import parse_patch


def get_file_sets(
    completion: vf.Messages, patch: str
) -> tuple[set[str], set[str]] | tuple[None, None]:
    """
    Extract file sets from completion messages and patch.

    Args:
        completion: The completion messages from the agent
        patch: The ground truth patch string

    Returns:
        Tuple of (result_files, patch_files) or (None, None) if extraction fails
    """
    file_paths, success = get_file_list_result(completion)

    # If no successful file list extraction, return None
    if not success or not file_paths:
        return None, None

    # Parse the patch to get file paths
    patch_info = parse_patch(patch)
    patch_files = set(patch_info.keys())

    # Get file paths from the XML file list
    result_files = set(file_paths)

    if not result_files:
        return None, None

    return result_files, patch_files


def calculate_precision(result_files: set[str], patch_files: set[str]) -> float:
    """
    Calculate precision: proportion of predicted files that are correct.

    Precision = |result_files ∩ patch_files| / |result_files|

    Args:
        result_files: Files identified by the agent
        patch_files: Files in the ground truth patch

    Returns:
        Precision score between 0.0 and 1.0
    """
    if not result_files:
        return 0.0

    matching_files = result_files.intersection(patch_files)
    return len(matching_files) / len(result_files)


def calculate_recall(result_files: set[str], patch_files: set[str]) -> float:
    """
    Calculate recall: proportion of ground truth files that were identified.

    Recall = |result_files ∩ patch_files| / |patch_files|

    Args:
        result_files: Files identified by the agent
        patch_files: Files in the ground truth patch

    Returns:
        Recall score between 0.0 and 1.0
    """
    if not patch_files:
        return 0.0

    matching_files = result_files.intersection(patch_files)
    return len(matching_files) / len(patch_files)


def calculate_f1(precision: float, recall: float) -> float:
    """
    Calculate F1 score: harmonic mean of precision and recall.

    F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        precision: Precision score
        recall: Recall score

    Returns:
        F1 score between 0.0 and 1.0
    """
    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)
