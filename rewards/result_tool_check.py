import verifiers as vf

from utils.get_result_tool_call import get_file_list_result


def result_tool_check(
    prompt, completion: vf.Messages, answer, state, task, info
) -> float:
    """
    Check if the file list XML is present and valid.
    """

    _, success = get_file_list_result(completion)
    return 1.0 if success else 0.0
