import verifiers as vf

from utils.parse_file_list_xml import get_file_list_from_messages


def file_list_check(prompt, completion: vf.Messages, answer, state, task, info) -> float:
    """
    Check if the file list XML is present and valid.
    """

    _, success = get_file_list_from_messages(completion)
    return 1.0 if success else 0.0
