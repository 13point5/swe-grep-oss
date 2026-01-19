import verifiers as vf

from utils.parse_file_list_xml import get_file_list_from_messages


def get_file_list_result(
    messages: vf.Messages,
) -> tuple[list[str] | None, bool]:
    """
    Get the file list from the messages using XML format.

    Returns:
        tuple: (file_paths, success) where success is True if valid file_list found
    """
    return get_file_list_from_messages(messages)


# Keep old function for backwards compatibility but mark as deprecated
def get_result_tool_call(
    messages: vf.Messages,
) -> tuple[list[str] | None, bool]:
    """
    DEPRECATED: Use get_file_list_result instead.
    Get the file list from the messages using XML format.

    Returns:
        tuple: (file_paths, success) where success is True if valid file_list found
    """
    return get_file_list_from_messages(messages)
