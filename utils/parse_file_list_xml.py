"""Parser for file_list XML format."""

import re
from typing import Optional

import verifiers as vf


def parse_file_list_xml(text: str) -> Optional[list[str]]:
    """
    Parse file list from XML format.

    Expected format:
    <file_list>
    path/to/file1.py
    path/to/file2.py
    </file_list>

    Args:
        text: Text containing the file_list XML tags

    Returns:
        List of file paths, or None if no valid file_list found
    """
    # Find the last occurrence of file_list tags
    pattern = r"<file_list>\s*(.*?)\s*</file_list>"
    matches = list(re.finditer(pattern, text, re.DOTALL))

    if not matches:
        return None

    # Use the last match (most recent file list)
    content = matches[-1].group(1).strip()

    if not content:
        return None

    # Split by newlines and filter empty lines
    file_paths = [
        line.strip() for line in content.split("\n") if line.strip()
    ]

    return file_paths if file_paths else None


def get_file_list_from_messages(
    messages: vf.Messages,
) -> tuple[list[str] | None, bool]:
    """
    Get file list from completion messages.

    Args:
        messages: The completion messages

    Returns:
        tuple: (file_paths, success) where success is True if valid file_list found
    """
    # Search through assistant messages for file_list
    for message in reversed(messages):
        if message.get("role") == "assistant":
            content = message.get("content", "")
            if isinstance(content, str) and "<file_list>" in content:
                file_paths = parse_file_list_xml(content)
                if file_paths is not None:
                    return file_paths, True

    return None, False
