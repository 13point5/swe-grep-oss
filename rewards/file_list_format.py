"""Format reward for file_list XML format."""

import re

import verifiers as vf


def file_list_format_reward(
    prompt, completion: vf.Messages, answer, state, task, info
) -> float:
    """
    Check if the completion follows the expected XML format for file list.

    Expected format:
    <file_list>
    path/to/file1.py
    path/to/file2.py
    </file_list>

    Scoring:
    - 0.0: No file_list tags found
    - 0.25: Opening tag found but no closing tag
    - 0.5: Both tags found but empty or malformed content
    - 0.75: Valid file_list with some content
    - 1.0: Valid file_list with properly formatted file paths
    """
    # Get assistant messages
    assistant_messages = [
        msg for msg in completion if msg.get("role") == "assistant"
    ]

    if not assistant_messages:
        return 0.0

    # Check the last assistant message for file_list
    for msg in reversed(assistant_messages):
        content = msg.get("content", "")
        if not isinstance(content, str):
            continue

        # Check for opening tag
        has_opening = "<file_list>" in content
        has_closing = "</file_list>" in content

        if not has_opening:
            continue

        if not has_closing:
            return 0.25

        # Extract content between tags
        pattern = r"<file_list>\s*(.*?)\s*</file_list>"
        match = re.search(pattern, content, re.DOTALL)

        if not match:
            return 0.5

        inner_content = match.group(1).strip()

        if not inner_content:
            return 0.5

        # Check if content looks like file paths
        lines = [line.strip() for line in inner_content.split("\n") if line.strip()]

        if not lines:
            return 0.5

        # Check if lines look like valid file paths
        valid_path_count = 0
        for line in lines:
            # Simple heuristic: file paths typically contain / or . and no spaces
            # Allow for various path formats
            if (
                ("/" in line or "." in line)
                and not line.startswith("<")
                and not line.endswith(">")
            ):
                valid_path_count += 1

        if valid_path_count == 0:
            return 0.5

        # All lines look like valid file paths
        if valid_path_count == len(lines):
            return 1.0

        # Some valid paths found
        return 0.75

    return 0.0
