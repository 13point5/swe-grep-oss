# XML File List Refactor

This document describes the migration from the `result` tool approach to an XML-based `<file_list>` format for returning file paths in the SWE-Grep evaluation system.

## Motivation

The previous approach required the model to call a dedicated `result` tool to return the list of relevant files. This has been replaced with a simpler XML format embedded directly in the model's text response.

**Before (result tool):**

```
Model calls: result(file_paths=["path/to/file1.py", "path/to/file2.py"])
```

**After (XML format):**

```
<file_list>
path/to/file1.py
path/to/file2.py
</file_list>
```

## Benefits

1. **Simpler output format** - No need for a dedicated tool; the model just outputs structured text
2. **Reduced tool overhead** - One less tool for the model to learn and call correctly
3. **More natural integration** - The file list appears as part of the model's response text
4. **Easier parsing** - Simple regex-based extraction vs. tool call argument parsing

## New File Structure

### Rewards

| File                         | Function           | Description                                |
| ---------------------------- | ------------------ | ------------------------------------------ |
| `rewards/file_list_check.py` | `file_list_check`  | Checks if a valid file list XML is present |
| `rewards/precision.py`       | `precision_reward` | File-level precision score                 |
| `rewards/recall.py`          | `recall_reward`    | File-level recall score                    |
| `rewards/f1.py`              | `f1_reward`        | File-level F1 score                        |

### Utilities

| File                           | Description                                         |
| ------------------------------ | --------------------------------------------------- |
| `utils/parse_file_list_xml.py` | Parser for extracting file paths from XML tags      |
| `utils/metrics.py`             | Functions for calculating precision, recall, and F1 |

## XML Format Specification

```xml
<file_list>
path/to/file1.py
path/to/file2.py
path/to/file3.py
</file_list>
```

- Each file path on its own line
- No other text inside the tags
- Whitespace around paths is trimmed
- The parser uses the **last** occurrence if multiple `<file_list>` blocks exist

## Rubric Configuration

The rubric in `swe_grep_oss.py` uses these rewards:

```python
rubric = vf.Rubric(
    funcs=[
        rewards.file_list_check,
        rewards.f1_reward,
        rewards.precision_reward,
        rewards.recall_reward,
    ],
    weights=[2.0, 1.0, 1.0, 1.0],
)
```
