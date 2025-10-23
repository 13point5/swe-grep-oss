# swe-grep-oss

### Overview

- **Environment ID**: `swe-grep-oss`
- **Short description**: Environment for evaluating and developing models like [SWE-grep](https://cognition.ai/blog/swe-grep)

### Datasets

- **Primary dataset(s)**: [SWE-Bench Lite](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite)

### Task

- **Type**: <single-turn | multi-turn | tool use>
- **Parser**: <e.g., ThinkParser, XMLParser, custom>
- **Rubric overview**: <briefly list reward functions and key metrics>

### Quickstart

Clone a sample SWE-Bench Lite repository:

```bash
uv run scripts/clone_repos.py --max-repos 1 --max-instances 1
```

Run an evaluation with your model of choice:

```bash
uv run vf-eval swe-grep-oss \
  --api-base-url https://generativelanguage.googleapis.com/v1beta/openai/ \
  --header 'Content-Type: application/json' \
  --api-key-var GEMINI_API_KEY \
  --model "gemini-2.5-flash" \
  --num-examples 1 \
  --rollouts-per-example 1
```
