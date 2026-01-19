# swe-grep-oss

### Overview

- **Environment ID**: `swe-grep-oss`
- **Short description**: Environment for evaluating and developing models like [SWE-grep](https://cognition.ai/blog/swe-grep)

![result](./docs/result.png)

### Datasets

- **Primary dataset(s)**: [SWE-Bench Lite](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite)

### Task

- **Type**: <single-turn | multi-turn | tool use>
- **Parser**: <e.g., ThinkParser, XMLParser, custom>
- **Rubric overview**: <briefly list reward functions and key metrics>

### Quickstart

Run an evaluation with your model of choice (repos are cloned automatically and deleted after each rollout):

```bash
uv run vf-eval swe-grep-oss \
  --api-base-url https://api.openai.com/v1 \
  --api-key-var OPENAI_API_KEY \
  --model "gpt-4o-mini" \
  --num-examples 2 \
  --rollouts-per-example 1
```
