import logging
import verifiers as vf
from datasets import load_dataset

import tools
from prompts.system_prompt import SYSTEM_PROMPT
from utils.get_instance import get_instance_path


logger = logging.getLogger("swe-grep-oss")


class SWEGrepEnv(vf.StatefulToolEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.add_tool(tools.bash, args_to_skip=["cwd"])
        self.add_tool(tools.result)

    # async def is_completed(
    #     self, messages: vf.types.Messages, state: vf.types.State, **kwargs
    # ) -> bool:
    #     """When overriding, call self.max_turns_reached(state) to check if turn limit reached."""
    #     max_turns_reached = await self.max_turns_reached(state)
    #     prompt_too_long = await self.prompt_too_long(state)
    #     if max_turns_reached or prompt_too_long:
    #         return True

    #     print("=" * 80)
    #     print("Messages")
    #     for message in messages:
    #         print(message)
    #     print("=" * 80)

    #     return False

    def update_tool_args(
        self,
        tool_name: str,
        tool_args: dict,
        messages: vf.types.Messages,
        state: vf.types.State,
        **kwargs,
    ) -> dict:
        if tool_name == "bash":
            repo_path = get_instance_path(
                {
                    "repo": state["info"]["repo"],
                    "instance_id": state["info"]["instance_id"],
                }
            )
            updated_tool_args = dict(tool_args)
            updated_tool_args["cwd"] = repo_path
            return updated_tool_args

        return tool_args


def load_environment(**kwargs):
    """Load and configure the environment."""
    # 1. Load dataset
    dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    dataset = dataset.map(
        lambda row: {
            # we can add metadata related to the dataset row here
            "info": {
                "repo": row["repo"],
                "instance_id": row["instance_id"],
            },
            "prompt": [{"role": "user", "content": row["problem_statement"]}],
            "answer": row["patch"],
        }
    )

    # 3. Define reward functions -- can automatically reference:
    # - parser, prompt, completion, answer, state , task, info
    # TODO: Implement the actual reward function
    def dummy_rubric_func(prompt, completion, answer, state, task, info):
        # Find the result tool call
        result_tool_call_id = None
        for message in completion:
            if message.get("role") == "assistant" and message.get("tool_calls"):
                for tool_call in message["tool_calls"]:
                    # Check if it's a result tool call
                    if (
                        hasattr(tool_call, "function")
                        and tool_call.function.name == "result"
                    ):
                        result_tool_call_id = tool_call.id
                        break
                if result_tool_call_id:
                    break

        # Check if there's a corresponding tool response with "Success"
        has_success = False
        if result_tool_call_id:
            for message in completion:
                if (
                    message.get("role") == "tool"
                    and message.get("tool_call_id") == result_tool_call_id
                ):
                    content = message.get("content", "")
                    if content == "Success":
                        has_success = True
                    break

        return 1.0 if has_success else 0.0

    # 4. Create rubric
    rubric = vf.Rubric(
        # funcs=[correct_answer, parser.get_format_reward_func()],
        # weights=[1.0, 0.2],
        funcs=[dummy_rubric_func],
        weights=[1.0],
    )

    # 5. Return configured environment
    return SWEGrepEnv(
        dataset=dataset,
        system_prompt=SYSTEM_PROMPT,
        rubric=rubric,
        max_turns=10,
        **kwargs,  # Pass through additional arguments
    )
