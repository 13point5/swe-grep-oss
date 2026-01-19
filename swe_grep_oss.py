import logging
import verifiers as vf
from datasets import load_dataset

import tools
import rewards
from prompts.system_prompt import SYSTEM_PROMPT
from utils.get_instance import get_instance_path
from utils.repo_manager import clone_repo, delete_repo
from utils.parse_file_list_xml import parse_file_list_xml


logger = logging.getLogger("swe-grep-oss")


class SWEGrepEnv(vf.StatefulToolEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Only add bash tool - file list is returned via XML in the response
        self.add_tool(tools.bash, args_to_skip=["cwd"])

    async def setup_state(self, state: vf.State) -> vf.State:
        """Clone the repository if it doesn't exist."""
        info = state["info"]
        repo_path, was_cloned = clone_repo(
            repo_name=info["repo"],
            commit_id=info["base_commit"],
            instance_id=info["instance_id"],
        )
        # Track whether we cloned it so we know to clean up later
        state["_repo_path"] = str(repo_path)
        state["_repo_was_cloned"] = was_cloned
        if was_cloned:
            logger.info(f"Cloned {info['repo']} for {info['instance_id']}")
        return state

    @vf.cleanup
    async def cleanup_repo(self, state: vf.State):
        """Delete the repository if we cloned it during this run."""
        if state.get("_repo_was_cloned", False):
            repo_path = state.get("_repo_path")
            if repo_path:
                delete_repo(repo_path)
                logger.info(f"Deleted cloned repo: {repo_path}")

    @vf.stop
    async def file_list_returned(self, state: vf.State) -> bool:
        """Stop when the model returns a file_list XML."""
        if len(state["trajectory"]) == 0:
            return False
        last_message = state["trajectory"][-1]["completion"][-1]
        if last_message.get("role") != "assistant":
            return False
        content = last_message.get("content", "")
        if isinstance(content, str) and "<file_list>" in content:
            file_paths = parse_file_list_xml(content)
            return file_paths is not None
        return False

    def update_tool_args(
        self,
        tool_name: str,
        tool_args: dict,
        messages: vf.Messages,
        state: vf.State,
        **kwargs,
    ) -> dict:
        if tool_name == "bash":
            # Use the repo path from state if available, otherwise fall back to get_instance_path
            repo_path = state.get("_repo_path")
            if not repo_path:
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

    # Load dataset
    dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    dataset = dataset.map(
        lambda row: {
            # we can add metadata related to the dataset row here
            "info": {
                "repo": row["repo"],
                "instance_id": row["instance_id"],
                "base_commit": row["base_commit"],
            },
            "prompt": [{"role": "user", "content": row["problem_statement"]}],
            "answer": row["patch"],
        }
    )

    # Define rubric with format reward
    rubric = vf.Rubric(
        funcs=[
            rewards.result_tool_check,
            rewards.result_tool_f1,
            rewards.result_tool_precision,
            rewards.result_tool_recall,
            rewards.file_list_format_reward,
        ],
        weights=[2.0, 1.0, 1.0, 1.0, 1.0],
    )

    # Load environment
    return SWEGrepEnv(
        dataset=dataset,
        system_prompt=SYSTEM_PROMPT,
        rubric=rubric,
        max_turns=10,
        **kwargs,  # Pass through additional arguments
    )
