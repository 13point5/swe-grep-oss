import logging
import verifiers as vf
from datasets import load_dataset

import tools
import rewards
from prompts.system_prompt import SYSTEM_PROMPT
from utils.repo_manager import clone_repo, delete_repo
from utils.parse_file_list_xml import parse_file_list_xml


logger = logging.getLogger("swe-grep-oss")


class SWEGrepEnv(vf.StatefulToolEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Only add bash tool - file list is returned via XML in the response
        self.add_tool(tools.bash, args_to_skip=["cwd"])

    async def setup_state(self, state: vf.State) -> vf.State:
        """Clone the repository into a rollout-specific directory."""
        info = state["info"]
        repo_path = clone_repo(
            repo_name=info["repo"],
            commit_id=info["base_commit"],
            instance_id=info["instance_id"],
        )
        state["_repo_path"] = str(repo_path)
        logger.info(
            "Cloned %s for %s into %s (clone root: %s)",
            info["repo"],
            info["instance_id"],
            repo_path,
            repo_path.parent,
        )
        return state

    @vf.cleanup
    async def cleanup_repo(self, state: vf.State):
        """Delete the rollout-specific cloned repository."""
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
            repo_path = state.get("_repo_path")
            if not repo_path:
                raise RuntimeError(
                    "Repository path missing from state. setup_state must run before tool use."
                )
            updated_tool_args = dict(tool_args)
            updated_tool_args["cwd"] = repo_path
            return updated_tool_args

        return tool_args


DATASET_CONFIGS = {
    "swe-bench-lite": {
        "path": "princeton-nlp/SWE-bench_Lite",
        "split": "test",
    },
    "swe-gym": {
        "path": "SWE-Gym/SWE-Gym",
        "split": "train",
    },
    "swe-gym-lite": {
        "path": "SWE-Gym/SWE-Gym-Lite",
        "split": "train",
    },
    "swe-gym-raw": {
        "path": "SWE-Gym/SWE-Gym-Raw",
        "split": "train",
    },
}


def load_environment(dataset_name: str = "swe-bench-lite", **kwargs):
    """Load and configure the environment.

    Args:
        dataset_name: Which dataset to use. Options:
            - "swe-bench-lite": SWE-bench Lite (300 instances)
            - "swe-gym": SWE-Gym (2,438 instances)
            - "swe-gym-lite": SWE-Gym Lite (230 instances)
            - "swe-gym-raw": SWE-Gym Raw (64,689 instances)
        **kwargs: Additional arguments passed to SWEGrepEnv
    """
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Available: {list(DATASET_CONFIGS.keys())}"
        )

    config = DATASET_CONFIGS[dataset_name]
    dataset = load_dataset(config["path"], split=config["split"])
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

    # Define rubric
    rubric = vf.Rubric(
        funcs=[
            rewards.file_list_check,
            rewards.f1_reward,
            rewards.precision_reward,
            rewards.recall_reward,
        ],
        weights=[1.0, 1.0, 0.0, 0.0],
    )

    # Load environment
    return SWEGrepEnv(
        dataset=dataset,
        system_prompt=SYSTEM_PROMPT,
        rubric=rubric,
        max_turns=10,
        **kwargs,  # Pass through additional arguments
    )
