import logging
import os
import shlex
import textwrap
from typing import Any

import verifiers as vf
from datasets import load_dataset

import tools
import rewards
from prompts.system_prompt import SYSTEM_PROMPT
from utils.repo_manager import clone_repo, delete_repo
from utils.parse_file_list_xml import parse_file_list_xml


logger = logging.getLogger("swe-grep-oss")

ENVIRONMENT_BACKEND_ENV_VAR = "SWE_GREP_ENV_BACKEND"
LOCAL_ENVIRONMENT_BACKEND = "local"
SANDBOX_ENVIRONMENT_BACKEND = "sandbox"

DEFAULT_SANDBOX_NAME = "swe-grep-oss"
DEFAULT_SANDBOX_DOCKER_IMAGE = "python:3.11-slim"
DEFAULT_SANDBOX_CPU_CORES = 1
DEFAULT_SANDBOX_MEMORY_GB = 2
DEFAULT_SANDBOX_DISK_SIZE_GB = 5
DEFAULT_SANDBOX_TIMEOUT_MINUTES = 60
DEFAULT_SANDBOX_SETUP_TIMEOUT_SECONDS = 300
DEFAULT_SANDBOX_REPO_PATH = "/workspace/repo"


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


class SWEGrepSandboxEnv(vf.SandboxEnv):
    def __init__(
        self,
        sandbox_name: str = DEFAULT_SANDBOX_NAME,
        docker_image: str = DEFAULT_SANDBOX_DOCKER_IMAGE,
        cpu_cores: int = DEFAULT_SANDBOX_CPU_CORES,
        memory_gb: int = DEFAULT_SANDBOX_MEMORY_GB,
        disk_size_gb: int = DEFAULT_SANDBOX_DISK_SIZE_GB,
        timeout_minutes: int = DEFAULT_SANDBOX_TIMEOUT_MINUTES,
        **kwargs,
    ):
        super().__init__(
            sandbox_name=sandbox_name,
            docker_image=docker_image,
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            disk_size_gb=disk_size_gb,
            timeout_minutes=timeout_minutes,
            **kwargs,
        )

    async def _run_setup_command(
        self,
        state: vf.State,
        command: str,
        timeout: int = DEFAULT_SANDBOX_SETUP_TIMEOUT_SECONDS,
    ) -> None:
        sandbox_state = state["sandbox_state"]
        sandbox_id = state["sandbox_id"]
        if not sandbox_state["ready"]:
            await self._wait_for_sandbox_ready(sandbox_state, sandbox_id)

        results = await self.sandbox_client.execute_command(
            sandbox_id,
            command,
            timeout=timeout,
        )
        if results.exit_code != 0:
            stdout = results.stdout.strip()
            stderr = (results.stderr or "").strip()
            output = stderr or stdout or f"Command failed with exit code {results.exit_code}"
            raise RuntimeError(f"Sandbox setup failed: {output}")

    async def setup_state(self, state: vf.State, **kwargs) -> vf.State:
        state = await super().setup_state(state, **kwargs)
        info = state["info"]
        state["working_dir"] = DEFAULT_SANDBOX_REPO_PATH
        try:
            await self._run_setup_command(
                state,
                _build_sandbox_setup_command(
                    repo_name=info["repo"],
                    commit_id=info["base_commit"],
                    repo_path=DEFAULT_SANDBOX_REPO_PATH,
                ),
            )
        except Exception:
            await self.destroy_sandbox(state)
            raise

        logger.info(
            "Prepared sandbox repo for %s at %s inside %s",
            info["instance_id"],
            info["base_commit"],
            DEFAULT_SANDBOX_REPO_PATH,
        )
        return state

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


def _build_dataset(dataset_name: str):
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Available: {list(DATASET_CONFIGS.keys())}"
        )

    config = DATASET_CONFIGS[dataset_name]
    dataset = load_dataset(config["path"], split=config["split"])
    return dataset.map(
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


def _build_rubric() -> vf.Rubric:
    return vf.Rubric(
        funcs=[
            rewards.file_list_check,
            rewards.f1_reward,
            rewards.precision_reward,
            rewards.recall_reward,
        ],
        weights=[1.0, 1.0, 0.0, 0.0],
    )


def _build_environment_kwargs(dataset_name: str) -> dict[str, Any]:
    return {
        "dataset": _build_dataset(dataset_name),
        "system_prompt": SYSTEM_PROMPT,
        "rubric": _build_rubric(),
        "max_turns": 10,
    }


def _build_sandbox_setup_command(
    repo_name: str,
    commit_id: str,
    repo_path: str = DEFAULT_SANDBOX_REPO_PATH,
) -> str:
    repo_url = f"https://github.com/{repo_name}.git"
    quoted_repo_path = shlex.quote(repo_path)
    return textwrap.dedent(
        f"""\
        set -euo pipefail

        if ! command -v git >/dev/null 2>&1 || ! command -v rg >/dev/null 2>&1 || ! command -v jq >/dev/null 2>&1; then
          export DEBIAN_FRONTEND=noninteractive
          apt-get update >/dev/null
          apt-get install -y --no-install-recommends ca-certificates git jq ripgrep >/dev/null
        fi

        rm -rf {quoted_repo_path}
        git init --quiet {quoted_repo_path}
        git -C {quoted_repo_path} remote add origin {shlex.quote(repo_url)}
        git -C {quoted_repo_path} fetch --quiet --depth 1 origin {shlex.quote(commit_id)}
        git -C {quoted_repo_path} -c advice.detachedHead=false checkout --quiet FETCH_HEAD
        """
    )


def resolve_environment_backend(environment_backend: str | None = None) -> str:
    backend = (
        environment_backend
        or os.getenv(ENVIRONMENT_BACKEND_ENV_VAR, LOCAL_ENVIRONMENT_BACKEND)
    ).strip().lower()
    if backend not in {LOCAL_ENVIRONMENT_BACKEND, SANDBOX_ENVIRONMENT_BACKEND}:
        raise ValueError(
            f"Unknown environment backend: {backend}. "
            f"Use one of: {LOCAL_ENVIRONMENT_BACKEND}, {SANDBOX_ENVIRONMENT_BACKEND}."
        )
    return backend


def load_local_environment(dataset_name: str = "swe-bench-lite", **kwargs):
    """Load the current direct-execution environment."""
    return SWEGrepEnv(
        **_build_environment_kwargs(dataset_name),
        **kwargs,
    )


def load_sandbox_environment(
    dataset_name: str = "swe-bench-lite",
    sandbox_name: str = DEFAULT_SANDBOX_NAME,
    docker_image: str = DEFAULT_SANDBOX_DOCKER_IMAGE,
    cpu_cores: int = DEFAULT_SANDBOX_CPU_CORES,
    memory_gb: int = DEFAULT_SANDBOX_MEMORY_GB,
    disk_size_gb: int = DEFAULT_SANDBOX_DISK_SIZE_GB,
    timeout_minutes: int = DEFAULT_SANDBOX_TIMEOUT_MINUTES,
    **kwargs,
):
    """Load a minimal sandbox-backed environment for bash-only tool use."""
    return SWEGrepSandboxEnv(
        sandbox_name=sandbox_name,
        docker_image=docker_image,
        cpu_cores=cpu_cores,
        memory_gb=memory_gb,
        disk_size_gb=disk_size_gb,
        timeout_minutes=timeout_minutes,
        **_build_environment_kwargs(dataset_name),
        **kwargs,
    )


def load_environment(
    dataset_name: str = "swe-bench-lite",
    environment_backend: str | None = None,
    **kwargs,
):
    """Load and configure the environment.

    Args:
        dataset_name: Which dataset to use. Options:
            - "swe-bench-lite": SWE-bench Lite (300 instances)
            - "swe-gym": SWE-Gym (2,438 instances)
            - "swe-gym-lite": SWE-Gym Lite (230 instances)
            - "swe-gym-raw": SWE-Gym Raw (64,689 instances)
        environment_backend: One of "local" or "sandbox". If unset, reads
            `SWE_GREP_ENV_BACKEND` and defaults to "local".
        **kwargs: Additional arguments passed through to the selected environment.
    """
    backend = resolve_environment_backend(environment_backend)
    if backend == SANDBOX_ENVIRONMENT_BACKEND:
        return load_sandbox_environment(dataset_name=dataset_name, **kwargs)
    return load_local_environment(dataset_name=dataset_name, **kwargs)
