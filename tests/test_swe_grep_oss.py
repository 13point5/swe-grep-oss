import os
import unittest
from unittest.mock import patch, sentinel

from swe_grep_oss import (
    DEFAULT_SANDBOX_CPU_CORES,
    DEFAULT_SANDBOX_DISK_SIZE_GB,
    DEFAULT_SANDBOX_DOCKER_IMAGE,
    DEFAULT_SANDBOX_MEMORY_GB,
    DEFAULT_SANDBOX_NAME,
    DEFAULT_SANDBOX_TIMEOUT_MINUTES,
    ENVIRONMENT_BACKEND_ENV_VAR,
    resolve_environment_backend,
    load_environment,
    load_sandbox_environment,
    _build_sandbox_setup_command,
)


class EnvironmentSelectionTests(unittest.TestCase):
    def test_resolve_environment_backend_defaults_to_local(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolve_environment_backend(), "local")

    def test_resolve_environment_backend_reads_env_var(self):
        with patch.dict(
            os.environ, {ENVIRONMENT_BACKEND_ENV_VAR: "sandbox"}, clear=True
        ):
            self.assertEqual(resolve_environment_backend(), "sandbox")

    def test_resolve_environment_backend_rejects_unknown_backend(self):
        with self.assertRaises(ValueError):
            resolve_environment_backend("nope")

    def test_load_environment_dispatches_to_local_backend(self):
        with patch(
            "swe_grep_oss.load_local_environment", return_value=sentinel.local_env
        ) as load_local, patch("swe_grep_oss.load_sandbox_environment") as load_sandbox:
            result = load_environment(dataset_name="swe-gym")

        self.assertIs(result, sentinel.local_env)
        load_local.assert_called_once_with(dataset_name="swe-gym")
        load_sandbox.assert_not_called()

    def test_load_environment_dispatches_to_sandbox_backend(self):
        with patch.dict(
            os.environ, {ENVIRONMENT_BACKEND_ENV_VAR: "sandbox"}, clear=True
        ), patch(
            "swe_grep_oss.load_local_environment"
        ) as load_local, patch(
            "swe_grep_oss.load_sandbox_environment",
            return_value=sentinel.sandbox_env,
        ) as load_sandbox:
            result = load_environment(dataset_name="swe-gym-raw")

        self.assertIs(result, sentinel.sandbox_env)
        load_sandbox.assert_called_once_with(dataset_name="swe-gym-raw")
        load_local.assert_not_called()


class SandboxEnvironmentTests(unittest.TestCase):
    def test_build_sandbox_setup_command_installs_shell_tools_and_fetches_commit(self):
        command = _build_sandbox_setup_command(
            repo_name="owner/repo",
            commit_id="abc123",
            repo_path="/workspace/repo",
        )

        self.assertIn(
            "apt-get install -y --no-install-recommends ca-certificates git jq ripgrep",
            command,
        )
        self.assertIn("git init --quiet /workspace/repo", command)
        self.assertIn("git -C /workspace/repo fetch --quiet --depth 1 origin abc123", command)
        self.assertIn(
            "git -C /workspace/repo -c advice.detachedHead=false checkout --quiet FETCH_HEAD",
            command,
        )

    def test_load_sandbox_environment_uses_minimal_default_specs(self):
        with patch(
            "swe_grep_oss._build_environment_kwargs",
            return_value={"dataset": "dataset", "system_prompt": "prompt", "rubric": "rubric"},
        ), patch(
            "swe_grep_oss.SWEGrepSandboxEnv", return_value=sentinel.sandbox_env
        ) as sandbox_env_cls:
            result = load_sandbox_environment(dataset_name="swe-gym")

        self.assertIs(result, sentinel.sandbox_env)
        sandbox_env_cls.assert_called_once_with(
            sandbox_name=DEFAULT_SANDBOX_NAME,
            docker_image=DEFAULT_SANDBOX_DOCKER_IMAGE,
            cpu_cores=DEFAULT_SANDBOX_CPU_CORES,
            memory_gb=DEFAULT_SANDBOX_MEMORY_GB,
            disk_size_gb=DEFAULT_SANDBOX_DISK_SIZE_GB,
            timeout_minutes=DEFAULT_SANDBOX_TIMEOUT_MINUTES,
            dataset="dataset",
            system_prompt="prompt",
            rubric="rubric",
        )


if __name__ == "__main__":
    unittest.main()
