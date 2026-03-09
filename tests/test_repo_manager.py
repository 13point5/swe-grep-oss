import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import subprocess

from utils.repo_manager import clone_repo, delete_repo


class RepoManagerTests(unittest.TestCase):
    def test_clone_repo_clones_into_rollout_specific_directory(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            collision_path = tmp_path / "owner_repo_instance-1_deadbeef"
            collision_path.mkdir()
            commands = []

            def fake_run(command, **kwargs):
                commands.append((command, kwargs))
                return None

            with patch(
                "utils.repo_manager.secrets.token_hex",
                side_effect=["deadbeef", "cafebabe"],
            ), patch("utils.repo_manager.subprocess.run", side_effect=fake_run):
                repo_path = clone_repo(
                    repo_name="owner/repo",
                    commit_id="abc123",
                    instance_id="instance-1",
                    output_dir=tmp_path,
                )

            expected_path = tmp_path.resolve() / "owner_repo_instance-1_cafebabe"
            self.assertEqual(repo_path, expected_path)
            self.assertEqual(
                commands,
                [
                    (
                        [
                            "git",
                            "clone",
                            "--quiet",
                            "--revision",
                            "abc123",
                            "--depth",
                            "1",
                            "https://github.com/owner/repo.git",
                            str(expected_path),
                        ],
                        {"check": True, "capture_output": True, "text": True},
                    ),
                ],
            )

    def test_clone_repo_falls_back_when_git_clone_revision_is_unsupported(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            commands = []

            def fake_run(command, **kwargs):
                commands.append((command, kwargs))
                if command[:3] == ["git", "clone", "--quiet"]:
                    raise subprocess.CalledProcessError(
                        returncode=129,
                        cmd=command,
                        stderr="error: unknown option `revision'\n",
                    )
                return None

            with patch(
                "utils.repo_manager.secrets.token_hex",
                return_value="cafebabe",
            ), patch("utils.repo_manager.subprocess.run", side_effect=fake_run):
                repo_path = clone_repo(
                    repo_name="owner/repo",
                    commit_id="abc123",
                    instance_id="instance-1",
                    output_dir=tmp_path,
                )

            expected_path = tmp_path.resolve() / "owner_repo_instance-1_cafebabe"
            self.assertEqual(repo_path, expected_path)
            self.assertEqual(
                commands,
                [
                    (
                        [
                            "git",
                            "clone",
                            "--quiet",
                            "--revision",
                            "abc123",
                            "--depth",
                            "1",
                            "https://github.com/owner/repo.git",
                            str(expected_path),
                        ],
                        {"check": True, "capture_output": True, "text": True},
                    ),
                    (
                        ["git", "init", "--quiet"],
                        {
                            "check": True,
                            "capture_output": True,
                            "text": True,
                            "cwd": expected_path,
                        },
                    ),
                    (
                        ["git", "remote", "add", "origin", "https://github.com/owner/repo.git"],
                        {
                            "check": True,
                            "capture_output": True,
                            "text": True,
                            "cwd": expected_path,
                        },
                    ),
                    (
                        ["git", "fetch", "--depth", "1", "origin", "abc123"],
                        {
                            "check": True,
                            "capture_output": True,
                            "text": True,
                            "cwd": expected_path,
                        },
                    ),
                    (
                        [
                            "git",
                            "-c",
                            "advice.detachedHead=false",
                            "checkout",
                            "--quiet",
                            "FETCH_HEAD",
                        ],
                        {
                            "check": True,
                            "capture_output": True,
                            "text": True,
                            "cwd": expected_path,
                        },
                    ),
                ],
            )

    def test_delete_repo_only_removes_requested_directory(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            first_rollout = tmp_path / "owner_repo_instance-1_deadbeef"
            second_rollout = tmp_path / "owner_repo_instance-1_cafebabe"
            first_rollout.mkdir()
            second_rollout.mkdir()

            deleted = delete_repo(first_rollout)

            self.assertTrue(deleted)
            self.assertFalse(first_rollout.exists())
            self.assertTrue(second_rollout.exists())


if __name__ == "__main__":
    unittest.main()
