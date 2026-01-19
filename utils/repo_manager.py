"""Utility functions for managing SWE-bench repository clones."""

import shutil
import subprocess
from pathlib import Path


def clone_repo(
    repo_name: str,
    commit_id: str,
    instance_id: str,
    output_dir: Path = Path("./swebench_repos"),
) -> tuple[Path, bool]:
    """
    Clone a repository at a specific commit if it doesn't already exist.

    Args:
        repo_name: Repository name in format 'owner/repo'
        commit_id: Commit hash to checkout
        instance_id: Instance ID for directory naming
        output_dir: Base output directory

    Returns:
        Tuple of (instance_path, was_cloned) where was_cloned is True if we
        cloned it (vs it already existing)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create instance directory name: repo_instance-id
    instance_dir_name = f"{repo_name.replace('/', '_')}_{instance_id}"
    instance_path = output_dir / instance_dir_name

    # Skip if already exists
    if instance_path.exists():
        return instance_path, False

    try:
        # Clone the repository
        subprocess.run(
            [
                "git",
                "clone",
                "--quiet",
                f"https://github.com/{repo_name}.git",
                str(instance_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        # Checkout the specific commit
        subprocess.run(
            ["git", "-C", str(instance_path), "checkout", "--quiet", commit_id],
            check=True,
            capture_output=True,
            text=True,
        )

        return instance_path, True

    except subprocess.CalledProcessError as e:
        # Clean up partial clone if it exists
        if instance_path.exists():
            shutil.rmtree(instance_path, ignore_errors=True)
        raise RuntimeError(f"Failed to clone {repo_name} at {commit_id}: {e.stderr}")


def delete_repo(instance_path: Path) -> bool:
    """
    Delete a cloned repository.

    Args:
        instance_path: Path to the instance directory

    Returns:
        True if deleted, False if it didn't exist
    """
    instance_path = Path(instance_path)
    if instance_path.exists():
        shutil.rmtree(instance_path, ignore_errors=True)
        return True
    return False
