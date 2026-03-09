"""Utility functions for managing SWE-bench repository clones."""

import secrets
import shutil
import subprocess
import tempfile
from pathlib import Path


def clone_repo(
    repo_name: str,
    commit_id: str,
    instance_id: str,
    output_dir: Path | str | None = None,
) -> Path:
    """
    Clone a repository at a specific commit into a rollout-specific directory.

    Args:
        repo_name: Repository name in format 'owner/repo'
        commit_id: Commit hash to checkout
        instance_id: Instance ID for directory naming
        output_dir: Base output directory

    Returns:
        Path to the rollout-specific clone directory.
    """
    # Resolve the shared temp-root that holds all rollout-specific clones.
    clone_root = Path(
        output_dir or (Path(tempfile.gettempdir()) / "swe-grep-oss-repos")
    ).expanduser().resolve()
    clone_root.mkdir(parents=True, exist_ok=True)

    # Allocate a unique directory for this rollout so concurrent runs never share a checkout.
    prefix = f"{repo_name.replace('/', '_')}_{instance_id}"
    for _ in range(10):
        instance_path = clone_root / f"{prefix}_{secrets.token_hex(4)}"
        if not instance_path.exists():
            break
    else:
        raise RuntimeError(
            f"Failed to allocate a unique clone directory for {repo_name} / {instance_id}"
        )

    try:
        # Clone only the requested commit to keep checkout time and disk usage low.
        subprocess.run(
            [
                "git",
                "clone",
                "--quiet",
                "--revision",
                commit_id,
                "--depth",
                "1",
                f"https://github.com/{repo_name}.git",
                str(instance_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return instance_path

    except subprocess.CalledProcessError as e:
        # Remove any partial checkout so failed rollouts do not leave broken directories behind.
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
