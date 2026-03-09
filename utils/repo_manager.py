"""Utility functions for managing SWE-bench repository clones."""

import secrets
import shutil
import subprocess
import tempfile
from pathlib import Path

REVISION_UNSUPPORTED_ERRORS = (
    "unknown option `revision'",
    "unknown option 'revision'",
)


def _run_git(command: list[str], cwd: Path | None = None) -> None:
    kwargs = {
        "check": True,
        "capture_output": True,
        "text": True,
    }
    if cwd is not None:
        kwargs["cwd"] = cwd
    subprocess.run(command, **kwargs)


def _clone_repo_with_revision(repo_url: str, commit_id: str, instance_path: Path) -> None:
    _run_git(
        [
            "git",
            "clone",
            "--quiet",
            "--revision",
            commit_id,
            "--depth",
            "1",
            repo_url,
            str(instance_path),
        ]
    )


def _clone_repo_with_fetch(repo_url: str, commit_id: str, instance_path: Path) -> None:
    instance_path.mkdir(parents=True, exist_ok=False)
    _run_git(["git", "init", "--quiet"], cwd=instance_path)
    _run_git(["git", "remote", "add", "origin", repo_url], cwd=instance_path)
    _run_git(["git", "fetch", "--depth", "1", "origin", commit_id], cwd=instance_path)
    _run_git(
        ["git", "-c", "advice.detachedHead=false", "checkout", "--quiet", "FETCH_HEAD"],
        cwd=instance_path,
    )


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

    repo_url = f"https://github.com/{repo_name}.git"
    try:
        # Clone only the requested commit to keep checkout time and disk usage low.
        _clone_repo_with_revision(repo_url, commit_id, instance_path)
        return instance_path

    except subprocess.CalledProcessError as e:
        if any(message in (e.stderr or "") for message in REVISION_UNSUPPORTED_ERRORS):
            if instance_path.exists():
                shutil.rmtree(instance_path, ignore_errors=True)
            try:
                # Older Git builds may not support `git clone --revision`.
                _clone_repo_with_fetch(repo_url, commit_id, instance_path)
                return instance_path
            except subprocess.CalledProcessError as fallback_error:
                if instance_path.exists():
                    shutil.rmtree(instance_path, ignore_errors=True)
                raise RuntimeError(
                    "Failed to clone "
                    f"{repo_name} at {commit_id} with the compatibility fallback: "
                    f"{fallback_error.stderr}"
                )

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
