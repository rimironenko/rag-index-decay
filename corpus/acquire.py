"""Clone the GitLab Handbook repo and check out the T0 commit.

Corpus repo: gitlab-com/content-sites/handbook. Content lives under content/handbook/.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import git

REPO_URL = "https://gitlab.com/gitlab-com/content-sites/handbook.git"
DEFAULT_BRANCH = "main"

REPO_ROOT = Path(__file__).parent.parent
CLONE_DIR = REPO_ROOT / "corpus" / "_cache" / "handbook_repo"
WORKTREE_DIR = REPO_ROOT / "corpus" / "_cache" / "worktree_t0"

# ~13 months before 2026-07-10 when the experiment has been started
T0_CUTOFF = "2025-06-01T00:00:00+00:00"


@dataclass
class T0Commit:
    sha: str
    commit_date: str  # ISO 8601


def clone_or_update(clone_dir: Path = CLONE_DIR, repo_url: str = REPO_URL) -> git.Repo:
    """Idempotent: fetch if already cloned with a matching remote, else clone fresh."""
    if clone_dir.exists() and (clone_dir / ".git").exists():
        repo = git.Repo(clone_dir)
        existing_url = repo.remotes.origin.url
        if existing_url != repo_url:
            raise RuntimeError(
                f"{clone_dir} already exists with a different origin "
                f"({existing_url!r} != {repo_url!r}) - refusing to reuse it."
            )
        repo.remotes.origin.fetch()
        return repo

    clone_dir.parent.mkdir(parents=True, exist_ok=True)
    return git.Repo.clone_from(repo_url, clone_dir, branch=DEFAULT_BRANCH)


def resolve_t0_commit(repo: git.Repo, since_iso: str = T0_CUTOFF) -> T0Commit:
    """First commit on/after `since_iso` on the default branch, chronological order."""
    since_dt = datetime.fromisoformat(since_iso)

    ref = f"origin/{DEFAULT_BRANCH}"
    commits = list(repo.iter_commits(ref, reverse=True))
    for commit in commits:
        committed_dt = datetime.fromtimestamp(commit.committed_date, tz=timezone.utc)
        if committed_dt >= since_dt:
            return T0Commit(sha=commit.hexsha, commit_date=committed_dt.isoformat())

    raise RuntimeError(f"No commit found on/after {since_iso} on {ref}")


def checkout_t0_worktree(
    repo: git.Repo, sha: str, worktree_dir: Path = WORKTREE_DIR
) -> Path:
    """Check out `sha` into a separate git worktree so `main` stays free for churn replay."""
    if worktree_dir.exists():
        # Idempotent: reuse if it's already checked out at the right commit.
        existing = git.Repo(worktree_dir)
        if existing.head.commit.hexsha == sha:
            return worktree_dir
        raise RuntimeError(
            f"{worktree_dir} exists but is checked out at "
            f"{existing.head.commit.hexsha}, not {sha} - remove it to re-checkout."
        )

    worktree_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(worktree_dir), sha],
        cwd=repo.working_dir,
        check=True,
        capture_output=True,
        text=True,
    )
    return worktree_dir


def acquire(since_iso: str = T0_CUTOFF) -> dict:
    """Orchestrates clone -> T0 resolution -> worktree checkout. Idempotent."""
    repo = clone_or_update()
    t0 = resolve_t0_commit(repo, since_iso)
    worktree_path = checkout_t0_worktree(repo, t0.sha)
    return {
        "sha": t0.sha,
        "commit_date": t0.commit_date,
        "worktree_path": str(worktree_path),
        "content_dir": str(worktree_path / "content" / "handbook"),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(acquire(), indent=2))
