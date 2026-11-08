from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import yaml

from .models import StoryConfig


def load_story_config(path: str | Path) -> StoryConfig:
    story_path = Path(path)
    data = yaml.safe_load(story_path.read_text(encoding="utf-8"))
    cfg = StoryConfig.model_validate(data)
    return resolve_story_paths(cfg, base_dir=story_path.parent)


def _resolve_path(path: Path | None, *, base_dir: Path) -> Path | None:
    if path is None or path.is_absolute():
        return path
    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists() or str(path).startswith(("samples/", "runs/", "local_fixtures/")):
        return cwd_candidate
    return base_dir / path


def resolve_story_paths(cfg: StoryConfig, *, base_dir: Path) -> StoryConfig:
    paths = cfg.paths.model_copy(
        update={
            "source_dir": _resolve_path(cfg.paths.source_dir, base_dir=base_dir),
            "glossary_path": _resolve_path(cfg.paths.glossary_path, base_dir=base_dir),
            "prompt_path": _resolve_path(cfg.paths.prompt_path, base_dir=base_dir),
            "expected_dir": _resolve_path(cfg.paths.expected_dir, base_dir=base_dir),
            "baseline_dir": _resolve_path(cfg.paths.baseline_dir, base_dir=base_dir),
            "runs_dir": _resolve_path(cfg.paths.runs_dir, base_dir=base_dir),
        }
    )
    return cfg.model_copy(update={"paths": paths})


def chapter_id(number: int) -> str:
    return f"{number:04d}"


def read_chapter(source_dir: Path, chapter: str) -> str:
    return (source_dir / f"{chapter}.txt").read_text(encoding="utf-8")


def make_run_id(slug: str) -> str:
    return datetime.now().strftime(f"%Y%m%d_%H%M%S_{slug}")


def prepare_run_dir(runs_dir: Path, run_id: str, *, overwrite: bool = False) -> Path:
    """Create one run directory without following user-controlled paths or links."""

    if not run_id or run_id in {".", ".."}:
        raise ValueError("run_id must be a non-empty directory name.")
    run_id_path = Path(run_id)
    if run_id_path.is_absolute() or run_id_path.name != run_id:
        raise ValueError("run_id must be a single directory name, not a path.")

    root = Path(runs_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    raw_run_dir = root / run_id
    if raw_run_dir.is_symlink():
        raise ValueError(f"Refusing to overwrite selected run symlink: {raw_run_dir}")
    run_dir = raw_run_dir.resolve()
    if run_dir.parent != root:
        raise ValueError("Resolved run directory must be directly under runs_dir.")

    if run_dir.exists():
        if not run_dir.is_dir():
            raise ValueError(f"Refusing to overwrite non-directory run path: {run_dir}")
        if overwrite:
            shutil.rmtree(run_dir)
        elif any(run_dir.iterdir()):
            raise FileExistsError(
                f"Run directory already exists: {run_dir}. Pass --overwrite to replace it."
            )

    run_dir.mkdir(parents=False, exist_ok=True)
    return run_dir
