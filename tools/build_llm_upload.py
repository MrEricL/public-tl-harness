"""Build a deterministic, text-only source bundle for LLM upload."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable


ROOT_FILES = (".gitignore", "DEMO_SCRIPT.md", "README.md", "pyproject.toml")
CONTENT_DIRS = ("agentic_translation", "samples", "templates", "tests", "tools")
EXCLUDED_PARTS = {
    "__pycache__",
    ".pytest_cache",
    ".agentic_cache",
    ".venv",
    "build",
    "dist",
    "runs",
}
EXCLUDED_NAMES = {"LLM_UPLOAD_MEGA.txt"}
TEXT_SUFFIXES = {
    ".css",
    ".html",
    ".j2",
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
SEPARATOR = "=" * 80


def _candidate_is_safe(path: Path, project_root: Path) -> bool:
    """Ensure a candidate stays inside the project without symlinked parents."""

    try:
        relative = path.relative_to(project_root)
    except ValueError:
        return False

    current = project_root
    for part in relative.parts:
        if part in {"", "."}:
            continue
        if part == "..":
            if current == project_root:
                return False
            current = current.parent
            continue
        current /= part
        if current.is_symlink():
            return False

    try:
        resolved = path.resolve(strict=False)
    except (OSError, RuntimeError):
        return False
    try:
        resolved.relative_to(project_root)
    except ValueError:
        return False
    return True


def _validate_output_path(project_root: Path, output_path: Path) -> Path:
    """Validate an output path lexically below project_root and return it."""

    try:
        relative = output_path.relative_to(project_root)
    except ValueError as exc:
        raise ValueError("output_path must be within project_root") from exc

    current = project_root
    for part in relative.parts:
        if part in {"", "."}:
            continue
        if part == "..":
            if current == project_root:
                raise ValueError("output_path must be within project_root")
            current = current.parent
            continue
        current /= part
        if current.is_symlink():
            raise ValueError(f"output_path must not contain symlinked ancestors: {current}")
    return output_path


def _rebase_output_path(
    raw_project_root: Path, resolved_project_root: Path, output_path: str | Path
) -> Path:
    """Rebase an output path under the resolved root without following aliases."""

    candidate = Path(output_path).expanduser()
    if not candidate.is_absolute():
        return resolved_project_root / candidate

    for root_spelling in (raw_project_root, resolved_project_root):
        try:
            relative = candidate.relative_to(root_spelling)
        except ValueError:
            continue
        return resolved_project_root / relative

    raise ValueError("output_path must be within project_root")


def _relative_name(path: Path, project_root: Path) -> str:
    """Return a project-relative POSIX path for a file."""

    return path.relative_to(project_root).as_posix()


def _is_included(path: Path, project_root: Path, output_path: Path) -> bool:
    """Return whether *path* is an eligible text source file."""

    if not _candidate_is_safe(path, project_root):
        return False
    if not path.is_file() or path == output_path:
        return False

    relative = path.relative_to(project_root)
    if path.name in EXCLUDED_NAMES or any(part in EXCLUDED_PARTS for part in relative.parts):
        return False

    return path.name == ".gitignore" or path.suffix.lower() in TEXT_SUFFIXES


def _included_files(project_root: Path, output_path: Path | None = None) -> list[Path]:
    """Collect eligible files from the declared root files and content dirs."""

    output_path = output_path or project_root / "LLM_UPLOAD_MEGA.txt"
    candidates: list[Path] = []
    for filename in ROOT_FILES:
        path = project_root / filename
        if _is_included(path, project_root, output_path):
            candidates.append(path)

    for dirname in CONTENT_DIRS:
        content_dir = project_root / dirname
        if not content_dir.is_dir() or content_dir.is_symlink():
            continue
        for path in content_dir.rglob("*"):
            if _is_included(path, project_root, output_path):
                candidates.append(path)

    return sorted(candidates, key=lambda path: _relative_name(path, project_root))


def _render_bundle(project_root: Path, source_files: Iterable[Path]) -> str:
    source_files = list(source_files)
    relative_names = [_relative_name(path, project_root) for path in source_files]

    lines = [
        "AGENTIC TRANSLATION HARNESS — COMPLETE LLM REVIEW BUNDLE",
        "",
        "Purpose: Review the portable source, fixtures, documentation, and tests.",
        "Important: Bundled replay fixtures are synthetic cache-only evidence.",
        "",
        f"FILE COUNT: {len(source_files)}",
        "",
        "FILE INDEX",
    ]
    lines.extend(f"- {name}" for name in relative_names)

    for path, relative_name in zip(source_files, relative_names):
        file_text = path.read_text(encoding="utf-8")
        lines.extend(("", SEPARATOR, f"BEGIN FILE: {relative_name}", SEPARATOR))
        if file_text:
            lines.append(file_text.rstrip("\n"))
        lines.extend((SEPARATOR, f"END FILE: {relative_name}", SEPARATOR, ""))

    return "\n".join(lines) + "\n"


def build_llm_upload(project_root: str | Path, output_path: str | Path | None = None) -> Path:
    """Generate and write ``LLM_UPLOAD_MEGA.txt`` for *project_root*.

    The returned path is absolute and resolved. Source labels inside the bundle
    remain project-relative so local filesystem paths are never exposed.
    """

    raw_root = Path(project_root).expanduser()
    if not raw_root.is_absolute():
        raw_root = Path.cwd() / raw_root
    resolved_root = raw_root.resolve()
    if not resolved_root.is_dir():
        raise NotADirectoryError(f"project_root is not a directory: {resolved_root}")

    destination = (
        resolved_root / "LLM_UPLOAD_MEGA.txt"
        if output_path is None
        else _rebase_output_path(raw_root, resolved_root, output_path)
    )
    destination = _validate_output_path(resolved_root, destination)
    destination = destination.resolve(strict=False)
    destination = _validate_output_path(resolved_root, destination)

    source_files = _included_files(resolved_root, destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(_render_bundle(resolved_root, source_files), encoding="utf-8")
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a deterministic LLM upload bundle.")
    parser.add_argument("project_root", nargs="?", default=".", help="Project root (default: current directory)")
    args = parser.parse_args(argv)
    print(build_llm_upload(args.project_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
