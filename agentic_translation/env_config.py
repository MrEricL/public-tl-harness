from __future__ import annotations

import os
from pathlib import Path


ENV_FILE_POINTER = "AGENTIC_TRANSLATION_ENV_FILE"
DEFAULT_ENV_FILENAMES = (".env", ".env.local", "agentic.env", "global_env")


def _strip_env_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or not key.replace("_", "").isalnum() or key[0].isdigit():
            continue
        values[key] = _strip_env_value(value)
    return values


def default_env_file_candidates(start: Path | None = None) -> list[Path]:
    base = (start or Path.cwd()).resolve()
    if base.is_file():
        base = base.parent
    candidates: list[Path] = []
    for directory in [base, *base.parents]:
        for filename in DEFAULT_ENV_FILENAMES:
            candidates.append(directory / filename)
        if directory == Path.home():
            break
    return candidates


def load_env_file(path: Path, *, override: bool = False) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Environment file does not exist: {path}")
    if not path.is_file():
        raise IsADirectoryError(f"Environment file is not a file: {path}")
    loaded: list[str] = []
    for key, value in parse_env_file(path).items():
        if override or key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded


def load_cli_env(explicit_env_file: Path | None = None) -> Path | None:
    env_file = explicit_env_file
    if env_file is None:
        pointer = os.environ.get(ENV_FILE_POINTER)
        if pointer:
            env_file = Path(pointer)
    if env_file is not None:
        load_env_file(env_file.expanduser())
        return env_file
    for candidate in default_env_file_candidates():
        if candidate.exists() and candidate.is_file():
            load_env_file(candidate)
            return candidate
    return None
