from __future__ import annotations

from pathlib import Path

import pytest

from agentic_translation.story import prepare_run_dir


@pytest.mark.parametrize("run_id", ["../victim", "a/b", ".", ".."])
def test_prepare_run_dir_rejects_non_component_run_ids(tmp_path: Path, run_id: str) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    victim = tmp_path / "victim"
    victim.mkdir()
    sentinel = victim / "sentinel.txt"
    sentinel.write_text("preserve", encoding="utf-8")

    with pytest.raises(ValueError, match="run_id"):
        prepare_run_dir(runs_dir, run_id, overwrite=run_id == "../victim")

    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_prepare_run_dir_rejects_absolute_run_id(tmp_path: Path) -> None:
    absolute_target = (tmp_path / "absolute").resolve()

    with pytest.raises(ValueError, match="run_id"):
        prepare_run_dir(tmp_path / "runs", str(absolute_target), overwrite=True)

    assert not absolute_target.exists()


def test_prepare_run_dir_rejects_selected_symlink(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_text("preserve", encoding="utf-8")
    link = runs_dir / "selected"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlinks unsupported: {exc}")

    with pytest.raises(ValueError, match="symlink"):
        prepare_run_dir(runs_dir, "selected", overwrite=True)

    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_prepare_run_dir_rejects_non_directory_overwrite_target(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    target = runs_dir / "selected"
    target.write_text("preserve", encoding="utf-8")

    with pytest.raises(ValueError, match="non-directory"):
        prepare_run_dir(runs_dir, "selected", overwrite=True)

    assert target.read_text(encoding="utf-8") == "preserve"


def test_prepare_run_dir_overwrites_only_valid_direct_child(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    target = runs_dir / "selected"
    target.mkdir(parents=True)
    (target / "old.txt").write_text("replace", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("preserve", encoding="utf-8")

    result = prepare_run_dir(runs_dir, "selected", overwrite=True)

    assert result == runs_dir.resolve() / "selected"
    assert result.is_dir()
    assert list(result.iterdir()) == []
    assert outside.read_text(encoding="utf-8") == "preserve"
