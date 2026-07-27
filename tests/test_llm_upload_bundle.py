import hashlib

import pytest
from tools.build_llm_upload import build_llm_upload


def test_build_llm_upload_contains_sorted_index_and_markers(tmp_path):
    files = {
        ".gitignore": "*.pyc\n",
        "README.md": "# Demo\n",
        "pyproject.toml": "[project]\nname = 'demo'\n",
        "agentic_translation/app.py": "print('hello')\n",
        "tests/test_app.py": "def test_app():\n    assert True\n",
    }
    for name, content in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    output_path = build_llm_upload(tmp_path)

    assert output_path == tmp_path / "LLM_UPLOAD_MEGA.txt"
    content = output_path.read_text(encoding="utf-8")
    assert "FILE COUNT: 5" in content

    index_section = content.split("FILE INDEX\n", 1)[1].split("\n\n", 1)[0]
    indexed_paths = [line.removeprefix("- ") for line in index_section.splitlines()]
    assert indexed_paths == sorted(files)

    for relative_path in sorted(files):
        assert f"BEGIN FILE: {relative_path}" in content
        assert f"END FILE: {relative_path}" in content


def test_build_llm_upload_excludes_generated_and_binary_paths(tmp_path):
    files = {
        ".gitignore": "*.pyc\n",
        "README.md": "# Demo\n",
        "pyproject.toml": "[project]\nname = 'demo'\n",
        "agentic_translation/app.py": "print('hello')\n",
        "tests/test_app.py": "def test_app():\n    assert True\n",
        "runs/output.txt": "should not be bundled\n",
        "agentic_translation/app.pyc": "binary-ish\n",
        "agentic_translation/.agentic_cache/hidden.txt": "cache secret\n",
        "LLM_UPLOAD_MEGA.txt": "stale bundle\n",
    }
    for name, content in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    output_path = build_llm_upload(tmp_path)
    bundled = output_path.read_text(encoding="utf-8")

    assert "runs/output.txt" not in bundled
    assert "agentic_translation/app.pyc" not in bundled
    assert "agentic_translation/.agentic_cache/hidden.txt" not in bundled
    assert "cache secret" not in bundled
    assert "stale bundle" not in bundled
    assert "FILE COUNT: 5" in bundled


def test_build_llm_upload_does_not_follow_symlinked_source(tmp_path):
    outside_secret = tmp_path.parent / f"{tmp_path.name}-outside_secret.txt"
    outside_secret.write_text("outside secret\n", encoding="utf-8")
    source_link = tmp_path / "agentic_translation" / "leak.txt"
    source_link.parent.mkdir(parents=True)
    try:
        source_link.symlink_to(outside_secret)
    except OSError as exc:
        pytest.skip(f"symlinks unsupported: {exc}")

    bundled = build_llm_upload(tmp_path).read_text(encoding="utf-8")

    assert "outside secret" not in bundled
    assert "agentic_translation/leak.txt" not in bundled


def test_build_llm_upload_rejects_symlinked_default_output(tmp_path):
    outside_output = tmp_path.parent / f"{tmp_path.name}-outside_output.txt"
    outside_output.write_text("preserve this content\n", encoding="utf-8")
    output_link = tmp_path / "LLM_UPLOAD_MEGA.txt"
    try:
        output_link.symlink_to(outside_output)
    except OSError as exc:
        pytest.skip(f"symlinks unsupported: {exc}")

    with pytest.raises(ValueError, match="symlink"):
        build_llm_upload(tmp_path)

    assert outside_output.read_text(encoding="utf-8") == "preserve this content\n"


def test_build_llm_upload_skips_symlinked_content_root(tmp_path):
    outside_source = tmp_path.parent / f"{tmp_path.name}-outside-source"
    outside_source.mkdir()
    secret = outside_source / "private.txt"
    secret.write_text("private secret\n", encoding="utf-8")
    content_root = tmp_path / "agentic_translation"
    try:
        content_root.symlink_to(outside_source, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlinks unsupported: {exc}")

    bundled = build_llm_upload(tmp_path).read_text(encoding="utf-8")

    assert "private secret" not in bundled
    assert "agentic_translation/private.txt" not in bundled


def test_build_llm_upload_rejects_symlinked_custom_output_parent(tmp_path):
    outside_output_parent = tmp_path.parent / f"{tmp_path.name}-outside-output"
    outside_output_parent.mkdir()
    output_parent = tmp_path / "custom-output-parent"
    try:
        output_parent.symlink_to(outside_output_parent, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlinks unsupported: {exc}")

    output_path = output_parent / "bundle.txt"
    with pytest.raises(ValueError, match="symlink"):
        build_llm_upload(tmp_path, output_path)

    assert not (outside_output_parent / "bundle.txt").exists()


def test_build_llm_upload_rebases_output_path_from_project_alias(tmp_path):
    project_alias = tmp_path.parent / f"{tmp_path.name}-project-alias"
    try:
        project_alias.symlink_to(tmp_path, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlinks unsupported: {exc}")

    aliased_output = project_alias / "normal-output" / "bundle.txt"
    output_path = build_llm_upload(project_alias, aliased_output)
    resolved_root = tmp_path.resolve()
    expected_output = resolved_root / "normal-output" / "bundle.txt"

    assert output_path == expected_output
    assert output_path.is_relative_to(resolved_root)
    assert output_path.exists()


def test_build_llm_upload_canonicalizes_dotdot_output_path(tmp_path):
    (tmp_path / "tools").mkdir()
    output_path = tmp_path / "tools" / ".." / "tools" / "custom.txt"

    first_output = build_llm_upload(tmp_path, output_path)
    first_text = first_output.read_text(encoding="utf-8")
    first_hash = hashlib.sha256(first_text.encode("utf-8")).hexdigest()
    second_output = build_llm_upload(tmp_path, output_path)
    second_text = second_output.read_text(encoding="utf-8")
    second_hash = hashlib.sha256(second_text.encode("utf-8")).hexdigest()

    expected_output = tmp_path.resolve() / "tools" / "custom.txt"
    assert first_output == expected_output
    assert second_output == expected_output
    assert first_text == second_text
    assert first_hash == second_hash
    assert "BEGIN FILE: tools/custom.txt" not in second_text
