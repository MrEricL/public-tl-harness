"""The folder demo advertises only entry points included in its installation."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentic_translation import demo


def test_reduced_installation_omits_unavailable_offline_command(monkeypatch, capsys):
    monkeypatch.setattr(demo, "find_spec", lambda name: None)
    with pytest.raises(SystemExit) as result:
        demo.main(["--help"])
    assert result.value.code == 0
    assert "--offline" not in capsys.readouterr().out
    monkeypatch.setattr(demo.sys.stdin, "isatty", lambda: False)
    assert demo.main([]) == 2
    error = capsys.readouterr().err
    assert "Pass --source FOLDER" in error
    assert "--offline" not in error


def test_full_installation_retains_offline_command(monkeypatch, capsys):
    inside = SimpleNamespace(origin=str(Path(demo.__file__).parent / "showcase.py"))
    monkeypatch.setattr(demo, "find_spec", lambda name: inside)
    with pytest.raises(SystemExit) as result:
        demo.main(["--help"])
    assert result.value.code == 0
    assert "--offline" in capsys.readouterr().out


def test_a_showcase_outside_this_package_does_not_enable_offline(monkeypatch, capsys, tmp_path):
    elsewhere = SimpleNamespace(origin=str(tmp_path / "agentic_translation" / "showcase.py"))
    monkeypatch.setattr(demo, "find_spec", lambda name: elsewhere)
    with pytest.raises(SystemExit):
        demo.main(["--help"])
    assert "--offline" not in capsys.readouterr().out
