"""Keep the published package version synchronized across metadata surfaces."""

import hashlib
import tomllib
from pathlib import Path

from agentic_translation import __version__


def test_public_version_is_synchronized() -> None:
    """The package metadata and runtime version must both report v0.2.0."""
    pyproject = Path(__file__).parents[1] / "pyproject.toml"
    metadata = tomllib.loads(pyproject.read_text(encoding="utf-8"))

    assert metadata["project"]["version"] == "0.2.0"
    assert __version__ == "0.2.0"


def test_private_package_metadata_and_license_contract() -> None:
    """Package metadata identifies the harness and ships its code license."""
    root = Path(__file__).parents[1]
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]

    assert project["name"] == "agentic-translation-harness"
    assert "translation" in project["description"].lower()
    assert "harness" in project["description"].lower()
    assert project["readme"] == "README.md"
    assert project["license"] == {"file": "LICENSE"}
    assert project["authors"] == [{"name": "MrEricL"}]
    assert project["urls"]["Repository"] == "https://github.com/MrEricL/public-tl-harness"
    assert project["urls"]["Issues"] == "https://github.com/MrEricL/public-tl-harness/issues"

    license_text = (root / "LICENSE").read_text(encoding="utf-8")
    assert "MIT License" in license_text
    assert "Copyright (c) 2026 MrEricL" in license_text
    assert "Scope:" not in license_text
    assert hashlib.sha256(license_text.encode("utf-8")).hexdigest() == (
        "82e7fc0e03f4639b0667f31219041a38678209e6078cca1af457981b12072ba2"
    )

    data_notice = (root / "DATA_NOTICE.md").read_text(encoding="utf-8")
    assert "not distributed for copyright reasons" in data_notice
    assert "aggregate results" in data_notice
    assert "MIT License" in data_notice
    assert "third-party dependencies" in data_notice
    assert "generated artifacts" in data_notice
    assert "assets that carry their own notices" in data_notice
