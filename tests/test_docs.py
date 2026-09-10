from pathlib import Path


def test_public_docs_are_portable_and_exclude_private_workspace_details() -> None:
    public_docs = [
        Path("README.md"),
        Path("DEMO_SCRIPT.md"),
        Path("DATA_NOTICE.md"),
        Path("CHANGELOG.md"),
        Path("docs/AUTOMATIC_REPAIR.md"),
        Path("docs/SESSION_IDENTITY.md"),
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in public_docs)

    for forbidden in (
        "/Users/",
        "recruiter-facing",
        "account is out of balance",
    ):
        assert forbidden not in combined


def test_data_notice_labels_the_new_fixture_as_synthetic() -> None:
    notice = Path("DATA_NOTICE.md").read_text(encoding="utf-8")
    assert "samples/synthetic_repair_demo/" in notice
    assert "newly authored synthetic test material" in notice
    assert "not copied or derived from a novel" in notice
