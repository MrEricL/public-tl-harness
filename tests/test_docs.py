from pathlib import Path


def test_public_docs_are_portable_and_exclude_private_workspace_details() -> None:
    public_docs = [
        Path("README.md"),
        Path("DEMO_SCRIPT.md"),
        Path("DATA_NOTICE.md"),
        Path("CHANGELOG.md"),
        Path("docs/AUTOMATIC_REPAIR.md"),
        Path("docs/SESSION_IDENTITY.md"),
        Path("docs/JEV_EXTENSION.md"),
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in public_docs)

    private_experiment_path = "experiments/" + "jev_quality_evaluation"
    gateway_secret_prefix = "v" + "ck_"
    for forbidden in (
        "/Users/",
        private_experiment_path,
        gateway_secret_prefix,
        "recruiter-facing",
        "account is out of balance",
    ):
        assert forbidden not in combined


def test_data_notice_labels_the_new_fixture_as_synthetic() -> None:
    notice = Path("DATA_NOTICE.md").read_text(encoding="utf-8")
    assert "samples/synthetic_repair_demo/" in notice
    assert "newly authored synthetic test material" in notice
    assert "not copied or derived from a novel" in notice


def test_public_jev_docs_keep_the_study_and_secret_boundary_explicit() -> None:
    jev_doc = Path("docs/JEV_EXTENSION.md").read_text(encoding="utf-8")
    normalized_jev_doc = " ".join(jev_doc.split())
    notice = Path("DATA_NOTICE.md").read_text(encoding="utf-8")

    for text in (
        "typesafe-ai/jev",
        "AI_GATEWAY_API_KEY",
        "off by default",
        "private development study",
        "not a public benchmark",
        "28,000-byte",
    ):
        assert text in normalized_jev_doc

    assert "raw Jev/Gateway payloads" in notice
    assert "no API key value is included" in notice
