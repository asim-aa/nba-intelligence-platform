from pathlib import Path


REQUIRED_PATHS = (
    "app/api",
    "app/dashboard",
    "data/raw",
    "data/interim",
    "data/samples",
    "pipelines/ingestion",
    "pipelines/transformations",
    "pipelines/features",
    "modeling/baselines",
    "modeling/training",
    "modeling/evaluation",
    "sql",
    "artifacts",
    "docs",
)


def test_required_project_paths_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    missing = [path for path in REQUIRED_PATHS if not (root / path).exists()]
    assert not missing, f"Missing required project paths: {missing}"


def test_phase_zero_spec_exists() -> None:
    root = Path(__file__).resolve().parents[1]
    assert (root / "docs" / "project_spec.md").is_file()
