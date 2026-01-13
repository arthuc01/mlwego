from pathlib import Path

from mlwego.evaluation.evaluator import validate_submission


def test_submission_validation_matches_sample(tmp_path: Path) -> None:
    run_dir = tmp_path
    (run_dir / "artifacts").mkdir()
    (run_dir / "data").mkdir()
    (run_dir / "data" / "sample_submission.csv").write_text("id,target\n1,0\n", encoding="utf-8")
    (run_dir / "artifacts" / "submission.csv").write_text("id,target\n1,0\n", encoding="utf-8")
    error = validate_submission(run_dir)
    assert error is None
