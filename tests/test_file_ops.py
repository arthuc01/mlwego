from pathlib import Path

from mlwego.workspace.file_ops import apply_patch


def test_apply_patch_creates_diff(tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"
    path.write_text("hello\n", encoding="utf-8")
    old, diff = apply_patch(path, "hello\nworld\n")
    assert old == "hello\n"
    assert diff
    assert "world" in path.read_text(encoding="utf-8")
