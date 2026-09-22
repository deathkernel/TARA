from pathlib import Path

from src.tools import FileTools


def test_file_tools_allow_list_and_read(tmp_path: Path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    (allowed / "hello.txt").write_text("hello", encoding="utf-8")
    tools = FileTools([allowed])

    listed = tools.list(allowed)
    assert listed.success
    assert listed.output == ("hello.txt",)
    assert tools.read(allowed / "hello.txt").output == "hello"


def test_file_tools_reject_traversal(tmp_path: Path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    tools = FileTools([allowed])

    result = tools.read(allowed / ".." / "outside.txt")
    assert not result.success
    assert "outside" in result.error.lower()


def test_destructive_operations_require_confirmation(tmp_path: Path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    target = allowed / "note.txt"
    tools = FileTools([allowed])

    assert not tools.write(target, "x").success
    assert tools.write(target, "x", confirm="WRITE").success
    assert not tools.delete(target).success
    assert tools.delete(target, confirm="DELETE").success


def test_move_stays_inside_allow_list(tmp_path: Path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    source = allowed / "a.txt"
    source.write_text("a", encoding="utf-8")
    tools = FileTools([allowed])

    result = tools.move(source, allowed / "nested" / "b.txt", confirm="MOVE")
    assert result.success
    assert (allowed / "nested" / "b.txt").read_text(encoding="utf-8") == "a"
