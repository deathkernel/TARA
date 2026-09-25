from pathlib import Path

from src.tools.file_tools import FileTools
from src.tools.terminal_tools import TerminalTools
from src.tools.app_tools import ApplicationTools, BrowserTools
from src.tools.input_tools import InputTools
from src.tools.controller import ToolController


def test_file_tool_boundary(tmp_path: Path):
    tools = FileTools([tmp_path])
    assert tools.write(tmp_path / "a.txt", "hello").success is False
    assert tools.write(tmp_path / "a.txt", "hello", confirm="WRITE").success
    assert tools.read(tmp_path / "a.txt").output == "hello"
    assert tools.read(tmp_path.parent / "a.txt").success is False
    assert tools.delete(tmp_path / "a.txt", confirm="DELETE").success


def test_terminal_allow_list(tmp_path: Path):
    tools = TerminalTools(["python"], working_root=tmp_path)
    assert tools.run(["echo", "x"]).success is False
    result = tools.run(["python", "-c", "print('ok')"])
    assert result.success and result.stdout.strip() == "ok"


def test_application_and_browser_are_injected():
    app, browser = ApplicationTools(), BrowserTools()
    app.register("open", lambda target: target)
    browser.register("navigate", lambda url: url)
    assert app.execute("open", target="editor").output == "editor"
    assert browser.execute("navigate", url="https://example.invalid").success


def test_input_backend_is_explicit():
    seen = []
    tools = InputTools(lambda action, args: seen.append((action, args)) or "ok")
    assert tools.click(10, 20).success
    assert tools.type_text("hi").success
    assert len(seen) == 2


def test_controller_selection_verification_and_stop():
    controller = ToolController()
    controller.register("ping", lambda: "pong")
    assert controller.decide("ping").allowed
    assert controller.execute("ping").output == "pong"
    controller.stop()
    assert not controller.execute("ping").success
    controller.resume()
    assert controller.execute("missing").success is False


def test_controller_rollback():
    state = []
    controller = ToolController()
    controller.register("add", lambda: state.append(1))
    result = controller.execute_with_rollback("add", rollback=lambda: state.pop())
    assert result.success and state == [1]
    assert controller.rollback_last() and state == []


def test_controller_supports_argument_policy_and_confirmation():
    controller = ToolController()
    controller.register(
        "delete",
        lambda path: f"deleted:{path}",
        validator=lambda args: (args["path"].startswith("/safe/"), "path outside safe root"),
        requires_confirmation=True,
    )
    blocked = controller.execute("delete", path="/safe/a")
    assert not blocked.success
    assert "confirmation" in blocked.error

    invalid = controller.execute("delete", path="/unsafe/a", confirmed=True)
    assert not invalid.success
    assert "safe root" in invalid.error

    allowed = controller.execute("delete", path="/safe/a", confirmed=True)
    assert allowed.success
    assert allowed.output == "deleted:/safe/a"
