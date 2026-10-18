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
    assert result.success is False
    assert "disabled" in (result.error or "")


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


def test_controller_logs_validator_failure(caplog):
    import logging

    controller = ToolController()

    def broken_validator(args):
        raise RuntimeError("validator unavailable")

    controller.register("check", lambda: "unused", validator=broken_validator)
    with caplog.at_level(logging.ERROR, logger="tara.tools"):
        result = controller.execute("check")

    assert not result.success
    assert any(
        getattr(record, "tara_operation", None) == "validate_tool:check"
        and getattr(record, "tara_error_type", None) == "RuntimeError"
        for record in caplog.records
    )


def test_controller_logs_tool_execution_failure(caplog):
    import logging

    controller = ToolController()

    def broken_tool():
        raise ValueError("tool unavailable")

    controller.register("broken", broken_tool)
    with caplog.at_level(logging.ERROR, logger="tara.tools"):
        result = controller.execute("broken")

    assert not result.success
    assert any(
        getattr(record, "tara_operation", None) == "execute_tool:broken"
        and getattr(record, "tara_error_type", None) == "ValueError"
        for record in caplog.records
    )


def test_failed_tool_compensation_is_not_left_on_rollback_stack(caplog):
    import logging

    controller = ToolController()
    rollback_calls = []

    def broken_tool():
        raise RuntimeError("operation failed")

    def broken_rollback():
        rollback_calls.append("rollback")
        raise OSError("compensation failed")

    controller.register("mutate", broken_tool)
    with caplog.at_level(logging.ERROR, logger="tara.tools"):
        result = controller.execute_with_rollback("mutate", rollback=broken_rollback)
        later_rollback = controller.rollback_last()

    assert not result.success
    assert rollback_calls == ["rollback"]
    assert later_rollback is False
    operations = {
        getattr(record, "tara_operation", None)
        for record in caplog.records
    }
    assert "execute_tool:mutate" in operations
    assert "rollback_tool:mutate" in operations


def test_controller_logs_rollback_last_failure(caplog):
    import logging

    controller = ToolController()
    controller.register("ok", lambda: "done")

    def broken_rollback():
        raise OSError("rollback unavailable")

    with caplog.at_level(logging.ERROR, logger="tara.tools"):
        result = controller.execute_with_rollback("ok", rollback=broken_rollback)
        rollback_result = controller.rollback_last()

    assert result.success
    assert rollback_result is False
    assert any(
        getattr(record, "tara_operation", None) == "rollback_last"
        and getattr(record, "tara_error_type", None) == "OSError"
        for record in caplog.records
    )
