"""Constrained file tools for TARA.

File access is opt-in: callers must provide one or more allowed root
 directories. Paths are resolved before access so traversal outside those
roots is rejected. Destructive operations are explicit and require a caller
confirmation token supplied by the host application.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


class FileToolError(RuntimeError):
    """Raised when a file-tool request violates its safety boundary."""


@dataclass(frozen=True)
class FileToolResult:
    operation: str
    success: bool
    path: str
    output: object = None
    error: str | None = None


class FileTools:
    """Allow-listed, bounded filesystem operations.

    No filesystem location is accessible until it is explicitly placed under
    an allowed root. Symlink escapes are rejected by resolving the target
    before the access check.
    """

    def __init__(self, allowed_roots: Iterable[str | Path], *, max_read_bytes: int = 1_048_576) -> None:
        roots = tuple(Path(root).expanduser().resolve() for root in allowed_roots)
        if not roots:
            raise ValueError("at least one allowed root is required")
        if max_read_bytes < 1:
            raise ValueError("max_read_bytes must be positive")
        self._roots = roots
        self.max_read_bytes = max_read_bytes

    @property
    def allowed_roots(self) -> tuple[Path, ...]:
        return self._roots

    def _resolve(self, path: str | Path) -> Path:
        candidate = Path(path).expanduser()
        resolved = candidate.resolve(strict=False)
        for root in self._roots:
            try:
                resolved.relative_to(root)
                return resolved
            except ValueError:
                continue
        raise FileToolError("path is outside the configured allow-list")

    def list(self, path: str | Path = ".") -> FileToolResult:
        try:
            target = self._resolve(path)
            if not target.is_dir():
                raise FileToolError("path is not a directory")
            entries = tuple(sorted(item.name for item in target.iterdir()))
            return FileToolResult("list", True, str(target), entries)
        except (OSError, FileToolError) as exc:
            return FileToolResult("list", False, str(path), error=str(exc))

    def read(self, path: str | Path) -> FileToolResult:
        try:
            target = self._resolve(path)
            if not target.is_file():
                raise FileToolError("path is not a file")
            size = target.stat().st_size
            if size > self.max_read_bytes:
                raise FileToolError("file exceeds the configured read limit")
            return FileToolResult("read", True, str(target), target.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, FileToolError) as exc:
            return FileToolResult("read", False, str(path), error=str(exc))

    def write(self, path: str | Path, content: str, *, confirm: str | None = None) -> FileToolResult:
        try:
            target = self._resolve(path)
            if confirm != "WRITE":
                raise FileToolError("write requires confirmation token WRITE")
            if not isinstance(content, str):
                raise FileToolError("content must be text")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return FileToolResult("write", True, str(target), len(content.encode("utf-8")))
        except (OSError, UnicodeError, FileToolError) as exc:
            return FileToolResult("write", False, str(path), error=str(exc))

    def delete(self, path: str | Path, *, confirm: str | None = None) -> FileToolResult:
        try:
            target = self._resolve(path)
            if confirm != "DELETE":
                raise FileToolError("delete requires confirmation token DELETE")
            if not target.exists() or not target.is_file():
                raise FileToolError("path is not an existing file")
            target.unlink()
            return FileToolResult("delete", True, str(target))
        except (OSError, FileToolError) as exc:
            return FileToolResult("delete", False, str(path), error=str(exc))

    def move(self, source: str | Path, destination: str | Path, *, confirm: str | None = None) -> FileToolResult:
        try:
            src = self._resolve(source)
            dst = self._resolve(destination)
            if confirm != "MOVE":
                raise FileToolError("move requires confirmation token MOVE")
            if not src.is_file():
                raise FileToolError("source is not an existing file")
            dst.parent.mkdir(parents=True, exist_ok=True)
            src.rename(dst)
            return FileToolResult("move", True, str(dst), str(src))
        except (OSError, FileToolError) as exc:
            return FileToolResult("move", False, str(destination), error=str(exc))
