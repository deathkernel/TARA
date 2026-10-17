"""Small, dependency-free, read-only GitHub REST API client."""
from __future__ import annotations

import base64
import json
import re
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


_REPOSITORY_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


class GitHubAPIError(RuntimeError):
    """A sanitized GitHub API or transport error."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class GitHubClient:
    """GitHub API client that intentionally exposes GET operations only."""

    API_ROOT = "https://api.github.com"

    def __init__(self, token: str | None = None, *, timeout: float = 10.0,
                 max_response_bytes: int = 2_000_000) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if max_response_bytes < 1:
            raise ValueError("max_response_bytes must be positive")
        self._token = token.strip() if token and token.strip() else None
        self.timeout = timeout
        self.max_response_bytes = max_response_bytes

    @staticmethod
    def _repository(repository: str) -> str:
        value = repository.strip() if isinstance(repository, str) else ""
        if not _REPOSITORY_RE.fullmatch(value):
            raise ValueError("repository must be in owner/name format")
        owner, name = value.split("/", 1)
        if owner in {".", ".."} or name in {".", ".."}:
            raise ValueError("invalid repository name")
        return f"{owner}/{name}"

    @staticmethod
    def _positive_number(number: int, name: str) -> int:
        if isinstance(number, bool) or not isinstance(number, int) or number < 1:
            raise ValueError(f"{name} must be a positive integer")
        return number

    def _request(self, path: str, *, accept: str = "application/vnd.github+json") -> bytes:
        if not path.startswith("/") or path.startswith("//"):
            raise ValueError("API path must be a relative GitHub API path")
        request = Request(
            self.API_ROOT + path,
            headers={
                "Accept": accept,
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "tara-github/0.1.0",
                **({"Authorization": f"Bearer {self._token}"} if self._token else {}),
            },
            method="GET",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                body = response.read(self.max_response_bytes + 1)
                if len(body) > self.max_response_bytes:
                    raise GitHubAPIError("GitHub response exceeded configured size limit")
                return body
        except HTTPError as exc:
            # Do not include request headers or token-bearing request objects in errors.
            if exc.code == 401:
                message = "GitHub authentication failed; check the supplied token"
            elif exc.code == 403:
                message = "GitHub access was denied or the API rate limit was reached"
            elif exc.code == 404:
                message = "GitHub resource was not found or is not accessible"
            elif exc.code == 429:
                message = "GitHub API rate limit was reached"
            else:
                message = f"GitHub API returned HTTP {exc.code}"
            raise GitHubAPIError(message, status=exc.code) from None
        except URLError as exc:
            reason = getattr(exc, "reason", None)
            if isinstance(reason, TimeoutError):
                message = "GitHub request timed out"
            else:
                message = "Could not connect to GitHub"
            raise GitHubAPIError(message) from None
        except TimeoutError:
            raise GitHubAPIError("GitHub request timed out") from None

    def _get_json(self, path: str) -> Any:
        try:
            return json.loads(self._request(path).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise GitHubAPIError("GitHub returned an invalid JSON response") from None

    def get_repository(self, repository: str) -> dict[str, Any]:
        """Return GitHub repository metadata."""
        repo = self._repository(repository)
        data = self._get_json(f"/repos/{repo}")
        if not isinstance(data, dict):
            raise GitHubAPIError("GitHub returned an unexpected repository response")
        return data

    def get_file(self, repository: str, path: str, *, ref: str | None = None) -> str:
        """Return the UTF-8 content of a repository file."""
        repo = self._repository(repository)
        if not isinstance(path, str) or not path.strip() or path.startswith("/"):
            raise ValueError("path must be a non-empty repository-relative path")
        encoded_path = quote(path.strip(), safe="/")
        query = ""
        if ref is not None:
            if not isinstance(ref, str) or not ref.strip():
                raise ValueError("ref must be a non-empty string when provided")
            from urllib.parse import urlencode
            query = "?" + urlencode({"ref": ref.strip()})
        data = self._get_json(f"/repos/{repo}/contents/{encoded_path}{query}")
        if not isinstance(data, dict) or data.get("type") != "file":
            raise GitHubAPIError("GitHub path does not identify a file")
        if data.get("encoding") == "base64" and isinstance(data.get("content"), str):
            try:
                return base64.b64decode(data["content"], validate=False).decode("utf-8")
            except (ValueError, UnicodeDecodeError):
                raise GitHubAPIError("GitHub file is not valid UTF-8 text") from None
        raise GitHubAPIError("GitHub did not return a supported text file")

    def list_issues(self, repository: str, *, state: str = "open", per_page: int = 30) -> list[dict[str, Any]]:
        """List issues, excluding pull requests returned by GitHub's issues endpoint."""
        repo = self._repository(repository)
        if state not in {"open", "closed", "all"}:
            raise ValueError("state must be open, closed, or all")
        per_page = min(self._positive_number(per_page, "per_page"), 100)
        from urllib.parse import urlencode
        rows = self._get_json(f"/repos/{repo}/issues?{urlencode({'state': state, 'per_page': per_page})}")
        if not isinstance(rows, list):
            raise GitHubAPIError("GitHub returned an unexpected issues response")
        return [row for row in rows if isinstance(row, dict) and "pull_request" not in row]

    def list_pull_requests(self, repository: str, *, state: str = "open",
                           per_page: int = 30) -> list[dict[str, Any]]:
        repo = self._repository(repository)
        if state not in {"open", "closed", "all"}:
            raise ValueError("state must be open, closed, or all")
        per_page = min(self._positive_number(per_page, "per_page"), 100)
        from urllib.parse import urlencode
        rows = self._get_json(f"/repos/{repo}/pulls?{urlencode({'state': state, 'per_page': per_page})}")
        if not isinstance(rows, list):
            raise GitHubAPIError("GitHub returned an unexpected pull request response")
        return rows

    def list_branches(self, repository: str, *, per_page: int = 30) -> list[dict[str, Any]]:
        repo = self._repository(repository)
        per_page = min(self._positive_number(per_page, "per_page"), 100)
        from urllib.parse import urlencode
        rows = self._get_json(f"/repos/{repo}/branches?{urlencode({'per_page': per_page})}")
        if not isinstance(rows, list):
            raise GitHubAPIError("GitHub returned an unexpected branches response")
        return rows

    def list_commits(self, repository: str, *, per_page: int = 30,
                     branch: str | None = None) -> list[dict[str, Any]]:
        repo = self._repository(repository)
        per_page = min(self._positive_number(per_page, "per_page"), 100)
        from urllib.parse import urlencode
        params: dict[str, str | int] = {"per_page": per_page}
        if branch is not None:
            if not isinstance(branch, str) or not branch.strip():
                raise ValueError("branch must be a non-empty string when provided")
            params["sha"] = branch.strip()
        rows = self._get_json(f"/repos/{repo}/commits?{urlencode(params)}")
        if not isinstance(rows, list):
            raise GitHubAPIError("GitHub returned an unexpected commits response")
        return rows

    def get_pull_request_diff(self, repository: str, number: int) -> str:
        """Return the unified diff for a pull request."""
        repo = self._repository(repository)
        number = self._positive_number(number, "number")
        body = self._request(f"/repos/{repo}/pulls/{number}", accept="application/vnd.github.diff")
        try:
            return body.decode("utf-8")
        except UnicodeDecodeError:
            raise GitHubAPIError("GitHub returned a non-UTF-8 pull request diff") from None
