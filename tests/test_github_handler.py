import base64
import json
from urllib.error import HTTPError
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tara_github import GitHubAPIError, GitHubClient, connect_github_to_brain, register_github_tools
from src.tools import ApplicationTools, ToolController
from src.tool_intelligence import ToolIntelligence


class FakeResponse:
    def __init__(self, body: bytes):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, size=-1):
        return self.body[:size]


class StubGitHubClient:
    def get_repository(self, repository):
        return {"full_name": repository}

    def get_file(self, repository, path, *, ref=None):
        return f"{repository}:{path}"

    def list_issues(self, repository, *, state="open", per_page=30):
        return []

    def list_pull_requests(self, repository, *, state="open", per_page=30):
        return []

    def list_branches(self, repository, *, per_page=30):
        return []

    def list_commits(self, repository, *, per_page=30, branch=None):
        return []

    def get_pull_request_diff(self, repository, number):
        return ""


def test_repository_path_is_validated_before_network():
    client = GitHubClient()
    with pytest.raises(ValueError):
        client.get_repository("../someone/repo")


def test_get_repository_parses_json():
    body = json.dumps({"full_name": "owner/repo", "default_branch": "main"}).encode()
    with patch("tara_github.client.urlopen", return_value=FakeResponse(body)) as request:
        result = GitHubClient().get_repository("owner/repo")
    assert result["default_branch"] == "main"
    assert request.call_args.args[0].full_url == "https://api.github.com/repos/owner/repo"
    assert request.call_args.args[0].get_method() == "GET"


def test_get_file_decodes_base64_text():
    body = json.dumps({
        "type": "file",
        "encoding": "base64",
        "content": base64.b64encode(b"hello TARA").decode(),
    }).encode()
    with patch("tara_github.client.urlopen", return_value=FakeResponse(body)):
        assert GitHubClient().get_file("owner/repo", "README.md") == "hello TARA"


def test_issue_list_excludes_pull_requests():
    body = json.dumps([
        {"number": 1, "title": "issue"},
        {"number": 2, "title": "pull", "pull_request": {"url": "https://api.github.com/..."}},
    ]).encode()
    with patch("tara_github.client.urlopen", return_value=FakeResponse(body)):
        rows = GitHubClient().list_issues("owner/repo")
    assert [row["number"] for row in rows] == [1]


def test_client_rejects_invalid_query_parameters():
    client = GitHubClient()
    with pytest.raises(ValueError):
        client.list_issues("owner/repo", state="merged")
    with pytest.raises(ValueError):
        client.list_commits("owner/repo", per_page=0)


def test_http_errors_are_sanitized():
    error = HTTPError("https://api.github.com/repos/owner/repo", 401, "Unauthorized", {}, None)
    with patch("tara_github.client.urlopen", side_effect=error):
        with pytest.raises(GitHubAPIError, match="authentication failed") as raised:
            GitHubClient(token="do-not-leak").get_repository("owner/repo")
    assert "do-not-leak" not in str(raised.value)


def test_adapter_registers_with_existing_tara_tool_boundaries():
    client = StubGitHubClient()
    app_tools = ApplicationTools()
    controller = ToolController()
    names = register_github_tools(client, application_tools=app_tools, controller=controller)
    assert set(names) == set(app_tools.names())
    result = app_tools.execute("github.get_repository", repository="owner/repo")
    assert result.success and result.output == {"full_name": "owner/repo"}
    controlled = controller.execute("github.get_file", repository="owner/repo", path="README.md")
    assert controlled.success and controlled.output == "owner/repo:README.md"


def test_adapter_requires_at_least_one_target():
    with pytest.raises(ValueError):
        register_github_tools(StubGitHubClient())


def test_brain_connection_registers_capabilities_and_actions():
    brain = SimpleNamespace(tools=ToolIntelligence())
    app_tools = ApplicationTools()
    controller = ToolController()
    names = connect_github_to_brain(
        brain, StubGitHubClient(), application_tools=app_tools, controller=controller
    )
    assert set(names) == set(app_tools.names())
    assert {item.name for item in brain.tools.registry.all()} == set(names)
    choice = brain.tools.choose("inspect repository", ("github", "read"))
    assert choice.selected in names
