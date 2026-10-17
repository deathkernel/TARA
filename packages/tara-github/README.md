# tara-github

A lightweight, dependency-free GitHub REST client for TARA and standalone Python use.

## Install

From the TARA repository root:

```bash
python -m pip install -e packages/tara-github
```

## Python API

```python
from tara_github import GitHubClient

github = GitHubClient()  # public read-only requests
repo = github.get_repository("deathkernel/TARA")
print(repo["full_name"], repo["default_branch"])
print(github.get_file("deathkernel/TARA", "README.md")[:200])
```

For higher rate limits or private repositories, pass a GitHub personal access token from your environment. Do not hard-code or commit tokens.

```python
import os
from tara_github import GitHubClient

github = GitHubClient(token=os.getenv("GITHUB_TOKEN"))
```

## TARA integration

```python
from tara_github import GitHubClient, connect_github_to_brain
from src.brain import TARABrain
from src.tools import ApplicationTools, ToolController

client = GitHubClient()
brain = TARABrain(model=your_model)
application_tools = ApplicationTools()
controller = ToolController()
connect_github_to_brain(
    brain, client, application_tools=application_tools, controller=controller
)

print(application_tools.names())
result = application_tools.execute(
    "github.get_repository", repository="deathkernel/TARA"
)
```

`connect_github_to_brain` registers the same actions with the brain's
`ToolIntelligence` capability registry so planning can select them. The host
application still decides how to map a selected action to validated arguments
and execute it through the registered boundary.

The first version is read-only. It never writes files, creates issues or pull requests, pushes commits, or changes repository settings. Write operations will be added separately behind explicit authorization and confirmation.
