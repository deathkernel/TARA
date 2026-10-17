"""Small read-only command-line interface for tara-github."""
from __future__ import annotations

import argparse
import json
import os
import sys

from .client import GitHubAPIError, GitHubClient


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tara-github", description="Read GitHub repository data")
    parser.add_argument("--token-env", default="GITHUB_TOKEN",
                        help="environment variable containing an optional GitHub token")
    sub = parser.add_subparsers(dest="command", required=True)
    repo = sub.add_parser("repo", help="show repository metadata")
    repo.add_argument("repository")
    file_cmd = sub.add_parser("file", help="print a text file")
    file_cmd.add_argument("repository")
    file_cmd.add_argument("path")
    file_cmd.add_argument("--ref")
    for command in ("issues", "pulls", "branches", "commits"):
        cmd = sub.add_parser(command, help=f"list repository {command}")
        cmd.add_argument("repository")
        if command in {"issues", "pulls"}:
            cmd.add_argument("--state", choices=("open", "closed", "all"), default="open")
        if command == "commits":
            cmd.add_argument("--branch")
        cmd.add_argument("--per-page", type=int, default=30)

    args = parser.parse_args(argv)
    client = GitHubClient(token=os.getenv(args.token_env))
    try:
        if args.command == "repo":
            result = client.get_repository(args.repository)
        elif args.command == "file":
            print(client.get_file(args.repository, args.path, ref=args.ref), end="")
            return 0
        elif args.command == "issues":
            result = client.list_issues(args.repository, state=args.state, per_page=args.per_page)
        elif args.command == "pulls":
            result = client.list_pull_requests(args.repository, state=args.state, per_page=args.per_page)
        elif args.command == "branches":
            result = client.list_branches(args.repository, per_page=args.per_page)
        else:
            result = client.list_commits(args.repository, per_page=args.per_page, branch=args.branch)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except (GitHubAPIError, ValueError) as exc:
        print(f"tara-github: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
