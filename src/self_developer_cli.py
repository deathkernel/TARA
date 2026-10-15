"""CMD-friendly CLI for TARA's controlled self-developer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .self_developer import SelfDeveloper, SelfDeveloperError


def _engine() -> SelfDeveloper:
    from .llm_provider_factory import provider_from_env
    return SelfDeveloper(Path.cwd(), provider=provider_from_env())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.self_developer_cli")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="show repository and self-developer locations")
    plan_parser = sub.add_parser("plan", help="ask Llama for an implementation plan")
    plan_parser.add_argument("task")
    run_parser = sub.add_parser("run", help="create and test an isolated candidate")
    run_parser.add_argument("task")
    run_parser.add_argument("--max-attempts", type=int, default=3)
    sub.add_parser("report", help="show the latest saved report")

    args = parser.parse_args(argv)
    try:
        engine = _engine()
        if args.command == "status":
            result = engine.status()
        elif args.command == "plan":
            result = engine.plan(args.task)
        elif args.command == "run":
            result = engine.run_task(args.task, max_attempts=args.max_attempts)
        else:
            result = engine.latest_report()
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result.get("status") != "failed" else 1
    except (SelfDeveloperError, OSError, ValueError) as exc:
        parser.exit(2, f"Self-Developer error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
