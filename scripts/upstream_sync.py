"""Resolve and lock all TARA upstream repositories."""

from __future__ import annotations

import argparse
import json
import sys

from src.upstream.registry import UpstreamRegistry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--write-lock", action="store_true")
    args = parser.parse_args()

    if not args.check and not args.write_lock:
        parser.error("choose --check or --write-lock")

    registry = UpstreamRegistry()
    try:
        snapshots = registry.resolve_all()
    except Exception as exc:
        print(f"upstream resolution failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    changes = registry.drift(snapshots)
    if args.check:
        if changes:
            print(json.dumps({"status": "drift", "changes": changes}, indent=2))
            return 1
        print("TARA upstream lock is current.")
        return 0

    registry.write_lock(snapshots)
    print(json.dumps({
        "status": "locked",
        "count": len(snapshots),
        "changed": len(changes),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
