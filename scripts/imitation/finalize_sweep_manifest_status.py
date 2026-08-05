"""Mark stale resumable-sweep entries as superseded without deleting historical evidence."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--active_num_envs", type=int, required=True)
    args = parser.parse_args()

    path = Path(args.manifest).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    changed = 0
    for entry in payload.get("entries", {}).values():
        if entry.get("status") == "running" and int(entry.get("num_envs", -1)) != args.active_num_envs:
            entry["status"] = "superseded"
            entry["superseded_at_utc"] = datetime.now(timezone.utc).isoformat()
            entry["superseded_reason"] = (
                f"Interrupted evaluation used num_envs={entry.get('num_envs')}; the complete frozen-protocol "
                f"replacement uses num_envs={args.active_num_envs} with identical seed and episode semantics."
            )
            changed += 1
    if changed:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            temp_name = stream.name
        os.replace(temp_name, path)
    print(json.dumps({"manifest": str(path), "superseded_entries": changed}, indent=2))


if __name__ == "__main__":
    main()
