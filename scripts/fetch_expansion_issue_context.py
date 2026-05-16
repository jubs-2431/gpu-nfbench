from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUEUE = ROOT / "annotation" / "gold_expansion_1000_queue.csv"
DEFAULT_OUT = ROOT / "data" / "expansion_issue_context"
ISSUE_RE = re.compile(r"https://github.com/([^/]+/[^/]+)/issues/(\d+)")


def gh_json(args: list[str]) -> object:
    result = subprocess.run(["gh", *args], check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "gh api failed").strip())
    return json.loads(result.stdout or "null")


def safe_name(repo: str, number: str) -> str:
    return repo.replace("/", "__") + f"__{number}"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch full issue/comment context for the 1000-row expansion queue.")
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.2)
    args = parser.parse_args()

    queue = args.queue if args.queue.is_absolute() else ROOT / args.queue
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    rows = read_csv(queue)
    if args.limit:
        rows = rows[: args.limit]

    manifest: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        match = ISSUE_RE.match(row.get("url", ""))
        if not match:
            continue
        repo, number = match.groups()
        stem = safe_name(repo, number)
        issue_path = out / f"{stem}.issue.json"
        comments_path = out / f"{stem}.comments.json"
        events_path = out / f"{stem}.events.json"
        fetch_error = ""
        try:
            if not issue_path.exists():
                issue = gh_json(
                    [
                        "api",
                        f"repos/{repo}/issues/{number}",
                        "--jq",
                        "{number,title,state,created_at,updated_at,closed_at,labels:[.labels[].name],body,html_url,comments,pull_request}",
                    ]
                )
                issue_path.write_text(json.dumps(issue, indent=2, sort_keys=True), encoding="utf-8")
            if not comments_path.exists():
                comments = gh_json(["api", f"repos/{repo}/issues/{number}/comments", "--paginate"])
                comments_path.write_text(json.dumps(comments, indent=2, sort_keys=True), encoding="utf-8")
            if not events_path.exists():
                events = gh_json(["api", f"repos/{repo}/issues/{number}/events", "--paginate"])
                events_path.write_text(json.dumps(events, indent=2, sort_keys=True), encoding="utf-8")
            print(f"{index}/{len(rows)} {repo}#{number}")
        except RuntimeError as exc:
            fetch_error = str(exc)
            print(f"{index}/{len(rows)} ERROR {repo}#{number}: {fetch_error[:180]}")
        manifest.append(
            {
                "expansion_id": row.get("expansion_id", ""),
                "repository": repo,
                "number": number,
                "url": row.get("url", ""),
                "issue_json": str(issue_path.relative_to(ROOT)) if issue_path.exists() else "",
                "comments_json": str(comments_path.relative_to(ROOT)) if comments_path.exists() else "",
                "events_json": str(events_path.relative_to(ROOT)) if events_path.exists() else "",
                "fetch_error": fetch_error,
            }
        )
        if args.sleep:
            time.sleep(args.sleep)

    manifest_path = out / "manifest.json"
    write_manifest(manifest_path, manifest)
    print(manifest_path)


if __name__ == "__main__":
    main()
