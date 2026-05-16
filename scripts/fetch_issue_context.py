from __future__ import annotations

import csv
import argparse
import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUBSET = ROOT / "data" / "processed" / "validation_subset.csv"
DEFAULT_OUT = ROOT / "data" / "validation_context"
ISSUE_RE = re.compile(r"https://github.com/([^/]+/[^/]+)/issues/(\d+)")


def gh_json(args: list[str]) -> object:
    result = subprocess.run(["gh", *args], check=False, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "gh api failed").strip())
    return json.loads(result.stdout or "null")


def safe_name(repo: str, number: str) -> str:
    return repo.replace("/", "__") + f"__{number}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch public issue and comment context with gh api.")
    parser.add_argument("--subset", type=Path, default=DEFAULT_SUBSET)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    subset = args.subset if args.subset.is_absolute() else ROOT / args.subset
    out = args.out if args.out.is_absolute() else ROOT / args.out

    out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(subset.open(newline="", encoding="utf-8")))
    manifest = []
    for i, row in enumerate(rows, start=1):
        match = ISSUE_RE.match(row["url"])
        if not match:
            continue
        repo, number = match.groups()
        stem = safe_name(repo, number)
        issue_path = out / f"{stem}.issue.json"
        comments_path = out / f"{stem}.comments.json"
        fetch_error = ""
        try:
            if not issue_path.exists():
                issue = gh_json(
                    [
                        "api",
                        f"repos/{repo}/issues/{number}",
                        "--jq",
                        "{number,title,state,created_at,updated_at,closed_at,labels:[.labels[].name],body,html_url,comments}",
                    ]
                )
                issue_path.write_text(json.dumps(issue, indent=2, sort_keys=True), encoding="utf-8")
            if not comments_path.exists():
                comments = gh_json(["api", f"repos/{repo}/issues/{number}/comments", "--paginate"])
                comments_path.write_text(json.dumps(comments, indent=2, sort_keys=True), encoding="utf-8")
            print(f"{i}/{len(rows)} {repo}#{number}")
        except RuntimeError as exc:
            fetch_error = str(exc)
            print(f"{i}/{len(rows)} ERROR {repo}#{number}: {fetch_error}")
        manifest.append(
            {
                "repository": repo,
                "number": number,
                "candidate_primary_failure": row["candidate_primary_failure"],
                "issue_json": str(issue_path.relative_to(ROOT)) if issue_path.exists() else "",
                "comments_json": str(comments_path.relative_to(ROOT)) if comments_path.exists() else "",
                "url": row["url"],
                "fetch_error": fetch_error,
            }
        )

    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(manifest_path)


if __name__ == "__main__":
    main()
