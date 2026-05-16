from __future__ import annotations

import argparse
import csv
import re
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LINKED_FIX_TABLE = ROOT / "tables" / "linked_fix_evidence_subset.csv"
DIFF_DIR = ROOT / "data" / "linked_pr_diffs"
MANIFEST = ROOT / "tables" / "linked_pr_diff_manifest.csv"
CHANGED_FILES = ROOT / "tables" / "linked_pr_changed_files.csv"
REPORT = ROOT / "reports" / "linked_pr_diff_fetch_report.md"

PR_URL_RE = re.compile(r"^https://github\.com/([^/\s]+)/([^/\s]+)/pull/(\d+)/?$")
DIFF_FILE_RE = re.compile(r"^diff --git a/(.*?) b/(.*?)$", re.MULTILINE)

MAX_DIFF_BYTES = 5_000_000
REQUEST_SLEEP_SECONDS = 0.25
USER_AGENT = "GPU-NFBench-linked-pr-diff-fetcher/1.0"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def split_urls(value: str) -> list[str]:
    urls: list[str] = []
    for part in (value or "").split("|"):
        part = part.strip().rstrip(".,)")
        if PR_URL_RE.match(part):
            urls.append(part.rstrip("/"))
    return urls


def safe_diff_name(url: str) -> str:
    match = PR_URL_RE.match(url)
    if not match:
        raise ValueError(f"Unsupported PR URL: {url}")
    owner, repo, number = match.groups()
    return f"{owner}__{repo}__pull_{number}.diff"


def fetch_diff(url: str, path: Path, refresh: bool) -> tuple[str, int | str, int, str]:
    if path.exists() and not refresh:
        return "cached", "", path.stat().st_size, ""

    diff_url = f"{url}.diff"
    request = urllib.request.Request(diff_url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > MAX_DIFF_BYTES:
                return "skipped_large", response.status, 0, f"Content-Length {content_length} exceeds cap"
            data = response.read(MAX_DIFF_BYTES + 1)
            if len(data) > MAX_DIFF_BYTES:
                return "skipped_large", response.status, 0, f"Diff exceeds {MAX_DIFF_BYTES} bytes"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            return "fetched", response.status, len(data), ""
    except urllib.error.HTTPError as exc:
        return "failed", exc.code, 0, str(exc)
    except urllib.error.URLError as exc:
        return "failed", "", 0, str(exc.reason)
    except TimeoutError as exc:
        return "failed", "", 0, str(exc)


def changed_files(path: Path) -> list[str]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    files = {new for _, new in DIFF_FILE_RE.findall(text)}
    return sorted(files)


def collect_pr_sources(rows: list[dict[str, str]]) -> dict[str, dict[str, set[str]]]:
    sources: dict[str, dict[str, set[str]]] = {}
    for row in rows:
        urls = split_urls(row.get("explicit_pull_urls", ""))
        urls.extend(split_urls(row.get("same_repo_fix_ref_urls", "")))
        for url in urls:
            entry = sources.setdefault(url, {"blind_ids": set(), "repositories": set()})
            entry["blind_ids"].add(row.get("blind_id", ""))
            entry["repositories"].add(row.get("repository", ""))
    return sources


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch public GitHub PR diffs referenced by GPU-NFBench linked-fix evidence.")
    parser.add_argument("--refresh", action="store_true", help="Re-download diffs even if cached.")
    args = parser.parse_args()

    rows = read_csv(LINKED_FIX_TABLE)
    sources = collect_pr_sources(rows)

    manifest_rows: list[dict[str, object]] = []
    changed_rows: list[dict[str, object]] = []

    for index, (url, source) in enumerate(sorted(sources.items())):
        match = PR_URL_RE.match(url)
        if not match:
            continue
        owner, repo, number = match.groups()
        local_path = DIFF_DIR / safe_diff_name(url)
        status, http_status, byte_count, error = fetch_diff(url, local_path, args.refresh)
        files = changed_files(local_path) if status in {"cached", "fetched"} else []

        manifest_rows.append(
            {
                "pull_url": url,
                "diff_url": f"{url}.diff",
                "local_diff_path": str(local_path.relative_to(ROOT)),
                "status": status,
                "http_status": http_status,
                "bytes": byte_count,
                "changed_file_count": len(files),
                "changed_files_sample": "|".join(files[:12]),
                "source_blind_ids": "|".join(sorted(v for v in source["blind_ids"] if v)),
                "source_repositories": "|".join(sorted(v for v in source["repositories"] if v)),
                "error": error,
            }
        )
        for file_name in files:
            changed_rows.append(
                {
                    "pull_url": url,
                    "repository": f"{owner}/{repo}",
                    "pull_number": number,
                    "changed_file": file_name,
                }
            )

        if index < len(sources) - 1:
            time.sleep(REQUEST_SLEEP_SECONDS)

    write_csv(
        MANIFEST,
        manifest_rows,
        [
            "pull_url",
            "diff_url",
            "local_diff_path",
            "status",
            "http_status",
            "bytes",
            "changed_file_count",
            "changed_files_sample",
            "source_blind_ids",
            "source_repositories",
            "error",
        ],
    )
    write_csv(CHANGED_FILES, changed_rows, ["pull_url", "repository", "pull_number", "changed_file"])

    fetched_like = [row for row in manifest_rows if row["status"] in {"cached", "fetched"}]
    failed = [row for row in manifest_rows if row["status"] == "failed"]
    skipped = [row for row in manifest_rows if row["status"] == "skipped_large"]
    zero_byte = [row for row in fetched_like if int(row["bytes"]) == 0]
    non_empty = [row for row in fetched_like if int(row["bytes"]) > 0]
    total_bytes = sum(int(row["bytes"]) for row in fetched_like)
    unique_changed_files = len({row["changed_file"] for row in changed_rows})

    lines = [
        "# Linked PR Diff Fetch Report",
        "",
        "This report fetches public GitHub `.diff` files for pull requests referenced by the linked-fix evidence table. It does not require `gh` authentication; unavailable, private, deleted, or non-pull-request references are retained in the manifest as failures instead of being silently dropped.",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| unique PR URLs attempted | {len(manifest_rows)} |",
        f"| fetched or cached diffs | {len(fetched_like)} |",
        f"| non-empty diffs | {len(non_empty)} |",
        f"| zero-byte HTTP-200 diffs | {len(zero_byte)} |",
        f"| failed fetches | {len(failed)} |",
        f"| skipped oversized diffs | {len(skipped)} |",
        f"| total fetched bytes | {total_bytes} |",
        f"| changed-file rows extracted | {len(changed_rows)} |",
        f"| unique changed files extracted | {unique_changed_files} |",
        "",
        "Generated artifacts:",
        "",
        "- `tables/linked_pr_diff_manifest.csv`",
        "- `tables/linked_pr_changed_files.csv`",
        "- `data/linked_pr_diffs/*.diff`",
        "",
        "The manifest keeps source `blind_id` links so a reviewer can trace each code diff back to the adjudicated benchmark row that referenced it.",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(REPORT)


if __name__ == "__main__":
    main()
