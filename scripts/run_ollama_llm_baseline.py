from __future__ import annotations

import argparse
import csv
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "evaluation" / "llm_baseline_prompts.jsonl"
OUT_DIR = ROOT / "evaluation"

ALLOWED_PRIMARY = {
    "nan_inf",
    "overflow_underflow",
    "precision_tolerance",
    "dtype_casting",
    "crash_compile",
    "performance_only",
    "not_numerical_failure",
    "needs_review",
}
ALLOWED_SECONDARY = {
    "memory_mask_bounds",
    "compiler_codegen",
    "async_race_ordering",
    "hardware_backend",
    "reduction_accumulation",
    "api_semantics",
    "environment_configuration",
    "unknown",
}


def read_prompts(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def truncate_user_content(content: str, max_chars: int) -> str:
    if len(content) <= max_chars:
        return content
    head = content[: int(max_chars * 0.72)]
    tail = content[-int(max_chars * 0.28) :]
    return f"{head}\n\n[...middle truncated for local baseline reproducibility...]\n\n{tail}"


def api_chat(model: str, system: str, user: str, host: str, timeout: int) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0,
            "num_predict": 220,
        },
    }
    request = urllib.request.Request(
        f"{host.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    return str(data.get("message", {}).get("content", ""))


def extract_json(text: str) -> dict[str, object]:
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def normalize_prediction(parsed: dict[str, object]) -> dict[str, str]:
    primary = str(parsed.get("primary_failure_label", "needs_review")).strip()
    if primary not in ALLOWED_PRIMARY:
        primary = "needs_review"

    raw_secondary = parsed.get("secondary_cause_labels", ["unknown"])
    if isinstance(raw_secondary, str):
        secondary_values = [part.strip() for part in raw_secondary.replace(",", "|").split("|")]
    elif isinstance(raw_secondary, list):
        secondary_values = [str(part).strip() for part in raw_secondary]
    else:
        secondary_values = []
    secondary = [value for value in secondary_values if value in ALLOWED_SECONDARY]
    if not secondary:
        secondary = ["unknown"]

    true_failure = str(parsed.get("is_true_numerical_failure", "unclear")).strip().lower()
    if true_failure not in {"yes", "no", "unclear"}:
        true_failure = "unclear"

    confidence = str(parsed.get("confidence", "low")).strip().lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "low"

    return {
        "primary_failure_label": primary,
        "secondary_cause_labels_pipe_separated": "|".join(dict.fromkeys(secondary)),
        "is_true_numerical_failure": true_failure,
        "confidence": confidence,
        "evidence_quote": str(parsed.get("evidence_quote", ""))[:500],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local Ollama LLM baseline on the GPU-NFBench prompt packet.")
    parser.add_argument("--model", default="llama3.2:3b")
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--prompts", type=Path, default=PROMPTS)
    parser.add_argument("--max-chars", type=int, default=5000)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    prompts = read_prompts(args.prompts)
    if args.limit:
        prompts = prompts[: args.limit]

    out_path = Path(args.out) if args.out else OUT_DIR / f"llm_baseline_predictions_ollama_{args.model.replace(':', '_')}.csv"
    rows: list[dict[str, object]] = []

    for index, prompt_row in enumerate(prompts, start=1):
        messages = prompt_row["messages"]
        if not isinstance(messages, list) or len(messages) < 2:
            continue
        system = str(messages[0].get("content", ""))
        user = truncate_user_content(str(messages[1].get("content", "")), args.max_chars)
        raw_response = ""
        error = ""
        started = time.time()
        try:
            raw_response = api_chat(args.model, system, user, args.host, args.timeout)
            parsed = extract_json(raw_response)
            normalized = normalize_prediction(parsed)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            error = str(exc)
            normalized = {
                "primary_failure_label": "needs_review",
                "secondary_cause_labels_pipe_separated": "unknown",
                "is_true_numerical_failure": "unclear",
                "confidence": "low",
                "evidence_quote": "",
            }
        elapsed = time.time() - started
        rows.append(
            {
                "blind_id": prompt_row["blind_id"],
                "model": args.model,
                "prompt_file": str(args.prompts.relative_to(ROOT) if args.prompts.is_absolute() and ROOT in args.prompts.parents else args.prompts),
                "max_chars": args.max_chars,
                "temperature": 0,
                "elapsed_seconds": f"{elapsed:.2f}",
                **normalized,
                "raw_response": raw_response[:2000],
                "error": error,
            }
        )
        print(f"{index}/{len(prompts)} {prompt_row['blind_id']} {normalized['primary_failure_label']} {elapsed:.1f}s", flush=True)
        if args.sleep:
            time.sleep(args.sleep)

    write_csv(
        out_path,
        rows,
        [
            "blind_id",
            "model",
            "prompt_file",
            "max_chars",
            "temperature",
            "elapsed_seconds",
            "primary_failure_label",
            "secondary_cause_labels_pipe_separated",
            "is_true_numerical_failure",
            "confidence",
            "evidence_quote",
            "raw_response",
            "error",
        ],
    )
    print(out_path)


if __name__ == "__main__":
    main()
