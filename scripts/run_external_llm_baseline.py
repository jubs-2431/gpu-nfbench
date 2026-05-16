from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "evaluation" / "llm_baseline_prompts.jsonl"

PRIMARY_LABELS = {
    "nan_inf",
    "overflow_underflow",
    "precision_tolerance",
    "dtype_casting",
    "crash_compile",
    "performance_only",
    "not_numerical_failure",
    "needs_review",
}
SECONDARY_LABELS = {
    "memory_mask_bounds",
    "compiler_codegen",
    "async_race_ordering",
    "hardware_backend",
    "reduction_accumulation",
    "api_semantics",
    "environment_configuration",
    "unknown",
}


def read_prompts(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def post_json(url: str, payload: dict[str, Any], headers: dict[str, str], timeout: int) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body[:1200]}") from exc


def call_openai_compat(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    timeout: int,
    response_format: bool,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    if response_format:
        payload["response_format"] = {"type": "json_object"}
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/jubs-2431",
        "X-Title": "GPU-NFBench external LLM baseline",
    }
    response = post_json(url, payload, headers, timeout)
    return response["choices"][0]["message"]["content"]


def call_gemini(
    *,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    timeout: int,
) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    system = "\n\n".join(message["content"] for message in messages if message["role"] == "system")
    user = "\n\n".join(message["content"] for message in messages if message["role"] == "user")
    payload = {
        "contents": [{"role": "user", "parts": [{"text": f"{system}\n\n{user}"}]}],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": max_tokens,
            "responseMimeType": "application/json",
        },
    }
    headers = {"Content-Type": "application/json"}
    response = post_json(url, payload, headers, timeout)
    return response["candidates"][0]["content"]["parts"][0]["text"]


def parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def normalize_prediction(raw: dict[str, Any]) -> dict[str, str]:
    primary = str(raw.get("primary_failure_label", "needs_review")).strip()
    if primary not in PRIMARY_LABELS:
        primary = "needs_review"

    secondary_raw = raw.get("secondary_cause_labels", ["unknown"])
    if isinstance(secondary_raw, str):
        secondary = [part.strip() for part in secondary_raw.replace(",", "|").split("|") if part.strip()]
    else:
        secondary = [str(part).strip() for part in secondary_raw if str(part).strip()]
    secondary = [label for label in secondary if label in SECONDARY_LABELS] or ["unknown"]

    true_failure = str(raw.get("is_true_numerical_failure", "unclear")).strip().lower()
    if true_failure not in {"yes", "no", "unclear"}:
        true_failure = "unclear"

    confidence = str(raw.get("confidence", "low")).strip().lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "low"

    return {
        "primary_failure_label": primary,
        "secondary_cause_labels": "|".join(secondary),
        "is_true_numerical_failure": true_failure,
        "evidence_quote": str(raw.get("evidence_quote", ""))[:500],
        "confidence": confidence,
    }


def output_path(provider: str, model: str) -> Path:
    safe_model = "".join(char if char.isalnum() or char in "._-" else "_" for char in model)
    return ROOT / "evaluation" / f"llm_baseline_predictions_{provider}_{safe_model}.csv"


def retry_delay(error: str, attempt: int) -> float:
    if "retry_after_seconds" in error:
        try:
            marker = '"retry_after_seconds":'
            after = error.split(marker, 1)[1].split(",", 1)[0].split("}", 1)[0]
            return max(float(after) + 1.0, 1.0)
        except (IndexError, ValueError):
            pass
    if "HTTP 429" in error:
        return min(30.0, 2.0 * attempt)
    return min(10.0, 1.0 * attempt)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GPU-NFBench prompts against an external LLM API.")
    parser.add_argument("--provider", choices=["openai_compat", "gemini"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="", help="OpenAI-compatible base URL, for example https://api.groq.com/openai/v1")
    parser.add_argument("--api-key-env", default="", help="Environment variable containing the API key.")
    parser.add_argument("--prompts", type=Path, default=PROMPTS)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--no-response-format", action="store_true", help="Do not request provider-side JSON mode.")
    args = parser.parse_args()

    env_name = args.api_key_env or ("GEMINI_API_KEY" if args.provider == "gemini" else "OPENAI_COMPAT_API_KEY")
    api_key = os.environ.get(env_name)
    if not api_key:
        raise SystemExit(f"Missing API key environment variable: {env_name}")
    if args.provider == "openai_compat" and not args.base_url:
        raise SystemExit("--base-url is required for --provider openai_compat")

    prompts = read_prompts(args.prompts)
    if args.limit:
        prompts = prompts[: args.limit]

    out = args.out or output_path(args.provider, args.model)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "blind_id",
        "primary_failure_label",
        "secondary_cause_labels",
        "is_true_numerical_failure",
        "evidence_quote",
        "confidence",
        "model",
        "provider",
        "elapsed_seconds",
        "error",
        "raw_response",
    ]
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for index, prompt in enumerate(prompts, start=1):
            started = time.time()
            row = {
                "blind_id": prompt["blind_id"],
                "primary_failure_label": "needs_review",
                "secondary_cause_labels": "unknown",
                "is_true_numerical_failure": "unclear",
                "evidence_quote": "",
                "confidence": "low",
                "model": args.model,
                "provider": args.provider,
                "elapsed_seconds": "0.000",
                "error": "",
                "raw_response": "",
            }
            for attempt in range(1, args.retries + 2):
                try:
                    if args.provider == "gemini":
                        content = call_gemini(
                            api_key=api_key,
                            model=args.model,
                            messages=prompt["messages"],
                            max_tokens=args.max_tokens,
                            timeout=args.timeout,
                        )
                    else:
                        content = call_openai_compat(
                            base_url=args.base_url,
                            api_key=api_key,
                            model=args.model,
                            messages=prompt["messages"],
                            max_tokens=args.max_tokens,
                            timeout=args.timeout,
                            response_format=not args.no_response_format,
                        )
                    row.update(normalize_prediction(parse_json_object(content)))
                    row["raw_response"] = content[:2000]
                    row["error"] = ""
                    break
                except Exception as exc:  # noqa: BLE001 - preserve row-level errors for auditability.
                    row["error"] = str(exc)[:1000]
                    if attempt <= args.retries:
                        time.sleep(retry_delay(str(exc), attempt))
            row["elapsed_seconds"] = f"{time.time() - started:.3f}"
            writer.writerow(row)
            fh.flush()
            print(f"{index}/{len(prompts)} {row['blind_id']} {row['primary_failure_label']} {row['error']}", file=sys.stderr)
            if args.sleep:
                time.sleep(args.sleep)
    print(out)


if __name__ == "__main__":
    main()
