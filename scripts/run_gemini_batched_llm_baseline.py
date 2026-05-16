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
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def chunks(rows: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [rows[index : index + size] for index in range(0, len(rows), size)]


def retry_delay(error: str, attempt: int) -> float:
    if "Please retry in" in error:
        try:
            after = error.split("Please retry in", 1)[1].split("s", 1)[0].strip()
            return max(float(after) + 2.0, 2.0)
        except (IndexError, ValueError):
            pass
    if "retry_after_seconds" in error:
        try:
            after = error.split('"retry_after_seconds":', 1)[1].split(",", 1)[0].split("}", 1)[0]
            return max(float(after) + 2.0, 2.0)
        except (IndexError, ValueError):
            pass
    return min(60.0, 5.0 * attempt)


def post_gemini(api_key: str, model: str, prompt: str, max_tokens: int, timeout: int) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": max_tokens,
            "responseMimeType": "application/json",
        },
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')[:1800]}") from exc
    try:
        return body["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as exc:
        raise RuntimeError(f"Gemini response missing text: {json.dumps(body)[:1800]}") from exc


def parse_json(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            obj = json.loads(text[start : end + 1])
            return obj.get("predictions", obj)
        raise


def normalize(raw: dict[str, Any], blind_id: str, model: str, elapsed: float, error: str = "", raw_response: str = "") -> dict[str, str]:
    primary = str(raw.get("primary_failure_label", "needs_review")).strip()
    if primary not in PRIMARY_LABELS:
        primary = "needs_review"
    secondary_raw = raw.get("secondary_cause_labels", ["unknown"])
    if isinstance(secondary_raw, str):
        secondary = [part.strip() for part in secondary_raw.replace(",", "|").split("|") if part.strip()]
    else:
        secondary = [str(part).strip() for part in secondary_raw if str(part).strip()]
    secondary = [label for label in secondary if label in SECONDARY_LABELS] or ["unknown"]
    true_raw = raw.get("is_true_numerical_failure", "unclear")
    if isinstance(true_raw, bool):
        true_failure = "yes" if true_raw else "no"
    else:
        true_failure = str(true_raw).strip().lower()
    if true_failure not in {"yes", "no", "unclear"}:
        true_failure = "unclear"
    confidence_raw = raw.get("confidence", "low")
    if isinstance(confidence_raw, (int, float)):
        confidence = "high" if confidence_raw >= 0.75 else "medium" if confidence_raw >= 0.4 else "low"
    else:
        confidence = str(confidence_raw).strip().lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "low"
    return {
        "blind_id": blind_id,
        "primary_failure_label": primary,
        "secondary_cause_labels": "|".join(secondary),
        "is_true_numerical_failure": true_failure,
        "evidence_quote": str(raw.get("evidence_quote", ""))[:500],
        "confidence": confidence,
        "model": model,
        "provider": "gemini_batched",
        "elapsed_seconds": f"{elapsed:.3f}",
        "error": error,
        "raw_response": raw_response[:2000],
    }


def batch_prompt(batch: list[dict[str, Any]]) -> str:
    allowed = (
        "Allowed primary labels: nan_inf, overflow_underflow, precision_tolerance, dtype_casting, "
        "crash_compile, performance_only, not_numerical_failure, needs_review\n"
        "Allowed secondary cause labels: memory_mask_bounds, compiler_codegen, async_race_ordering, "
        "hardware_backend, reduction_accumulation, api_semantics, environment_configuration, unknown\n"
    )
    rows = []
    for item in batch:
        user = "\n".join(message["content"] for message in item["messages"] if message["role"] == "user")
        rows.append(f"### ROW {item['blind_id']}\n{user}")
    return (
        "You classify public GPU/kernel issue reports for a research benchmark. "
        "Use only the supplied text. Return JSON only: an array of objects, one per input row. "
        "Each object must include blind_id, primary_failure_label, secondary_cause_labels, "
        "is_true_numerical_failure, evidence_quote, and confidence. Do not include markdown.\n\n"
        f"{allowed}\n\n"
        + "\n\n".join(rows)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GPU-NFBench prompts against Gemini in multi-row batches.")
    parser.add_argument("--model", default="gemini-3-flash-preview")
    parser.add_argument("--api-key-env", default="GEMINI_API_KEY")
    parser.add_argument("--prompts", type=Path, default=PROMPTS)
    parser.add_argument("--out", type=Path, default=ROOT / "evaluation" / "llm_baseline_predictions_gemini_batched.csv")
    parser.add_argument("--start", type=int, default=0, help="Zero-based row offset in the prompt packet.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--sleep", type=float, default=2.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Missing API key environment variable: {args.api_key_env}")

    prompts = read_prompts(args.prompts)
    if args.start:
        prompts = prompts[args.start :]
    if args.limit:
        prompts = prompts[: args.limit]
    args.out.parent.mkdir(parents=True, exist_ok=True)
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
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for batch_index, batch in enumerate(chunks(prompts, args.batch_size), start=1):
            started = time.time()
            text = ""
            error = ""
            for attempt in range(1, args.retries + 2):
                try:
                    text = post_gemini(api_key, args.model, batch_prompt(batch), args.max_tokens, args.timeout)
                    error = ""
                    break
                except Exception as exc:  # noqa: BLE001
                    error = str(exc)
                    if attempt <= args.retries:
                        time.sleep(retry_delay(error, attempt))
            elapsed = time.time() - started
            predictions: dict[str, dict[str, Any]] = {}
            if not error:
                try:
                    parsed = parse_json(text)
                    if isinstance(parsed, dict):
                        parsed = parsed.get("predictions", [])
                    predictions = {
                        str(row.get("blind_id", "")): row
                        for row in parsed
                        if isinstance(row, dict) and row.get("blind_id")
                    }
                except Exception as exc:  # noqa: BLE001
                    error = f"parse_error: {exc}"
            for item in batch:
                blind_id = item["blind_id"]
                row = normalize(
                    predictions.get(blind_id, {}),
                    blind_id,
                    args.model,
                    elapsed,
                    error=error[:1000] if blind_id not in predictions else "",
                    raw_response=text,
                )
                writer.writerow(row)
            fh.flush()
            print(f"batch {batch_index} rows={len(batch)} predictions={len(predictions)} error={error[:160]}", file=sys.stderr)
            if args.sleep:
                time.sleep(args.sleep)
    print(args.out)


if __name__ == "__main__":
    main()
