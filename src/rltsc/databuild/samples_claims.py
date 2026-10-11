import argparse
import asyncio
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

from rltsc.databuild.samples_simu import append_log, index_by_sample_id, read_json, write_json, write_jsonl
from rltsc.process.prompts import PROMPTS
from rltsc.trainer.parser import STEP_DISPLAY_NAMES, parse_processlight_response
from rltsc.verify import ClaimStatus, verify_claims

STEP_NAMES = ("phase_mapping", "local_demand", "downstream", "neighbor_coordination", "flow_trend", "fairness", "final_answer")
EVIDENCE_STEPS = STEP_NAMES[:-1]
STEP_NAME_MAP = dict(zip(STEP_NAMES, STEP_DISPLAY_NAMES, strict=True))
REPAIR_SYSTEM_PROMPT = "Regenerate one complete ProcessLight response using the original traffic facts and validation errors. Return the corrected response only."
GOLDEN_SYSTEM_PROMPT = "Write a concise, complete ProcessLight traffic-signal reasoning response from verified claims. Return the response only."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--response-model")
    parser.add_argument("--base-url", default=os.environ["OPENAI_BASE_URL"] if "OPENAI_BASE_URL" in os.environ else None)
    parser.add_argument("--api-key", default=os.environ["OPENAI_API_KEY"] if "OPENAI_API_KEY" in os.environ else None)
    parser.add_argument("--concurrency", type=int, default=150)
    parser.add_argument("--sample-concurrency", type=int, default=100)
    parser.add_argument("--response-concurrency", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--repair-budget", type=int, default=8)
    parser.add_argument("--response-repair-budget", type=int, default=3)
    parser.add_argument("--max-claims-per-step", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--request-retries", type=int, default=2)
    parser.add_argument("--extra-body", type=json.loads)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if not args.api_key:
        parser.error("Set OPENAI_API_KEY or --api-key")
    for name in ("concurrency", "sample_concurrency", "response_concurrency", "max_tokens", "max_claims_per_step"):
        if getattr(args, name) < 1:
            parser.error(f"{name} must be positive")
    if any(getattr(args, name) < 0 for name in ("repair_budget", "response_repair_budget", "request_retries")) or (args.limit is not None and args.limit < 0):
        parser.error("Budgets, retries, and limit must be nonnegative")
    if not math.isfinite(args.timeout) or args.timeout <= 0 or not math.isfinite(args.temperature) or args.temperature < 0:
        parser.error("Invalid timeout or temperature")
    if args.extra_body is not None and not isinstance(args.extra_body, dict):
        parser.error("--extra-body must be an inline JSON object")
    args.response_model = args.response_model if args.response_model else args.model
    return args


def output_format(max_claims: int) -> str:
    return PROMPTS["FORMAT"].replace("no more than 4 atomic <claim>", f"no more than {max_claims} atomic <claim>")


def prompt_messages(sample: dict[str, Any], max_claims: int) -> list[dict[str, str]]:
    messages = [dict(message) for message in sample["prompt"]]
    for message in messages:
        if message["role"] == "user" and "## Output Prompt" in message["content"]:
            message["content"] = message["content"].split("## Output Prompt", 1)[0].rstrip() + "\n\n" + output_format(max_claims)
    return messages


def build_simulator_guidance(sample: dict[str, Any]) -> str:
    return f"""## Simulator-Guided Target
The configured simulation evaluation selected {sample["simulation"]["best_phase"]} as the best phase.
Reason toward this target using verifiable traffic-state evidence.
Do not place the simulator conclusion inside <claim>; claims must be supported by the original state tables."""


def guided_messages(sample: dict[str, Any], max_claims: int = 4) -> list[dict[str, str]]:
    messages = prompt_messages(sample, max_claims)
    messages.append({"role": "user", "content": build_simulator_guidance(sample) + "\n\n" + output_format(max_claims)})
    return messages


def verdict_dicts(verdicts: list[Any]) -> list[dict[str, Any]]:
    return [{"claim": verdict.claim, "status": verdict.status.value, "reason": verdict.reason} for verdict in verdicts]


def evaluate_completion(sample: dict[str, Any], completion: str, source: str, attempt_index: int, max_claims: int = 4) -> dict[str, Any]:
    parsed = parse_processlight_response(completion)
    raw_claims = {step: parsed.claims[display] for step, display in STEP_NAME_MAP.items()}
    claims = {step: [] for step in STEP_NAMES}
    unknown = {step: [] for step in STEP_NAMES}
    flat = [claim for step in EVIDENCE_STEPS for claim in raw_claims[step]]
    verdicts = verdict_dicts(verify_claims(flat, sample["prompt_state_dict"])) if flat else []
    statuses = {verdict["claim"]: verdict["status"] for verdict in verdicts}
    for step in EVIDENCE_STEPS:
        for claim in raw_claims[step]:
            if statuses[claim] == ClaimStatus.CORRECT.value:
                claims[step].append(claim)
            elif statuses[claim] == ClaimStatus.UNKNOWN.value:
                unknown[step].append(claim)
    errors = [{"status": "format", "reason": reason} for reason in parsed.format_errors]
    if parsed.phase != sample["simulation"]["best_phase"]:
        errors.append({"status": "phase", "reason": f"Expected phase {sample['simulation']['best_phase']}, got {parsed.phase}"})
    if "Final Answer" in parsed.sections and "<claim" in parsed.sections["Final Answer"].text.lower():
        errors.append({"status": "format", "reason": "Final Answer must not contain claims"})
    if not flat:
        errors.append({"status": "claim", "reason": "No nonempty claims"})
    for step in EVIDENCE_STEPS:
        if len(raw_claims[step]) > max_claims:
            errors.append({"status": "format", "reason": f"Too many claims in {step}: limit={max_claims}"})
    errors.extend(verdict for verdict in verdicts if verdict["status"] == ClaimStatus.INCORRECT.value)
    return {
        "sample_id": sample["sample_id"],
        "source": source,
        "attempt_index": attempt_index,
        "completion": completion,
        "phase": parsed.phase,
        "raw_claims": raw_claims,
        "claims": claims,
        "unknown_claims": unknown,
        "claim_verdicts": verdicts,
        "format_errors": parsed.format_errors,
        "errors": errors,
        "success": not errors,
    }


def correct_claim_fallback(sample: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any] | None:
    claims = {step: [] for step in STEP_NAMES}
    unknown = {step: [] for step in STEP_NAMES}
    best_record = None
    best_count = -1
    for record in records:
        if "phase" not in record or record["phase"] != sample["simulation"]["best_phase"]:
            continue
        count = sum(len(record["claims"][step]) for step in EVIDENCE_STEPS)
        if count > best_count:
            best_record, best_count = record, count
        for step in EVIDENCE_STEPS:
            for claim in record["claims"][step]:
                if claim not in claims[step]:
                    claims[step].append(claim)
            for claim in record["unknown_claims"][step]:
                if claim not in unknown[step]:
                    unknown[step].append(claim)
    if best_record is None or not any(claims[step] for step in EVIDENCE_STEPS):
        return None
    return {
        "sample_id": sample["sample_id"],
        "source": "correct_claim_fallback",
        "attempt_index": best_record["attempt_index"],
        "completion": best_record["completion"],
        "phase": sample["simulation"]["best_phase"],
        "claims": claims,
        "unknown_claims": unknown,
        "errors": [],
        "success": True,
    }


def build_repair_prompt(sample: dict[str, Any], record: dict[str, Any], max_claims: int = 4) -> str:
    return f"""Original traffic-signal messages:
{json.dumps(prompt_messages(sample, max_claims), ensure_ascii=False)}

{build_simulator_guidance(sample)}

Previous complete response:
{record["completion"]}

Validation errors:
{json.dumps(record["errors"], ensure_ascii=False)}

Regenerate the entire response. Preserve the required section order and use only verifiable state facts.
The final section must contain <phase>{sample["simulation"]["best_phase"]}</phase> and no <claim>.

{output_format(max_claims)}"""


async def call_model(client: Any, model: str, messages: list[dict[str, str]], args: argparse.Namespace, semaphore: asyncio.Semaphore) -> str:
    request = {"model": model, "messages": messages, "temperature": args.temperature, "max_tokens": args.max_tokens}
    if args.extra_body is not None:
        request["extra_body"] = args.extra_body
    async with semaphore:
        response = await client.chat.completions.create(**request)
    message = response.choices[0].message
    content = message.content
    if content is None or not content.strip():
        content = getattr(message, "reasoning_content", None)
    if content is None or not content.strip():
        raise ValueError("Empty LLM response")
    return content


async def generate_claims(sample: dict[str, Any], args: argparse.Namespace, client: Any, semaphore: asyncio.Semaphore) -> tuple[dict[str, Any] | None, list[dict[str, Any]], str]:
    records = []
    for attempt in range(args.repair_budget + 1):
        source = "rollout" if attempt == 0 else "repair"
        completion = ""
        try:
            messages = (
                guided_messages(sample, args.max_claims_per_step)
                if attempt == 0
                else [
                    {"role": "system", "content": REPAIR_SYSTEM_PROMPT},
                    {"role": "user", "content": build_repair_prompt(sample, records[-1], args.max_claims_per_step)},
                ]
            )
            completion = await call_model(client, args.model, messages, args, semaphore)
            record = evaluate_completion(sample, completion, source, attempt, args.max_claims_per_step)
        except Exception as exc:
            record = {
                "sample_id": sample["sample_id"],
                "source": source,
                "attempt_index": attempt,
                "completion": completion,
                "errors": [{"status": "error", "reason": str(exc)}],
                "success": False,
            }
        if attempt:
            record["input_errors"] = records[-1]["errors"]
        records.append(record)
        if record["success"]:
            return record, records, "initial_success" if attempt == 0 else "repair_success"
        if not completion:
            break
    fallback = correct_claim_fallback(sample, records)
    if fallback is not None:
        return fallback, records, "correct_claim_fallback"
    reason = "llm_error_without_completion" if not records[-1]["completion"] else "repair_budget_exhausted_without_candidate"
    return None, records, reason


def natural_language_text(text: str) -> str:
    return re.sub(r"<(?:claim|phase)>.*?</(?:claim|phase)>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()


def validate_response(response: str, sample: dict[str, Any], claims: dict[str, list[str]]) -> list[dict[str, Any]]:
    parsed = parse_processlight_response(response)
    errors = [{"status": "format", "reason": reason} for reason in parsed.format_errors]
    if parsed.phase != sample["simulation"]["best_phase"]:
        errors.append({"status": "phase", "reason": "Response does not match best phase"})
    if "Final Answer" in parsed.sections and "<claim" in parsed.sections["Final Answer"].text.lower():
        errors.append({"status": "format", "reason": "Final Answer must not contain claims"})
    for step in EVIDENCE_STEPS:
        title = STEP_NAME_MAP[step]
        if title in parsed.sections and not natural_language_text(parsed.sections[title].text):
            errors.append({"status": "format", "reason": f"Missing natural-language reasoning: {title}"})
        missing = [claim for claim in claims[step] if claim not in parsed.claims[title]]
        if missing:
            errors.append({"status": "missing_claims", "reason": f"{title}: {missing}"})
    flat = [claim for title in STEP_DISPLAY_NAMES[:-1] for claim in parsed.claims[title]]
    errors.extend(verdict for verdict in verdict_dicts(verify_claims(flat, sample["prompt_state_dict"])) if verdict["status"] != ClaimStatus.CORRECT.value)
    return errors


def build_golden_prompt(sample: dict[str, Any], claims: dict[str, list[str]], max_claims: int) -> str:
    return f"""Original traffic-signal messages:
{json.dumps(prompt_messages(sample, max_claims), ensure_ascii=False)}

Best phase: {sample["simulation"]["best_phase"]}
Verified claims by step:
{json.dumps(claims, ensure_ascii=False)}

Write all seven sections in order: {", ".join(STEP_DISPLAY_NAMES)}.
Include every supplied claim verbatim in its corresponding section, wrapped in <claim>...</claim>.
Each evidence section must also contain 1-2 concise natural-language reasoning sentences outside claim tags.
Use <claim></claim> for a step with no claims. Additional claims must be verifiable from the original state.
The Final Answer section must contain <phase>{sample["simulation"]["best_phase"]}</phase> and no <claim>.
Return the complete response only."""


async def generate_response(
    sample: dict[str, Any], claims: dict[str, list[str]], args: argparse.Namespace, client: Any, semaphore: asyncio.Semaphore
) -> tuple[str | None, list[dict[str, Any]], str]:
    flat = [claim for step in EVIDENCE_STEPS for claim in claims[step]]
    if not flat or any(verdict.status != ClaimStatus.CORRECT for verdict in verify_claims(flat, sample["prompt_state_dict"])):
        return None, [], "No verified input claims"
    records = []
    prompt = build_golden_prompt(sample, claims, args.max_claims_per_step)
    for attempt in range(args.response_repair_budget + 1):
        response = ""
        try:
            content = prompt
            if attempt:
                content += f"\n\nPrevious response:\n{records[-1]['response']}\n\nValidation errors:\n{json.dumps(records[-1]['errors'], ensure_ascii=False)}\nRegenerate the complete response."
            response = await call_model(client, args.response_model, [{"role": "system", "content": GOLDEN_SYSTEM_PROMPT}, {"role": "user", "content": content}], args, semaphore)
            errors = validate_response(response, sample, claims)
        except Exception as exc:
            errors = [{"status": "error", "reason": str(exc)}]
        records.append({"sample_id": sample["sample_id"], "attempt_index": attempt, "response": response, "errors": errors, "success": not errors})
        if not errors:
            return response, records, "initial" if attempt == 0 else "repair"
        if not response:
            return None, records, "LLM error without response"
    sections = []
    for step, title in STEP_NAME_MAP.items():
        sections.append(f"## {title}")
        if step == "final_answer":
            sections.append(f"Select the verified best phase.\n<phase>{sample['simulation']['best_phase']}</phase>")
        else:
            sections.append("Use the verified traffic facts for this decision step.")
            sections.extend([f"<claim>{claim}</claim>" for claim in claims[step]] if claims[step] else ["<claim></claim>"])
    response = "\n".join(sections)
    errors = validate_response(response, sample, claims)
    return (None, records, json.dumps(errors)) if errors else (response, records, "fallback")


async def process_sample(
    sample: dict[str, Any], args: argparse.Namespace, client: Any, claim_semaphore: asyncio.Semaphore, response_semaphore: asyncio.Semaphore
) -> dict[str, Any]:
    best, records, reason = await generate_claims(sample, args, client, claim_semaphore)
    result = {
        "sample_id": sample["sample_id"],
        "claims": None,
        "rejected": None,
        "unresolved": None,
        "sft": None,
        "golden": None,
        "golden_failed": None,
        "records": records,
        "golden_records": [],
    }
    if best is None:
        failure = {"sample_id": sample["sample_id"], "reject_reason": reason, "completion": records[-1]["completion"], "errors": records[-1]["errors"]}
        result["rejected"] = failure
        result["unresolved"] = dict(failure)
        return result
    result["claims"] = {
        "sample_id": sample["sample_id"],
        "response": best["completion"],
        "claim_source": best["source"],
        "claims": best["claims"],
        "unknown_claims": best["unknown_claims"],
        "llm_call_count": len(records),
        "repair_count": len(records) - 1,
        "rollout_stop_reason": reason,
    }
    response, golden_records, golden_source = await generate_response(sample, best["claims"], args, client, response_semaphore)
    result["golden_records"] = golden_records
    if response is None:
        result["golden_failed"] = {"sample_id": sample["sample_id"], "fail_reason": golden_source, "attempts": golden_records}
    else:
        result["golden"] = {"sample_id": sample["sample_id"], "response": response, "source": golden_source}
        result["sft"] = {
            "data_source": "processlight",
            "prompt": prompt_messages(sample, args.max_claims_per_step),
            "response": response,
            "extra_info": {"sample_id": sample["sample_id"], "best_phase": sample["simulation"]["best_phase"]},
        }
    return result


def write_stage_outputs(output: Path, results: list[dict[str, Any]]) -> None:
    for field, filename in (
        ("claims", "claims_samples.json"),
        ("rejected", "claims_rejected.json"),
        ("unresolved", "claims_unresolved.json"),
        ("sft", "sft_samples.json"),
        ("golden_failed", "golden_failed.json"),
    ):
        write_json(output / filename, [result[field] for result in results if result[field] is not None])
    golden = [result["golden"] for result in results if result["golden"] is not None]
    write_json(output / "importance_data.json", [{"sample_id": row["sample_id"], "response": row["response"]} for row in golden])
    write_jsonl(output / "claims_rollouts.jsonl", [record for result in results for record in result["records"] if record["source"] == "rollout"])
    write_jsonl(output / "claims_repairs.jsonl", [record for result in results for record in result["records"] if record["source"] == "repair"])
    write_jsonl(output / "golden_samples.jsonl", golden)
    write_jsonl(output / "golden_repairs.jsonl", [record for result in results for record in result["golden_records"] if record["attempt_index"] > 0])


async def run_all(samples: list[dict[str, Any]], args: argparse.Namespace, output: Path) -> list[dict[str, Any]]:
    import httpx
    from openai import AsyncOpenAI

    log = output / "claims_log.txt"
    log.write_text("", encoding="utf-8")
    results = []
    write_stage_outputs(output, results)
    if not samples:
        return results
    claim_semaphore = asyncio.Semaphore(args.concurrency)
    response_semaphore = asyncio.Semaphore(args.response_concurrency)
    sample_semaphore = asyncio.Semaphore(args.sample_concurrency)
    http_client = httpx.AsyncClient(timeout=args.timeout, trust_env=False)
    async with AsyncOpenAI(api_key=args.api_key, base_url=args.base_url, timeout=args.timeout, max_retries=args.request_retries, http_client=http_client) as client:

        async def run(sample: dict[str, Any]) -> dict[str, Any]:
            async with sample_semaphore:
                return await process_sample(sample, args, client, claim_semaphore, response_semaphore)

        started = time.monotonic()
        tasks = [asyncio.create_task(run(sample)) for sample in samples]
        for task in asyncio.as_completed(tasks):
            result = await task
            results.append(result)
            write_stage_outputs(output, results)
            message = f"completed={len(results)}/{len(samples)} sample={result['sample_id']} claims={result['claims'] is not None} sft={result['sft'] is not None} elapsed={time.monotonic() - started:.1f}s"
            append_log(log, message)
            print(message, flush=True)
    return results


def main() -> None:
    args = parse_args()
    samples = read_json(args.input)
    if not isinstance(samples, list):
        raise ValueError("Input must be a JSON array")
    samples = samples[: args.limit] if args.limit is not None else samples
    index_by_sample_id(samples, "claims input")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    results = asyncio.run(run_all(samples, args, output))
    kept = sum(result["claims"] is not None for result in results)
    sft = sum(result["sft"] is not None for result in results)
    print(f"kept={kept} rejected={len(results) - kept} sft={sft} output={output}")


if __name__ == "__main__":
    main()
