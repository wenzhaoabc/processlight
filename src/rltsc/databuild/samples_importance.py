import argparse
import asyncio
import math
import os
import time
from pathlib import Path
from typing import Any

from rltsc.databuild.samples_claims import EVIDENCE_STEPS, STEP_NAME_MAP, STEP_NAMES
from rltsc.databuild.samples_simu import PHASE_ORDER, append_log, index_by_sample_id, read_json, write_json
from rltsc.processlight_contract import attach_simulation_to_prompt_state, validate_training_prompt_state
from rltsc.trainer.parser import STEP_DISPLAY_NAMES, parse_processlight_response

PHASE_LABELS = ("A", "B", "C", "D")
PHASE_TOKEN_MAPPING = dict(zip(PHASE_LABELS, PHASE_ORDER, strict=True))
IMPORTANCE_STEPS = EVIDENCE_STEPS
IMPORTANCE_LLM_CONCURRENCY = 128
IMPORTANCE_LOGPROBS = 8


class ImportanceSampleError(ValueError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--origin", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--claims", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default=os.environ["LOCAL_MODEL_NAME"] if "LOCAL_MODEL_NAME" in os.environ else None)
    parser.add_argument("--base-url", default=os.environ["LOCAL_BASE_URL"] if "LOCAL_BASE_URL" in os.environ else None)
    parser.add_argument("--api-key", default=os.environ["LOCAL_API_KEY"] if "LOCAL_API_KEY" in os.environ else "EMPTY")
    parser.add_argument("--concurrency", type=int, default=IMPORTANCE_LLM_CONCURRENCY)
    parser.add_argument("--logprobs", type=int, default=IMPORTANCE_LOGPROBS)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if not args.model or not args.base_url:
        parser.error("Set --model and --base-url, or LOCAL_MODEL_NAME and LOCAL_BASE_URL")
    if args.concurrency < 1 or args.logprobs < 1 or not math.isfinite(args.timeout) or args.timeout <= 0 or (args.limit is not None and args.limit < 0):
        parser.error("Invalid concurrency, logprobs, timeout, or limit")
    args.base_url = args.base_url.rstrip("/")
    if not args.base_url.endswith("/v1"):
        args.base_url += "/v1"
    return args


def tokenize_base_url(base_url: str) -> str:
    return base_url[:-3] if base_url.endswith("/v1") else base_url


def kl_divergence(left: dict[str, float], right: dict[str, float]) -> float:
    epsilon = 1e-12
    return sum(left[label] * math.log((left[label] + epsilon) / (right[label] + epsilon)) for label in PHASE_LABELS)


def jsd(left: dict[str, float], right: dict[str, float]) -> float:
    middle = {label: (left[label] + right[label]) / 2 for label in PHASE_LABELS}
    return max(0.0, (kl_divergence(left, middle) + kl_divergence(right, middle)) / 2)


def softmax_from_logprobs(logprobs: dict[str, float]) -> dict[str, float]:
    if set(logprobs) != set(PHASE_LABELS) or not all(math.isfinite(value) for value in logprobs.values()):
        raise ValueError("Expected finite logprobs for A/B/C/D")
    maximum = max(logprobs.values())
    values = {label: math.exp(logprobs[label] - maximum) for label in PHASE_LABELS}
    total = sum(values.values())
    return {label: values[label] / total for label in PHASE_LABELS}


def normalize_importance(scores: dict[str, float]) -> dict[str, float]:
    if set(scores) != set(IMPORTANCE_STEPS) or any(not math.isfinite(value) or value < 0 for value in scores.values()):
        raise ValueError("Expected finite nonnegative scores for all six evidence steps")
    total = sum(scores.values())
    if total <= 1e-12:
        return {step: 1.0 / len(scores) for step in scores}
    return {step: scores[step] / total for step in scores}


def build_importance_prompt(sample: dict[str, Any], origin: dict[str, Any], masked_step: str | None = None) -> str:
    parsed = parse_processlight_response(sample["response"])
    evidence = []
    for step in IMPORTANCE_STEPS:
        title = STEP_NAME_MAP[step]
        if title not in parsed.sections:
            raise ImportanceSampleError(f"Missing response section: {title}")
        if step != masked_step:
            evidence.append(f"## {title}\n{parsed.sections[title].text}")
    state_messages = [message["content"] for message in origin["prompt"] if message["role"] == "user"]
    if not state_messages:
        raise ImportanceSampleError("Missing traffic-state user prompt")
    sections = ["Traffic signal state:", "\n\n".join(state_messages), "", "Use A/B/C/D as internal phase labels:"]
    sections.extend(f"{label} = {phase}" for label, phase in PHASE_TOKEN_MAPPING.items())
    sections.extend(["", "Use the following reasoning response evidence to predict the final phase label."])
    for text in evidence:
        sections.extend(["", text])
    sections.extend(["", "Complete the final phase label only.", "## Final Answer", "<phase>"])
    return "\n".join(sections)


async def tokenize_one(client: Any, base_url: str, model: str, text: str) -> int:
    response = await client.post(f"{tokenize_base_url(base_url)}/tokenize", json={"model": model, "prompt": text})
    response.raise_for_status()
    tokens = response.json()["tokens"]
    if not isinstance(tokens, list) or not tokens or len(tokens) > 2:
        raise ValueError(f"Phase label must be a single token: {text!r}, tokens={tokens}")
    if len(tokens) == 2 and tokens[0] not in {0, 1, 2, 128000}:
        raise ValueError(f"Unexpected leading token: {text!r}, tokens={tokens}")
    return int(tokens[-1])


async def resolve_phase_tokens(client: Any, base_url: str, model: str) -> tuple[dict[str, str], dict[str, int]]:
    import httpx

    texts = {label: f" {label}" for label in PHASE_LABELS}
    tokens = {}
    for label in PHASE_LABELS:
        try:
            tokens[label] = await tokenize_one(client, base_url, model, texts[label])
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 404:
                raise
            return texts, {label: -1 for label in PHASE_LABELS}
    return texts, tokens


def prompt_token_logprob(prompt_logprobs: list[Any], token_id: int, token_text: str) -> float:
    if not prompt_logprobs or not isinstance(prompt_logprobs[-1], dict):
        raise ValueError("Missing final prompt token logprobs")
    entry = prompt_logprobs[-1]
    token_key = str(token_id)
    if token_key in entry:
        return float(entry[token_key]["logprob"])
    for item in entry.values():
        if item["decoded_token"] == token_text:
            return float(item["logprob"])
    raise ValueError(f"Missing prompt logprob for {token_text!r}, token_id={token_id}")


async def phase_distribution(
    client: Any, base_url: str, model: str, prompt: str, token_texts: dict[str, str], token_ids: dict[str, int], logprobs: int, semaphore: asyncio.Semaphore
) -> dict[str, float]:
    async def score(label: str) -> tuple[str, float]:
        body = {"model": model, "prompt": prompt + token_texts[label], "max_tokens": 1, "temperature": 0, "prompt_logprobs": 1, "logprobs": logprobs}
        async with semaphore:
            response = await client.post(f"{base_url}/completions", json=body)
        response.raise_for_status()
        values = response.json()["choices"][0]["prompt_logprobs"]
        return label, prompt_token_logprob(values, token_ids[label], token_texts[label])

    return softmax_from_logprobs(dict(await asyncio.gather(*(score(label) for label in PHASE_LABELS))))


async def process_sample(
    sample: dict[str, Any],
    origin: dict[str, Any],
    client: Any,
    base_url: str,
    model: str,
    token_texts: dict[str, str],
    token_ids: dict[str, int],
    logprobs: int,
    semaphore: asyncio.Semaphore,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    try:
        full_prompt = build_importance_prompt(sample, origin)
        masked_prompts = {step: build_importance_prompt(sample, origin, step) for step in IMPORTANCE_STEPS}
    except ImportanceSampleError as exc:
        return None, {"sample_id": sample["sample_id"], "reject_reason": str(exc)}
    full = await phase_distribution(client, base_url, model, full_prompt, token_texts, token_ids, logprobs, semaphore)

    async def masked(step: str) -> tuple[str, dict[str, float]]:
        return step, await phase_distribution(client, base_url, model, masked_prompts[step], token_texts, token_ids, logprobs, semaphore)

    masks = dict(await asyncio.gather(*(masked(step) for step in IMPORTANCE_STEPS)))
    return {
        "sample_id": sample["sample_id"],
        "origin_probs": full,
        "masked_probs": masks,
        "step_importance": normalize_importance({step: jsd(full, masks[step]) for step in IMPORTANCE_STEPS}),
    }, None


async def run_importance(
    samples: list[dict[str, Any]],
    origin_by_id: dict[str, dict[str, Any]],
    model: str,
    base_url: str,
    api_key: str,
    log_path: Path,
    *,
    concurrency: int = IMPORTANCE_LLM_CONCURRENCY,
    logprobs: int = IMPORTANCE_LOGPROBS,
    timeout: float = 120.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import httpx

    kept, rejected = [], []
    write_json(log_path.parent / "importance_samples.json", kept)
    write_json(log_path.parent / "importance_rejected.json", rejected)
    if not samples:
        return kept, rejected
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(headers={"Authorization": f"Bearer {api_key}"}, timeout=timeout, trust_env=False, limits=limits) as client:
        try:
            token_texts, token_ids = await resolve_phase_tokens(client, base_url, model)
        except Exception as exc:
            rejected = [{"sample_id": sample["sample_id"], "reject_reason": f"Phase token resolution failed: {exc}"} for sample in samples]
            write_json(log_path.parent / "importance_rejected.json", rejected)
            append_log(log_path, f"rejected={len(rejected)} reason=phase_token_resolution")
            return kept, rejected
        semaphore = asyncio.Semaphore(concurrency)
        sample_semaphore = asyncio.Semaphore(concurrency)

        async def run(sample: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
            async with sample_semaphore:
                try:
                    return await process_sample(sample, origin_by_id[sample["sample_id"]], client, base_url, model, token_texts, token_ids, logprobs, semaphore)
                except Exception as exc:
                    return None, {"sample_id": sample["sample_id"], "reject_reason": str(exc)}

        started = time.monotonic()
        tasks = [asyncio.create_task(run(sample)) for sample in samples]
        for task in asyncio.as_completed(tasks):
            row, failure = await task
            if row is not None:
                kept.append(row)
            if failure is not None:
                rejected.append(failure)
            write_json(log_path.parent / "importance_samples.json", kept)
            write_json(log_path.parent / "importance_rejected.json", rejected)
            message = f"completed={len(kept) + len(rejected)}/{len(samples)} kept={len(kept)} rejected={len(rejected)} elapsed={time.monotonic() - started:.1f}s"
            append_log(log_path, message)
            print(message, flush=True)
    return kept, rejected


def build_grpo_sample(origin: dict[str, Any], claims: dict[str, Any], importance: dict[str, Any]) -> dict[str, Any]:
    simulation = origin["simulation"]
    returns = simulation["phase_return"]
    if set(returns) != set(PHASE_ORDER) or not all(math.isfinite(float(value)) for value in returns.values()) or simulation["best_phase"] not in PHASE_ORDER:
        raise ValueError("Invalid simulation phase returns")
    weights = importance["step_importance"]
    if set(weights) != set(IMPORTANCE_STEPS) or any(not math.isfinite(float(value)) or float(value) < 0 for value in weights.values()) or abs(sum(weights.values()) - 1.0) > 1e-3:
        raise ValueError("Invalid step importance")
    state = attach_simulation_to_prompt_state(origin["prompt_state_dict"], simulation)
    validate_training_prompt_state(state)
    return {
        "data_source": "processlight",
        "prompt": origin["prompt"],
        "solution": {"best_phase": simulation["best_phase"], "phase_return": dict(returns)},
        "extra_info": {
            "sample_id": origin["sample_id"],
            "step_names": list(STEP_DISPLAY_NAMES),
            "prompt_state_dict": state,
            "oracle_claims": {STEP_NAME_MAP[step]: claims["claims"][step] for step in STEP_NAMES},
            "unknown_claims": {STEP_NAME_MAP[step]: claims["unknown_claims"][step] for step in STEP_NAMES},
            "step_importance": {STEP_NAME_MAP[step]: weights[step] for step in IMPORTANCE_STEPS},
        },
    }


def merge_grpo(
    origins: dict[str, dict[str, Any]], claims: list[dict[str, Any]], importance: list[dict[str, Any]], output: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    importance_by_id = index_by_sample_id(importance, "importance results")
    kept, rejected = [], []
    write_json(output / "grpo_samples.json", kept)
    write_json(output / "grpo_rejected.json", rejected)
    for row in claims:
        sample_id = row["sample_id"]
        try:
            kept.append(build_grpo_sample(origins[sample_id], row, importance_by_id[sample_id]))
        except Exception as exc:
            rejected.append({"sample_id": sample_id, "reject_reason": str(exc)})
        write_json(output / "grpo_samples.json", kept)
        write_json(output / "grpo_rejected.json", rejected)
    return kept, rejected


def main() -> None:
    args = parse_args()
    origins = index_by_sample_id(read_json(args.origin), "origin input")
    samples = read_json(args.input)
    claims = read_json(args.claims)
    index_by_sample_id(samples, "importance input")
    index_by_sample_id(claims, "claims input")
    if args.limit is not None:
        samples = samples[: args.limit]
        selected = {row["sample_id"] for row in samples}
        claims = [row for row in claims if row["sample_id"] in selected]
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    log = output / "importance_log.txt"
    log.write_text("", encoding="utf-8")
    kept, rejected = asyncio.run(
        run_importance(samples, origins, args.model, args.base_url, args.api_key, log, concurrency=args.concurrency, logprobs=args.logprobs, timeout=args.timeout)
    )
    grpo, grpo_rejected = merge_grpo(origins, claims, kept, output)
    append_log(log, f"importance={len(kept)} rejected={len(rejected)} grpo={len(grpo)} grpo_rejected={len(grpo_rejected)}")
    print(f"importance={len(kept)} rejected={len(rejected)} grpo={len(grpo)} grpo_rejected={len(grpo_rejected)} output={output}")


if __name__ == "__main__":
    main()
