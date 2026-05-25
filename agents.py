import re
import time
import ollama as ollama_client
from groq import Groq
from typing import Optional
from config import (
    GROQ_API_KEY,
    GROQ_AGENTS,
    OLLAMA_AGENTS,
    REQUEST_TIMEOUT_SEC,
    MAX_RETRIES,
    RETRY_BACKOFF_SEC,
    RATE_LIMIT_MAX_SLEEP_SEC,
    STOP_ON_RATE_LIMIT,
    MAX_TOKENS_AGENT,
)

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

try:
    ollama_client_obj = ollama_client.Client(timeout=REQUEST_TIMEOUT_SEC)
except Exception:
    ollama_client_obj = None


def _call_with_retries(call_fn, agent_name: str):
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return call_fn()
        except Exception as exc:
            last_error = exc
            if attempt < MAX_RETRIES:
                err_text = str(exc)
                wait_sec = RETRY_BACKOFF_SEC * attempt
                rate_limit_match = re.search(
                    r"Please try again in (?:(\d+)m)?([0-9]+(?:\.[0-9]+)?)s",
                    err_text,
                )
                if rate_limit_match:
                    mins = int(rate_limit_match.group(1) or 0)
                    secs = float(rate_limit_match.group(2))
                    parsed_wait = mins * 60 + secs + 1.0
                    wait_sec = max(wait_sec, min(parsed_wait, RATE_LIMIT_MAX_SLEEP_SEC))
                print(f"\n  [WARN] {agent_name}: attempt {attempt}/{MAX_RETRIES} failed ({exc}); retrying in {wait_sec:.1f}s")
                time.sleep(wait_sec)
    raise last_error


def _normalize_mcq_token(token: Optional[str],
                         mcq_num_choices: Optional[int] = None) -> Optional[str]:
    if token is None:
        return None
    n = int(mcq_num_choices or 4)
    n = max(2, min(n, 26))

    allowed = [chr(ord("A") + i) for i in range(n)]
    token = str(token).strip().upper()
    if token in allowed:
        return token

    try:
        value = float(token)
        if not value.is_integer():
            return None
        idx = int(value)
    except Exception:
        return None

    if 1 <= idx <= n:
        return allowed[idx - 1]
    if 0 <= idx < n:
        return allowed[idx]
    return None


def parse_answer(raw_text: str,
                 expected_format: Optional[str] = None,
                 mcq_num_choices: Optional[int] = None) -> Optional[str]:
    """
    Try multiple patterns to extract the answer.
    1. Look for ANSWER: <value>
    2. Look for last number in text
    3. Look for A/B/C/D letter
    """
    # Pattern 1: explicit ANSWER: tag
    match = re.search(r'ANSWER:\s*([A-Za-z]|\-?\d+\.?\d*)', raw_text)
    if match:
        token = match.group(1).strip().upper()
        if expected_format == "mcq":
            return _normalize_mcq_token(token, mcq_num_choices=mcq_num_choices)
        return token

    # Pattern 2: last number in the text
    numbers = re.findall(r'\-?\d+\.?\d*', raw_text)
    if numbers:
        if expected_format == "mcq":
            normalized = _normalize_mcq_token(
                numbers[-1],
                mcq_num_choices=mcq_num_choices,
            )
            if normalized is not None:
                return normalized
        return numbers[-1]

    # Pattern 3: standalone option letter
    match = re.search(r'\b([A-Z])\b', raw_text.upper())
    if match:
        token = match.group(1).upper()
        if expected_format == "mcq":
            return _normalize_mcq_token(token, mcq_num_choices=mcq_num_choices)
        return token

    return None


def ask_agent(agent_name: str,
              question: str,
              ask_confidence: bool = False,
              expected_format: Optional[str] = None,
              mcq_num_choices: Optional[int] = None) -> dict:
    if ask_confidence:
        prompt = (
            f"Answer the following question carefully.\n"
            f"At the end of your response write exactly:\n"
            f"ANSWER: <your final answer, number or letter only>\n"
            f"CONFIDENCE: <integer 0-100>\n\n"
            f"Question: {question}"
        )
    else:
        answer_schema = "<your final answer, number or letter only>"
        if expected_format == "mcq":
            n = int(mcq_num_choices or 4)
            n = max(2, min(n, 26))
            last_letter = chr(ord("A") + n - 1)
            answer_schema = f"<one letter only: A to {last_letter}>"
        prompt = (
            f"Answer the following question carefully.\n"
            f"At the end write exactly:\n"
            f"ANSWER: {answer_schema}\n\n"
            f"Question: {question}"
        )

    raw_text = ""
    try:
        if agent_name in GROQ_AGENTS:
            model = GROQ_AGENTS[agent_name]

            if groq_client is None:
                raise RuntimeError("GROQ_API_KEY is not set. Export GROQ_API_KEY before running.")

            def groq_once():
                req = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                    "max_tokens": MAX_TOKENS_AGENT,
                }
                try:
                    return groq_client.chat.completions.create(
                        timeout=REQUEST_TIMEOUT_SEC,
                        **req,
                    )
                except TypeError:
                    return groq_client.chat.completions.create(**req)

            response = _call_with_retries(groq_once, agent_name)
            raw_text = response.choices[0].message.content

        elif agent_name in OLLAMA_AGENTS:
            model = OLLAMA_AGENTS[agent_name]

            def ollama_once():
                if ollama_client_obj is not None:
                    return ollama_client_obj.chat(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        options={"temperature": 0.0},
                    )
                return ollama_client.chat(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": 0.0},
                )

            response = _call_with_retries(ollama_once, agent_name)
            raw_text = response["message"]["content"]
        else:
            raise ValueError(f"Unknown agent name: {agent_name}")

    except Exception as e:
        err_text = str(e).lower()
        if STOP_ON_RATE_LIMIT and (
            "rate_limit_exceeded" in err_text
            or "rate limit reached" in err_text
        ):
            raise
        print(f"\n  [ERROR] {agent_name}: {e}")
        time.sleep(3)
        return {"answer": None, "confidence": 0.5, "raw": ""}

    answer = parse_answer(
        raw_text,
        expected_format=expected_format,
        mcq_num_choices=mcq_num_choices,
    )

    confidence = 0.5
    if ask_confidence:
        conf_match = re.search(r'CONFIDENCE:\s*(\d+)', raw_text)
        if conf_match:
            confidence = min(int(conf_match.group(1)), 100) / 100.0

    return {"answer": answer, "confidence": confidence, "raw": raw_text}


def extract_gsm8k_answer(solution: str) -> str:
    match = re.search(r'####\s*(\-?\d[\d,]*)', solution)
    if match:
        return match.group(1).replace(",", "")
    return None


def extract_mcq_answer(answer) -> str:
    if isinstance(answer, int):
        return ["A", "B", "C", "D"][answer] if answer < 4 else None
    if isinstance(answer, str):
        match = re.search(r'[A-Da-d]', answer)
        return match.group(0).upper() if match else None
    return None
