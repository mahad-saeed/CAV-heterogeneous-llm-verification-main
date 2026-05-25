import os


# API credentials (set in your shell/environment)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

# Runtime reliability / reproducibility settings
REQUEST_TIMEOUT_SEC = float(os.getenv("REQUEST_TIMEOUT_SEC", "45"))
MAX_RETRIES         = int(os.getenv("MAX_RETRIES", "3"))
RETRY_BACKOFF_SEC   = float(os.getenv("RETRY_BACKOFF_SEC", "1.5"))
RATE_LIMIT_MAX_SLEEP_SEC = float(os.getenv("RATE_LIMIT_MAX_SLEEP_SEC", "420"))
STOP_ON_RATE_LIMIT = os.getenv("STOP_ON_RATE_LIMIT", "1") == "1"
RANDOM_SEED         = int(os.getenv("RANDOM_SEED", "42"))
CALIBRATION_BINS    = int(os.getenv("CALIBRATION_BINS", "10"))
STRICT_HETEROGENEITY = os.getenv("STRICT_HETEROGENEITY", "0") == "1"
MIN_UNIQUE_MODELS    = int(os.getenv("MIN_UNIQUE_MODELS", "4"))
PARAPHRASE_MODEL     = os.getenv("PARAPHRASE_MODEL", "llama-3.1-8b-instant")
MAX_TOKENS_AGENT     = int(os.getenv("MAX_TOKENS_AGENT", "512"))
MAX_TOKENS_PARAPHRASE = int(os.getenv("MAX_TOKENS_PARAPHRASE", "512"))
STRICT_BENCHMARK = os.getenv("STRICT_BENCHMARK", "1") == "1"
MAX_AGENT_ANSWER_RETRIES = int(os.getenv("MAX_AGENT_ANSWER_RETRIES", "3"))
MAX_MISSING_AGENTS_PER_QUESTION = int(
    os.getenv("MAX_MISSING_AGENTS_PER_QUESTION", "1")
)
FAIL_ON_MISSING_AGENTS = os.getenv("FAIL_ON_MISSING_AGENTS", "0") == "1"

# Toggle local inference. Default is API-only to reduce laptop load.
USE_LOCAL_OLLAMA = os.getenv("USE_LOCAL_OLLAMA", "0") == "1"


def _validate_model_diversity(agent_map: dict, group_name: str) -> None:
    unique_models = {m for m in agent_map.values() if isinstance(m, str) and m.strip()}
    if len(unique_models) >= MIN_UNIQUE_MODELS:
        return

    msg = (
        f"{group_name} has only {len(unique_models)} unique model(s). "
        f"Set MODEL_* vars to at least {MIN_UNIQUE_MODELS} unique models "
        f"for stronger heterogeneous claims."
    )
    if STRICT_HETEROGENEITY:
        raise ValueError(msg)
    print(f"[WARN] {msg}")

# Agent definitions
# In API-only mode (default), all tiers are called via Groq API.
# Override model names using env vars MODEL_VERY_STRONG, MODEL_STRONG, etc.
if not USE_LOCAL_OLLAMA:
    GROQ_AGENTS = {
        "very_strong": os.getenv("MODEL_VERY_STRONG", "openai/gpt-oss-120b"),
        "strong":      os.getenv("MODEL_STRONG", "llama-3.3-70b-versatile"),
        "medium":      os.getenv("MODEL_MEDIUM", "qwen/qwen3-32b"),
        "weak":        os.getenv("MODEL_WEAK", "openai/gpt-oss-20b"),
        "very_weak":   os.getenv("MODEL_VERY_WEAK", "llama-3.1-8b-instant"),
    }
    OLLAMA_AGENTS = {}
    _validate_model_diversity(GROQ_AGENTS, "GROQ_AGENTS")
else:
    GROQ_AGENTS = {
        "very_strong": os.getenv("MODEL_VERY_STRONG", "llama-3.3-70b-versatile"),
        "strong":      os.getenv("MODEL_STRONG", "llama-3.1-8b-instant"),
    }

    OLLAMA_AGENTS = {
        "medium": os.getenv("MODEL_MEDIUM", "qwen2.5:7b"),
        "weak": os.getenv("MODEL_WEAK", "qwen2.5:1.5b"),
        "very_weak": os.getenv("MODEL_VERY_WEAK", "gemma2:2b"),
    }
    _validate_model_diversity({**GROQ_AGENTS, **OLLAMA_AGENTS}, "ALL_AGENTS")

ALL_AGENTS = list(GROQ_AGENTS.keys()) + list(OLLAMA_AGENTS.keys())

# Experiment settings
CALIBRATION_SIZE  = int(os.getenv("CALIBRATION_SIZE", "50"))
CONSISTENCY_SIZE  = int(os.getenv("CONSISTENCY_SIZE", "30"))
NUM_PARAPHRASES   = int(os.getenv("NUM_PARAPHRASES", "3"))
TEST_SIZE         = int(os.getenv("TEST_SIZE", "200"))
ALPHA             = float(os.getenv("ALPHA", "0.5"))

# Benchmarks to evaluate on
BENCHMARKS = ["gsm8k", "mmlu", "truthful_qa"]
