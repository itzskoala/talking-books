from litellm import completion

from encoding import LiteLLMEmbeddings

EMBEDDING_MODELS = [
    "openrouter/openai/text-embedding-3-small",
]

CHAT_MODELS = [
    "openrouter/google/gemini-3.5-flash-lite",       # gemini, via OpenRouter
    "openrouter/openai/gpt-4o-mini",                 # chatgpt, via OpenRouter
    "ollama/gpt-oss:20b",                             # ollama cloud - NOT via OpenRouter, see chat() below
    "openrouter/meta-llama/llama-3.2-3b-instruct",   # huggingface (open-weight), via OpenRouter
]

OLLAMA_CLOUD_API_BASE = "https://ollama.com"


def chat(model, messages, **kwargs):
    """Wraps litellm.completion. Ollama Cloud models need an explicit api_base pointed at
    ollama.com (litellm then picks up OLLAMA_API_KEY automatically) - every other model here
    is OpenRouter-routed and only needs OPENROUTER_API_KEY, which litellm already reads itself.

    Caps max_tokens by default - without it, some models ask OpenRouter for their full context
    window as the output budget, which 402s if your account credit can't cover that worst case."""
    if model.startswith("ollama/"):
        kwargs["api_base"] = OLLAMA_CLOUD_API_BASE
    kwargs.setdefault("max_tokens", 1024)
    return completion(model=model, messages=messages, **kwargs)


def embeddings_for(model):
    return LiteLLMEmbeddings(model)
