from litellm import completion

from encoding import LiteLLMEmbeddings

EMBEDDING_MODELS = [
    "openrouter/openai/text-embedding-3-small",
]

CHAT_MODELS = [
    "openrouter/google/gemini-3.5-flash-lite",
    "openrouter/openai/gpt-4o-mini",
]

OLLAMA_CLOUD_API_BASE = "https://ollama.com"


def chat(model, messages, **kwargs):
    if model.startswith("ollama/"):
        kwargs["api_base"] = OLLAMA_CLOUD_API_BASE
    kwargs.setdefault("max_tokens", 1024)
    return completion(model=model, messages=messages, **kwargs)


def embeddings_for(model):
    return LiteLLMEmbeddings(model)
