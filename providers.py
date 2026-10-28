import os

import litellm
from litellm import completion

from encoding import LiteLLMEmbeddings

if os.environ.get("LITELLM_DEBUG") == "true":
    litellm.turn_on_debug()

EMBEDDING_MODELS = [
    "openrouter/openai/text-embedding-3-small",
    # "openrouter/openai/text-embedding-3-large",
    "openrouter/openai/text-embedding-3-small",
    # "openrouter/openai/text-embedding-3-large",
    "openrouter/qwen/qwen3-embedding-8b",
    "openrouter/google/gemini-embedding-001",
    "openrouter/google/gemini-embedding-2",
    "openrouter/baai/bge-m3",
    "openrouter/perplexity/pplx-embed-v1-0.6b",
]

CHAT_MODELS = [
    # Paid - commented out for now
    "openrouter/google/gemini-3.5-flash-lite",
    "openrouter/openai/gpt-4o-mini",
    # "openrouter/anthropic/claude-sonnet-4.5",
    # "openrouter/meta-llama/llama-3.3-70b-instruct",
    # "openrouter/qwen/qwen3-coder",
    # "openrouter/mistralai/mixtral-8x22b-instruct",
    # "openrouter/google/gemma-3-27b-it",
    "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free",
]

OLLAMA_CLOUD_API_BASE = "https://ollama.com"


def chat(model, messages, **kwargs):
    if model.startswith("ollama/"):
        kwargs["api_base"] = OLLAMA_CLOUD_API_BASE
    kwargs.setdefault("max_tokens", 1024)
    return completion(model=model, messages=messages, **kwargs)


def embeddings_for(model):
    return LiteLLMEmbeddings(model)
