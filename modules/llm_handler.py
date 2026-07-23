"""
LLM handler.

Generates answers using a hosted, free-tier instruction-tuned language
model through the Hugging Face Inference Providers API. This keeps
memory usage minimal on the local application, since no model weights
are loaded or run in-process. An optional Hugging Face API token can
be supplied via the HF_TOKEN environment variable to raise rate
limits; the app also functions without one, subject to lower
anonymous rate limits.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

from huggingface_hub import InferenceClient

DEFAULT_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

SYSTEM_INSTRUCTIONS = (
    "You are an enterprise document assistant. Answer the user's question "
    "using only the provided context. If the context does not contain "
    "sufficient information to answer the question, state that clearly "
    "instead of guessing. Respond in a professional, concise, and factual "
    "tone. Do not use emojis or informal language."
)

NO_CONTEXT_MESSAGE = (
    "I could not find sufficient information in the uploaded documents "
    "to answer this question."
)


@dataclass
class ContextChunk:
    text: str
    source: str


class LLMGenerationError(Exception):
    """Raised when the language model fails to generate a response."""


class LLMHandler:
    """Wraps the Hugging Face Inference Providers API for grounded answer generation."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self.model_name = model_name
        api_token = os.getenv("HF_TOKEN") or None
        self.client = InferenceClient(model=model_name, token=api_token)

    def generate_answer(self, question: str, context_chunks: List[ContextChunk]) -> str:
        if not context_chunks:
            return NO_CONTEXT_MESSAGE

        context_text = "\n\n".join(
            f"[Source: {chunk.source}]\n{chunk.text}" for chunk in context_chunks
        )

        messages = [
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
            {
                "role": "user",
                "content": (
                    f"Context:\n{context_text}\n\n"
                    f"Question: {question}\n\n"
                    "Provide a clear and professional answer based solely on the context above."
                ),
            },
        ]

        try:
            response = self.client.chat_completion(
                messages=messages,
                max_tokens=512,
                temperature=0.2,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            raise LLMGenerationError(
                "An unexpected error occurred while generating a response. Please try again."
            ) from exc
