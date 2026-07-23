"""
LLM handler.

Generates answers using a small, free, instruction-tuned language
model that runs entirely on the local machine through the
transformers library. No external API, account, or token is
required. The model is downloaded once from Hugging Face on first
use and cached locally for subsequent runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

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
    """Raised when the language model fails to load or generate a response."""


class LLMHandler:
    """Loads and runs a small instruction-tuned model locally for grounded answer generation."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self.model_name = model_name
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
            )
            self.model.eval()
        except Exception as exc:
            raise LLMGenerationError(
                f"Unable to load the local language model: {model_name}."
            ) from exc

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
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(prompt, return_tensors="pt")

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=400,
                    temperature=0.2,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
            answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            return answer.strip() or NO_CONTEXT_MESSAGE
        except Exception as exc:
            raise LLMGenerationError(
                "An unexpected error occurred while generating a response. Please try again."
            ) from exc
