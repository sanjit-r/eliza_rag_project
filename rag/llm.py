"""
llm.py — Single Claude API call for answer generation.

The final answer is always produced in exactly one API request,
satisfying the assessment constraint.

Default model: claude-sonnet-4-6
  - Strong reasoning quality for financial analysis
  - 200K context window (easily accommodates ~15K tokens of retrieved context)
"""

from __future__ import annotations

import logging
import os

import anthropic

from rag.prompt import SYSTEM_PROMPT, build_prompt

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 4096


def generate_answer(
    question: str,
    chunks: list[dict],
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> tuple[str, dict]:
    """
    Make a single Claude API call and return (answer_text, usage_stats).

    Args:
        question:   The business question to answer.
        chunks:     List of retrieved chunk dicts (with text, metadata, score).
        model:      Claude model ID.
        max_tokens: Maximum tokens for the generated answer.

    Returns:
        (answer_text, usage_dict) where usage_dict has input_tokens / output_tokens.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY environment variable is not set. "
            "Copy .env.example to .env and add your key."
        )

    client = anthropic.Anthropic(api_key=api_key)
    user_message = build_prompt(question, chunks)

    logger.info("Calling %s with %d chunks (~%d chars of context) ...",
                model, len(chunks), len(user_message))

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    answer_text = response.content[0].text
    usage = {
        "input_tokens":  response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "model":         response.model,
    }

    logger.info("Response: %d input tokens, %d output tokens",
                usage["input_tokens"], usage["output_tokens"])

    return answer_text, usage
