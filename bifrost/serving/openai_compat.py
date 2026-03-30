"""
OpenAI-compatible chat completions API layer.

Translates between the OpenAI ``/v1/chat/completions`` request/response
schema and the internal HLM model interface.  Supports both synchronous
and streaming (SSE) responses.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:12]}"


def _unix_ts() -> int:
    return int(time.time())


def _messages_to_prompt(messages: List[Dict[str, str]]) -> str:
    """Flatten OpenAI-style messages into a single prompt string."""
    parts: List[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            parts.append(f"[system] {content}")
        elif role == "assistant":
            parts.append(f"[assistant] {content}")
        else:
            parts.append(content)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Synchronous response
# ---------------------------------------------------------------------------
def _build_response(
    completion_id: str,
    text: str,
    model_name: str = "clifford-hlm",
    finish_reason: str = "stop",
) -> Dict[str, Any]:
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": _unix_ts(),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": -1,
            "completion_tokens": -1,
            "total_tokens": -1,
        },
    }


# ---------------------------------------------------------------------------
# Streaming chunks
# ---------------------------------------------------------------------------
def _build_stream_chunks(
    completion_id: str,
    text: str,
    model_name: str = "clifford-hlm",
    chunk_size: int = 4,
) -> List[str]:
    """Split *text* into SSE-ready JSON chunks."""
    chunks: List[str] = []

    # role chunk
    chunks.append(
        json.dumps(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": _unix_ts(),
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant"},
                        "finish_reason": None,
                    }
                ],
            }
        )
    )

    # content chunks
    for i in range(0, len(text), chunk_size):
        piece = text[i : i + chunk_size]
        chunks.append(
            json.dumps(
                {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": _unix_ts(),
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": piece},
                            "finish_reason": None,
                        }
                    ],
                }
            )
        )

    # final chunk
    chunks.append(
        json.dumps(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": _unix_ts(),
                "model": model_name,
                "choices": [
                    {"index": 0, "delta": {}, "finish_reason": "stop"}
                ],
            }
        )
    )

    return chunks


# ---------------------------------------------------------------------------
# Handler called by the main server
# ---------------------------------------------------------------------------
def handle_chat_completions(handler: Any, body: Dict, model: Any) -> None:
    """Process a ``/v1/chat/completions`` request.

    Parameters
    ----------
    handler : _CliffordHandler
        The HTTP request handler (has ``_send_json`` and ``_send_sse``).
    body : dict
        Parsed JSON request body.
    model : object
        Model with a ``.generate()`` method.
    """
    messages = body.get("messages", [])
    prompt = _messages_to_prompt(messages)
    max_tokens = body.get("max_tokens", 128)
    temperature = body.get("temperature", 1.0)
    top_p = body.get("top_p", 1.0)
    stream = body.get("stream", False)

    generated = model.generate(
        prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p
    )

    cid = _make_id()

    if stream:
        chunks = _build_stream_chunks(cid, generated)
        handler._send_sse(chunks)
    else:
        resp = _build_response(cid, generated)
        handler._send_json(resp)
