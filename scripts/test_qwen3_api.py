#!/usr/bin/env python3
"""Smoke-test an OpenAI-compatible Qwen3 chat-completions endpoint."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from typing import Any, Dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test Qwen3 OpenAI-compatible API connectivity")
    parser.add_argument("--model", default="Qwen3-30B-A3B-Instruct-2507", help="Qwen3 model name served by the API")
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible base URL, for example https://xxx/v1")
    parser.add_argument("--api-key", required=True, help="API key for the Qwen3 service")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling probability")
    parser.add_argument("--max-tokens", type=int, default=128, help="Maximum completion tokens")
    return parser.parse_args()


def validate_response(response: Any) -> Dict[str, Any]:
    if not getattr(response, "choices", None):
        raise RuntimeError("API response did not contain any choices")
    message = response.choices[0].message
    content = getattr(message, "content", None)
    if not content or not content.strip():
        raise RuntimeError("API response message content was empty")

    usage = getattr(response, "usage", None)
    return {
        "content": content.strip(),
        "usage": {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
            "total_tokens": getattr(usage, "total_tokens", 0) if usage else 0,
        },
    }


def main() -> int:
    args = parse_args()
    if importlib.util.find_spec("openai") is None:
        print("ERROR: missing dependency 'openai'. Install it with `pip install openai`.", file=sys.stderr)
        return 2

    from openai import OpenAI

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    messages = [
        {"role": "system", "content": "You are a concise test assistant."},
        {"role": "user", "content": "Reply with exactly: Qwen3 API OK"},
    ]

    try:
        response = client.chat.completions.create(
            model=args.model,
            messages=messages,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )
        parsed = validate_response(response)
    except Exception as exc:
        print(f"ERROR: Qwen3 API validation failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps({"status": "ok", "model": args.model, **parsed}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
