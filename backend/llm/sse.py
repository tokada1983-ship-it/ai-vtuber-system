"""OpenAI互換 SSE ストリームのパースヘルパ"""
from __future__ import annotations

import json
from typing import AsyncIterator

import httpx


async def iter_openai_sse(response: httpx.Response) -> AsyncIterator[str]:
    """
    OpenAI互換の Server-Sent Events から content デルタを取り出す。
    `data: {...}` 行を読み、choices[0].delta.content を順にyieldする。
    reasoning_content しか返さないモデルにも対応する。
    """
    async for line in response.aiter_lines():
        line = line.strip()
        if not line or not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            continue
        choices = chunk.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        content = delta.get("content")
        if not content:
            # 一部のreasoningモデルは content が空で reasoning_content に流す
            content = delta.get("reasoning_content")
        if content:
            yield content


async def iter_ndjson(response: httpx.Response, *, key_path: tuple[str, ...]) -> AsyncIterator[str]:
    """
    改行区切りJSON（Ollama形式）から指定パスの文字列を取り出す。
    key_path=("message", "content") で {"message":{"content":"..."}} を拾う。
    """
    async for line in response.aiter_lines():
        line = line.strip()
        if not line:
            continue
        try:
            chunk = json.loads(line)
        except json.JSONDecodeError:
            continue
        node = chunk
        for key in key_path:
            if not isinstance(node, dict):
                node = None
                break
            node = node.get(key)
        if isinstance(node, str) and node:
            yield node
        if chunk.get("done"):
            break
