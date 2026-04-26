from __future__ import annotations

import json
import re
from typing import Any, TypeVar

from openai import OpenAI
from pydantic import BaseModel

from news_quant.config import (
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MAX_RETRIES,
    OPENAI_MODEL,
    OPENAI_TIMEOUT_SECONDS,
)

T = TypeVar("T", bound=BaseModel)


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    matched = re.search(r"\{[\s\S]*\}", text)
    if not matched:
        raise ValueError("响应中未找到 JSON 对象")
    return json.loads(matched.group())


def call_llm_json(system: str, user: str, response_model: type[T]) -> T:
    client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        timeout=OPENAI_TIMEOUT_SECONDS,
        max_retries=OPENAI_MAX_RETRIES,
    )
    completion = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    raw = completion.choices[0].message.content or "{}"
    data = _extract_json(raw)
    return response_model.model_validate(data)
