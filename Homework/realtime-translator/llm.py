"""Groq(무료 API) 클라이언트 — 번역과 답변 추천.

- translate_en_ko : 영어 발화를 자연스러운 한국어로 (속도 우선 모델)
- suggest_replies : 최근 대화(컨텍스트 버퍼)를 바탕으로 영어 답변 3개 + 한국어 설명
"""
import json

from groq import Groq

import config

_client = Groq(api_key=config.GROQ_API_KEY) if config.key_ready() else None


def available() -> bool:
    return _client is not None


def translate_en_ko(text: str) -> str:
    text = (text or "").strip()
    if not _client or not text:
        return ""
    resp = _client.chat.completions.create(
        model=config.GROQ_FAST_MODEL,
        temperature=0.2,
        max_tokens=400,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a real-time interpreter. Translate the user's English "
                    "into natural, spoken Korean. Output ONLY the Korean translation — "
                    "no quotes, no notes, no romanization."
                ),
            },
            {"role": "user", "content": text},
        ],
    )
    return resp.choices[0].message.content.strip()


def translate_en_ko_stream(text: str, context=None):
    """영어 -> 한국어를 스트리밍 번역. 최근 대화(context)를 참고해 일관되게 번역.

    context: [(speaker, text), ...] 최근 발화. 대명사·이름·용어 일관성에만 사용하고
    번역 대상은 마지막 새 문장(text)뿐이다. 누적 한국어 텍스트를 토큰마다 yield.
    """
    text = (text or "").strip()
    if not _client or not text:
        return

    ctx_lines = ""
    if context:
        prior = [t for (_s, t) in context if t.strip() and t.strip() != text][-6:]
        ctx_lines = "\n".join(prior)

    system = (
        "You are a real-time interpreter for an English video meeting. "
        "Translate the speaker's English into natural, spoken Korean. "
        "Use the earlier conversation ONLY as context so pronouns, names, numbers and "
        "terms stay consistent with what was said before. "
        "Output ONLY the Korean translation of the new line — no quotes, no romanization, "
        "and never translate or repeat the context lines."
    )
    if ctx_lines:
        user = f"Earlier conversation:\n{ctx_lines}\n\nTranslate ONLY this new line:\n{text}"
    else:
        user = text

    stream = _client.chat.completions.create(
        model=config.GROQ_FAST_MODEL,
        temperature=0.2,
        max_tokens=220,
        stream=True,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    acc = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        if delta:
            acc += delta
            yield acc


def suggest_replies(context):
    """context: [(speaker, text), ...]  ->  [{"en":..., "ko":...}, ...] (최대 3개)"""
    if not _client or not context:
        return []
    convo = "\n".join(f"{spk}: {txt}" for spk, txt in context)
    system = (
        "You help me respond in an English video meeting. Based on the recent "
        "conversation, propose 3 short, natural English replies I could say next. "
        "Each reply gets a Korean translation so I understand it. "
        'Respond as JSON: {"replies":[{"en":"...","ko":"..."}, ...]}'
    )
    resp = _client.chat.completions.create(
        model=config.GROQ_FAST_MODEL,
        temperature=0.5,
        max_tokens=400,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Recent conversation:\n{convo}"},
        ],
    )
    try:
        data = json.loads(resp.choices[0].message.content)
        replies = data.get("replies", [])
        return [r for r in replies if r.get("en")][:3]
    except (json.JSONDecodeError, AttributeError, TypeError):
        return []


def _parse_reply_lines(acc: str):
    """'English ||| 한국어' 형식의 완성된 줄들을 [{en, ko}]로 파싱."""
    out = []
    for ln in acc.split("\n"):
        ln = ln.strip()
        if "|||" not in ln:
            continue
        en, _, ko = ln.partition("|||")
        en = en.strip().lstrip("0123456789.-) ").strip()
        if en:
            out.append({"en": en, "ko": ko.strip()})
    return out[:3]


def suggest_replies_stream(context):
    """답변 추천을 스트리밍. 한 줄(=답변 1개)이 완성될 때마다 현재까지의 리스트를 yield.

    JSON을 한 번에 받지 않고 'EN ||| KO' 줄 형식으로 받아, 영어가 끝나는 즉시(||| 도달)
    화면에 표시 → 첫 답변이 매우 빠르게 보인다.
    """
    if not _client or not context:
        return
    convo = "\n".join(f"{spk}: {txt}" for spk, txt in context)
    system = (
        "You help me reply in an English video meeting. Based on the recent conversation, "
        "give exactly 3 short, natural English replies I could say next. "
        "Output EXACTLY 3 lines, ONE reply per line, in this exact format:\n"
        "English reply ||| 한국어 번역\n"
        "No numbering, no bullets, no extra text, no blank lines."
    )
    stream = _client.chat.completions.create(
        model=config.GROQ_FAST_MODEL,
        temperature=0.5,
        max_tokens=300,
        stream=True,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Recent conversation:\n{convo}"},
        ],
    )
    acc = ""
    prev_count = 0
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        if not delta:
            continue
        acc += delta
        replies = _parse_reply_lines(acc)
        if len(replies) > prev_count:
            prev_count = len(replies)
            yield replies  # 새 답변(영어)이 완성되는 즉시 표시
    final = _parse_reply_lines(acc)
    if final:
        yield final  # 한국어까지 채워 최종 표시
