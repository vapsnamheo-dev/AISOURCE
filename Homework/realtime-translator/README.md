# 실시간 회의 통역 오버레이 (meet-overlay)

Zoom / Google Meet **상대방 목소리(시스템 오디오)** 를 실시간으로 캡처해서
- 왼쪽: **영어 → 한국어 자막**을 실시간으로 보여주고
- 오른쪽: **컨텍스트(최근 대화)를 기억**해 **영어 답변 3개 + 한국어 설명**을 핫키로 추천

하는 **반투명·항상 위·나만 보이는** 데스크톱 오버레이입니다.

> 모든 처리는 **무료**입니다. 음성인식(STT)은 로컬 `faster-whisper`,
> 번역·답변 추천은 **Groq 무료 API**를 사용합니다. (유료 과금 없음)

---

## 1. 설치

```powershell
cd C:\AISOURCE\meet-overlay
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2. 무료 키 설정

1. https://console.groq.com/keys 에서 **무료** API 키 발급
2. `.env` 파일을 열어 키 교체:
   ```
   GROQ_API_KEY=gsk_...본인키...
   ```
   (`.env`는 `.gitignore`로 커밋되지 않습니다.)

## 3. 실행

```powershell
python main.py
```
- 스피커/헤드셋이 **기본 출력 장치**인지 확인하세요 (루프백 캡처 대상).
- Zoom/Meet 통화를 시작하면 왼쪽에 한국어 자막이 흐릅니다.
- 답변이 필요할 때 **`Ctrl+Space`**(또는 우측 상단 **답변 추천** 버튼) → 영어 답변 3개.

---

## 속도 튜닝 (목표: 1초 이내)

`.env` 또는 `config.py`에서 조정:

| 항목 | 빠르게 | 정확하게 |
|---|---|---|
| `WHISPER_MODEL` | `tiny` / `base` | `small` |
| GPU | NVIDIA GPU면 자동 `cuda` 사용 (가장 큰 효과) | - |
| `PARTIAL_INTERVAL` | 낮추면 자막 더 자주 갱신 | 높이면 부하↓ |
| `SILENCE_HANG_SEC` | 낮추면 문장 더 빨리 확정 | 높이면 끊김↓ |

- **GPU가 있으면** `base` 모델로 짧은 발화는 ~1초 내 자막이 가능합니다.
- **CPU만 있으면** `tiny`/`base` 권장. 1~2초 수준이 현실적입니다.
- 무음 임계치 `SILENCE_RMS`는 환경 소음에 맞춰 조정하세요(자막이 안 뜨면 낮춤).

## 구조

```
main.py          진입점 · 스레드 파이프라인 연결
audio_capture.py WASAPI 루프백(시스템 오디오) 캡처 → 16kHz mono
stt.py           faster-whisper 스트리밍 STT (부분/확정 자막)
llm.py           Groq: 영→한 번역 + 영어 답변 추천
overlay.py       PyQt6 반투명 오버레이 UI
config.py        설정(.env 로드)
```

## 주의

- 음성인식·번역 품질은 통화 오디오 상태와 모델 크기에 좌우됩니다.
- 회의 녹음/자막화는 **상대방 동의**가 필요한 경우가 많습니다. 법적·윤리적 책임은 사용자에게 있습니다.
- Groq 무료 티어에는 분당 요청 한도가 있습니다(개인 사용엔 충분).
