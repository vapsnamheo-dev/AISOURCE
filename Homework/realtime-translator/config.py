"""앱 설정 — 값은 .env로 덮어쓸 수 있습니다."""
import os

from dotenv import load_dotenv

load_dotenv()

# ---- Groq (무료 API) ----
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip()         # 답변 추천(품질)
GROQ_FAST_MODEL = os.getenv("GROQ_FAST_MODEL", "llama-3.1-8b-instant").strip()  # 번역(속도)
PLACEHOLDER_KEY = "여기에_본인_Groq_무료키를_붙여넣으세요"

# ---- STT (faster-whisper, 로컬) ----
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base").strip()
SOURCE_LANG = "en"   # 상대방 언어(영어)
TARGET_LANG = "ko"   # 자막 언어(한국어)

# ---- 오디오/스트리밍 ----
SAMPLE_RATE = 16000        # faster-whisper 입력 레이트
CHUNK_SECONDS = 0.20       # 캡처 단위
SILENCE_RMS = 0.012        # 무음 판정 임계치(환경에 맞게 조정)
SILENCE_HANG_SEC = 0.45    # 이만큼 무음이면 한 문장 종료(낮을수록 번역 빨리 시작)
PARTIAL_INTERVAL = 0.6     # 부분 인식(흐름 자막) 주기
MAX_UTTERANCE_SEC = 6.0    # 한 문장 최대 길이(강제 종료)

# ---- 답변 추천 ----
CONTEXT_TURNS = 8          # 답변 추천에 참고할 최근 발화 수
SUGGEST_HOTKEY = "ctrl+space"


def key_ready() -> bool:
    return bool(GROQ_API_KEY) and GROQ_API_KEY != PLACEHOLDER_KEY
