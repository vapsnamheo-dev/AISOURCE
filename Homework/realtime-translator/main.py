"""실시간 회의 통역 오버레이 — 진입점.

파이프라인(모두 별도 스레드로 분리해 지연 최소화):
  [오디오 캡처] -> audio_q -> [STT(faster-whisper)] -> (영어 부분/확정)
        확정 영어 -> 컨텍스트 버퍼 + translate_q -> [Groq 번역] -> 한국어 자막
  핫키(Ctrl+Space) -> [Groq 답변 추천] -> 영어 답변 3개

실행:  python main.py
"""
import os
import queue
import sys
import threading
from collections import deque

from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtWidgets import QApplication

import config
import llm
from audio_capture import LoopbackCapture
from overlay import OverlayWindow
from stt import StreamingSTT


class Bridge(QObject):
    """워커 스레드 -> GUI 스레드로 안전하게 신호 전달."""

    partial = pyqtSignal(str)
    subtitle_en = pyqtSignal(int, str)  # 영어 확정 문장 즉시 표시 (id, en)
    subtitle_ko = pyqtSignal(int, str)  # 한국어 번역 스트리밍 채움 (id, ko)
    replies = pyqtSignal(list)
    status = pyqtSignal(str)
    question_detected = pyqtSignal()  # 상대가 질문하면 자동 답변 추천 트리거


_Q_STARTERS = (
    "what", "why", "how", "when", "where", "who", "which", "whose", "whom",
    "do you", "did you", "are you", "is ", "can you", "could you", "would you",
    "will you", "have you", "has ", "should", "may i", "shall", "is there",
    "are there", "isn't", "aren't", "don't you", "right?",
)


def _is_question(text: str) -> bool:
    """상대 발화가 질문인지 간단 판별(물음표 또는 의문사 시작)."""
    t = (text or "").strip().lower()
    if not t:
        return False
    if "?" in t:
        return True
    return any(t.startswith(w) for w in _Q_STARTERS)


def audio_worker(bridge: Bridge, audio_q: "queue.Queue"):
    try:
        cap = LoopbackCapture()
    except Exception as exc:  # 캡처 장치 문제
        bridge.status.emit(f"오디오 캡처 실패: {exc}")
        return
    bridge.status.emit(f"캡처 중 · {cap.device['name'][:34]}")
    try:
        for chunk in cap.frames():
            audio_q.put(chunk)
    except Exception as exc:
        bridge.status.emit(f"오디오 중단: {exc}")
    finally:
        cap.close()


def stt_worker(bridge: Bridge, audio_q: "queue.Queue", context: deque, translate_q: "queue.Queue"):
    counter = {"n": 0}

    def on_partial(text: str):
        bridge.partial.emit(text)

    def on_final(text: str):
        counter["n"] += 1
        eid = counter["n"]
        context.append(("Them", text))
        bridge.subtitle_en.emit(eid, text)             # 영어 먼저 즉시 표시
        translate_q.put((eid, text, list(context)))    # 번역(컨텍스트 스냅샷 포함)
        # 자동 토글 ON이면 매 문장마다 답변 추천을 최신으로 갱신(다음 문장으로 넘어감)
        bridge.question_detected.emit()

    bridge.status.emit(f"STT 모델 로딩 중… (첫 실행은 whisper-{config.WHISPER_MODEL} 다운로드)")
    try:
        stt = StreamingSTT(on_partial, on_final)
    except Exception as exc:
        bridge.status.emit(f"STT 로드 실패: {exc}")
        return
    bridge.status.emit(f"STT 준비됨 · {stt.device} · whisper-{config.WHISPER_MODEL}")
    while True:
        try:
            # 0.25초 안에 청크가 없으면 = 무음 → 진행 중 문장 확정
            chunk = audio_q.get(timeout=0.25)
        except queue.Empty:
            try:
                stt.flush_if_idle()
            except Exception as exc:
                bridge.status.emit(f"STT 오류: {exc}")
            continue
        try:
            stt.feed(chunk)
        except Exception as exc:
            bridge.status.emit(f"STT 오류: {exc}")


def translate_worker(bridge: Bridge, translate_q: "queue.Queue"):
    while True:
        eid, en, ctx = translate_q.get()
        try:
            got = False
            for partial_ko in llm.translate_en_ko_stream(en, ctx):
                got = True
                bridge.subtitle_ko.emit(eid, partial_ko)  # 토큰마다 갱신(스트리밍)
            if not got:
                bridge.subtitle_ko.emit(eid, "")
        except Exception as exc:
            bridge.subtitle_ko.emit(eid, f"[번역 오류] {exc}")


def main():
    app = QApplication(sys.argv)

    bridge = Bridge()
    win = OverlayWindow(hotkey=config.SUGGEST_HOTKEY.replace("ctrl", "Ctrl").title())
    bridge.partial.connect(win.show_partial, Qt.ConnectionType.QueuedConnection)
    bridge.subtitle_en.connect(win.start_entry, Qt.ConnectionType.QueuedConnection)
    bridge.subtitle_ko.connect(win.set_translation, Qt.ConnectionType.QueuedConnection)
    bridge.replies.connect(win.show_replies, Qt.ConnectionType.QueuedConnection)
    bridge.status.connect(win.set_status, Qt.ConnectionType.QueuedConnection)
    win.show()

    if not llm.available():
        bridge.status.emit("⚠ .env의 GROQ_API_KEY 미설정 — 번역/추천 불가")

    context: deque = deque(maxlen=config.CONTEXT_TURNS)
    audio_q: queue.Queue = queue.Queue()
    translate_q: queue.Queue = queue.Queue()

    threading.Thread(target=audio_worker, args=(bridge, audio_q), daemon=True).start()
    threading.Thread(
        target=stt_worker, args=(bridge, audio_q, context, translate_q), daemon=True
    ).start()
    threading.Thread(target=translate_worker, args=(bridge, translate_q), daemon=True).start()

    # ---- 답변 추천 (자동 토글 ON이면 질문마다 / 수동: 핫키·버튼) ----
    suggest_busy = threading.Event()
    suggest_pending = threading.Event()  # 생성 중 새 질문이 들어오면 표시

    def request_suggestions():
        if not llm.available():
            bridge.status.emit("⚠ GROQ_API_KEY 미설정 — 추천 불가")
            return
        if suggest_busy.is_set():
            # 생성 중에 다음 질문이 오면 → 끝난 뒤 최신 질문으로 한 번 더(자동으로 넘어감)
            suggest_pending.set()
            return
        suggest_busy.set()
        bridge.status.emit("답변 생성 중…")

        def work():
            try:
                while True:
                    suggest_pending.clear()
                    last = []
                    for partial in llm.suggest_replies_stream(list(context)):
                        last = partial
                        bridge.replies.emit(partial)  # 한 줄 완성될 때마다 즉시 표시
                    bridge.replies.emit(last)         # 최종(한국어까지) 확정
                    if not suggest_pending.is_set():
                        break  # 그 사이 새 질문이 없으면 종료
                bridge.status.emit("답변 추천 완료")
            except Exception as exc:
                bridge.status.emit(f"추천 오류: {exc}")
            finally:
                suggest_busy.clear()

        threading.Thread(target=work, daemon=True).start()

    def on_auto_question():
        # '자동 답변' 토글이 ON일 때만 질문에 자동 응답
        if win.auto_enabled():
            request_suggestions()

    bridge.question_detected.connect(
        on_auto_question, Qt.ConnectionType.QueuedConnection
    )

    win.suggest_requested.connect(request_suggestions)

    try:
        import keyboard

        keyboard.add_hotkey(config.SUGGEST_HOTKEY, request_suggestions)
    except Exception as exc:
        bridge.status.emit(f"핫키 비활성({exc}) — 버튼으로 사용하세요")

    # 테스트용: SMOKE_SECONDS 환경변수가 있으면 그 시간 뒤 자동 종료(스모크 테스트)
    smoke = os.getenv("SMOKE_SECONDS")
    if smoke:
        from PyQt6.QtCore import QTimer

        QTimer.singleShot(int(float(smoke) * 1000), app.quit)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
