"""faster-whisper 기반 스트리밍 STT.

오디오 청크를 계속 feed 하면, 말하는 동안 주기적으로 '부분 인식'(on_partial)을
내보내 흐름 자막을 만들고, 무음이 일정 시간 이어지면 한 문장을 확정(on_final)한다.
모두 로컬에서 동작하므로 API 비용이 없다. GPU가 있으면 자동으로 사용한다.
"""
import time

import numpy as np
from faster_whisper import WhisperModel

from config import (
    MAX_UTTERANCE_SEC,
    PARTIAL_INTERVAL,
    SAMPLE_RATE,
    SILENCE_HANG_SEC,
    SILENCE_RMS,
    SOURCE_LANG,
    WHISPER_MODEL,
)


def _load_model():
    """GPU(cuda)를 먼저 시도하고, 실패하면 CPU(int8)로 폴백."""
    for device, compute in (("cuda", "float16"), ("cpu", "int8")):
        try:
            model = WhisperModel(WHISPER_MODEL, device=device, compute_type=compute)
            return model, device
        except Exception:
            continue
    raise RuntimeError("faster-whisper 모델을 로드하지 못했습니다.")


class StreamingSTT:
    def __init__(self, on_partial, on_final):
        self.model, self.device = _load_model()
        self.on_partial = on_partial
        self.on_final = on_final
        self._buf = np.zeros(0, dtype=np.float32)
        self._speaking = False
        self._last_voice_t = 0.0
        self._last_partial_t = 0.0

    @staticmethod
    def _rms(x: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0

    def _transcribe(self) -> str:
        if self._buf.size < SAMPLE_RATE * 0.2:  # 0.2초 미만이면 스킵
            return ""
        segments, _ = self.model.transcribe(
            self._buf,
            language=SOURCE_LANG,
            beam_size=1,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        return "".join(seg.text for seg in segments).strip()

    def feed(self, chunk: np.ndarray):
        """오디오 청크 1개를 처리. (WASAPI 루프백은 소리가 있을 때만 청크를 준다)"""
        now = time.time()
        voiced = self._rms(chunk) > SILENCE_RMS

        if voiced:
            if not self._speaking:
                self._speaking = True
                self._buf = np.zeros(0, dtype=np.float32)
                self._last_partial_t = now
            self._last_voice_t = now
            self._buf = np.concatenate([self._buf, chunk])
        elif self._speaking:
            # 말끝의 짧은 무음도 버퍼에 포함(자연스러운 문장 경계)
            self._buf = np.concatenate([self._buf, chunk])

        if not self._speaking:
            return

        duration = self._buf.size / SAMPLE_RATE

        # 흐름 자막용 부분 인식
        if voiced and (now - self._last_partial_t) > PARTIAL_INTERVAL:
            self._last_partial_t = now
            text = self._transcribe()
            if text:
                self.on_partial(text)

        # 너무 길어지면 강제로 한 문장 확정
        if duration > MAX_UTTERANCE_SEC:
            self._finalize()

    def flush_if_idle(self):
        """청크가 한동안 안 들어올 때(=무음) 호출. 진행 중 문장을 확정한다.

        루프백은 무음 구간에 데이터를 주지 않으므로, '청크 부재'를 무음으로 보고
        호출자(메인 루프)가 주기적으로 이 메서드를 불러 문장 경계를 만든다.
        """
        if self._speaking and (time.time() - self._last_voice_t) > SILENCE_HANG_SEC:
            self._finalize()

    def _finalize(self):
        text = self._transcribe()
        self._speaking = False
        self._buf = np.zeros(0, dtype=np.float32)
        if text:
            self.on_final(text)
