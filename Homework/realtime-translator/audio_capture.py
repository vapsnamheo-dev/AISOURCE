"""WASAPI 루프백으로 시스템 오디오(=Zoom/Meet 상대방 소리)를 캡처한다.

마이크가 아니라 '스피커로 나가는 소리'를 그대로 가져오므로, 회의 상대방의
음성을 그대로 받아 STT에 넘길 수 있다. 16kHz mono float32 청크를 yield 한다.
"""
import numpy as np
import pyaudiowpatch as pyaudio
from scipy.signal import resample_poly

from config import CHUNK_SECONDS, SAMPLE_RATE


class LoopbackCapture:
    def __init__(self):
        self._p = pyaudio.PyAudio()
        self._stream = None
        self.device = self._find_loopback()

    def _find_loopback(self):
        p = self._p
        wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        default_out = p.get_device_info_by_index(wasapi["defaultOutputDevice"])
        # 기본 출력장치에 대응하는 루프백 장치를 찾는다
        if default_out.get("isLoopbackDevice"):
            return default_out
        for lb in p.get_loopback_device_info_generator():
            if default_out["name"] in lb["name"]:
                return lb
        raise RuntimeError(
            "WASAPI 루프백 장치를 찾지 못했습니다. 스피커/헤드셋이 기본 출력인지 확인하세요."
        )

    def frames(self):
        """16kHz mono float32 numpy 청크를 계속 yield."""
        dev = self.device
        rate = int(dev["defaultSampleRate"])
        channels = int(dev["maxInputChannels"]) or 2
        frames_per_buffer = max(1, int(rate * CHUNK_SECONDS))

        self._stream = self._p.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=rate,
            input=True,
            input_device_index=dev["index"],
            frames_per_buffer=frames_per_buffer,
        )
        while True:
            data = self._stream.read(frames_per_buffer, exception_on_overflow=False)
            audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            if channels > 1:
                audio = audio.reshape(-1, channels).mean(axis=1)
            if rate != SAMPLE_RATE:
                audio = resample_poly(audio, SAMPLE_RATE, rate).astype(np.float32)
            yield audio

    def close(self):
        if self._stream is not None:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
        self._p.terminate()
