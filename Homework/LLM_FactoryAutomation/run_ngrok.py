# -*- coding: utf-8 -*-
"""FEMTO-ST Streamlit 앱 — ngrok public URL 배포 런처.

사용법:
    python run_ngrok.py                         # 기본: streamlit_rag.py
    python run_ngrok.py --app femto             # streamlit_femto.py
    python run_ngrok.py --app unified           # streamlit_unified.py
    python run_ngrok.py --app rag               # streamlit_rag.py (기본)
    python run_ngrok.py --port 8502             # 포트 지정

ngrok 인증 토큰 (최초 1회):
    python run_ngrok.py --auth <YOUR_TOKEN>
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

VENV_PYTHON     = Path(r"C:\AISOURCE\.venv\Scripts\python.exe")
VENV_STREAMLIT  = Path(r"C:\AISOURCE\.venv\Scripts\streamlit.exe")
PROJECT_ROOT    = Path(__file__).resolve().parent

APP_MAP = {
    "rag":     PROJECT_ROOT / "app" / "streamlit_rag.py",
    "femto":   PROJECT_ROOT / "app" / "streamlit_femto.py",
    "unified": PROJECT_ROOT / "app" / "streamlit_unified.py",
    "app":     PROJECT_ROOT / "app" / "streamlit_app.py",
}

# 앱별 기본 포트 매핑 (명시적 --port가 없을 때 사용)
APP_DEFAULT_PORTS = {
    "app": 8501,    # DL 메인 앱
    "rag": 8502,    # LLM / RAG 전용 앱은 다른 포트(별도 URL)
    "femto": 8503,
    "unified": 8504,
}


def set_auth_token(token: str) -> None:
    from pyngrok import ngrok as _ngrok
    _ngrok.set_auth_token(token)
    print("[ngrok] 인증 토큰 저장 완료.")


def run(app_key: str = "rag", port: int | None = None) -> None:
    app_path = APP_MAP.get(app_key)
    if app_path is None or not app_path.exists():
        print(f"[ERROR] 앱을 찾을 수 없습니다: {app_key} -> {app_path}")
        sys.exit(1)

    # 포트 결정: 명시적 포트가 없으면 앱별 기본 포트 사용
    if port is None:
        port = APP_DEFAULT_PORTS.get(app_key, 8501)

    streamlit_cmd = str(VENV_STREAMLIT) if VENV_STREAMLIT.exists() else "streamlit"

    print(f"\n{'='*60}")
    print(f" FEMTO-ST Streamlit + ngrok 배포 런처")
    print(f"{'='*60}")
    print(f" 앱:   {app_path.name}")
    print(f" 포트: {port}")
    print(f"{'='*60}\n")

    # ── 1. Streamlit 서버 시작 ────────────────────────────────────────────
    proc = subprocess.Popen(
        [streamlit_cmd, "run", str(app_path),
         "--server.port", str(port),
         "--server.headless", "true",
         "--server.fileWatcherType", "none"],
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    print(f"[Streamlit] 서버 시작 중... (PID={proc.pid})")
    time.sleep(3)

    if proc.poll() is not None:
        out, err = proc.communicate()
        print("[ERROR] Streamlit 서버가 즉시 종료되었습니다.")
        print(err.decode("utf-8", errors="replace"))
        sys.exit(1)

    # ── 2. ngrok 터널 생성 ────────────────────────────────────────────────
    try:
        from pyngrok import ngrok
        tunnel = ngrok.connect(port, "http")
        public_url = tunnel.public_url
    except Exception as e:
        proc.terminate()
        print(f"[ERROR] ngrok 터널 생성 실패: {e}")
        print("  -> ngrok 인증 토큰이 필요한 경우: python run_ngrok.py --auth <TOKEN>")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f" Public URL: {public_url}")
    print(f" 로컬 URL:   http://localhost:{port}")
    print(f"{'='*60}")
    print("\n Ctrl+C 로 종료\n")

    # ── 3. 서버 유지 ──────────────────────────────────────────────────────
    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\n[종료] Streamlit + ngrok 터널 닫는 중...")
    finally:
        ngrok.disconnect(public_url)
        ngrok.kill()
        proc.terminate()
        print("[종료] 완료.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FEMTO-ST Streamlit + ngrok public URL 배포 런처",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--app",  default="rag",
                        choices=list(APP_MAP.keys()),
                        help="실행할 앱 (기본: rag)")
    parser.add_argument("--port", type=int, default=None,
                        help="Streamlit 포트 (기본: 앱별 기본 포트)")
    parser.add_argument("--auth", metavar="TOKEN",
                        help="ngrok 인증 토큰 설정 후 종료")
    args = parser.parse_args()

    if args.auth:
        set_auth_token(args.auth)
        return

    run(app_key=args.app, port=args.port)


if __name__ == "__main__":
    main()
