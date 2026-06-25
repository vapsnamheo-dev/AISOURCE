"""반투명 · 항상 위 · 나만 보이는 오버레이 UI (PyQt6).

좌측: 실시간 한국어 자막(확정 문장) + 현재 인식 중인 영어(흐름)
우측: 영어 답변 추천 카드(영어 + 한국어 설명)
창은 프레임이 없고 드래그로 옮길 수 있다.
"""
import html

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

STYLE = """
#root { background: rgba(6, 7, 11, 0.98); border: 1px solid rgba(120,140,200,0.32);
        border-radius: 16px; }
#bar { background: transparent; }
#title { color: #aeb9ff; font-size: 13px; font-weight: 700; }
#status { color: #7c8aa8; font-size: 12px; }
QPushButton#act { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 #5b8bff, stop:1 #45d0a8); color:#0c0e14; border:none;
        border-radius:9px; padding:6px 14px; font-weight:700; font-size:12px; }
QPushButton#act:hover { opacity:0.9; }
QPushButton#close { background: rgba(255,255,255,0.08); color:#cfd6e6; border:none;
        border-radius:9px; padding:6px 10px; font-size:12px; }
QPushButton#close:hover { background: rgba(255,90,90,0.5); }
QPushButton#toggle { background: rgba(255,255,255,0.08); color:#cfd6e6; border:none;
        border-radius:9px; padding:6px 12px; font-size:12px; }
QPushButton#toggle:checked { background:#45d0a8; color:#08121a; font-weight:700; }
#panelTitle { color:#8a96b4; font-size:11px; font-weight:700; letter-spacing:1px; }
#subs { background: rgba(0,0,0,0.55); border:1px solid rgba(120,140,200,0.20);
        border-radius:12px; color:#ffffff; font-size:16px; padding:10px; }
#interim { color:#b6c0dc; font-size:14px; font-style:italic; padding:2px 4px; }
#repCard { background: rgba(34,42,74,0.65); border:1px solid rgba(120,140,200,0.38);
        border-radius:12px; }
#repNum { color:#7aa2ff; font-weight:800; font-size:13px; }
#repEn { color:#ffffff; font-size:15px; font-weight:700; }
#repKo { color:#b3bedd; font-size:12px; }
#hint { color:#6c7894; font-size:11px; }
"""


class OverlayWindow(QWidget):
    suggest_requested = pyqtSignal()

    def __init__(self, hotkey: str = "Ctrl+Space"):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.resize(940, 380)
        self._drag_pos = None
        self._hotkey = hotkey
        self._entries = []  # 자막 항목 [{id, en, ko}]
        self._build()
        self.setStyleSheet(STYLE)

    # ---------- 구성 ----------
    def _build(self):
        root = QFrame(self)
        root.setObjectName("root")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(root)

        v = QVBoxLayout(root)
        v.setContentsMargins(14, 12, 14, 12)
        v.setSpacing(10)

        # 상단 바
        bar = QHBoxLayout()
        title = QLabel("🎧 실시간 회의 통역 오버레이")
        title.setObjectName("title")
        self.status = QLabel("준비됨")
        self.status.setObjectName("status")
        self.auto_btn = QPushButton("자동 답변: OFF")
        self.auto_btn.setObjectName("toggle")
        self.auto_btn.setCheckable(True)
        self.auto_btn.toggled.connect(
            lambda on: self.auto_btn.setText("자동 답변: ON" if on else "자동 답변: OFF")
        )
        suggest_btn = QPushButton(f"답변 추천  ({self._hotkey})")
        suggest_btn.setObjectName("act")
        suggest_btn.clicked.connect(self.suggest_requested.emit)
        close_btn = QPushButton("✕")
        close_btn.setObjectName("close")
        close_btn.clicked.connect(self.close)
        bar.addWidget(title)
        bar.addSpacing(10)
        bar.addWidget(self.status)
        bar.addStretch(1)
        bar.addWidget(self.auto_btn)
        bar.addWidget(suggest_btn)
        bar.addWidget(close_btn)
        v.addLayout(bar)

        # 본문 2분할
        body = QHBoxLayout()
        body.setSpacing(12)

        # 좌: 한국어 자막
        left = QVBoxLayout()
        lt = QLabel("실시간 자막 (한국어 + English)")
        lt.setObjectName("panelTitle")
        self.subs = QTextEdit()
        self.subs.setObjectName("subs")
        self.subs.setReadOnly(True)
        self.interim = QLabel("")
        self.interim.setObjectName("interim")
        self.interim.setWordWrap(True)
        left.addWidget(lt)
        left.addWidget(self.subs, 1)
        left.addWidget(self.interim)

        # 우: 영어 답변 추천
        right = QVBoxLayout()
        rt = QLabel("영어 답변 추천")
        rt.setObjectName("panelTitle")
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        holder = QWidget()
        self.replies_box = QVBoxLayout(holder)
        self.replies_box.setSpacing(8)
        self.replies_box.addStretch(1)
        scroll.setWidget(holder)
        hint = QLabel(f"상대 말이 쌓인 뒤 '{self._hotkey}'를 누르면 답변을 추천합니다.")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        right.addWidget(rt)
        right.addWidget(scroll, 1)
        right.addWidget(hint)

        body.addLayout(left, 1)
        body.addLayout(right, 1)
        v.addLayout(body, 1)

    # ---------- 슬롯 (다른 스레드에서 signal로 호출) ----------
    def set_status(self, text: str):
        self.status.setText(text)

    def auto_enabled(self) -> bool:
        return self.auto_btn.isChecked()

    def show_partial(self, en: str):
        self.interim.setText("🎙 " + en)

    def start_entry(self, eid: int, en: str):
        # 영어 확정 문장을 즉시 표시(한국어는 스트리밍으로 곧 채워짐)
        self.interim.setText("")
        self._entries.append({"id": eid, "en": en or "", "ko": ""})
        self._entries = self._entries[-40:]
        self._render_subs()

    def set_translation(self, eid: int, ko: str):
        for e in self._entries:
            if e["id"] == eid:
                e["ko"] = ko or ""
                break
        self._render_subs()

    def _render_subs(self):
        parts = []
        for e in self._entries:
            ko = html.escape(e["ko"]) if e["ko"] else "…"
            en = html.escape(e["en"])
            parts.append(
                '<div style="margin:0 0 9px 0">'
                f'<span style="color:#ffffff">{ko}</span><br>'
                f'<span style="color:#8a97bd; font-size:13px">{en}</span></div>'
            )
        self.subs.setHtml("".join(parts))
        sb = self.subs.verticalScrollBar()
        sb.setValue(sb.maximum())

    def show_replies(self, replies: list):
        # 기존 카드 제거(마지막 stretch는 유지)
        while self.replies_box.count() > 1:
            item = self.replies_box.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        if not replies:
            empty = QLabel("추천할 답변이 없습니다. 대화가 더 쌓인 뒤 다시 시도하세요.")
            empty.setObjectName("repKo")
            empty.setWordWrap(True)
            self.replies_box.insertWidget(0, empty)
            return
        for i, r in enumerate(replies, 1):
            self.replies_box.insertWidget(i - 1, self._reply_card(i, r.get("en", ""), r.get("ko", "")))

    def _reply_card(self, num: int, en: str, ko: str) -> QWidget:
        card = QFrame()
        card.setObjectName("repCard")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(4)
        head = QHBoxLayout()
        n = QLabel(f"{num}")
        n.setObjectName("repNum")
        en_lbl = QLabel(en)
        en_lbl.setObjectName("repEn")
        en_lbl.setWordWrap(True)
        en_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        head.addWidget(n)
        head.addSpacing(6)
        head.addWidget(en_lbl, 1)
        ko_lbl = QLabel(ko)
        ko_lbl.setObjectName("repKo")
        ko_lbl.setWordWrap(True)
        lay.addLayout(head)
        lay.addWidget(ko_lbl)
        return card

    # ---------- 드래그 이동 ----------
    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = e.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, e):
        if self._drag_pos is not None and e.buttons() & Qt.MouseButton.LeftButton:
            self.move(e.globalPosition().toPoint() - self._drag_pos)

    def mouseReleaseEvent(self, e):
        self._drag_pos = None
