# ════════════════════════════════════════════════════════════════════
# [역할] SQLAlchemy DB 모델(물리 ERD) — machine·sensor·prediction·model_registry·failure_type.
# [단계] DB·예측 (보고서 7장)
# [작업 메모] 기본 SQLite, DATABASE_URL로 교체 가능.
# (이 헤더는 이전/이번 작업 맥락을 요약한 주석입니다 — 기능에는 영향 없음)
# ════════════════════════════════════════════════════════════════════
"""DB 모델(SQLAlchemy) · 04_DB설계 물리 ERD 대응."""
from __future__ import annotations
from datetime import datetime, timezone


def _utcnow():
    return datetime.now(timezone.utc)
from sqlalchemy import (create_engine, Integer, SmallInteger, String, Numeric,
                        DateTime, ForeignKey, LargeBinary)
from sqlalchemy.orm import (DeclarativeBase, Mapped, mapped_column,
                            relationship, sessionmaker)
from src import config





class Base(DeclarativeBase):
    pass


class Machine(Base):
    __tablename__ = "machine"
    machine_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    product_id: Mapped[str] = mapped_column(String(20), nullable=False)
    type: Mapped[str] = mapped_column(String(1), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    readings: Mapped[list["SensorReading"]] = relationship(back_populates="machine")


class SensorReading(Base):
    __tablename__ = "sensor_reading"
    reading_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    machine_id: Mapped[int] = mapped_column(ForeignKey("machine.machine_id"), nullable=False)
    air_temperature: Mapped[float] = mapped_column(Numeric(6, 2))
    process_temperature: Mapped[float] = mapped_column(Numeric(6, 2))
    rotational_speed: Mapped[int] = mapped_column(Integer)
    torque: Mapped[float] = mapped_column(Numeric(6, 2))
    tool_wear: Mapped[int] = mapped_column(Integer)
    recorded_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    machine: Mapped["Machine"] = relationship(back_populates="readings")
    predictions: Mapped[list["Prediction"]] = relationship(back_populates="reading")


class ModelRegistry(Base):
    __tablename__ = "model_registry"
    model_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(50), nullable=False)
    version: Mapped[str] = mapped_column(String(20), nullable=False)
    algorithm: Mapped[str] = mapped_column(String(30))
    accuracy: Mapped[float] = mapped_column(Numeric(5, 4), nullable=True)
    f1_score: Mapped[float] = mapped_column(Numeric(5, 4), nullable=True)
    roc_auc: Mapped[float] = mapped_column(Numeric(5, 4), nullable=True)
    pr_auc: Mapped[float] = mapped_column(Numeric(5, 4), nullable=True)
    threshold: Mapped[float] = mapped_column(Numeric(5, 4), default=0.5)
    is_active: Mapped[int] = mapped_column(SmallInteger, default=1)
    # 학습된 모델 번들(model+scaler+columns)을 직렬화하여 DB에 저장 (BLOB)
    artifact: Mapped[bytes] = mapped_column(LargeBinary, nullable=True)
    trained_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    predictions: Mapped[list["Prediction"]] = relationship(back_populates="model")


class FailureType(Base):
    __tablename__ = "failure_type"
    code: Mapped[str] = mapped_column(String(10), primary_key=True)
    name: Mapped[str] = mapped_column(String(50), nullable=False)


class Prediction(Base):
    __tablename__ = "prediction"
    prediction_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    reading_id: Mapped[int] = mapped_column(ForeignKey("sensor_reading.reading_id"), nullable=False)
    model_id: Mapped[int] = mapped_column(ForeignKey("model_registry.model_id"), nullable=True)
    pred_label: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    pred_proba: Mapped[float] = mapped_column(Numeric(5, 4))
    failure_type_code: Mapped[str] = mapped_column(ForeignKey("failure_type.code"), nullable=True)
    predicted_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    reading: Mapped["SensorReading"] = relationship(back_populates="predictions")
    model: Mapped["ModelRegistry"] = relationship(back_populates="predictions")


class ThresholdHistory(Base):
    """운영자가 '저장' 버튼으로 확정한 임계값 변경 이력."""
    __tablename__ = "threshold_history"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_id: Mapped[int] = mapped_column(ForeignKey("model_registry.model_id"), nullable=True)
    old_value: Mapped[float] = mapped_column(Numeric(5, 4))
    new_value: Mapped[float] = mapped_column(Numeric(5, 4))
    changed_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


# ── [PostgreSQL 영구저장 시 활성화] ─────────────────────────────────────────
# 외부 DB(Supabase 등) 연동 시 아래 싱글톤 블록을 활성화하고
# 아래 단순 버전 get_engine / init_db 를 주석 처리하세요.
#
# _ENGINE = None
#
# def get_engine(url: str | None = None):
#     global _ENGINE
#     if _ENGINE is None or url is not None:
#         effective_url = url or config.DATABASE_URL
#         # pool_pre_ping: 끊긴 연결 자동 재연결 (Supabase idle timeout 대응)
#         _ENGINE = create_engine(effective_url, future=True, pool_pre_ping=True)
#     return _ENGINE
#
# def init_db(url: str | None = None):
#     engine = get_engine(url)
#     Base.metadata.create_all(engine)
# ────────────────────────────────────────────────────────────────────────────


def get_engine(url: str | None = None):
    return create_engine(url or config.DATABASE_URL, future=True)


def init_db(engine=None):
    engine = engine or get_engine()
    Base.metadata.create_all(engine)
    # 룩업 시드
    Session = sessionmaker(bind=engine)
    with Session() as s:
        if s.get(FailureType, "TWF") is None:
            s.add_all([FailureType(code=c, name=n) for c, n in config.FAILURE_TYPES.items()])
            s.commit()
    return engine


def get_session(engine=None):
    engine = engine or get_engine()
    return sessionmaker(bind=engine, future=True)()


if __name__ == "__main__":
    init_db()
    print("DB 초기화 완료:", config.DATABASE_URL)
