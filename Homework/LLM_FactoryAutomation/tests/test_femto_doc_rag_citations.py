"""tests/test_femto_doc_rag_citations.py — RAG 출처 근거(citation) 단위 테스트."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.femto_doc_rag import (
    TEXT_FILE_PATH,
    _format_citation,
    _load_guide_documents,
    _parse_guide_sections,
)

SAMPLE_TEXT = """제목 문서
개정일: 2026-06-01 (v1.0)
================================================

1. 첫째 절
--------------------------------
첫째 절 본문입니다.
Q1. FAQ 안의 숫자는 절 구분과 섞이면 안 된다.
1) 괄호 목록도 절 구분과 섞이면 안 된다.

2. 둘째 절
--------------------------------
둘째 절 본문입니다.
"""


def test_parse_guide_sections_splits_by_numbered_header():
    sections = _parse_guide_sections(SAMPLE_TEXT)
    assert [s["section_no"] for s in sections] == ["1", "2"]
    assert sections[0]["section_title"] == "첫째 절"
    assert sections[1]["section_title"] == "둘째 절"


def test_parse_guide_sections_ignores_faq_and_paren_numbering():
    sections = _parse_guide_sections(SAMPLE_TEXT)
    assert "Q1." in sections[0]["content"]
    assert "1)" in sections[0]["content"]
    # FAQ/괄호 목록 항목이 별도 절로 쪼개지면 안 된다 (섹션은 2개만 존재)
    assert len(sections) == 2


def test_format_citation_includes_source_section_and_revision():
    citation = _format_citation({
        "source_file": "bearing_maintenance_guide.txt",
        "section_no": "3",
        "section_title": "경고 신호 및 판정 기준",
        "revision_date": "2026-06-01",
    })
    assert citation == "[출처: bearing_maintenance_guide.txt · § 3. 경고 신호 및 판정 기준 · 개정 2026-06-01]"


def test_format_citation_falls_back_when_metadata_missing():
    citation = _format_citation({})
    assert citation == "[출처: 정비 지식 문서]"


def test_load_guide_documents_reads_real_guide_file():
    docs = _load_guide_documents(TEXT_FILE_PATH)
    assert len(docs) >= 6  # 실제 가이드는 6개 절로 구성
    assert all(d.metadata["revision_date"] == "2026-06-01" for d in docs)
    assert all(d.metadata["source_file"] == "bearing_maintenance_guide.txt" for d in docs)
    section_titles = [d.metadata["section_title"] for d in docs]
    assert "베어링 열화의 기본 개념" in section_titles
