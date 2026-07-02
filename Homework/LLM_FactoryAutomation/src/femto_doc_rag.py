# -*- coding: utf-8 -*-
"""FEMTO-ST 베어링 정비 지식 문서 검색 — Level 2 RAG (Chroma + LangChain + Ollama).

src/femto_rag_search.py(Level 1, FAISS + 12-dim 수치 특성 벡터)와 달리
이 모듈은 자연어 정비 지식 문서(산출물/bearing_maintenance_guide.txt)를
청크 단위로 쪼개 임베딩하고, Chroma 벡터 DB에 저장한 뒤 Ollama LLM으로
질의응답(RAG)하는 "문서 기반" 검색을 담당한다.

사전 설치:
    pip install langchain langchain-community chromadb sentence-transformers langchain-text-splitters

Ollama 사전 준비 (로컬 실행):
    ollama pull gemma2
    ollama serve   # http://localhost:11434 에서 대기

실행:
    python -m src.femto_doc_rag           # 인덱스 빌드
    python -m src.femto_doc_rag --demo    # 빌드 + 샘플 질의
    python -m src.femto_doc_rag --ask "온도만 상승하면 어떻게 해야 하나요?"

출력:
    models/chroma_store/   (Chroma 벡터 DB, 디스크 영구 저장)
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT = Path(__file__).resolve().parent.parent

# ── 설정 ─────────────────────────────────────────────────────────────────────
TEXT_FILE_PATH = ROOT / "산출물" / "bearing_maintenance_guide.txt"
PERSIST_DIRECTORY = ROOT / "models" / "chroma_store"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL_NAME = "gemma4:e2b"

RAG_PROMPT = ChatPromptTemplate.from_template(
    """당신은 베어링 설비 예지보전(PdM) 정비 지식 도우미입니다.
아래 [참고 문서]만 근거로 질문에 답하세요. 문서에 없는 내용은 "문서에서 찾을 수
없습니다"라고 답하세요.

[참고 문서]
{context}

[질문]
{question}
"""
)


def _load_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


# ─────────────────────────────────────────────────────────────────────────────
# 인덱스 빌드
# ─────────────────────────────────────────────────────────────────────────────

def build_index(verbose: bool = True) -> Chroma:
    """정비 지식 문서를 청크 분할·임베딩하여 Chroma 벡터 DB를 (재)구축한다."""
    if not TEXT_FILE_PATH.exists():
        raise FileNotFoundError(f"문서 없음: {TEXT_FILE_PATH}")

    # 재실행 시 오래된 벡터 저장소가 섞이지 않도록 삭제 후 재생성
    if PERSIST_DIRECTORY.exists():
        shutil.rmtree(PERSIST_DIRECTORY)
    PERSIST_DIRECTORY.parent.mkdir(parents=True, exist_ok=True)

    loader = TextLoader(str(TEXT_FILE_PATH), encoding="utf-8")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        # 청크 크기를 늘려 더 많은 컨텍스트가 포함되도록 함
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )
    texts = text_splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(
        texts,
        _load_embeddings(),
        persist_directory=str(PERSIST_DIRECTORY),
    )

    if verbose:
        print(f"[DocRAG] 문서: {TEXT_FILE_PATH.name}")
        print(f"[DocRAG] -> 총 {len(texts)}개의 청크로 분할되었습니다.")
        print(f"[DocRAG] 저장 -> {PERSIST_DIRECTORY}")

    return vectorstore


# ─────────────────────────────────────────────────────────────────────────────
# 인덱스 로드
# ─────────────────────────────────────────────────────────────────────────────

def load_index() -> Chroma:
    """저장된 Chroma 벡터 DB를 로드한다. 없으면 새로 빌드한다."""
    if not PERSIST_DIRECTORY.exists():
        return build_index(verbose=True)
    return Chroma(
        persist_directory=str(PERSIST_DIRECTORY),
        embedding_function=_load_embeddings(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# RAG 질의응답
# ─────────────────────────────────────────────────────────────────────────────

def _format_docs(docs) -> str:
    return "\n\n".join(d.page_content for d in docs)


def retrieve_docs(question: str, k: int = 3, vectorstore: Chroma | None = None) -> list[str]:
    """질문과 관련된 정비 지식 문서 청크만 검색해 반환한다 (Ollama 불필요).

    femto_llm_report.py 등 다른 LLM(Claude API 등)에 검색 결과만 컨텍스트로
    넘기고 싶을 때 사용한다. ask()와 달리 로컬 LLM 호출이 없어 가볍고,
    Ollama 서버가 꺼져 있어도 동작한다.
    """
    if vectorstore is None:
        vectorstore = load_index()

    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(question)
    return [d.page_content for d in docs]


def ask(question: str, k: int = 3, vectorstore: Chroma | None = None) -> str:
    """질문에 대해 문서 기반 RAG 답변을 생성한다."""
    if vectorstore is None:
        vectorstore = load_index()

    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    llm = Ollama(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL_NAME)

    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain.invoke(question)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _demo() -> None:
    sample_questions = [
        "온도만 상승하고 진동은 정상이라면 어떻게 해야 하나요?",
        "베어링 위험 단계로 판단하는 기준은 무엇인가요?",
    ]
    vectorstore = load_index()
    for q in sample_questions:
        print(f"\n[질문] {q}")
        print(f"[답변] {ask(q, vectorstore=vectorstore)}")


def run() -> None:
    parser = argparse.ArgumentParser(
        description="FEMTO 정비 지식 문서 RAG — Level 2 (Chroma + Ollama)"
    )
    parser.add_argument("--demo", action="store_true", help="빌드 후 샘플 질의 실행")
    parser.add_argument("--ask", type=str, default=None, help="단일 질문 실행")
    args = parser.parse_args()

    print("=" * 60)
    print("FEMTO-ST 정비 지식 문서 RAG  (Level 2 — Chroma + Ollama)")
    print("=" * 60)
    build_index(verbose=True)

    if args.ask:
        print(f"\n[질문] {args.ask}")
        print(f"[답변] {ask(args.ask)}")
    elif args.demo:
        _demo()


if __name__ == "__main__":
    run()
