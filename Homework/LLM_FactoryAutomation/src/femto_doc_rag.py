# -*- coding: utf-8 -*-
"""FEMTO-ST 베어링 정비 지식 문서 검색 — Level 2 RAG (Chroma + LangChain + Ollama).

src/femto_rag_search.py(Level 1, FAISS + 12-dim 수치 특성 벡터)와 달리
이 모듈은 자연어 정비 지식 문서(산출물/bearing_maintenance_guide.txt)를
청크 단위로 쪼개 임베딩하고, Chroma 벡터 DB에 저장한 뒤 Ollama LLM으로
질의응답(RAG)하는 "문서 기반" 검색을 담당한다.

사전 설치:
    pip install langchain langchain-community chromadb sentence-transformers langchain-text-splitters rank_bm25

Hybrid RAG (v0.6):
    retrieve_docs()/ask()는 기본적으로 BM25(키워드 매칭) + Chroma 벡터 검색을
    EnsembleRetriever로 결합한다. "40도 초과", "1.5배" 같은 정확한 수치·임계값
    표현은 BM25가, 의미적으로 유사한 문장은 벡터 검색이 담당해 서로 보완한다.
    use_hybrid=False로 넘기면 기존처럼 벡터 검색만 사용한다.

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
OLLAMA_TEMPERATURE = 0.0  # RAG는 사실 기반 답변만 해야 하므로 0(결정론적 출력)으로 설정

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
    if not TEXT_FILE_PATH.parent.exists():
        raise FileNotFoundError(
            f"문서 폴더 없음: {TEXT_FILE_PATH.parent} — 폴더를 생성하고 "
            f"{TEXT_FILE_PATH.name}을(를) 넣어주세요."
        )
    if not TEXT_FILE_PATH.exists():
        raise FileNotFoundError(f"문서 없음: {TEXT_FILE_PATH}")

    # 재실행 시 오래된 벡터 저장소가 섞이지 않도록 삭제 후 재생성
    if PERSIST_DIRECTORY.exists():
        shutil.rmtree(PERSIST_DIRECTORY)
    PERSIST_DIRECTORY.parent.mkdir(parents=True, exist_ok=True)

    loader = TextLoader(str(TEXT_FILE_PATH), encoding="utf-8")
    documents = loader.load()

    if not documents or not any(d.page_content.strip() for d in documents):
        raise ValueError(f"문서 내용이 비어 있습니다: {TEXT_FILE_PATH}")

    text_splitter = RecursiveCharacterTextSplitter(
        # 청크 크기를 늘려 더 많은 컨텍스트가 포함되도록 함
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )
    texts = text_splitter.split_documents(documents)

    if not texts:
        raise ValueError(f"청크 분할 결과가 비어 있습니다: {TEXT_FILE_PATH}")

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


def _get_retriever(vectorstore: Chroma, k: int, use_hybrid: bool):
    """벡터 검색기, 또는 BM25+벡터를 결합한 Hybrid 검색기를 반환한다."""
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    if not use_hybrid:
        return vector_retriever

    try:
        # langchain>=1.0: EnsembleRetriever가 langchain_classic으로 이동
        from langchain_classic.retrievers import EnsembleRetriever
    except ImportError:
        from langchain.retrievers import EnsembleRetriever
    from langchain_community.retrievers import BM25Retriever

    stored = vectorstore.get(include=["documents"])
    bm25_texts = [t for t in stored.get("documents", []) if t and t.strip()]
    if not bm25_texts:
        # 인덱스에 문서가 없으면 BM25를 구성할 수 없으므로 벡터 검색만 사용
        return vector_retriever

    bm25_retriever = BM25Retriever.from_texts(bm25_texts)
    bm25_retriever.k = k

    return EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5],  # 키워드(BM25) 50% + 의미(벡터) 50%
    )


def retrieve_docs(
    question: str,
    k: int = 3,
    vectorstore: Chroma | None = None,
    use_hybrid: bool = True,
) -> list[str]:
    """질문과 관련된 정비 지식 문서 청크만 검색해 반환한다 (Ollama 불필요).

    femto_llm_report.py 등 다른 LLM(Claude API 등)에 검색 결과만 컨텍스트로
    넘기고 싶을 때 사용한다. ask()와 달리 로컬 LLM 호출이 없어 가볍고,
    Ollama 서버가 꺼져 있어도 동작한다.

    use_hybrid=True(기본값)면 BM25(키워드)+벡터 검색을 결합한 Hybrid RAG를
    사용한다. 수치·임계값 등 정확한 표현이 포함된 질문의 검색 정확도가 오른다.
    """
    if vectorstore is None:
        vectorstore = load_index()

    retriever = _get_retriever(vectorstore, k=k, use_hybrid=use_hybrid)
    docs = retriever.invoke(question)
    return [d.page_content for d in docs]


def ask(
    question: str,
    k: int = 3,
    vectorstore: Chroma | None = None,
    use_hybrid: bool = True,
) -> str:
    """질문에 대해 문서 기반 RAG 답변을 생성한다."""
    if vectorstore is None:
        vectorstore = load_index()

    retriever = _get_retriever(vectorstore, k=k, use_hybrid=use_hybrid)
    llm = Ollama(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL_NAME, temperature=OLLAMA_TEMPERATURE)

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
