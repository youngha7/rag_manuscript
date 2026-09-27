import logging
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config.settings as settings

if TYPE_CHECKING:
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


def _get_embeddings() -> "HuggingFaceEmbeddings":
    from langchain_huggingface import HuggingFaceEmbeddings
    device = settings.EMBEDDING_DEVICE
    logger.info("[Embedding] 모델 로드: %s (device=%s)", settings.EMBEDDING_MODEL, device)
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        model_kwargs={"device": device},
    )
    logger.info("[Embedding] 모델 로드 완료")
    return embeddings


def load_pdf_docs(pdf_path: str) -> List[Document]:
    """단일 PDF 파일을 로드하여 Document 목록을 반환한다.

    MEDICAL_RAG_PDF_OCR=true 인 경우:
    - 텍스트 레이어가 없는 페이지는 PyMuPDF로 이미지 렌더링 후
      rapidocr-onnxruntime으로 OCR을 수행한다.
    - 텍스트 레이어가 있는 페이지는 그대로 사용한다.
    """
    from langchain_community.document_loaders import PyPDFLoader

    path = Path(pdf_path)
    logger.info("[PDF] 로드 시작: %s (OCR=%s)", path.name, settings.PDF_OCR_ENABLED)

    if not settings.PDF_OCR_ENABLED:
        docs = PyPDFLoader(str(path)).load()
        logger.info("[PDF] 로드 완료 (일반 모드): %s → %d 페이지", path.name, len(docs))
        return docs

    # OCR 모드: 텍스트가 없는 페이지만 rapidocr로 추출
    logger.info("[OCR] OCR 모드 활성화 — rapidocr_onnxruntime import 시도")
    try:
        import fitz  # PyMuPDF
        import numpy as np
        from rapidocr_onnxruntime import RapidOCR
        _ocr_engine = RapidOCR()
        logger.info("[OCR] rapidocr_onnxruntime import 성공")
    except Exception as e:
        logger.warning(
            "[OCR] rapidocr_onnxruntime import 실패 → 일반 PyPDFLoader로 fallback\n"
            "  원인: %s\n%s",
            e, traceback.format_exc()
        )
        docs = PyPDFLoader(str(path)).load()
        logger.info("[PDF] fallback 로드 완료: %s → %d 페이지", path.name, len(docs))
        return docs

    try:
        doc = fitz.open(str(path))
        logger.info("[OCR] fitz.open 성공: %s (%d 페이지)", path.name, len(doc))
        documents: List[Document] = []

        for page_num, page in enumerate(doc):
            text = page.get_text().strip()

            if not text:
                logger.debug("[OCR] 페이지 %d: 텍스트 없음 → OCR 시도", page_num)
                try:
                    mat = fitz.Matrix(2, 2)
                    pix = page.get_pixmap(matrix=mat)
                    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                        pix.height, pix.width, pix.n
                    )
                    if pix.n == 4:
                        arr = arr[:, :, :3]

                    result, _ = _ocr_engine(arr)
                    if result:
                        text = " ".join(line[1] for line in result)
                        logger.debug("[OCR] 페이지 %d: OCR 성공 (%d자)", page_num, len(text))
                    else:
                        logger.debug("[OCR] 페이지 %d: OCR 결과 없음", page_num)
                except Exception as e:
                    logger.warning("[OCR] 페이지 %d OCR 실패: %s", page_num, e)
            else:
                logger.debug("[OCR] 페이지 %d: 텍스트 레이어 사용 (%d자)", page_num, len(text))

            if text:
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": str(path), "page": page_num},
                    )
                )

        logger.info("[OCR] 완료: %s → %d 페이지 추출", path.name, len(documents))
        return documents

    except Exception as e:
        logger.error(
            "[OCR] fitz 처리 실패 → PyPDFLoader fallback\n  원인: %s\n%s",
            e, traceback.format_exc()
        )
        docs = PyPDFLoader(str(path)).load()
        logger.info("[PDF] fallback 로드 완료: %s → %d 페이지", path.name, len(docs))
        return docs


def load_and_split_pdfs(folder_path: str) -> List[Document]:
    """폴더 내 전체 PDF를 로드하고 청크로 분할한다."""
    pdf_paths = sorted(Path(folder_path).glob("**/*.pdf"))
    logger.info("[Index] PDF 검색 경로: %s → %d개 발견", folder_path, len(pdf_paths))

    if not pdf_paths:
        raise RuntimeError(
            f"'{folder_path}' 폴더에 PDF 파일이 없습니다. "
            "MSD 매뉴얼 PDF를 data 폴더에 넣은 후 다시 실행하세요."
        )

    docs: List[Document] = []
    for path in pdf_paths:
        loaded = load_pdf_docs(str(path))
        logger.info("[Index] %s → %d 페이지", path.name, len(loaded))
        docs.extend(loaded)

    logger.info("[Index] 전체 %d 페이지 로드 완료, 청크 분할 시작", len(docs))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_MAX_CHARS,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    logger.info("[Index] 청크 분할 완료: %d 청크 (chunk_size=%d, overlap=%d)",
                len(chunks), settings.CHUNK_MAX_CHARS, settings.CHUNK_OVERLAP)
    return chunks


def build_faiss_db(docs: List[Document]) -> "FAISS":
    """Document 목록으로 FAISS DB를 생성합니다."""
    from langchain_community.vectorstores import FAISS
    logger.info("[FAISS] 인덱스 빌드 시작: %d 청크", len(docs))
    db = FAISS.from_documents(docs, _get_embeddings())
    logger.info("[FAISS] 인덱스 빌드 완료")
    return db


def save_faiss_db(db: "FAISS", index_path: str) -> None:
    """FAISS DB를 로컬에 저장합니다."""
    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    db.save_local(index_path)
    logger.info("[FAISS] 인덱스 저장 완료: %s", index_path)


def load_faiss_db(index_path: str) -> "FAISS":
    """로컬에 저장된 FAISS DB를 로드합니다."""
    from langchain_community.vectorstores import FAISS
    logger.info("[FAISS] 인덱스 로드 시작: %s", index_path)
    db = FAISS.load_local(
        index_path,
        _get_embeddings(),
        allow_dangerous_deserialization=True,
    )
    logger.info("[FAISS] 인덱스 로드 완료")
    return db


def retrieve(db: "FAISS", query: str, top_k: int = None) -> List[Document]:
    """FAISS DB에서 유사도 높은 상위 top_k 문서를 반환합니다."""
    k = top_k or settings.RAG_TOP_K
    logger.debug("[Retrieve] query='%s', top_k=%d", query[:50], k)
    retriever = db.as_retriever(search_kwargs={"k": k})
    results = retriever.invoke(query)
    logger.debug("[Retrieve] 결과: %d 청크", len(results))
    return results
