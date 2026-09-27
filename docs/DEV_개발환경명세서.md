# 개발 환경 명세서 (Development Environment Specification)

**프로젝트명**: 의료 정보 자기교정 RAG 시스템  
**문서버전**: v4.0  
**작성일**: 2026-08-27  
**작성자**: 연구자

---

## 1. 개발 환경 개요

본 시스템은 Python 기반의 LLM 응용 시스템으로, LangGraph 워크플로우, FAISS 벡터 검색, RAGAS 자동 평가, Flesch-Kincaid Grade Level 가독성 측정, Streamlit UI를 핵심 기술 스택으로 한다. 외부 LLM API(OpenAI)와 Oracle Database를 활용한다.

---

## 2. 하드웨어 환경

| 항목 | 사양 |
|------|------|
| **운영체제** | Rocky Linux 9.5 (Blue Onyx), kernel 5.14.0-503.23.2.el9_5.x86_64 |
| **CPU** | x86-64 아키텍처 |
| **GPU** | NVIDIA GeForce RTX 3070 Ti (VRAM 8GB) |
| **GPU 드라이버** | 570.86.15 |
| **CUDA (드라이버 최대 지원)** | 12.8 |
| **CUDA** | 활성화 (torch.cuda.is_available() = True) |

> **비고**:
> - 임베딩 모델(sentence-transformers)만 GPU를 사용하며, `config/settings.py`의 `EMBEDDING_DEVICE`가 `torch.cuda.is_available()`로 자동 감지한다 (`MEDICAL_RAG_EMBEDDING_DEVICE=cuda|cpu|mps|auto`로 강제 지정 가능).
> - **torch 버전 상한 주의**: PyPI의 torch 기본 빌드는 2.11 이상부터 CUDA 13.0 런타임을 번들한다. 이 서버의 드라이버(570.x)는 CUDA 12.8까지만 지원하므로 2.11+ 설치 시 `torch.cuda.is_available()`가 `False`로 떨어진다. `pyproject.toml`/`requirements.txt`에서 `torch>=2.2.0,<2.11.0`으로 상한을 고정해 CUDA 12.8 런타임(torch 2.10.0)이 선택되도록 해두었다.
> - GPU VRAM이 8GB로 크지 않으므로, GPU에 올라가는 모델(임베딩 등)은 반드시 전역 싱글턴으로 캐싱해 재사용해야 한다. 요청마다 새로 인스턴스화하면 배치 실행(예: 480회 ablation study) 중 GPU 메모리가 누적되어 OOM이 발생한다.

---

## 3. 소프트웨어 환경

### 3.1 언어 및 런타임

| 항목 | 버전 |
|------|------|
| **Python** | 3.12.14 |
| **패키지 관리** | pip (`requirements.txt`) / Poetry 2.2.1 (`pyproject.toml`, `poetry.lock`) |
| **가상환경** | Poetry가 자동 생성 (`~/.cache/pypoetry/virtualenvs/`) |

> `pyproject.toml`의 `python` 제약은 `>=3.11,<3.13`이다. 운영 서버의 시스템 Python은 3.9뿐이라 이 범위를 만족하지 못했으므로, [uv](https://github.com/astral-sh/uv)의 `uv python install 3.12`로 sudo 없이 사용자 로컬에 CPython 3.12.14를 설치한 뒤 `poetry env use <해당 python 경로>`로 Poetry 가상환경을 그 인터프리터로 고정했다. Poetry 자체도 서버에 없었다면 `pip install --user pipx && pipx install poetry`로 설치한다.
>
> `requirements.txt`와 `pyproject.toml`은 동일한 의존성 목록을 유지한다 — pip 기반 배포와 Poetry 기반 개발 환경 어느 쪽으로 설치해도 같은 패키지 셋이 깔리도록 두 파일을 함께 갱신한다.

---

## 4. 핵심 라이브러리 버전

### 4.1 UI 및 설정

| 패키지 | 버전 | 용도 |
|--------|------|------|
| `streamlit` | 1.62.0 | 웹 UI 프레임워크 |
| `python-dotenv` | 1.2.3 | `.env` 환경변수 로드 |
| `pydantic` | 2.13.4 | 데이터 모델 검증 |

### 4.2 LangChain / LangGraph

| 패키지 | 버전 | 용도 |
|--------|------|------|
| `langgraph` | 0.6.11 | StateGraph 기반 LLM 워크플로우 |
| `langchain` | 0.3.30 | LangChain 코어 (직접 의존성 아님 — `langchain-community`/`ragas` 등을 통한 전이 의존성) |
| `langchain-core` | 0.3.86 | 프롬프트, 파서, 메시지 추상화 |
| `langchain-community` | 0.3.31 | 커뮤니티 도구 및 통합 |
| `langchain-openai` | 0.3.35 | OpenAI ChatOpenAI 래퍼 |
| `langchain-text-splitters` | 0.3.11 | RecursiveCharacterTextSplitter |
| `langchain-huggingface` | 0.3.1 | HuggingFace 임베딩 연동 |

### 4.3 LLM API

| 패키지 | 버전 | 용도 |
|--------|------|------|
| `openai` | 2.54.0 | OpenAI API 클라이언트 (`langchain-openai` 경유). `ragas`의 Anthropic 어댑터가 의존하는 `instructor`가 `openai>=2.0.0,<3.0.0`을 요구해 2.x로 고정됨 |
| `anthropic` | 0.125.0 | RAGAS 판정 LLM(Claude) 클라이언트. 답변 생성 LLM과 무관하게 항상 이 클라이언트로 F/AR/CP를 채점 |
| `tiktoken` | 0.14.0 | OpenAI 토크나이저 |
| `tenacity` | 8.5.0 | API 재시도 로직 |

### 4.4 RAG / 벡터 검색

| 패키지 | 버전 | 용도 |
|--------|------|------|
| `ragas` | 0.4.3 | RAGAS 공식 평가 프레임워크 (Faithfulness / AR / CP). 판정 LLM은 Claude 고정 |
| `faiss-cpu` | 1.15.0 | Facebook AI 유사도 검색 |
| `sentence-transformers` | 3.4.1 | **BAAI/bge-base-en-v1.5** 임베딩 모델 (768차원). GPU에서 CUDA로 자동 실행 |
| `huggingface-hub` | 0.36.2 | HuggingFace 모델 다운로드 |
| `torch` | 2.10.0+cu128 | PyTorch (sentence-transformers 의존성). 버전 상한(`<2.11.0`) 이유는 2절 참고 |
| `numpy` | 2.5.2 | 수치 연산 (벡터 처리) |
| `scikit-learn` | 1.9.0 | 유사도 계산 보조 |

### 4.5 PDF 처리 및 OCR

| 패키지 | 버전 | 용도 |
|--------|------|------|
| `pymupdf` (fitz) | 1.28.2 | PDF 텍스트 추출 및 페이지 렌더링 |
| `pypdf` | 4.3.1 | PDF 메타데이터 처리 보조 |
| `rapidocr-onnxruntime` | 1.4.4 | 스캔 PDF OCR (ONNX 런타임 기반, CPU) |

### 4.6 웹검색

| 패키지 | 버전 | 용도 |
|--------|------|------|
| `duckduckgo-search` | 8.1.1 | Tier 2 웹검색 (DuckDuckGo API) |

### 4.7 데이터베이스

| 패키지 | 버전 | 용도 |
|--------|------|------|
| `oracledb` | — | Oracle Database 직접 연결 (감사 로그 저장) |

### 4.8 데이터 처리 및 시각화

| 패키지 | 버전 | 용도 |
|--------|------|------|
| `pandas` | 2.3.3 | 감사 로그 데이터 처리 및 집계 |
| `matplotlib` | 3.11.1 | 정적 차트 렌더링 (영어 텍스트 전용 — 폰트 렌더링 안정성) |
| `seaborn` | 0.13.2 | 통계 시각화 (박스플롯, 막대차트 등) |
| `scipy` | 1.18.1 | KDE 분포 계산 (`gaussian_kde`) |

### 4.8.1 Jupyter 실행 환경 (main.ipynb)

| 패키지 | 버전 | 용도 |
|--------|------|------|
| `ipykernel` | 7.3.0 | Poetry 가상환경을 Jupyter 커널로 등록 |
| `ipywidgets` | 8.1.9 | tqdm 등의 위젯 기반 진행률 표시 (`IProgress not found` 경고 방지) |

> 커널 등록: `poetry run python -m ipykernel install --user --name=<커널명> --display-name="<표시명>"`. Poetry 가상환경이 재생성되면(예: Python 버전 변경) 커널을 다시 등록해야 VSCode/Jupyter에서 인식된다.

### 4.9 성능평가 전용 지표 (Hit Rate / MRR / DeepEval RAG 지표)

`disease`(STQS-240/ablation 정답 라벨)가 있는 요청에서만 critic_agent가 추가로 계산해 DB에 기록하는 지표. Self-Correction Loop 게이트와는 무관.

| 패키지 | 버전 | 용도 |
|--------|------|------|
| `deepeval` | 4.2.0 | Faithfulness / Answer Relevancy 2종을 채점하는 RAG 평가 프레임워크. RAGAS(Claude)와 독립된 교차검증용. 판정 LLM은 `deepeval.models.OpenAIModel`(`DEEPEVAL_LLM_MODEL`, 기본 gpt-4o-mini) |
| `google-genai` | 1.75.0 | **현재 미사용 (레거시 의존성)**. DeepEval이 원래 `GeminiModel`을 썼을 때의 잔재 — pyproject.toml/requirements.txt에는 아직 남아있지만 코드에서 `google.genai`를 import하는 곳은 없음 (정리 필요) |

> 2026-08-30: 판정 프레임워크를 TruLens → DeepEval로 교체했다 (판정 LLM은 처음엔 Gemini로 동일 유지). 이전에 쓰던 `litellm`/`trulens-core`/`trulens-providers-litellm`은 제거했고, Oracle `rag_audit_log_bak`에 `deepeval_*` 3개 컬럼을 추가했다. `trulens_*` 3개 컬럼은 과거 실험 데이터(1,264건)를 포함해 완전히 삭제했다(`db/create_table.sql` 참고). RAGAS ContextualPrecision/Recall은 reference(expected_output)가 필요해 제외했다 — 이 파이프라인은 정답 answer를 따로 두지 않는다.
>
> (날짜 미상, 8-30 이후): 비용·단순화 목적으로 DeepEval 판정 LLM을 Gemini → **OpenAI gpt-4o-mini**로 통일했다. 이 시점부터 RAGAS(Claude)·DeepEval(gpt-4o-mini)·답변 생성(OpenAI 기본) 중 RAGAS만 독립 provider이고, DeepEval과 답변 생성은 같은 OpenAI 계열이 될 수 있어 판정 LLM 간 완전한 독립성은 더 이상 보장되지 않는다 (`infra/evaluator.py` 주석 참고).
>
> 2026-09-21: `deepeval_contextual_relevancy`를 더 이상 사용하지 않기로 하고 제거했다. DeepEval은 faithfulness/answer_relevancy 2종만 유지 (`infra/evaluator.py: compute_deepeval_scores()`).

> IR Hit Rate/MRR은 `context_sources`(검색 청크 출처 파일명)와 `disease`명을 문자열로 매칭하는 순수 로직으로, 별도 패키지가 필요 없다.

---

## 5. 외부 서비스

### 5.1 LLM API

| 서비스 | 모델 | 역할 | 필수 여부 |
|--------|------|------|----------|
| **OpenAI API** | gpt-4o-mini (코드 기본값은 gpt-4o) | RAG 엔진 (답변 생성) | 필수 (기본) |
| **OpenAI API** | gpt-4o-mini | 사용자 분류, 쿼리 최적화 | 필수 (기본) |
| **Anthropic API** | claude-haiku-4-5-20251001 (`ANTHROPIC_MODEL`) | RAGAS 판정 LLM (F/AR/CP). 답변 생성 LLM(OpenAI/Gemini 토글)과 무관하게 항상 사용 | **필수** — 없으면 critic_agent가 평가 자체를 못 함 |
| **OpenAI API** | gpt-4o-mini (`DEEPEVAL_LLM_MODEL`) | DeepEval RAG 지표 판정 LLM (Faithfulness/Answer Relevancy). `disease`가 있는 STQS/ablation 요청에서만 사용 | STQS/ablation 성능평가 시에만 필요 (일반 운영은 생략) |
| **Google Gemini API** | gemini-2.5-flash (`GEMINI_AUX_MODEL`) | 답변 생성/분류/쿼리최적화 LLM 백엔드 토글 (`llm_provider="gemini"` 지정 시). DeepEval 판정에는 더 이상 사용하지 않음 | 선택 — Gemini 백엔드를 쓸 때만 필요 |

### 5.2 데이터베이스

| 서비스 | 유형 | 용도 |
|--------|------|------|
| **Oracle Database** | Oracle DB | 감사 로그(`rag_audit_log_bak`) 저장 및 대시보드 조회. N+1행 설계, fk_grade 컬럼 포함 |

### 5.3 임베딩 모델

| 모델 | 차원 | 다운로드 경로 | 용도 |
|------|------|--------------|------|
| `BAAI/bge-base-en-v1.5` | 768 | HuggingFace Hub | PDF 청크 및 쿼리 벡터화 (코사인 유사도) |

> `config/settings.py`의 기본값은 `sentence-transformers/all-MiniLM-L6-v2`이지만, `.env`의 `MEDICAL_RAG_EMBEDDING_MODEL`로 재정의한다. 현재 배포는 BAAI/bge-base-en-v1.5를 사용하며, 이 모델로 빌드된 FAISS 인덱스가 `db/msd_faiss.index/`에 커밋되어 있다.

---

## 6. 환경 설정 파일

### 6.1 `.env` 파일 구조

```env
# ── LLM API 키 ──────────────────────────────
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...                 # 필수 — RAGAS 판정 LLM(Claude) 전용
GEMINI_API_KEY=AIza...                       # 선택 — 답변 생성 LLM을 Gemini 백엔드로 쓸 때만 필요 (DeepEval 판정에는 더 이상 사용 안 함)

# ── 모델 설정 ────────────────────────────────
OPENAI_MODEL=gpt-4o-mini                     # 코드 기본값은 gpt-4o
MEDICAL_RAG_TRANSLATE_MODEL=gpt-4o-mini
MEDICAL_RAG_CLASSIFIER_MODEL=gpt-4o-mini
ANTHROPIC_MODEL=claude-haiku-4-5-20251001    # RAGAS 판정 LLM (core/llm_client.py: ragas_model())
GEMINI_AUX_MODEL=gemini-2.5-flash            # 답변 생성 LLM 백엔드 토글용 (llm_provider="gemini" 지정 시). DeepEval 판정에는 미사용
MEDICAL_RAG_DEEPEVAL_LLM_MODEL=gpt-4o-mini   # DeepEval 판정 LLM (infra/evaluator.py: _get_deepeval_judge_model())
# MEDICAL_RAG_RAGAS_LLM_MODEL은 더 이상 사용되지 않음 (구버전: RAGAS 판정에 OpenAI/Gemini 토글 사용 — 현재는 ANTHROPIC_MODEL로 고정)

# ── 데이터베이스 ─────────────────────────────
ORACLE_USER=<사용자명>
ORACLE_PASSWORD=<비밀번호>
ORACLE_DSN=<host>:<port>/<service_name>

# ── 경로 설정 ────────────────────────────────
MEDICAL_RAG_DATA_DIR=data
MEDICAL_RAG_INDEX_PATH=db/msd_faiss.index

# ── 임베딩 모델 ──────────────────────────────
MEDICAL_RAG_EMBEDDING_MODEL=sentence-transformers/BAAI/bge-base-en-v1.5

# ── Self-Corrective Loop 임계값 ──────────────
MEDICAL_RAG_FAITHFULNESS_THRESHOLD=0.8
MEDICAL_RAG_AR_THRESHOLD=0.8
MEDICAL_RAG_CP_THRESHOLD=0.8
MEDICAL_RAG_MAX_LOOPS=3

# ── 즉시 에스컬레이션 임계값 ─────────────────
MEDICAL_RAG_CRITICAL_AR_THRESHOLD=0.3
MEDICAL_RAG_CRITICAL_F_THRESHOLD=0.3
MEDICAL_RAG_CRITICAL_CP_THRESHOLD=0.3

# ── 청크 분할 파라미터 ───────────────────────
MEDICAL_RAG_CHUNK_MAX_CHARS=1000
MEDICAL_RAG_CHUNK_OVERLAP=60
MEDICAL_RAG_TOP_K=3

# ── RAGAS 파라미터 ───────────────────────────
MEDICAL_RAG_RAGAS_STRICTNESS=3
MEDICAL_RAG_RAGAS_ANSWER_MAX_CHARS=1500
MEDICAL_RAG_RAGAS_CONTEXT_MAX_CHARS=2000
MEDICAL_RAG_RAGAS_LLM_MAX_TOKENS=8192

# ── OCR 설정 ─────────────────────────────────
MEDICAL_RAG_PDF_OCR=true                     # 스캔 PDF OCR 활성화

# ── 웹검색 ───────────────────────────────────
MEDICAL_RAG_WEB_SEARCH_MAX_RESULTS=3

# ── 로그 수준 ────────────────────────────────
MEDICAL_RAG_LOG_LEVEL=INFO
```

### 6.2 주요 디렉토리 사전 준비

```
rag_nonmun/
├── data/          # MSD Manual PDF 파일 배치
└── db/            # FAISS 인덱스 (msd_faiss.index/ 폴더로 저장)
    └── msd_faiss.index/
        ├── index.faiss   # FAISS 벡터 인덱스 바이너리
        └── index.pkl     # 청크 텍스트 및 출처 메타데이터
```

---

## 7. 설치 및 실행 방법

### 7.1 패키지 설치

```bash
# pip 사용
pip install -r requirements.txt

# Poetry 사용
poetry install   # 기존 poetry.lock 그대로 설치
poetry update    # 의존성을 새로 해석해 poetry.lock 갱신 + 설치 (버전 범위 변경 시)
```

> **신규 Linux 서버에 Poetry가 없는 경우**: `pip install --user pipx && pipx install poetry`로 설치한다. 또한 서버 기본 Python이 `pyproject.toml`의 `python = ">=3.11,<3.13"` 범위 밖(예: 3.9만 있는 경우)이면, `pip install --user uv && uv python install 3.12`로 sudo 없이 CPython 3.12를 받은 뒤 `poetry env use $(uv python find 3.12)`로 Poetry가 그 인터프리터를 쓰도록 지정하고 나서 `poetry install`/`poetry update`를 실행한다.
>
> **주의**: 의존성 버전 범위가 넓은 상태에서 `poetry update`를 돌리면 SAT resolver가 backtracking으로 오래 걸리거나 끝나지 않을 수 있다. `pyproject.toml`의 `[tool.poetry.dependencies]`는 실제 코드가 쓰는 패키지만(= `requirements.txt`와 동일 목록) 유지하도록 관리한다 — 예전 템플릿에서 물려받은 미사용 패키지(llama-index, pinecone, milvus, chromadb, redis, konlpy 등)를 다시 추가하지 않는다.

### 7.2 환경변수 설정

```bash
# .env 파일 생성 후 API 키 입력
cp .env.example .env
# OPENAI_API_KEY, ANTHROPIC_API_KEY, ORACLE_USER/PASSWORD/DSN 필수 입력
# GEMINI_API_KEY는 답변 생성 LLM을 Gemini 백엔드로 쓸 때만 필요 (DeepEval 판정에는 미사용)
```

### 7.3 Oracle 테이블 생성

```sql
-- db/create_table.sql 스크립트를 Oracle SQL Developer 또는 sqlplus에서 실행
-- docs/DB_데이터베이스설계서.md 섹션 3.1 참조
```

### 7.4 데이터 준비

```
1. MSD Manual PDF 파일을 data/ 폴더에 배치
2. 사이드바 "인덱스 전체 재빌드" 버튼으로 FAISS 인덱스 빌드
   또는 BAAI/bge-base-en-v1.5로 빌드된 기존 인덱스를 db/ 폴더에 배치
   (앱 시작 시 기존 인덱스가 있으면 로드만 수행, 자동 재빌드 없음)
```

### 7.5 앱 실행

```bash
# Streamlit 직접 실행
streamlit run app.py

# launch.py 스크립트 사용
python launch.py
```

### 7.6 성능 평가 일괄 실험

```bash
# Jupyter Notebook으로 실행 (2조건 × 240질문 = 480건 자동 실험)
jupyter notebook main.ipynb

# VSCode에서 실행할 경우, Poetry 가상환경을 커널로 먼저 등록
poetry run python -m ipykernel install --user --name=rag-nonmun --display-name="rag_nonmun (poetry py3.12)"
# main.ipynb 우측 상단 커널 선택 → "rag_nonmun (poetry py3.12)" 선택
```

---

## 8. 개발 도구

| 도구 | 버전/종류 | 용도 |
|------|-----------|------|
| **IDE** | VSCode | 코드 편집, 디버깅 |
| **버전관리** | Git | 소스 코드 관리 |
| **패키지관리** | pip / Poetry 2.2.1 | 의존성 관리 |
| **Python 버전 관리** | [uv](https://github.com/astral-sh/uv) `python install` | sudo 권한 없는 Linux 서버에 프로젝트가 요구하는 Python 버전을 사용자 로컬로 설치 |
| **DB 클라이언트** | Oracle SQL Developer / DBeaver | 감사 로그 조회 및 관리 |
| **실험 환경** | Jupyter Notebook / VSCode + ipykernel | 시스템 성능 평가 실험 (main.ipynb) |

---

## 9. 기술 스택 요약

```
┌─────────────────────────────────────────────────────────┐
│                   기술 스택 요약                           │
├──────────────┬──────────────────────────────────────────┤
│ 언어         │ Python 3.12.14 (운영 서버, Poetry)        │
│ UI           │ Streamlit 1.62.0                         │
│ 워크플로우   │ LangGraph 0.6.11                         │
│ LLM          │ OpenAI gpt-4o-mini                       │
│ LLM 프레임워크│ LangChain 0.3.x                         │
│ 임베딩       │ BAAI/bge-base-en-v1.5 (768차원, GPU)      │
│ 벡터DB       │ FAISS 1.15.0 (코사인 유사도)              │
│ 평가 (품질)  │ RAGAS 0.4.3 (F / AR / CP / Q_total)      │
│ 평가 (가독성)│ Flesch-Kincaid Grade Level               │
│              │   (Consumer ≤9 / Professional 14–18)     │
│ PDF 처리     │ PyMuPDF 1.28.2 + RapidOCR 1.4.4         │
│ 웹검색       │ DuckDuckGo Search 8.1.1                  │
│ 데이터베이스 │ Oracle Database (oracledb)               │
│              │   N+1행 설계 / fk_grade 컬럼 포함         │
│ 시각화       │ Matplotlib 3.11 / Seaborn 0.13           │
│              │   (matplotlib: 영어 텍스트 전용)           │
│ 딥러닝       │ PyTorch 2.10.0+cu128 (GPU, CUDA 12.8)   │
│ 실험 환경    │ Jupyter Notebook / VSCode+ipykernel      │
└──────────────┴──────────────────────────────────────────┘
```

---

*본 문서는 논문 연구 목적의 시스템 산출물이며, 실제 임상 적용을 위한 의학적 검증은 포함하지 않습니다.*
