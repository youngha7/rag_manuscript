# 시스템 아키텍처 설계서 (System Architecture Design)

**프로젝트명**: 의료 정보 자기교정 RAG 시스템  
**문서버전**: v3.0  
**작성일**: 2026-05-16  
**작성자**: 연구자

---

## 1. 문서 개요

본 문서는 LangGraph 기반 의료 정보 자기교정 RAG 시스템의 전체 아키텍처를 정의한다. 시스템의 컴포넌트 구성, 계층 구조, 데이터 흐름, 모듈 간 의존관계, 핵심 설계 결정사항을 포함한다.

**버전별 변경 이력:**

| 버전 | 주요 변경사항 |
|------|--------------|
| v1.0 | 초기 아키텍처 설계 |
| v2.0 | Proposal System vs Baseline 2조건, STQS-40, 요청당 1행 감사 로그, FK Grade 기초 설계 |
| v3.0 | 감사 로그 N+1행 설계(save_loop_log 추가), fk_grade 컬럼 도입, Consumer/Professional 가독성 프롬프트, 성능 시각화 7개 섹션 확장 |
| v3.1 (2026-08-30) | 성능평가 전용 교차검증 프레임워크를 TruLens RAG Triad(3종, Gemini 판정) → DeepEval(2종: Faithfulness/Answer Relevancy, gpt-4o-mini 판정)으로 교체. Contextual Relevancy는 이후(v3.2) 제거 |
| v3.2 (2026-09-21) | 성능 시각화 8개 섹션으로 확장(DeepEval 비교, IR Hit Rate/MRR 섹션 추가) 및 번호 재정렬 |

---

## 2. 아키텍처 개요

### 2.1 아키텍처 스타일

본 시스템은 다음 아키텍처 패턴을 복합적으로 적용한다:

| 패턴 | 적용 범위 |
|------|-----------|
| **그래프 기반 워크플로우** | LangGraph StateGraph를 이용한 에이전트 파이프라인 |
| **계층형 아키텍처** | UI → 비즈니스 로직(Graph) → 인프라 → 외부 API |
| **Self-Corrective Loop** | RAGAS 평가 결과에 따라 동적으로 라우팅되는 피드백 루프 |
| **다중 계층 폴백** | Tier 0 → Tier 1 → Tier 2 → Fallback 순의 계단식 에스컬레이션 |
| **조건 비교** | Proposal System(A) vs Baseline(E) 2가지 조건 비교 |

### 2.2 전체 시스템 구성도

```
┌─────────────────────────────────────────────────────────────────────┐
│                          사용자 (Browser)                             │
└───────────────────────────────┬─────────────────────────────────────┘
                                │  HTTP (Streamlit)
┌───────────────────────────────▼─────────────────────────────────────┐
│                       Presentation Layer                             │
│  ┌──────────────┐  ┌───────────────┐  ┌──────────────────────────┐  │
│  │  app.py      │  │  ui/sidebar   │  │  ui/dashboard            │  │
│  │  (진입점)    │  │  (설정 패널)  │  │  ├ log_viewer            │  │
│  │  조건 A 고정 │  │               │  │  └ performance_viz       │  │
│  └──────┬───────┘  └───────────────┘  └──────────────────────────┘  │
└─────────┼───────────────────────────────────────────────────────────┘
          │  run_medical_self_corrective_rag(ablation_condition="A")
          │
          │  main.ipynb: run_medical_self_corrective_rag(ablation_condition="A"/"E")
┌─────────▼───────────────────────────────────────────────────────────┐
│                      Business Logic Layer                            │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │              LangGraph StateGraph (graph.py)                  │  │
│  │                                                               │  │
│  │  [level_classifier] ──► [query_rewriter] ──► [rag_engine]    │  │
│  │         │                      ▲                  │           │  │
│  │         │                      │ (Self-Corrective │           │  │
│  │         │                      │   Loop / A)      ▼           │  │
│  │         │                  [critic] ◄──────────────           │  │
│  │         │              (A/E 조건별 라우팅)                     │  │
│  │         │               ┌──────┴──────┐                       │  │
│  │         │               ▼             ▼                       │  │
│  │         │     [output (_output_node)] [fallback]              │  │
│  │         │     FK Grade 계산            save_audit_log          │  │
│  │         │     output_agent(출처/면책)  (fk_grade=None)        │  │
│  │         │     save_audit_log(fk_grade)                        │  │
│  │         │               │             │                       │  │
│  │         └───────────────┴─────────────┘                       │  │
│  │                         END                                   │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────────┐   │
│  │ agents/      │  │ core/        │  │ config/settings.py      │   │
│  │ classifier   │  │ llm_client   │  │ (모든 임계값·모델 설정)   │   │
│  │ rewriter     │  │ (OpenAI)     │  └─────────────────────────┘   │
│  │ rag_engine   │  └──────────────┘                                 │
│  │ critic       │                                                    │
│  │ output       │                                                    │
│  └──────────────┘                                                    │
└─────────────────────────────────────────────────────────────────────┘
          │
┌─────────▼───────────────────────────────────────────────────────────┐
│                       Infrastructure Layer                           │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │ infra/           │  │ infra/           │  │ tools/           │  │
│  │ vector_store.py  │  │ audit_logger.py  │  │ vector_search.py │  │
│  │ (FAISS 인덱스)   │  │ (N+1행 저장)     │  │ web_search.py   │  │
│  │                  │  │ evaluator.py     │  │                  │  │
│  │                  │  │ (RAGAS + FK)     │  │                  │  │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘  │
└───────────┼─────────────────────┼─────────────────────┼────────────┘
            │                     │                     │
┌───────────▼─────────────────────▼─────────────────────▼────────────┐
│                        External Services                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────┐  │
│  │ FAISS Index  │  │   Oracle     │  │  OpenAI API  │  │DuckDuck│  │
│  │ (로컬 폴더)  │  │  Database    │  │              │  │Go API  │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  └────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. 계층별 상세 설계

### 3.1 Presentation Layer (UI)

| 모듈 | 파일 | 역할 |
|------|------|------|
| 진입점 | `app.py` | Streamlit 앱 초기화, 세션 상태 관리, 라우팅 (항상 조건 A) |
| 사이드바 | `ui/sidebar.py` | 페르소나 선택, LLM 백엔드 선택, 인덱스 재빌더, 대시보드 메뉴 |
| 헤더 | `ui/header.py` | 시스템 설명 및 플로우 안내 |
| PDF 업로더 | `ui/pdf_uploader.py` | PDF 업로드 및 인덱싱 UI |
| 스텝 렌더러 | `ui/step_renderers.py` | LangGraph 노드별 실시간 상태 표시 |
| 점수 카드 | `ui/score_card.py` | F/AR/CP 점수 카드 렌더링 |
| 결과 패널 | `ui/result_panel.py` | 최종 답변 표시 |
| 로그 뷰어 | `ui/dashboard/log_viewer.py` | 감사 로그 목록/상세 조회 |
| 성능 시각화 | `ui/dashboard/performance_viz.py` | Proposal System vs Baseline 8개 섹션 시각화 차트 |

**Streamlit 세션 상태 관리:**

```python
SESSION_DEFAULTS = {
    "result": "",
    "logs": [],
    "scores": {},
    "detected_level": "",
    "search_tier": 0,
    "llm_provider": "openai",
    "dashboard_menu": "",
}
```

---

### 3.2 Business Logic Layer (LangGraph 워크플로우)

#### 3.2.1 GraphState 설계

시스템 전체를 관통하는 단일 상태 객체. 모든 노드는 이 상태를 읽고 업데이트한다.

```python
class GraphState(TypedDict):
    # ── 요청 식별 ──────────────────────────────────────────────────────
    request_id:               str          # 요청 고유 UUID
    question:                 str          # 원본 한국어 질문
    user_level:               str          # "Professional" | "Consumer"
    queries:                  List[str]    # 최적화된 영문 쿼리 이력
    context:                  List[str]    # 검색된 컨텍스트 청크
    context_sources:          List[str]    # 출처 메타데이터
    answer:                   str          # 현재 생성된 답변
    # ── RAGAS 평가 결과 ────────────────────────────────────────────────
    critic_score:             float        # Faithfulness (0~1)
    answer_relevance_score:   float        # Answer Relevance (0~1); AR은 항상 queries[0] 기준 평가
    context_precision_score:  float        # Context Precision (0~1)
    critic_feedback:          str          # 기준 미달 지표·원인을 자연어로 요약한 재쿼리 힌트
    # ── 성능평가 전용 지표 (게이트 무관, disease 있는 STQS/ablation 행만 값 존재) ──
    hit_rate_score:            float        # IR Hit Rate (0/1), 일반 운영 시 None
    mrr_score:                 float        # IR MRR (0~1), 일반 운영 시 None
    deepeval_faithfulness:      float        # DeepEval — RAGAS F 교차검증 (gpt-4o-mini 판정)
    deepeval_answer_relevancy:  float        # DeepEval — RAGAS AR 교차검증 (gpt-4o-mini 판정)
    # ── 티어 및 루프 ───────────────────────────────────────────────────
    search_tier:              int          # 현재 검색 티어 (0/1/2)
    loop_count:               int          # 현재 Tier 재시도 횟수
    tier_path:                str          # 에스컬레이션 경로: "0"/"0→1"/"0→1→2"
    self_correction_count:    int          # Tier 0 자가 교정 누적 횟수
    eval_count:               int          # critic 평가 누적 횟수
    # ── Best Answer 추적 ──────────────────────────────────────────────────
    best_answer:              str          # 루프 전체에서 Q_total이 가장 높은 답변 (Fallback 우선 사용)
    best_q_total:             float        # 해당 답변의 Q_total (0.4·F + 0.4·AR + 0.2·CP)
    # ── 시스템 정보 ────────────────────────────────────────────────────
    llm_provider:             str          # "openai"
    workflow_start_time:      float        # time.time() 워크플로우 시작 시각
    log:                      List[str]    # 실행 로그
    # ── 실험 조건 메타데이터 ──────────────────────────────────────────
    ablation_condition:       str          # "A"(Proposal System)/"E"(Baseline), ""=일반 운영
    query_index:              int          # STQS-240 질문 번호 (1-240), None=일반 운영
    disease:                  str          # 질환명
    query_level_label:        str          # "P"/"C" 정답 레이블, ""=일반 운영
```

#### 3.2.2 LangGraph 노드 설계

```
level_classifier ──► query_rewriter ──► rag_engine ──► critic
                           ▲                              │
                           │    (Self-Corrective,         │ 성공: F≥0.8∧AR≥0.8∧CP≥0.8
                           │     Proposal System만 적용)  │
                           └──────────────────────────────┤
                                                          │ 기준 미달 (조건별 분기)
                                                          │
                                                   save_loop_log()  ← 매 평가마다 (is_final=FALSE)
                                                          │
                                                     ┌────▼────┐
                                              ┌──────│ output  │──► save_audit_log(fk_grade) ──► END
                                              │      └─────────┘   (is_final=TRUE)
                                              │      ┌──────────┐
                                              └──────│ fallback │──► save_audit_log(fk_grade=None) ──► END
                                                     └──────────┘   (is_final=TRUE, is_fallback=TRUE)
```

| 노드 | 함수 | 역할 |
|------|------|------|
| `level_classifier` | `agents/classifier.py` | LLM으로 사용자 수준 분류 (조건 E는 Baseline 고정) |
| `query_rewriter` | `agents/rewriter.py` | 한국어 질문 → 영문 의료 검색 쿼리 최적화 |
| `rag_engine` | `agents/rag_engine.py` | Tier별 검색 및 수준별 맞춤 답변 합성 (FK Grade 목표 적용) |
| `critic` | `agents/critic.py` + `graph.py` | RAGAS 3중 평가 + A/E 조건별 Self-Corrective 라우팅 + save_loop_log() |
| `output` | `agents/output.py` + `graph.py` | FK Grade 계산 → output_agent(출처·면책 조항 추가) → save_audit_log(fk_grade) |
| `fallback` | `graph.py` | 원문 제시 → save_audit_log(is_fallback=True, fk_grade=None) |

#### 3.2.3 Self-Corrective Loop 라우팅 로직

```
critic 노드 평가 후 → save_loop_log() (is_final=FALSE) → 라우팅 결정:

[조건 E — Baseline]
  → 항상 즉시 output (평가 결과 무관)

[Tier 1 평가 — AR만 사용]
  AR ≥ 0.8 → output
  AR < 0.8 → search_tier=2, goto query_rewriter

[성공 조건 — F∧AR∧CP 모두 충족]
  F ≥ 0.8 AND AR ≥ 0.8 AND CP ≥ 0.8
    └─► output → END

[Tier 0 실패 — 조건 A: Proposal System (기본 동작)]
  is_critically_low (AR<0.3 OR F<0.3∧CP<0.2) → 즉시 search_tier=1
  loop >= MAX_LOOPS → search_tier=1, tier_path="0→1" → query_rewriter
  그 외 → self_correction_count+1, loop+1 → query_rewriter (Tier 0 재시도)

[Tier 2 소진]
  모든 Tier 소진 → fallback → END
```

#### 3.2.4 RAG Engine Tier별 동작

| Tier | 검색 소스 | 에이전트 방식 | 평가 지표 | Temperature | FK Grade 목표 |
|------|-----------|--------------|-----------|-------------|--------------|
| **0** | FAISS VectorDB (MSD Manual) | ReAct 에이전트 + `search_msd_manual` 도구 | F + AR + CP | 0.0 | Consumer ≤9, Pro ≥12 |
| **1** | LLM 학습데이터 | 도구 없음, LLM 직접 생성 | AR만 | 0.1 | Consumer ≤9, Pro ≥12 |
| **2** | DuckDuckGo 웹검색 | ReAct 에이전트 + `search_web` 도구 | F + AR + CP | 0.1 | Consumer ≤9, Pro ≥12 |

**사용자 수준별 프롬프트 언어 규칙:**

| 수준 | FK Grade 목표 | 주요 규칙 |
|------|--------------|-----------|
| Consumer | ≤ 9 | 문장당 최대 15단어, 1~2음절 일상어, 의료 용어 시 괄호 설명, 불릿 포인트, 능동태 |
| Professional | ≥ 12 | 문장당 20단어 이상 복합 문장, 임상·약리 전문 용어, 라틴/그리스어 어근 미설명, Pathophysiology/Diagnostic Criteria/Therapeutic Approach/Clinical Considerations 구조 |

---

### 3.3 Infrastructure Layer

#### 3.3.1 벡터 스토어 (`infra/vector_store.py`)

```
PDF 파일 (data/)
    │
    ▼
[PyMuPDF (fitz)]  텍스트 추출
    │
    ├─ 텍스트 있음 → 직접 사용
    └─ 텍스트 없음 (스캔 PDF) → [RapidOCR] → 텍스트 인식
    │
    ▼
[RecursiveCharacterTextSplitter]
  chunk_size=1000, chunk_overlap=60  (.env MEDICAL_RAG_CHUNK_MAX_CHARS 기준, 코드 기본값은 500)
    │
    ▼
[URL 청크 필터링]  (http://, https:// 포함 청크 제거)
    │
    ▼
[BAAI/bge-base-en-v1.5]  임베딩 (768차원)
    │
    ▼
[LangChain FAISS]  벡터 인덱스 저장
  db/msd_faiss.index/
    ├── index.faiss   # 벡터 인덱스 바이너리
    └── index.pkl     # Document 메타데이터
```

**로드 정책**: 앱 시작 시 `db/msd_faiss.index/`가 존재하면 로드만 수행. 자동 재빌드 없음.

#### 3.3.2 감사 로거 (`infra/audit_logger.py`)

**요청당 N+1행 설계 (v3.0 변경)**:
- **save_loop_log()**: critic 평가 완료마다 호출. `is_final=FALSE`, `final_answer=NULL`, `fk_grade=NULL`로 중간 상태 INSERT
- **save_audit_log()**: output_node 또는 fallback_node 완료 후 단 1회 호출. `is_final=TRUE`, `final_answer`, `fk_grade` 포함 INSERT
- **UPDATE 없음**: INSERT only 패턴
- **스레드 안전**: `threading.local()`로 스레드별 oracledb 커넥션 관리

```
critic_node 완료 (매 회차)
    │
    ▼
save_loop_log(state, request_id, eval_count)
    │
    ├─ q_total = 0.4·F + 0.4·AR + 0.2·CP (Tier 1은 NULL)
    ├─ is_final=FALSE, fk_grade=NULL
    └─ INSERT INTO rag_audit_log_bak (중간 행)

output_node 완료
    │
    ├─ fk_grade = flesch_kincaid_grade_en(state["answer"])  ← 출처·면책 조항 추가 전 영어 원문
    ├─ output_agent(state)  ← 출처·면책 조항 추가
    │
    ▼
save_audit_log(state, request_id, fk_grade=fk)
    │
    ├─ is_final=TRUE, final_answer 포함
    ├─ fk_grade = 영어 원문 FK Grade Level
    └─ INSERT INTO rag_audit_log_bak (최종 행)
```

**fk_grade NULL 조건:**
- is_final=FALSE 행: 항상 NULL (중간 평가)
- is_final=TRUE, is_fallback=FALSE (output 경로): fk 값
- is_final=TRUE, is_fallback=TRUE, best_answer 존재: fk 값 (best_answer 기준)
- is_final=TRUE, is_fallback=TRUE, best_answer 없음(원문 청크 그대로 제시): NULL

#### 3.3.3 RAGAS 및 FK Grade 평가 (`infra/evaluator.py`)

**RAGAS 평가 (Streamlit 이벤트 루프 충돌 방지):**

```
Streamlit 메인 스레드
    │  (자체 asyncio 이벤트 루프 보유)
    │
    ▼
ThreadPoolExecutor (max_workers=1)
    │  별도 워커 스레드
    ▼
asyncio.new_event_loop()  (새 이벤트 루프 생성)
    │
    ▼
asyncio.gather(
    faith.ascore(),   # Faithfulness
    arel.ascore(),    # Answer Relevance
    cpre.ascore()     # Context Precision
)
    │  timeout=120초
    ▼
OfficialRagasScores(faithfulness, answer_relevance, context_precision, hallu_flags)
```

**판정 LLM: Claude 고정 (`ragas_async_client()` / `ragas_model()`)**

`llm_factory(ragas_model(), provider="anthropic", client=AsyncAnthropic(...))`로 구성한다. `core/llm_client.py`의 `ContextVar[llm_provider]`(답변 생성용 OpenAI/Gemini 토글)와 완전히 분리되어 있어, 어떤 백엔드로 답변을 생성하든 채점은 항상 Claude(`ANTHROPIC_MODEL`, 기본 `claude-haiku-4-5-20251001`)가 수행한다. 같은 모델이 생성과 채점을 겸할 때 생기는 순환성(circularity) 편향을 피하기 위한 설계.

> **호환성 이슈**: 설치된 `ragas`(0.4.3)의 Anthropic 어댑터는 OpenAI/Google과 달리 `temperature`/`top_p`를 무조건 pass-through한다. Claude 5세대 모델은 두 값을 동시에 받으면 400 에러를 내므로, `llm_factory()` 반환 객체의 `llm.model_args`에서 `temperature`·`top_p` 키를 제거하고 모델 기본 샘플링에 맡긴다.

**성능평가 전용 지표 (`compute_ir_metrics`, `compute_deepeval_scores`)**

`disease`(STQS-240/ablation 정답 라벨)가 있는 요청에서만 critic_agent가 호출하며, Self-Correction Loop 게이트에는 관여하지 않고 DB 기록·성능 시각화 전용으로만 쓰인다.

| 함수 | 방식 | 판정 LLM |
|------|------|----------|
| `compute_ir_metrics(disease, context_sources)` | `context_sources` 파일명에 `disease`명 포함 여부로 Hit Rate(0/1)·MRR(1/rank) 계산 | 없음 (순수 문자열 매칭) |
| `compute_deepeval_scores(question, answer, context_chunks)` | DeepEval(Faithfulness/Answer Relevancy)을 RAGAS와 별개 프레임워크로 채점, `ThreadPoolExecutor(max_workers=2)`로 2개 지표 동시 호출(timeout=90초). Contextual Relevancy는 더 이상 사용하지 않음 | **gpt-4o-mini**(`DEEPEVAL_LLM_MODEL`, `deepeval.models.OpenAIModel` 경유) |

실패 시 두 함수 모두 `0.0`이 아닌 `None`을 반환해 DB에 `NULL`로 남긴다 (측정 실패를 낮은 점수로 오인하지 않도록).

**Flesch-Kincaid Grade Level 계산 (`flesch_kincaid_grade_en`):**

```python
# 공식: 0.39*(words/sentences) + 11.8*(syllables/words) - 15.59
# 영어 음절 수: 모음 그룹(vowel group) 개수로 근사 계산, 묵음 e 제거
# 입력: 영어 원문 (graph.py _output_node에서 output_agent 호출 전에 계산)
# 출력: Grade Level 값 (Consumer 목표 ≤9, Professional 목표 ≥12)
```

---

### 3.4 Core Layer

#### 3.4.1 LLM 클라이언트 (`core/llm_client.py`)

**LLM 백엔드 설계**: ContextVar를 이용한 스레드-안전 provider 관리 (현재 OpenAI만 사용)

```
ContextVar[llm_provider]  ← "openai"
    │
    └─ "openai"  → ChatOpenAI(model=gpt-4o-mini, api_key=OPENAI_API_KEY)
```

**모델 역할 분리:**

| 역할 | 모델 | 비고 |
|------|------|------|
| 사용자 분류 | gpt-4o-mini | `ContextVar[llm_provider]` 토글 (OpenAI/Gemini) |
| 쿼리 최적화 | gpt-4o-mini | 〃 |
| RAG 엔진 (답변 생성) | gpt-4o-mini (코드 기본값은 gpt-4o) | 〃 |
| RAGAS 평가 (F/AR/CP) | **claude-haiku-4-5-20251001** (`ANTHROPIC_MODEL`) | `ContextVar[llm_provider]`와 무관, 항상 Claude 고정 |
| DeepEval (성능평가 전용) | **gpt-4o-mini** (`DEEPEVAL_LLM_MODEL`) | `disease` 있는 STQS/ablation 요청에서만 호출. Faithfulness/Answer Relevancy 2종 |

---

## 4. 데이터 흐름 설계

### 4.1 정상 흐름 (Tier 0 성공, 조건 A)

```
사용자 질문 (한국어)
    │
    ▼
[level_classifier]
  LLM → user_level: "Consumer" | "Professional"
    │
    ▼
[query_rewriter]
  LLM → queries: ["common cold symptoms consumer"]
    │
    ▼
[rag_engine - Tier 0]
  ReAct 에이전트 → search_msd_manual("common cold symptoms consumer")
  → context: [청크1, 청크2] (BAAI/bge 코사인 유사도)
  → answer: "[Consumer Summary] Common cold symptoms include..."
    (Consumer 프롬프트: 15단어 이하 문장, 일상어 사용 — FK ≤9 목표)
    │
    ▼
[critic (_critic_node)]
  RAGAS 평가 (ThreadPoolExecutor)
  → critic_score(F): 0.91, answer_relevance_score(AR): 0.88, context_precision_score(CP): 0.85
  → 조건 A: F≥0.8 AND AR≥0.8 AND CP≥0.8 → output으로 라우팅
  → save_loop_log() → INSERT is_final=FALSE (loop_number=1)
    │
    ▼
[output (_output_node)]
  fk = flesch_kincaid_grade_en(state["answer"])  ← 영어 원문 기준, 출처·면책 추가 전
  output_agent(state) → 출처·면책 조항 추가 → answer: "[Consumer Summary] Common cold symptoms include...\n\nSource: MSD Manual - ...\n\nThis answer is generated based on the MSD Manual..."
  save_audit_log(state, request_id, fk_grade=fk)
  → INSERT is_final=TRUE (tier_path="0", q_total=0.884, fk_grade=8.5)
    │
    ▼
최종 답변 (영문, 출처·면책 포함) + 점수 카드 표시
```

### 4.2 에스컬레이션 흐름 (조건 A: Tier 0 → 1 → 2)

```
[critic] AR=0.19 < 0.3 (is_critically_low → 즉시 에스컬레이션)
  → save_loop_log() → INSERT is_final=FALSE
  → search_tier=1, tier_path="0→1", loop_count=0
    │
    ▼
[query_rewriter] 재최적화
    │
    ▼
[rag_engine - Tier 1]
  LLM 학습데이터 직접 생성
    │
    ▼
[critic] AR=0.54 < 0.8 (Tier 1: AR만 평가, 기준 미달)
  → save_loop_log() → INSERT is_final=FALSE (F=NULL, CP=NULL — Tier 1)
  → search_tier=2, tier_path="0→1→2", loop_count=0
    │
    ▼
[rag_engine - Tier 2]
  DuckDuckGo 웹검색 → answer 합성
    │
    ▼
[critic] F=0.84, AR=0.82, CP=0.81 → 성공
  → save_loop_log() → INSERT is_final=FALSE
  → output으로 라우팅
    │
    ▼
[output]
  fk = flesch_kincaid_grade_en(state["answer"])
  save_audit_log(state, request_id, fk_grade=fk)
  → INSERT is_final=TRUE (tier_path="0→1→2", is_escalated=true, final_tier=2)
```

### 4.3 Baseline 흐름 (조건 E: 즉시 출력)

```
main.ipynb → run_medical_self_corrective_rag(ablation_condition="E", query_index=7, ...)

[rag_engine - Tier 0] → answer 생성
    │
    ▼
[critic] F=0.62, AR=0.58 → 평가 완료
  조건 E (Baseline): 결과 무관 즉시 output
  → save_loop_log() → INSERT is_final=FALSE
    │
    ▼
[output]
  fk = flesch_kincaid_grade_en(state["answer"])
  save_audit_log(...)
  → INSERT is_final=TRUE (ablation_condition="E", query_index=7,
             tier_path="0", self_correction_count=0, is_escalated=false)
```

---

## 5. 컴포넌트 의존관계

```
app.py  (ablation_condition="A" 고정)
  └─ ui/sidebar.py
  └─ ui/step_renderers.py
  └─ ui/score_card.py
  └─ ui/result_panel.py
  └─ ui/pdf_uploader.py
  └─ graph.py (run_medical_self_corrective_rag)
       └─ agents/classifier.py
       │    └─ core/llm_client.py
       │    └─ utils/json_parser.py
       └─ agents/rewriter.py
       │    └─ core/llm_client.py
       └─ agents/rag_engine.py
       │    └─ core/llm_client.py
       │    └─ tools/vector_search.py
       │    │    └─ infra/vector_store.py
       │    └─ tools/web_search.py
       └─ agents/critic.py
       │    └─ infra/evaluator.py        ← RAGAS + flesch_kincaid_grade_en + IR/DeepEval(성능평가 전용)
       │         ├─ core/llm_client.py (RAGAS 판정 — Claude, AsyncAnthropic)
       │         └─ deepeval.models.OpenAIModel (DeepEval 판정 — gpt-4o-mini, OpenAI 직접 호출)
       └─ agents/output.py
       │    └─ core/llm_client.py
       └─ infra/audit_logger.py         ← save_loop_log + save_audit_log (N+1행 INSERT)
       └─ infra/evaluator.py            ← flesch_kincaid_grade_en (graph.py _output_node 호출)
       └─ config/settings.py (모든 임계값·모델명)

main.ipynb  (ablation_condition="A"/"E", STQS-240 × 2조건 = 480실험)
  └─ graph.py (run_medical_self_corrective_rag)  [동일 의존관계]

ui/dashboard/performance_viz.py
  └─ oracledb → Oracle Database (직접 쿼리)
  └─ matplotlib, seaborn (영어 텍스트만 사용 — 폰트 렌더링 안정성)
  └─ 8개 섹션 시각화 (RAGAS 비교 / Tier 분포+표 / 루프 수렴 / FK Grade / 처리 시간 / 분류기 / DeepEval 비교 / IR Hit Rate·MRR)

medical_rag_graph.py  (하위 호환 re-export 모듈 — 직접 사용 지양)
```

---

## 6. 설정 관리 설계

모든 임계값과 모델명은 `config/settings.py`에서 환경변수로 중앙 관리한다.

| 설정 항목 | 환경변수 | 기본값 | 용도 |
|-----------|---------|--------|------|
| `FAITHFULNESS_THRESHOLD` | `MEDICAL_RAG_FAITHFULNESS_THRESHOLD` | `0.8` | Self-Corrective Loop 성공 기준 (F) |
| `AR_THRESHOLD` | `MEDICAL_RAG_AR_THRESHOLD` | `0.8` | Self-Corrective Loop 성공 기준 (AR) / Tier 1 기준 |
| `CP_THRESHOLD` | `MEDICAL_RAG_CP_THRESHOLD` | `0.8` | Self-Corrective Loop 성공 기준 (CP) |
| `CRITICAL_AR_THRESHOLD` | `MEDICAL_RAG_CRITICAL_AR_THRESHOLD` | `0.3` | 즉시 에스컬레이션 조건 (AR) |
| `CRITICAL_F_THRESHOLD` | `MEDICAL_RAG_CRITICAL_F_THRESHOLD` | `0.3` | 즉시 에스컬레이션 조건 (F) |
| `CRITICAL_CP_THRESHOLD` | `MEDICAL_RAG_CRITICAL_CP_THRESHOLD` | `0.2` | 즉시 에스컬레이션 조건 (CP) |
| `MAX_LOOPS` | `MEDICAL_RAG_MAX_LOOPS` | `3` | Tier당 최대 재시도 횟수 |
| `OPENAI_MODEL` | `OPENAI_MODEL` | `gpt-4o` (현재 `.env`: `gpt-4o-mini`) | RAG 엔진 모델 |
| `EMBEDDING_MODEL` | `MEDICAL_RAG_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | 임베딩 모델 (배포: BAAI/bge-base-en-v1.5) |
| `CHUNK_MAX_CHARS` | `MEDICAL_RAG_CHUNK_MAX_CHARS` | `500` (현재 `.env`: `1000`) | 청크 최대 길이 |
| `CHUNK_OVERLAP` | `MEDICAL_RAG_CHUNK_OVERLAP` | `60` | 청크 오버랩 크기 |
| `RAG_TOP_K` | `MEDICAL_RAG_TOP_K` | `5` (현재 `.env`: `3`) | VectorDB 검색 상위 K개 |
| `PDF_OCR_ENABLED` | `MEDICAL_RAG_PDF_OCR` | `false` | 스캔 PDF OCR 활성화 |

---

## 7. 디렉토리 구조

```
rag_nonmun/
├── app.py                    # Streamlit 진입점 (항상 조건 A 실행)
├── graph.py                  # LangGraph 그래프 빌드 및 실행
├── main.ipynb                # 성능 평가 일괄 실험 (2조건 × 240질문 = 480건)
├── medical_rag_graph.py      # 하위 호환 re-export 모듈 (직접 사용 지양)
├── launch.py                 # CLI 실행 스크립트
│
├── agents/                   # LangGraph 노드 에이전트
│   ├── classifier.py         # 사용자 수준 분류
│   ├── rewriter.py           # 쿼리 최적화
│   ├── rag_engine.py         # 검색 및 답변 합성 (Tier 0/1/2, FK Grade 목표 프롬프트)
│   ├── critic.py             # RAGAS 평가 + 라우팅 판단
│   └── output.py             # 출처·면책 조항 추가 및 최종 답변 완성
│
├── core/
│   └── llm_client.py         # LLM 클라이언트 (OpenAI)
│
├── infra/
│   ├── vector_store.py       # FAISS 인덱스 빌드 및 관리 (BAAI/bge-base-en-v1.5)
│   ├── evaluator.py          # RAGAS(Claude 판정) + flesch_kincaid_grade_en() + IR/DeepEval(gpt-4o-mini 판정, 성능평가 전용)
│   └── audit_logger.py       # Oracle DB 감사 로그 저장 (N+1행: save_loop_log + save_audit_log)
│
├── tools/
│   ├── vector_search.py      # FAISS 검색 도구 (LangChain Tool)
│   └── web_search.py         # DuckDuckGo 웹검색 도구
│
├── models/
│   └── state.py              # GraphState TypedDict 정의
│
├── config/
│   └── settings.py           # 전체 환경변수 설정 중앙 관리
│
├── ui/
│   ├── constants.py          # SESSION_DEFAULTS, TIER_CONFIGS
│   ├── sidebar.py            # 사이드바 UI
│   ├── header.py             # 헤더 및 안내
│   ├── pdf_uploader.py       # PDF 업로드 UI
│   ├── step_renderers.py     # LangGraph 단계별 실시간 렌더링
│   ├── score_card.py         # RAGAS 점수 카드
│   ├── result_panel.py       # 최종 답변 패널
│   └── dashboard/
│       ├── log_viewer.py     # 로그 조회 화면
│       ├── log_list.py       # 로그 목록 컴포넌트
│       ├── log_detail.py     # 로그 상세 컴포넌트
│       ├── log_query.py      # 로그 DB 쿼리
│       └── performance_viz.py # 성능 시각화 대시보드 (8개 섹션)
│
├── utils/
│   └── json_parser.py        # LLM JSON 응답 파싱 유틸리티
│
├── data/                     # MSD 매뉴얼 PDF 원본
├── db/                       # FAISS 인덱스 폴더
│   └── msd_faiss.index/
│       ├── index.faiss       # 벡터 인덱스 바이너리 (BAAI/bge-base-en-v1.5, 768차원)
│       └── index.pkl         # 청크 텍스트 및 출처 메타데이터
├── docs/                     # 시스템 산출 문서
├── .env                      # API 키 및 환경변수 (비공개)
├── requirements.txt          # Python 패키지 의존성
└── pyproject.toml            # Poetry 프로젝트 설정
```

---

## 8. 핵심 설계 결정사항

### 8.1 LangGraph StateGraph 채택

**결정**: 순수 Python 파이프라인 대신 LangGraph StateGraph 사용  
**이유**: 조건부 라우팅(Self-Corrective Loop, 에스컬레이션, Proposal System/Baseline 분기)을 선언적으로 정의할 수 있고, 노드 실행 순서를 그래프 구조로 명확히 표현 가능. `stream_mode="updates"`로 실시간 UI 업데이트 지원.

### 8.2 RAGAS 비동기 평가 격리

**결정**: Streamlit 메인 스레드와 별도로 ThreadPoolExecutor + 새 asyncio 이벤트 루프에서 RAGAS 실행  
**이유**: Streamlit은 자체 asyncio 이벤트 루프를 보유하며, `asyncio.run()`을 직접 호출 시 "This event loop is already running" 오류 발생. 별도 스레드에서 새 루프 생성으로 충돌 방지.

### 8.3 영문 쿼리 최적화

**결정**: 사용자 한국어 질문을 영문으로 변환하여 FAISS 검색  
**이유**: MSD Manual PDF가 전부 영어로 작성되어 있으므로, 한국어 쿼리로 검색 시 임베딩 유사도가 낮음. 영문 의료 학술 용어로 변환하여 검색 정확도 향상.

### 8.4 요청당 N+1행 감사 로그 (v3.0 변경)

**결정**: 완료 후 단 1회 INSERT하는 v2.0 설계에서 평가마다 INSERT(is_final=FALSE) + 완료 시 1회 INSERT(is_final=TRUE)하는 N+1행 설계로 전환  
**이유**: v2.0의 단일 행 설계로는 Self-Corrective Loop의 중간 RAGAS 점수 변화를 추적할 수 없음. N+1행 설계로 루프별 점수 변화, 에스컬레이션 시점, 자가 교정 효과를 세밀하게 분석 가능.

### 8.5 Tier 1 AR 단독 평가

**결정**: Tier 1(LLM 학습데이터)에서는 AR만으로 성공 여부를 판단  
**이유**: Tier 1은 외부 컨텍스트 청크가 없으므로 Faithfulness(근거성)와 Context Precision(청크 유효성)을 의미있게 계산할 수 없음. 중간 로그에도 F, CP는 NULL로 저장.

### 8.6 BAAI/bge-base-en-v1.5 임베딩 모델

**결정**: 기본값인 all-MiniLM-L6-v2 대신 BAAI/bge-base-en-v1.5 사용  
**이유**: bge-base 모델은 영문 의료 도메인에서 더 높은 검색 정확도를 제공하며(768차원 vs 384차원), 정규화된 벡터를 생성하므로 L2 인덱스가 코사인 유사도와 동치.

### 8.7 oracledb 직접 연결

**결정**: ORM 대신 oracledb로 Oracle Database 직접 연결  
**이유**: 감사 로그는 INSERT only 패턴이며, ORM 도입 시 의존성이 늘어남. 스레드별 커넥션 관리로 Streamlit 멀티스레드 환경에 대응.

### 8.8 싱글턴 그래프 인스턴스

LangGraph 그래프는 모듈 수준 싱글턴으로 관리하여 매 요청마다 재컴파일하지 않는다:

```python
_compiled_graph = None

def _get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph
```

### 8.9 FK Grade 출처·면책 조항 추가 전 계산

**결정**: Flesch-Kincaid Grade Level을 `output_agent`(출처·면책 조항 추가) 호출 전 영어 원문 기준으로 계산  
**이유**: FK Grade는 영어 텍스트 기반 가독성 지표이며, `output_agent`가 덧붙이는 출처·면책 문구가 섞이면 값이 왜곡됨. `graph.py`의 `_output_node`에서 `flesch_kincaid_grade_en(state["answer"])`를 `output_agent` 호출 전에 실행하고, `save_audit_log(fk_grade=fk)`로 전달함. Fallback에서 원문 청크를 그대로 제시하는 경우(best_answer 없음)는 FK 계산이 의미 없으므로 NULL 저장.

### 8.10 matplotlib 영어 텍스트 전용

**결정**: 성능 시각화 대시보드의 matplotlib 차트 텍스트를 모두 영어로 작성  
**이유**: 배포 환경(Streamlit Cloud 포함)에서 한국어 폰트가 설치되지 않아 matplotlib 차트에 깨진 문자(□□□)가 출력됨. Streamlit `st.dataframe()`은 한국어를 정상 렌더링하므로 표(DataFrame)는 한국어 허용, 차트(matplotlib)는 영어만 사용.

### 8.11 RAGAS 판정 LLM을 Claude로 고정 (순환성 회피)

**결정**: RAGAS 판정 LLM(`ragas_async_client()`/`ragas_model()`)을 답변 생성 LLM(OpenAI/Gemini 토글)과 무관하게 항상 Claude(`ANTHROPIC_MODEL`)로 고정  
**이유**: Critic Agent가 런타임 Quality Gate로 RAGAS를 쓰는 동시에 같은 지표로 최종 성능평가까지 하면, "같은 모델이 최적화하고 같은 모델이 채점"하는 self-grading 순환성 편향이 학술적으로 비판받을 수 있음. 판정 LLM을 답변 생성 LLM과 별도 provider로 분리해 이를 회피.

### 8.12 성능평가 전용 지표 추가 — IR Hit Rate/MRR, TruLens RAG Triad

**결정**: `disease`(STQS-240/ablation 정답 라벨)가 있는 요청에서만 critic_agent가 IR Hit Rate/MRR과 TruLens RAG Triad(Gemini 판정)를 추가로 계산해 DB에 기록. Self-Correction Loop 게이트(`check_faithfulness`/`is_critically_low`)에는 관여하지 않음  
**이유**: (1) RAGAS만으로 최종 성능평가를 하면 8.11과 동일한 순환성 문제가 남음 — TruLens라는 별도 프레임워크·별도 판정 모델(Gemini)로 F/AR/CP를 교차검증. (2) 전통적 IR 지표(Hit Rate/MRR)로 검색 성능 자체를 LLM 판정 없이 독립적으로 검증. (3) `disease`가 없는 일반 운영 쿼리는 ground truth가 없거나(IR) 교차검증 목적이 없어(TruLens) 계산을 생략해 불필요한 LLM 호출 비용을 막음.

### 8.13 판정 프레임워크 교체 — TruLens → DeepEval, Contextual Relevancy 제거 (2026-08-30 / 2026-09-21)

**결정 1 (2026-08-30)**: 8.12에서 도입한 TruLens RAG Triad(Context Relevance/Groundedness/Answer Relevance 3종, Gemini 판정)를 DeepEval(Faithfulness/Answer Relevancy/Contextual Relevancy 3종, 처음엔 Gemini 판정 유지)로 교체. `litellm`/`trulens-core`/`trulens-providers-litellm` 의존성 제거, `deepeval` 패키지 도입. 이후 판정 LLM도 비용·단순화 목적으로 Gemini → **OpenAI gpt-4o-mini**(`DEEPEVAL_LLM_MODEL`)로 통일 — 이 시점부터 판정 LLM 간 독립성(circularity 회피 목적)은 더 이상 완전히 보장되지 않는다.  
**결정 2 (2026-09-21)**: DeepEval 3종 중 `Contextual Relevancy`를 사용 중단 — Faithfulness/Answer Relevancy 2종만 유지. 대시보드 Table 7(DeepEval Metric Comparison), Oracle 컬럼 `deepeval_contextual_relevancy`도 함께 제거.  
**영향**: `GraphState`·`rag_audit_log_bak` 스키마·`ui/dashboard/performance_viz.py` 모두 2개 지표(`deepeval_faithfulness`, `deepeval_answer_relevancy`) 기준으로 갱신됨 (§3.2.1, §3.3, DB_데이터베이스설계서.md v5.3~v5.6 참고).

---

## 9. 보안 및 운영 설계

### 9.1 API 키 관리

- 모든 API 키는 `.env` 파일에만 저장, 소스코드 하드코딩 금지
- `python-dotenv`로 환경변수 로드, Streamlit Cloud는 `st.secrets`에서 자동 주입

### 9.2 오류 처리 전략

| 오류 유형 | 처리 방식 |
|-----------|-----------|
| LLM API 오류 | `max_retries=6` 자동 재시도 |
| RAGAS 평가 실패 | 각 메트릭별 개별 try/except, 실패 시 0.0 반환 (게이트 판단에 쓰이므로 안전한 기본값) |
| RAGAS 타임아웃 | `future.result(timeout=120)` 초과 시 0.0 반환 |
| DeepEval 평가 실패/타임아웃 | 지표별 개별 try/except, `future.result(timeout=90)`. 게이트에 쓰이지 않으므로 0.0이 아닌 `None` 반환 → DB NULL |
| DB 커넥션 오류 | 커넥션 초기화 후 로그만 기록, 시스템 계속 실행 |
| 모든 Tier 소진 | Fallback 노드로 라우팅, 원문 제시 |

---

*본 문서는 논문 연구 목적의 시스템 산출물이며, 실제 임상 적용을 위한 의학적 검증은 포함하지 않습니다.*
