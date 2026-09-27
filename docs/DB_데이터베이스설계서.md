# 데이터베이스 설계서 (Database Design Document)

**프로젝트명**: 의료 정보 자기교정 RAG 시스템  
**문서버전**: v5.2  
**작성일**: 2026-06-20  
**작성자**: 연구자

---

## 1. 개요

### 1.1 목적

본 문서는 의료 정보 자기교정 RAG 시스템에서 사용하는 데이터베이스의 논리적·물리적 설계를 정의한다. 시스템은 Oracle Database를 사용하여 모든 질의 처리 이력과 RAGAS 평가 결과를 감사 로그로 저장한다.

**버전별 변경 이력:**

| 버전 | 주요 변경사항 |
|------|--------------|
| v1.0 | 루프마다 INSERT + 최종 UPDATE 설계 |
| v2.0 | 완료 후 단 1회 INSERT (tier_path, self_correction_count 추가) |
| v3.0 | 평가마다 INSERT + 완료 시 1회 INSERT (loop_number, is_final 추가) |
| v4.0 | `fk_grade` 컬럼 추가 (Flesch-Kincaid Grade Level, is_final=TRUE 행만 기록) |
| v5.0 | `hallucination_detected`, `hallucination_count` 컬럼 제거 |
| v5.1 | `final_answer` VARCHAR2→CLOB, `query_index` 범위 1-240으로 확대, PostgreSQL → Oracle Database 마이그레이션 |
| v5.2 | 성능평가 전용 지표 5종 추가: `ir_hit_rate`, `ir_mrr`, `trulens_context_relevance`, `trulens_groundedness`, `trulens_answer_relevance`. critic 루프의 Quality Gate(F/AR/CP)와는 분리되어 기록 전용으로만 사용. 동시에 RAGAS 판정 LLM을 **Claude**로 고정(답변 생성 LLM과 분리)하고 TruLens 판정 LLM은 **Gemini**로 지정 — RAGAS 단독·동일 모델 평가의 순환성(circularity) 비판을 피하기 위한 독립 교차검증 설계 |
| v5.3 (2026-08-30) | 판정 프레임워크를 TruLens → DeepEval로 교체 (판정 LLM은 Gemini로 동일 유지). `trulens_context_relevance`/`trulens_groundedness`/`trulens_answer_relevance` 3개 컬럼 삭제, `deepeval_contextual_relevancy`/`deepeval_faithfulness`/`deepeval_answer_relevancy` 3개 컬럼 추가 |
| v5.4/v5.5 | `deepeval_contextual_precision`/`deepeval_contextual_recall`로 실험했다가 `deepeval_contextual_relevancy`로 롤백. DeepEval 판정 LLM을 Gemini → **OpenAI gpt-4o-mini**(`DEEPEVAL_LLM_MODEL`)로 통일 (RAGAS 판정 LLM도 이미 Claude 고정, 비용·단순화 목적) |
| v5.6 (2026-09-21) | `deepeval_contextual_relevancy` 컬럼 삭제 — DeepEval은 Faithfulness/Answer Relevancy 2종만 유지. **실제 운영 테이블은 `rag_audit_log_bak`** (`infra/audit_logger.py`가 INSERT하는 대상). 아래 3.1의 `rag_audit_log` 정의는 동일 스키마의 참고용 템플릿이며, 이 문서의 DDL 예시를 실제 적용할 때는 테이블명을 `rag_audit_log_bak`로 바꿔야 한다 |

### 1.2 데이터베이스 구성

| 항목 | 내용 |
|------|------|
| **DBMS** | Oracle Database |
| **연결 방식** | oracledb 직접 연결 (ORACLE_USER / ORACLE_PASSWORD / ORACLE_DSN) |
| **스키마** | 기본 사용자 스키마 |
| **테이블 수** | 1개 (실제 운영 테이블은 `rag_audit_log_bak` — `infra/audit_logger.py`가 INSERT하는 대상. `rag_audit_log`는 동일 스키마의 미사용 템플릿) |
| **로컬 스토리지** | FAISS 인덱스 폴더 (`db/msd_faiss.index/`) |

> **참고**: FAISS 벡터 인덱스는 관계형 DB가 아닌 로컬 바이너리 파일로 관리하며, 본 문서의 논리/물리 모델은 Oracle 감사 로그 테이블을 대상으로 한다.

---

## 2. 논리적 데이터 모델 (Logical Data Model)

### 2.1 핵심 설계 원칙: 평가 회차마다 1행 (v3.0), fk_grade 추가 (v4.0)

매 critic 평가 후 **즉시 1행 INSERT(is_final=FALSE)**하고, output/fallback 완료 시 **추가로 1행 INSERT(is_final=TRUE)**한다.

| v1.0 (구) | v2.0 | v3.0 | v4.0 | v5.1 (현재) |
|-----------|------|------|------|-------------|
| 루프마다 INSERT + 최종 UPDATE | 완료 후 단 1회 INSERT | 평가마다 INSERT + 완료 시 1회 INSERT | v3.0 + `fk_grade` 컬럼 추가 | Oracle DB, final_answer CLOB, query_index 1-240 |
| tier_id + loop_count 컬럼 | final_tier + tier_path 컬럼 | loop_number + is_final 컬럼 추가 | is_final=TRUE 행에만 fk_grade 기록 | 동일 |
| request_id당 N행 | request_id당 1행 | request_id당 N+1행 (N=평가 횟수) | 동일 | 동일 |

**저장 흐름**:
- `_critic_node` 완료마다 → `save_loop_log()` → is_final=**0**, loop_number=eval_count, final_answer=NULL, fk_grade=NULL
- `_output_node` 완료 시 → `save_audit_log(fk_grade=fk)` → is_final=**1**, final_answer(CLOB) 포함, **fk_grade 포함**
- `_fallback_node` 완료 시 → `save_audit_log(fk_grade=None)` → is_final=**1**, is_fallback=1, fk_grade=NULL

**fk_grade 계산 시점**: `graph.py`의 `_output_node`에서 `output_agent` 호출(출처·면책 조항 추가) *전*에 영어 원문으로 `flesch_kincaid_grade_en()`을 실행한다. `output_agent`가 출처·면책 문구를 덧붙이면 순수 영어 답변이 아니게 되므로 반드시 그 전에 계산해야 한다.

### 2.2 엔터티 정의

#### 엔터티: 감사 로그 (Audit Log)

하나의 의료 질의 요청에 대한 처리 결과를 담는 레코드 (request_id당 N+1행).

| 속성명 | 설명 | 타입 | 필수 |
|--------|------|------|------|
| 로그ID | 행 고유 식별자 (시퀀스 자동 증가) | NUMBER | ✅ |
| 요청ID | 워크플로우 전체 UUID | VARCHAR2(36) | ✅ |
| 생성일시 | INSERT 시각 (UTC) | TIMESTAMP WITH TIME ZONE | ✅ |
| **루프번호** | critic 평가 누적 횟수 (1부터 시작) | NUMBER(10) | ✅ |
| **최종여부** | 최종 출력 완료 행이면 1 | NUMBER(1) | ✅ |
| **실험 조건** | 시스템 조건 ('A'=Proposal System, 'E'=Baseline), 일반 운영 시 NULL | CHAR(1) | - |
| **질문 번호** | STQS-240 질문 순번 (1-240), 일반 운영 시 NULL | NUMBER(5) | - |
| **질환명** | STQS-240 대상 질환, 일반 운영 시 NULL | VARCHAR2(50) | - |
| **정답 레이블** | STQS-240 수준 정답 ('P'/'C'), 일반 운영 시 NULL | CHAR(1) | - |
| 사용자수준 | 분류된 사용자 유형 | VARCHAR2(20) | ✅ |
| 원본질문 | 한국어 원문 질문 | VARCHAR2(1000) | ✅ |
| 최적화쿼리 | 최종 영문 검색 쿼리 | VARCHAR2(1000) | - |
| **최종 티어** | 실제 답변이 생성된 검색 계층 | NUMBER(5) | ✅ |
| **티어 경로** | 에스컬레이션 경로 ("0"/"0→1"/"0→1→2") | VARCHAR2(20) | ✅ |
| **에스컬레이션여부** | Tier 이동 발생 여부 | NUMBER(1) | ✅ |
| Fallback여부 | Fallback으로 처리된 여부 | NUMBER(1) | ✅ |
| **자가교정횟수** | Tier 0 내 Self-Corrective Loop 누적 횟수 | NUMBER(5) | ✅ |
| Faithfulness | RAGAS F 점수 (0~1) | BINARY_DOUBLE | - |
| AnswerRelevance | RAGAS AR 점수 (0~1) | BINARY_DOUBLE | - |
| ContextPrecision | RAGAS CP 점수 (0~1) | BINARY_DOUBLE | - |
| **Q_total** | 종합 품질 점수 (0.4·F + 0.4·AR + 0.2·CP) | BINARY_DOUBLE | - |
| 검색문서수 | 검색된 컨텍스트 청크 수 | NUMBER(10) | - |
| LLM모델 | 사용된 LLM 백엔드 식별자 | VARCHAR2(50) | - |
| 실행시간 | 전체 워크플로우 소요 시간 (ms) | NUMBER(10) | - |
| 최종답변 | 출처·면책 조항이 포함된 최종 영어 답변 | **CLOB** | - |
| **FK Grade** | Flesch-Kincaid Grade Level (영어 원문 기준) | BINARY_DOUBLE | - |
| **IR Hit Rate** | top-k 검색 결과 내 정답 문서(disease 라벨 기준) 적중 여부 (0/1). 일반 운영 시 NULL | BINARY_DOUBLE | - |
| **IR MRR** | 정답 문서가 처음 등장한 순위의 역수 (1/rank), 미적중 시 0. 일반 운영 시 NULL | BINARY_DOUBLE | - |
| DeepEval Faithfulness | DeepEval 기반 근거성 (RAGAS F 교차검증용, 판정 LLM: gpt-4o-mini) | BINARY_DOUBLE | - |
| DeepEval Answer Relevancy | DeepEval 기반 답변 관련성 (RAGAS AR 교차검증용, 판정 LLM: gpt-4o-mini) | BINARY_DOUBLE | - |

### 2.3 ERD (단일 테이블)

```
┌────────────────────────────────────────────────────────────┐
│                    감사 로그 (Audit Log)                     │
├────────────────────────────────────────────────────────────┤
│ PK  log_id               NUMBER          NOT NULL          │
│     request_id           VARCHAR2(36)    NOT NULL          │
│     created_at           TIMESTAMP(TZ)   NOT NULL          │
│     loop_number          NUMBER(10)      NOT NULL          │
│     is_final             NUMBER(1)       NOT NULL DEFAULT 0│
│  ── 실험 조건 ─────────────────────────────────────────    │
│     ablation_condition   CHAR(1)         NULL  ('A'/'E')   │
│     query_index          NUMBER(5)       NULL  (1-240)     │
│     disease              VARCHAR2(50)    NULL              │
│     query_level_label    CHAR(1)         NULL  ('P'/'C')   │
│  ── 요청 ──────────────────────────────────────────────    │
│     user_level           VARCHAR2(20)    NOT NULL          │
│     original_query       VARCHAR2(1000)  NOT NULL          │
│     optimized_query      VARCHAR2(1000)  NULL              │
│  ── 티어 라우팅 ────────────────────────────────────────    │
│     final_tier           NUMBER(5)       NOT NULL DEFAULT 0│
│     tier_path            VARCHAR2(20)    NOT NULL DEFAULT'0'│
│     is_escalated         NUMBER(1)       NOT NULL DEFAULT 0│
│     is_fallback          NUMBER(1)       NOT NULL DEFAULT 0│
│     self_correction_count NUMBER(5)      NOT NULL DEFAULT 0│
│  ── RAGAS 평가 ─────────────────────────────────────────   │
│     ragas_f              BINARY_DOUBLE   NULL  (0~1)       │
│     ragas_ar             BINARY_DOUBLE   NULL  (0~1)       │
│     ragas_cp             BINARY_DOUBLE   NULL  (0~1)       │
│     q_total              BINARY_DOUBLE   NULL  (0~1)       │
│  ── 시스템 ─────────────────────────────────────────────   │
│     retrieved_doc_count  NUMBER(10)      NULL              │
│     llm_model            VARCHAR2(50)    NULL              │
│     execution_time_ms    NUMBER(10)      NULL              │
│     final_answer         CLOB            NULL              │
│     fk_grade             BINARY_DOUBLE   NULL              │
└────────────────────────────────────────────────────────────┘
```

### 2.4 비즈니스 규칙

| 규칙 ID | 규칙 내용 |
|---------|-----------|
| BR-01 | user_level은 'Professional', 'Consumer', 'Baseline' 중 하나여야 한다 |
| BR-02 | final_tier는 0, 1, 2 중 하나여야 한다 |
| BR-04 | F, AR, CP, q_total은 0.0~1.0 범위여야 한다 |
| BR-05 | q_total = 0.4·F + 0.4·AR + 0.2·CP (audit_logger가 자동 계산) |
| BR-06 | ablation_condition은 'A'(Proposal System), 'E'(Baseline) 또는 NULL(일반 운영)이어야 한다 |
| BR-07 | query_level_label은 'P', 'C' 또는 NULL이어야 한다 |
| BR-08 | request_id당 N+1행이 존재한다 (N=critic 평가 횟수, 중간 N행 + 최종 1행) |
| BR-09 | is_escalated = (tier_path != "0") 로 자동 결정된다 |
| BR-10 | is_final=0 행: critic 노드 완료 직후 저장 (final_answer=NULL, fk_grade=NULL) |
| BR-11 | is_final=1 행: output/fallback 노드 완료 직후 저장 (final_answer(CLOB) 포함) |
| BR-12 | 최종 결과 집계 시 반드시 WHERE is_final = 1 필터를 사용한다 |
| BR-13 | fk_grade는 is_final=1이고 is_fallback=0인 행에만 값이 존재한다. 중간 행과 fallback 행은 NULL |
| BR-14 | fk_grade는 output_agent의 출처·면책 조항 추가 전 영어 원문 답변을 기준으로 계산한다 (Flesch-Kincaid Grade Level 공식) |
| BR-15 | Tier 1 행: F, CP, q_total은 NULL (컨텍스트 없음). AR만 유효 |

### 2.5 데이터 흐름 시나리오

#### 시나리오 A: Proposal System — Tier 0 첫 번째 평가 성공 (재시도 없음)

```
request_id = "abc-001", ablation_condition = "A"
→ 저장 행: 2행 (중간 1행 + 최종 1행)

| loop_number | is_final | final_tier | ragas_f | ragas_ar | q_total | fk_grade | final_answer |
|-------------|----------|------------|---------|----------|---------|----------|--------------|
|      1      |    0     |     0      |  0.930  |  0.921   |  0.908  |   NULL   |    NULL      |  ← save_loop_log
|      1      |    1     |     0      |  0.930  |  0.921   |  0.908  |  11.20   |  "Hypothyroidism..."  |  ← save_audit_log
```

#### 시나리오 B: Proposal System — Tier 0 자가 교정 2회 후 Tier 1 에스컬레이션 → 성공

```
request_id = "xyz-002", ablation_condition = "A"
→ 저장 행: 4행 (중간 3행 + 최종 1행)

| loop_number | is_final | final_tier | tier_path | ragas_f | ragas_ar | self_correction_count | fk_grade | final_answer |
|-------------|----------|------------|-----------|---------|----------|-----------------------|----------|--------------|
|      1      |    0     |     0      |    "0"    |  0.61   |  0.57    |           0           |   NULL   |     NULL     |
|      2      |    0     |     0      |    "0"    |  0.72   |  0.68    |           1           |   NULL   |     NULL     |
|      3      |    0     |     1      |   "0→1"   |  NULL   |  0.840   |           2           |   NULL   |     NULL     |  ← Tier1: F/CP=NULL
|      3      |    1     |     1      |   "0→1"   |  NULL   |  0.840   |           2           |   NULL   |  "Coronary artery..." |  ← Tier1 최종행: fk_grade=NULL
```

#### 시나리오 C: Proposal System — Tier 0 성공 (Consumer, fk_grade 낮음)

```
request_id = "cons-003", ablation_condition = "A", user_level="Consumer"
→ 저장 행: 2행

| loop_number | is_final | user_level | ragas_f | ragas_ar | q_total | fk_grade | final_answer |
|-------------|----------|------------|---------|----------|---------|----------|--------------|
|      1      |    0     |  Consumer  |  0.88   |  0.85    |  0.872  |   NULL   |     NULL     |
|      1      |    1     |  Consumer  |  0.88   |  0.85    |  0.872  |  8.50    |  "The common cold..."  |  ← fk_grade ≤ 9 목표
```

---

## 3. 물리적 데이터 모델 (Physical Data Model)

### 3.1 테이블 정의: `rag_audit_log`

```sql
-- ============================================================
-- rag_audit_log 테이블 생성 스크립트 (Oracle)
-- 버전: v5.6
-- ============================================================

BEGIN EXECUTE IMMEDIATE 'DROP TABLE rag_audit_log'; EXCEPTION WHEN OTHERS THEN IF SQLCODE != -942 THEN RAISE; END IF; END;
/
BEGIN EXECUTE IMMEDIATE 'DROP SEQUENCE rag_audit_log_seq'; EXCEPTION WHEN OTHERS THEN IF SQLCODE != -2289 THEN RAISE; END IF; END;
/

CREATE SEQUENCE rag_audit_log_seq
    START WITH 1
    INCREMENT BY 1
    NOCACHE
    NOCYCLE;
/

CREATE TABLE rag_audit_log (
    log_id                  NUMBER              NOT NULL,
    request_id              VARCHAR2(36)        NOT NULL,
    created_at              TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL,

    -- 루프 추적
    loop_number             NUMBER(10)          DEFAULT 1   NOT NULL,
    is_final                NUMBER(1)           DEFAULT 0   NOT NULL CHECK (is_final IN (0, 1)),

    -- 실험 조건
    ablation_condition      CHAR(1),
    query_index             NUMBER(5),                      -- STQS 질문 번호 (1-240)
    disease                 VARCHAR2(50),
    query_level_label       CHAR(1),                        -- 'P'/'C'

    -- 요청 정보
    user_level              VARCHAR2(20)        NOT NULL,
    original_query          VARCHAR2(1000)      NOT NULL,
    optimized_query         VARCHAR2(1000),

    -- 티어 라우팅
    final_tier              NUMBER(5)           DEFAULT 0   NOT NULL,
    tier_path               VARCHAR2(20)        DEFAULT '0' NOT NULL,
    is_escalated            NUMBER(1)           DEFAULT 0   NOT NULL CHECK (is_escalated IN (0, 1)),
    is_fallback             NUMBER(1)           DEFAULT 0   NOT NULL CHECK (is_fallback IN (0, 1)),
    self_correction_count   NUMBER(5)           DEFAULT 0   NOT NULL,

    -- RAGAS 평가 (Tier 1 행: ragas_f, ragas_cp, q_total = NULL)
    ragas_f                 BINARY_DOUBLE,
    ragas_ar                BINARY_DOUBLE,
    ragas_cp                BINARY_DOUBLE,
    q_total                 BINARY_DOUBLE,

    -- 시스템
    retrieved_doc_count     NUMBER(10),
    llm_model               VARCHAR2(50),
    execution_time_ms       NUMBER(10),
    final_answer            CLOB,               -- is_final=0 행은 NULL

    -- 가독성 지표
    fk_grade                BINARY_DOUBLE,      -- Flesch-Kincaid Grade Level (영어 원문 기준)

    -- 성능평가 전용 지표 (v5.6 기준, critic 루프 게이트와 무관, disease 라벨 있는 STQS 행만 기록)
    ir_hit_rate              BINARY_DOUBLE,      -- 0 또는 1 (top-k 내 정답 문서 적중 여부)
    ir_mrr                   BINARY_DOUBLE,      -- 0~1 (1/rank, 정답 문서 첫 등장 순위 역수)
    deepeval_faithfulness         BINARY_DOUBLE, -- DeepEval — RAGAS F 교차검증
    deepeval_answer_relevancy     BINARY_DOUBLE, -- DeepEval — RAGAS AR 교차검증

    CONSTRAINT pk_rag_audit_log PRIMARY KEY (log_id)
);
/

CREATE OR REPLACE TRIGGER rag_audit_log_bir
    BEFORE INSERT ON rag_audit_log
    FOR EACH ROW
BEGIN
    IF :NEW.log_id IS NULL THEN
        :NEW.log_id := rag_audit_log_seq.NEXTVAL;
    END IF;
END;
/
```

### 3.1.1 기존 테이블 마이그레이션 이력 (v5.1 → v5.6, ALTER TABLE)

이미 데이터가 쌓인 운영 테이블(`rag_audit_log_bak`)에는 위 `CREATE TABLE`(DROP 포함) 스크립트를 재실행하지 말고, 아래 `ALTER TABLE`로 컬럼만 변경한다. 전체 이력은 `db/create_table_bak.sql` 참고.

```sql
-- v5.1 → v5.2: 성능평가 전용 지표 5종 추가 (IR + TruLens, 이후 폐기됨)
ALTER TABLE rag_audit_log_bak ADD (
    ir_hit_rate               BINARY_DOUBLE,
    ir_mrr                    BINARY_DOUBLE,
    trulens_context_relevance BINARY_DOUBLE,
    trulens_groundedness      BINARY_DOUBLE,
    trulens_answer_relevance  BINARY_DOUBLE
);
/

-- v5.2 → v5.3: 판정 프레임워크 TruLens → DeepEval 교체
ALTER TABLE rag_audit_log_bak DROP (
    trulens_context_relevance,
    trulens_groundedness,
    trulens_answer_relevance
);
/
ALTER TABLE rag_audit_log_bak ADD (
    deepeval_contextual_relevancy  BINARY_DOUBLE,
    deepeval_faithfulness          BINARY_DOUBLE,
    deepeval_answer_relevancy      BINARY_DOUBLE
);
/

-- v5.5 → v5.6: Contextual Relevancy 사용 중단
ALTER TABLE rag_audit_log_bak DROP (
    deepeval_contextual_relevancy
);
/
```

기존 행은 컬럼 추가 시 전부 NULL로 채워지며(신규 컬럼 NULL 허용), 재계산 없이도 이후 INSERT부터 값이 채워진다. 컬럼 삭제(DROP) 시 해당 컬럼의 과거 데이터는 복구 불가능하게 사라진다.

### 3.2 컬럼 상세 명세

| 컬럼명 | 데이터 타입 | NULL 허용 | 기본값 | 설명 |
|--------|------------|----------|--------|------|
| `log_id` | NUMBER | NOT NULL | 시퀀스 자동 증가 | 행 고유 식별자 (PK) |
| `request_id` | VARCHAR2(36) | NOT NULL | — | 워크플로우 전체 고유 ID |
| `created_at` | TIMESTAMP WITH TIME ZONE | NOT NULL | `SYSTIMESTAMP` | INSERT 시각 (UTC) |
| `loop_number` | NUMBER(10) | NOT NULL | 1 | critic 평가 누적 회차 (save_audit_log에서는 최종 eval_count) |
| `is_final` | NUMBER(1) | NOT NULL | 0 | 1=최종 결과 행(final_answer 포함), 0=중간 평가 행 |
| `ablation_condition` | CHAR(1) | NULL | NULL | 시스템 조건 ('A'=Proposal System, 'E'=Baseline). NULL=일반 운영 |
| `query_index` | NUMBER(5) | NULL | NULL | STQS-240 질문 번호 (1-240). NULL=일반 운영 |
| `disease` | VARCHAR2(50) | NULL | NULL | STQS-240 질환명. NULL=일반 운영 |
| `query_level_label` | CHAR(1) | NULL | NULL | STQS-240 수준 정답 레이블 ('P'/'C'). NULL=일반 운영 |
| `user_level` | VARCHAR2(20) | NOT NULL | — | 'Professional', 'Consumer', 'Baseline' 중 하나 |
| `original_query` | VARCHAR2(1000) | NOT NULL | — | 사용자가 입력한 한국어 원본 질문 |
| `optimized_query` | VARCHAR2(1000) | NULL | — | 마지막 루프에서 사용된 영문 최적화 검색 쿼리 |
| `final_tier` | NUMBER(5) | NOT NULL | 0 | 최종 답변이 생성된 검색 계층 (0/1/2) |
| `tier_path` | VARCHAR2(20) | NOT NULL | '0' | 에스컬레이션 경로. "0" / "0→1" / "0→1→2" |
| `is_escalated` | NUMBER(1) | NOT NULL | 0 | Tier 에스컬레이션 발생 여부 (tier_path != "0") |
| `is_fallback` | NUMBER(1) | NOT NULL | 0 | 모든 Tier 소진으로 Fallback 처리 여부 |
| `self_correction_count` | NUMBER(5) | NOT NULL | 0 | Tier 0 내 자가 교정 누적 횟수 |
| `ragas_f` | BINARY_DOUBLE | NULL | — | RAGAS Faithfulness (0~1). 판정 LLM: **Claude**(`ANTHROPIC_MODEL`, 답변 생성 LLM과 무관하게 고정). Tier 1 행은 NULL |
| `ragas_ar` | BINARY_DOUBLE | NULL | — | RAGAS Answer Relevance (0~1). 판정 LLM: Claude. 모든 Tier 사용 |
| `ragas_cp` | BINARY_DOUBLE | NULL | — | RAGAS Context Precision (0~1). 판정 LLM: Claude. Tier 1 행은 NULL |
| `q_total` | BINARY_DOUBLE | NULL | — | 종합 품질 점수: 0.4·F + 0.4·AR + 0.2·CP. Tier 1 행은 NULL |
| `retrieved_doc_count` | NUMBER(10) | NULL | — | 검색된 컨텍스트 청크 수 |
| `llm_model` | VARCHAR2(50) | NULL | — | 답변 생성에 사용된 LLM 백엔드 ('openai'/'gemini'). RAGAS 판정(Claude)·DeepEval 판정(gpt-4o-mini)과는 별개 |
| `execution_time_ms` | NUMBER(10) | NULL | — | 전체 워크플로우 소요 시간 (ms). is_final=0 행은 NULL |
| `final_answer` | **CLOB** | NULL | — | 출처·면책 조항이 포함된 최종 영어 답변. is_final=0 행은 NULL |
| `fk_grade` | BINARY_DOUBLE | NULL | — | Flesch-Kincaid Grade Level. is_final=1이고 is_fallback=0인 행만 기록. Consumer 목표 ≤9, Professional 목표 ≥12 |
| `ir_hit_rate` | BINARY_DOUBLE | NULL | — | 전통적 IR 지표. top-k 검색 결과 중 `disease` 라벨과 매칭되는 출처 문서가 하나라도 있으면 1, 없으면 0. `disease`가 NULL인 일반 운영 행은 ground truth가 없으므로 NULL. LLM 불필요(문자열 매칭). **critic 루프 게이트에는 미사용, 성능평가 전용** |
| `ir_mrr` | BINARY_DOUBLE | NULL | — | Mean Reciprocal Rank. `disease` 라벨과 매칭되는 첫 출처 문서의 순위 역수(1/rank), 미적중 시 0. `disease`가 NULL인 행은 NULL. **성능평가 전용** |
| `deepeval_faithfulness` | BINARY_DOUBLE | NULL | — | DeepEval. 판정 LLM: **gpt-4o-mini**(`DEEPEVAL_LLM_MODEL`, OpenAI 직접 호출). RAGAS `ragas_f`와 별도 프레임워크로 교차검증. **성능평가 전용, 게이트 미사용** |
| `deepeval_answer_relevancy` | BINARY_DOUBLE | NULL | — | DeepEval. 판정 LLM: gpt-4o-mini. RAGAS `ragas_ar`와 별도 프레임워크로 교차검증. **성능평가 전용, 게이트 미사용** |

### 3.3 인덱스 (Indexes)

```sql
CREATE INDEX idx_audit_ablation    ON rag_audit_log (ablation_condition);
CREATE INDEX idx_audit_request     ON rag_audit_log (request_id);
CREATE INDEX idx_audit_disease     ON rag_audit_log (disease);
CREATE INDEX idx_audit_query_index ON rag_audit_log (query_index);
CREATE INDEX idx_audit_final       ON rag_audit_log (request_id, is_final);
/
```

| 인덱스명 | 대상 컬럼 | 용도 |
|---------|----------|------|
| `pk_rag_audit_log` | `log_id` | PK 조회 |
| `idx_audit_ablation` | `ablation_condition` | 조건별 성능 비교 집계 |
| `idx_audit_request` | `request_id` | 단일 요청 조회 |
| `idx_audit_disease` | `disease` | 질환별 분석 |
| `idx_audit_query_index` | `query_index` | STQS-240 질문 번호별 분석 |
| `idx_audit_final` | `(request_id, is_final)` | 최종 결과만 조회 시 성능 향상 |

---

## 4. 주요 쿼리 패턴 (Oracle SQL)

### 4.1 시스템 조건별 성능 비교 (Proposal System vs Baseline)

```sql
SELECT
    ablation_condition,
    COUNT(*)                                        AS n,
    ROUND(AVG(ragas_f),  3)                         AS avg_f,
    ROUND(AVG(ragas_ar), 3)                         AS avg_ar,
    ROUND(AVG(ragas_cp), 3)                         AS avg_cp,
    ROUND(AVG(q_total),  3)                         AS avg_q,
    ROUND(AVG(self_correction_count), 2)            AS avg_loops,
    SUM(CASE WHEN is_fallback = 1 THEN 1 ELSE 0 END)
        * 100.0 / COUNT(*)                          AS fallback_rate_pct
FROM rag_audit_log
WHERE ablation_condition IS NOT NULL
  AND is_final = 1
GROUP BY ablation_condition
ORDER BY ablation_condition;
```

### 4.2 사용자 수준 분류 정확도 (조건별)

```sql
SELECT
    ablation_condition,
    query_level_label                               AS answer_label,
    user_level                                      AS predicted_level,
    COUNT(*)                                        AS n,
    SUM(CASE
        WHEN (query_level_label = 'P' AND user_level = 'Professional')
          OR (query_level_label = 'C' AND user_level = 'Consumer')
        THEN 1 ELSE 0 END) * 100.0 / COUNT(*)       AS accuracy_pct
FROM rag_audit_log
WHERE query_level_label IS NOT NULL
  AND is_final = 1
GROUP BY ablation_condition, query_level_label, user_level
ORDER BY ablation_condition, query_level_label;
```

### 4.3 Self-Corrective Loop 효과 (자가 교정 횟수별 품질)

```sql
SELECT
    self_correction_count,
    COUNT(*)                            AS n,
    ROUND(AVG(ragas_f),  3)             AS avg_f,
    ROUND(AVG(ragas_ar), 3)             AS avg_ar,
    ROUND(AVG(q_total),  3)             AS avg_q
FROM rag_audit_log
WHERE ablation_condition = 'A'
  AND is_final = 1
GROUP BY self_correction_count
ORDER BY self_correction_count;
```

### 4.4 FK Grade 분포 (수준별 가독성 목표 달성률)

```sql
SELECT
    user_level,
    COUNT(*)                                            AS n,
    ROUND(AVG(fk_grade), 3)                             AS avg_fk_grade,
    ROUND(STDDEV(fk_grade), 3)                          AS std_fk_grade,
    SUM(CASE WHEN user_level = 'Consumer'     AND fk_grade <= 9  THEN 1 ELSE 0 END)
        * 100.0 / NULLIF(COUNT(*), 0)                   AS within_target_pct,
    SUM(CASE WHEN user_level = 'Professional' AND fk_grade >= 12 THEN 1 ELSE 0 END)
        * 100.0 / NULLIF(COUNT(*), 0)                   AS within_target_pct_pro
FROM rag_audit_log
WHERE is_final = 1
  AND is_fallback = 0
  AND fk_grade IS NOT NULL
GROUP BY user_level;
```

### 4.5 감사 로그 목록 조회 (페이지네이션)

```sql
SELECT
    log_id,
    request_id,
    CAST(created_at AT TIME ZONE 'Asia/Seoul' AS TIMESTAMP) AS created_at_kst,
    ablation_condition,
    user_level,
    original_query,
    final_tier,
    tier_path,
    ragas_f, ragas_ar, ragas_cp, q_total,
    fk_grade,
    is_escalated, is_fallback, execution_time_ms
FROM rag_audit_log
WHERE is_final = 1
ORDER BY created_at DESC
FETCH FIRST 20 ROWS ONLY;
```

### 4.6 단일 요청 상세 조회

```sql
SELECT *
FROM rag_audit_log
WHERE request_id = 'xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx'
ORDER BY loop_number;
```

---

## 5. 로컬 파일 스토리지 (FAISS 인덱스)

관계형 DB 외에 벡터 검색을 위한 로컬 파일 스토리지를 사용한다.

| 항목 | 내용 |
|------|------|
| **저장 경로** | `db/msd_faiss.index/` (폴더) |
| **파일 구성** | `index.faiss` (벡터 인덱스 바이너리) + `index.pkl` (청크 메타데이터) |
| **생성 방법** | 사이드바 "인덱스 전체 재빌드" 버튼 또는 pre-built 인덱스 배치 |
| **로드 정책** | 앱 시작 시 기존 인덱스가 있으면 로드만 수행 (자동 재빌드 없음) |
| **임베딩 모델** | `BAAI/bge-base-en-v1.5` (768차원, 코사인 유사도) |
| **청크 크기** | 최대 1000자, 60자 오버랩 (`.env`의 `MEDICAL_RAG_CHUNK_MAX_CHARS` 기준, 코드 기본값은 500자) |
| **URL 필터링** | http://, https:// 포함 청크는 인덱싱 제외 |

### FAISS 인덱스 구조

```
db/
└── msd_faiss.index/
    ├── index.faiss    # LangChain FAISS 벡터 인덱스 (IndexFlatL2 + bge-base 정규화 벡터 = 코사인 동치)
    └── index.pkl      # LangChain Document 객체 (청크 텍스트 + 출처 메타데이터)
```

---

## 6. 데이터 생명주기

```
[질의 제출]
    │
    ▼
[워크플로우 실행]
  level_classifier → query_rewriter → rag_engine → critic
  (Self-Corrective Loop / 에스컬레이션 / A·E 조건 분기)
    │
    ▼
[critic 평가 완료 (매 회차마다)]
    │  save_loop_log() → INSERT is_final=0
    │  (loop_number, RAGAS 점수, tier 정보 포함 / final_answer=NULL / fk_grade=NULL)
    ▼
[output 완료]
    │  fk_grade = flesch_kincaid_grade_en(영어 원문)  ← 출처·면책 조항 추가 전 계산
    │  output_agent() → 출처·면책 조항 추가
    │  save_audit_log(fk_grade=fk) → INSERT is_final=1
    │  (final_answer(CLOB), execution_time_ms, fk_grade 포함)
    │  UPDATE 없음
    ▼
[fallback 완료]
    │  save_audit_log(is_fallback=1, fk_grade=None) → INSERT is_final=1
    │  (fk_grade=NULL: 혼합 언어 답변은 FK 계산 불가)
    ▼
[감사 로그 완성 — request_id당 N+1행]
    │
    ├─ 대시보드 로그 조회 (fetch_logs / fetch_detail)
    ├─ 성능 시각화 집계 (performance_viz._load_data)
    └─ 성능 비교 분석 (Proposal System / Baseline 집계 쿼리)
```

---

## 7. 데이터 보존 및 관리

| 항목 | 정책 |
|------|------|
| **보존 기간** | 연구 기간 전체 (별도 삭제 정책 없음) |
| **백업** | Oracle Database 백업 정책에 따름 |
| **접근 권한** | ORACLE_USER / ORACLE_DSN 설정에 따라 접근 제한 |
| **개인정보** | 사용자 식별 정보 미포함 (request_id는 랜덤 UUID) |

---

*본 문서는 논문 연구 목적의 시스템 산출물이며, 실제 임상 적용을 위한 의학적 검증은 포함하지 않습니다.*
