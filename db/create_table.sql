-- ============================================================
-- rag_audit_log 테이블 생성 스크립트 (Oracle)
-- 프로젝트: 의료 정보 자기교정 RAG 시스템
-- 버전: v5.7  (ir_hit_rate/ir_mrr 컬럼 제거, IR 지표 폐기)
-- ============================================================

-- 기존 객체 안전 삭제
BEGIN EXECUTE IMMEDIATE 'DROP TABLE rag_audit_log'; EXCEPTION WHEN OTHERS THEN IF SQLCODE != -942 THEN RAISE; END IF; END;
/
BEGIN EXECUTE IMMEDIATE 'DROP SEQUENCE rag_audit_log_seq'; EXCEPTION WHEN OTHERS THEN IF SQLCODE != -2289 THEN RAISE; END IF; END;
/

-- 시퀀스
CREATE SEQUENCE rag_audit_log_seq
    START WITH 1
    INCREMENT BY 1
    NOCACHE
    NOCYCLE;
/

-- 테이블
CREATE TABLE rag_audit_log (
    log_id                  NUMBER              NOT NULL,
    request_id              VARCHAR2(36)        NOT NULL,
    created_at              TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL,

    -- 루프 추적
    loop_number             NUMBER(10)          DEFAULT 1   NOT NULL,
    is_final                NUMBER(1)           DEFAULT 0   NOT NULL CHECK (is_final IN (0, 1)),

    -- 실험 조건 ('A'=Proposal System, 'E'=Baseline, NULL=일반 운영)
    ablation_condition      CHAR(1),
    query_index             NUMBER(5),                      -- STQS 질문 번호 (1-240)
    disease                 VARCHAR2(50),
    query_level_label       CHAR(1),                        -- 'P'/'C'

    -- 요청 정보
    user_level              VARCHAR2(20)        NOT NULL,   -- 'Professional'/'Consumer'
    original_query          VARCHAR2(1000)      NOT NULL,
    optimized_query         VARCHAR2(1000),

    -- 티어 라우팅
    final_tier              NUMBER(5)           DEFAULT 0   NOT NULL,
    tier_path               VARCHAR2(20)        DEFAULT '0' NOT NULL,   -- '0'/'0→1'/'0→1→2'
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
    execution_time_ms       NUMBER(10),         -- is_final=0 행은 NULL
    final_answer            CLOB,               -- is_final=0 행은 NULL

    -- 가독성 지표 (is_final=1 & is_fallback=0 행만 기록)
    fk_grade                BINARY_DOUBLE,      -- Flesch-Kincaid Grade Level (영어 원문 기준)

    -- 성능평가 전용 지표 (critic 루프 게이트와 무관, disease 라벨 있는 STQS 행만 기록)
    deepeval_faithfulness          BINARY_DOUBLE,  -- DeepEval — RAGAS F 교차검증
    deepeval_answer_relevancy      BINARY_DOUBLE,  -- DeepEval — RAGAS AR 교차검증

    CONSTRAINT pk_rag_audit_log PRIMARY KEY (log_id)
);
/

-- PK 자동 증가 트리거
CREATE OR REPLACE TRIGGER rag_audit_log_bir
    BEFORE INSERT ON rag_audit_log
    FOR EACH ROW
BEGIN
    IF :NEW.log_id IS NULL THEN
        :NEW.log_id := rag_audit_log_seq.NEXTVAL;
    END IF;
END;
/

-- 인덱스
CREATE INDEX idx_audit_ablation    ON rag_audit_log (ablation_condition);
CREATE INDEX idx_audit_request     ON rag_audit_log (request_id);
CREATE INDEX idx_audit_disease     ON rag_audit_log (disease);
CREATE INDEX idx_audit_query_index ON rag_audit_log (query_index);
CREATE INDEX idx_audit_final       ON rag_audit_log (request_id, is_final);
/

-- ============================================================
-- 기존 운영 테이블(rag_audit_log_bak) 마이그레이션 이력 (v5.2 → v5.3, 2026-08-30)
-- 1) deepeval_* 3종 컬럼 추가
-- 2) trulens_* 3종 컬럼(및 과거 실험 데이터)은 완전 삭제 — 위 CREATE TABLE 정의와
--    동일한 최종 스키마로 맞춘 것이므로, 신규 배포 시에는 이 단계가 필요 없다.
-- ============================================================
-- ALTER TABLE rag_audit_log_bak ADD (
--     deepeval_contextual_relevancy  BINARY_DOUBLE,
--     deepeval_faithfulness          BINARY_DOUBLE,
--     deepeval_answer_relevancy      BINARY_DOUBLE
-- );
-- /
-- ALTER TABLE rag_audit_log_bak DROP (
--     trulens_context_relevance,
--     trulens_groundedness,
--     trulens_answer_relevance
-- );
-- /

-- ============================================================
-- 롤백 (v5.4/v5.5 → v5.3 복귀, 2026-09-03)
-- 세션 중 deepeval_contextual_precision/recall로 실험했다가 CR 사용 시점으로
-- 되돌림. precision/recall 컬럼은 삭제, contextual_relevancy 컬럼 재생성.
-- 단, contextual_relevancy의 과거 데이터는 이미 삭제되어 복구 불가 — 빈 컬럼만 복원.
-- ============================================================
-- ALTER TABLE rag_audit_log_bak DROP (
--     deepeval_contextual_precision,
--     deepeval_contextual_recall
-- );
-- /
-- ALTER TABLE rag_audit_log_bak ADD (
--     deepeval_contextual_relevancy  BINARY_DOUBLE
-- );
-- /

-- ============================================================
-- v5.6 (2026-09-21): deepeval_contextual_relevancy 제거
-- 더 이상 사용하지 않기로 결정 — DeepEval은 faithfulness/answer_relevancy
-- 2종만 유지.
-- ============================================================
-- ALTER TABLE rag_audit_log_bak DROP (
--     deepeval_contextual_relevancy
-- );
-- /

-- ============================================================
-- v5.7 (2026-09-30): ir_hit_rate/ir_mrr 제거 (라이브 DB에 이미 반영됨)
-- IR(Hit Rate/MRR) 지표를 더 이상 사용하지 않기로 결정 — 논문 지표에서 제외.
-- ============================================================
-- ALTER TABLE rag_audit_log_bak DROP (
--     ir_hit_rate,
--     ir_mrr
-- );
-- /
