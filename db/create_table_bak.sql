-- ============================================================
-- rag_audit_log_bak 테이블 생성 스크립트 (Oracle)
-- 실제 운영 중인 테이블 (infra/audit_logger.py가 INSERT하는 대상).
-- 버전: v5.7  (ir_hit_rate/ir_mrr 컬럼 제거, IR 지표 폐기)
-- ============================================================

-- 기존 객체 안전 삭제
BEGIN EXECUTE IMMEDIATE 'DROP TABLE rag_audit_log_bak'; EXCEPTION WHEN OTHERS THEN IF SQLCODE != -942 THEN RAISE; END IF; END;
/
BEGIN EXECUTE IMMEDIATE 'DROP SEQUENCE rag_audit_log_bak_seq'; EXCEPTION WHEN OTHERS THEN IF SQLCODE != -2289 THEN RAISE; END IF; END;
/

-- 시퀀스
CREATE SEQUENCE rag_audit_log_bak_seq
    START WITH 1
    INCREMENT BY 1
    NOCACHE
    NOCYCLE;
/

-- 테이블
CREATE TABLE rag_audit_log_bak (
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

    CONSTRAINT pk_rag_audit_log_bak PRIMARY KEY (log_id)
);
/

-- PK 자동 증가 트리거
CREATE OR REPLACE TRIGGER rag_audit_log_bak_bir
    BEFORE INSERT ON rag_audit_log_bak
    FOR EACH ROW
BEGIN
    IF :NEW.log_id IS NULL THEN
        :NEW.log_id := rag_audit_log_bak_seq.NEXTVAL;
    END IF;
END;
/

-- 인덱스
CREATE INDEX idx_bak_ablation    ON rag_audit_log_bak (ablation_condition);
CREATE INDEX idx_bak_request     ON rag_audit_log_bak (request_id);
CREATE INDEX idx_bak_disease     ON rag_audit_log_bak (disease);
CREATE INDEX idx_bak_query_index ON rag_audit_log_bak (query_index);
CREATE INDEX idx_bak_final       ON rag_audit_log_bak (request_id, is_final);
/

-- ============================================================
-- 참고: 라이브 rag_audit_log_bak는 이미 trulens_* 3종 → deepeval_* 3종으로
-- 교체된 상태(2026-08-30, DB 직접 조회로 확인)지만 이 파일에는 그 ALTER가
-- 기록된 적이 없었음 — 위 CREATE TABLE 정의를 라이브 스키마와 일치하도록
-- 갱신(v5.4)하며 함께 정리함.
-- ============================================================

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
-- 2종만 유지. 컬럼 값은 전부 NULL이었으므로(v5.4/v5.5 시점에 추가된 컬럼이라
-- 기록된 실제 데이터 없음) 데이터 손실 없이 삭제 가능.
-- ============================================================
-- ALTER TABLE rag_audit_log_bak DROP (
--     deepeval_contextual_relevancy
-- );
-- /

-- ============================================================
-- v5.7 (2026-09-30): ir_hit_rate/ir_mrr 제거 (라이브 DB에 이미 반영됨)
-- IR(Hit Rate/MRR) 지표를 더 이상 사용하지 않기로 결정 — 논문 지표에서 제외.
-- 관련 코드(agents/critic.py의 compute_ir_metrics 호출, infra/evaluator.py의
-- IRMetrics/compute_ir_metrics, models/state.py의 hit_rate_score/mrr_score,
-- infra/audit_logger.py의 INSERT 컬럼)도 함께 제거함.
-- ============================================================
-- ALTER TABLE rag_audit_log_bak DROP (
--     ir_hit_rate,
--     ir_mrr
-- );
-- /
