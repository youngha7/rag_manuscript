"""Oracle rag_audit_log_bak 테이블에 RAGAS 평가 결과를 저장하는 모듈.

설계 원칙 (v3.0):
- save_loop_log : critic 평가 완료마다 무조건 INSERT (is_final=0). loop_number=eval_count.
- save_audit_log: output_node / fallback_node 완료 후 1회 INSERT (is_final=1). final_answer 포함.
- request_id당 N+1행 (N=critic 평가 횟수). 루프별 점수 변화 추적 가능.
- UPDATE 없음. INSERT only.
"""
import logging
import threading
import time
from typing import Optional

try:
    import oracledb
    _ORACLEDB_AVAILABLE = True
except ImportError:
    oracledb = None  # type: ignore
    _ORACLEDB_AVAILABLE = False
    import logging as _logging
    _logging.getLogger(__name__).warning(
        "oracledb 패키지가 없습니다. audit_log INSERT가 비활성화됩니다. "
        "`pip install oracledb` 후 재시작하세요."
    )

import config.settings as settings
from models.state import GraphState

logger = logging.getLogger(__name__)

_local = threading.local()

_INSERT_SQL = """
    INSERT INTO rag_audit_log_bak (
        request_id, loop_number, is_final,
        ablation_condition, query_index, disease,
        query_level_label, user_level,
        original_query, optimized_query,
        final_tier, tier_path,
        is_escalated, is_fallback, self_correction_count,
        ragas_f, ragas_ar, ragas_cp, q_total,
        retrieved_doc_count, llm_model, execution_time_ms,
        final_answer, fk_grade,
        ir_hit_rate, ir_mrr,
        deepeval_faithfulness, deepeval_answer_relevancy
    ) VALUES (
        :request_id, :loop_number, :is_final,
        :ablation_condition, :query_index, :disease,
        :query_level_label, :user_level,
        :original_query, :optimized_query,
        :final_tier, :tier_path,
        :is_escalated, :is_fallback, :self_correction_count,
        :ragas_f, :ragas_ar, :ragas_cp, :q_total,
        :retrieved_doc_count, :llm_model, :execution_time_ms,
        :final_answer, :fk_grade,
        :ir_hit_rate, :ir_mrr,
        :deepeval_faithfulness, :deepeval_answer_relevancy
    )
"""


def _get_conn():
    """스레드-로컬 oracledb 커넥션 반환. 끊어졌으면 재연결."""
    if not _ORACLEDB_AVAILABLE:
        raise RuntimeError("oracledb 패키지가 설치되어 있지 않습니다.")
    conn = getattr(_local, "conn", None)
    if conn is not None:
        try:
            conn.ping()
        except Exception:
            conn = None
    if conn is None:
        if not settings.ORACLE_DSN:
            raise RuntimeError("ORACLE_DSN이 설정되지 않았습니다.")
        conn = oracledb.connect(
            user=settings.ORACLE_USER,
            password=settings.ORACLE_PASSWORD,
            dsn=settings.ORACLE_DSN,
        )
        conn.autocommit = True
        _local.conn = conn
    return conn


def _build_tier_path(log: list, final_tier: int) -> str:
    """Extract escalation path from state['log'] as fallback when tier_path not set."""
    path = [0]
    for line in log:
        if ("Tier 1 escalation" in line or "Tier 1 에스컬레이션" in line) and 1 not in path:
            path.append(1)
        if ("Tier 2 escalation" in line or "Tier 2 에스컬레이션" in line) and 2 not in path:
            path.append(2)
    return "→".join(str(t) for t in path)


def _build_row(
    state: GraphState,
    request_id: str,
    loop_number: int,
    is_final: bool,
    is_fallback: bool,
    execution_time_ms: Optional[int],
    final_answer: Optional[str],
    fk_grade: Optional[float] = None,
) -> dict:
    """공통 row dict 생성. BOOLEAN → NUMBER(1) 변환 포함."""
    final_tier = state["search_tier"]
    ar = float(state.get("answer_relevance_score", 0.0) or 0.0)

    f_raw  = state.get("critic_score")
    cp_raw = state.get("context_precision_score")

    if final_tier == 1:
        # Tier 1은 context 없이 LLM 지식만 사용 → F·CP 무의미, Q_total = AR만으로 산출
        f       = None
        cp      = None
        q_total = round(ar, 6)
    elif f_raw is None or cp_raw is None:
        f       = None
        cp      = None
        q_total = None
    else:
        f       = float(f_raw)
        cp      = float(cp_raw)
        q_total = round(0.4 * f + 0.4 * ar + 0.2 * cp, 6)

    tier_path    = (state.get("tier_path") or "").strip()
    if not tier_path:
        tier_path = _build_tier_path(state.get("log", []), final_tier)
    is_escalated = (tier_path != "0")


    return {
        "request_id":            request_id,
        "loop_number":           loop_number,
        "is_final":              1 if is_final else 0,
        "ablation_condition":    state.get("ablation_condition") or None,
        "query_index":           state.get("query_index") or None,
        "disease":               state.get("disease") or None,
        "query_level_label":     state.get("query_level_label") or None,
        "user_level":            state.get("user_level") or "",
        "original_query":        (state["question"] or "")[:1000],
        "optimized_query":       (state["queries"][-1] if state.get("queries") else None or "")[:1000] or None,
        "final_tier":            final_tier,
        "tier_path":             tier_path,
        "is_escalated":          1 if is_escalated else 0,
        "is_fallback":           1 if is_fallback else 0,
        "self_correction_count": int(state.get("self_correction_count", 0)),
        "ragas_f":               f,
        "ragas_ar":              ar,
        "ragas_cp":              cp,
        "q_total":               q_total,
        "retrieved_doc_count":   len(state.get("context") or []),
        "llm_model":             state.get("llm_provider") or "openai",
        "execution_time_ms":     execution_time_ms,
        "final_answer":          final_answer,
        "fk_grade":              fk_grade,
        "ir_hit_rate":                    state.get("hit_rate_score"),
        "ir_mrr":                         state.get("mrr_score"),
        "deepeval_faithfulness":          state.get("deepeval_faithfulness"),
        "deepeval_answer_relevancy":      state.get("deepeval_answer_relevancy"),
    }


def _execute_insert(row: dict) -> Optional[int]:
    """row dict를 INSERT한다."""
    try:
        conn = _get_conn()
        with conn.cursor() as cur:
            cur.execute(_INSERT_SQL, row)

        def _fmt(v) -> str:
            return f"{v:.3f}" if v is not None else "null"
        logger.info(
            "[AuditLog] INSERT OK | rid=%s | loop=%d | is_final=%s | "
            "cond=%s | tier=%s | F=%s AR=%s CP=%s Q=%s",
            row["request_id"], row["loop_number"], row["is_final"],
            row["ablation_condition"], row["tier_path"],
            _fmt(row["ragas_f"]), _fmt(row["ragas_ar"]),
            _fmt(row["ragas_cp"]), _fmt(row["q_total"]),
        )
        return None
    except Exception as exc:
        logger.error("[AuditLog] INSERT 실패: %s", exc, exc_info=True)
        _local.conn = None
        return None


_VALID_LEVELS = {"Professional", "Consumer", "Baseline"}


def save_loop_log(
    state: GraphState,
    request_id: str,
    loop_number: int,
) -> Optional[int]:
    """중간 루프 RAGAS 결과를 INSERT한다 (is_final=0)."""
    user_level = state.get("user_level") or ""
    if user_level not in _VALID_LEVELS:
        logger.warning("[AuditLog] user_level=%r 유효하지 않음, loop_log 스킵.", user_level)
        return None

    row = _build_row(
        state=state,
        request_id=request_id,
        loop_number=loop_number,
        is_final=False,
        is_fallback=False,
        execution_time_ms=None,
        final_answer=None,
    )
    return _execute_insert(row)


def save_audit_log(
    state: GraphState,
    request_id: str,
    is_fallback: bool = False,
    execution_time_ms: Optional[int] = None,
    fk_grade: Optional[float] = None,
) -> Optional[int]:
    """최종 결과를 INSERT한다 (is_final=1)."""
    user_level = state.get("user_level") or ""
    if user_level not in _VALID_LEVELS:
        logger.warning("[AuditLog] user_level=%r 유효하지 않음, audit_log 스킵.", user_level)
        return None

    if execution_time_ms is None:
        start = state.get("workflow_start_time")
        if start:
            execution_time_ms = int((time.time() - start) * 1000)

    loop_number = int(state.get("eval_count", 1))

    row = _build_row(
        state=state,
        request_id=request_id,
        loop_number=loop_number,
        is_final=True,
        is_fallback=is_fallback,
        execution_time_ms=execution_time_ms,
        final_answer=state.get("answer") or None,
        fk_grade=fk_grade,
    )
    return _execute_insert(row)
