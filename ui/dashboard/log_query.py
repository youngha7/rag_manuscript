"""rag_audit_log_bak 테이블 조회 함수 모음 (Oracle)."""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any

try:
    import oracledb
    _ORACLEDB_AVAILABLE = True
except ImportError:
    oracledb = None  # type: ignore
    _ORACLEDB_AVAILABLE = False

import pandas as pd

import config.settings as settings

logger = logging.getLogger(__name__)

PAGE_SIZE = 20

TIER_LABEL = {0: "Tier 0 · VectorDB", 1: "Tier 1 · LLM", 2: "Tier 2 · Web"}


def _conn():
    if not _ORACLEDB_AVAILABLE:
        raise RuntimeError("oracledb 패키지가 설치되어 있지 않습니다. `pip install oracledb` 후 재시작하세요.")
    return oracledb.connect(
        user=settings.ORACLE_USER,
        password=settings.ORACLE_PASSWORD,
        dsn=settings.ORACLE_DSN,
    )


def _in_clause(col: str, values: list, params: dict, prefix: str) -> str:
    """Oracle IN 절 생성. 각 값을 named bind variable로 등록한다."""
    placeholders = []
    for i, v in enumerate(values):
        key = f"{prefix}_{i}"
        params[key] = v
        placeholders.append(f":{key}")
    return f"{col} IN ({', '.join(placeholders)})"


def _read_value(v):
    """CLOB 등 LOB locator는 연결이 살아있는 동안 즉시 읽어 문자열로 변환한다.

    연결을 닫은 뒤 지연 접근하면 DPY-1001(not connected to database)이 발생한다.
    """
    if oracledb is not None and isinstance(v, oracledb.LOB):
        return v.read()
    return v


def _rows_to_dicts(cur) -> list[dict]:
    cols = [d[0].lower() for d in cur.description]
    return [
        {col: _read_value(v) for col, v in zip(cols, row)}
        for row in cur.fetchall()
    ]


# ── 목록 조회 ─────────────────────────────────────────────────────────────────

def fetch_logs(
    date_from: date | None = None,
    date_to: date | None = None,
    user_levels: list[str] | None = None,
    tiers: list[int] | None = None,
    escalated: bool | None = None,
    fallback: bool | None = None,
    ragas_f_min: float = 0.0,
    ragas_f_max: float = 1.0,
    keyword: str = "",
    ablation_conditions: list[str] | None = None,
    diseases: list[str] | None = None,
    query_level_labels: list[str] | None = None,
    page: int = 1,
) -> tuple[pd.DataFrame, int]:
    """필터 조건으로 rag_audit_log_bak를 페이지네이션 조회한다.

    Returns:
        (DataFrame, 전체 건수)
    """
    where: list[str] = []
    params: dict[str, Any] = {}

    if date_from:
        where.append("created_at >= :date_from")
        params["date_from"] = datetime(
            date_from.year, date_from.month, date_from.day, tzinfo=timezone.utc
        )
    if date_to:
        next_day = date_to + timedelta(days=1)
        where.append("created_at < :date_to")
        params["date_to"] = datetime(
            next_day.year, next_day.month, next_day.day, tzinfo=timezone.utc
        )
    if user_levels:
        where.append(_in_clause("user_level", user_levels, params, "ul"))
    if tiers:
        where.append(_in_clause("final_tier", tiers, params, "tier"))
    if escalated is not None:
        where.append("is_escalated = :is_escalated")
        params["is_escalated"] = 1 if escalated else 0
    if fallback is not None:
        where.append("is_fallback = :is_fallback")
        params["is_fallback"] = 1 if fallback else 0
    if ablation_conditions:
        where.append(_in_clause("ablation_condition", ablation_conditions, params, "abl"))
    if diseases:
        where.append(_in_clause("disease", diseases, params, "dis"))
    if query_level_labels:
        where.append(_in_clause("query_level_label", query_level_labels, params, "qll"))

    where.append("(ragas_f IS NULL OR ragas_f BETWEEN :ragas_f_min AND :ragas_f_max)")
    params["ragas_f_min"] = ragas_f_min
    params["ragas_f_max"] = ragas_f_max

    if keyword.strip():
        where.append("UPPER(original_query) LIKE UPPER(:keyword)")
        params["keyword"] = f"%{keyword.strip()}%"

    where.append("is_final = 1")
    where_sql = "WHERE " + " AND ".join(where)

    count_sql = f"SELECT COUNT(*) FROM rag_audit_log_bak {where_sql}"
    data_sql = f"""
        SELECT
            log_id,
            request_id,
            created_at,
            ablation_condition,
            query_index,
            disease,
            query_level_label,
            user_level,
            original_query,
            final_tier,
            tier_path,
            loop_number,
            self_correction_count,
            ragas_f,
            ragas_ar,
            ragas_cp,
            q_total,
            fk_grade,
            ir_hit_rate,
            ir_mrr,
            deepeval_faithfulness,
            deepeval_answer_relevancy,
            is_escalated,
            is_fallback,
            execution_time_ms
        FROM rag_audit_log_bak
        {where_sql}
        ORDER BY created_at DESC
        OFFSET :offset ROWS FETCH NEXT :page_size ROWS ONLY
    """

    try:
        conn = _conn()
        with conn.cursor() as cur:
            cur.execute(count_sql, params)
            total: int = cur.fetchone()[0]

            data_params = dict(params)
            data_params["offset"] = (page - 1) * PAGE_SIZE
            data_params["page_size"] = PAGE_SIZE
            cur.execute(data_sql, data_params)
            rows = _rows_to_dicts(cur)
        conn.close()

        if not rows:
            return pd.DataFrame(), total

        df = pd.DataFrame(rows)
        df["request_id_short"] = df["request_id"].astype(str).str[:8] + "..."
        df["tier_label"] = df["final_tier"].map(TIER_LABEL)
        df["created_at"] = (
            pd.to_datetime(df["created_at"], utc=True)
            .dt.tz_convert("Asia/Seoul")
            .dt.strftime("%Y-%m-%d %H:%M")
        )
        return df, total

    except Exception as e:
        logger.error("fetch_logs 실패: %s", e, exc_info=True)
        return pd.DataFrame(), 0


# ── 상세 조회 ─────────────────────────────────────────────────────────────────

def fetch_detail(request_id: str) -> dict:
    """단일 request_id의 전체 루프 이력과 메타 정보를 반환한다.

    Returns:
        {
            "meta": {...},          # 첫 번째 행의 요청 정보
            "loops": DataFrame,     # 전체 루프 행 (loop_number 순)
            "queries": [...],       # optimized_query 이력 (중복 제거 순서 유지)
            "final_answer": str,    # 마지막 행의 final_answer
        }
    """
    sql = """
        SELECT
            log_id,
            request_id,
            created_at,
            ablation_condition,
            query_index,
            disease,
            query_level_label,
            user_level,
            original_query,
            optimized_query,
            final_answer,
            loop_number,
            is_final,
            final_tier,
            tier_path,
            self_correction_count,
            ragas_f,
            ragas_ar,
            ragas_cp,
            q_total,
            fk_grade,
            ir_hit_rate,
            ir_mrr,
            deepeval_faithfulness,
            deepeval_answer_relevancy,
            is_escalated,
            is_fallback,
            retrieved_doc_count,
            llm_model,
            execution_time_ms
        FROM rag_audit_log_bak
        WHERE request_id = :request_id
        ORDER BY loop_number ASC
    """
    try:
        conn = _conn()
        with conn.cursor() as cur:
            cur.execute(sql, {"request_id": request_id})
            dicts = _rows_to_dicts(cur)
        conn.close()

        if not dicts:
            return {}

        first = dicts[0]

        loops_df = pd.DataFrame(dicts)
        loops_df["tier_label"] = loops_df["final_tier"].map(TIER_LABEL)
        loops_df["created_at"] = (
            pd.to_datetime(loops_df["created_at"], utc=True)
            .dt.tz_convert("Asia/Seoul")
            .dt.strftime("%Y-%m-%d %H:%M:%S")
        )

        seen: set[str] = set()
        queries: list[str] = []
        for d in dicts:
            q = (d.get("optimized_query") or "").strip()
            if q and q not in seen:
                seen.add(q)
                queries.append(q)

        # Oracle stores is_final as NUMBER(1); treat 1 as True
        final_row = next((d for d in dicts if d.get("is_final") == 1), dicts[-1])
        final_answer = final_row.get("final_answer") or ""

        return {
            "meta": {
                "request_id":         str(first["request_id"]),
                "created_at":         pd.to_datetime(first["created_at"], utc=True)
                    .tz_convert("Asia/Seoul")
                    .strftime("%Y-%m-%d %H:%M:%S"),
                "user_level":         first["user_level"],
                "llm_model":          first["llm_model"] or "",
                "original_query":     first["original_query"],
                "ablation_condition": first.get("ablation_condition") or "",
                "query_index":        first.get("query_index"),
                "disease":            first.get("disease") or "",
                "query_level_label":  first.get("query_level_label") or "",
            },
            "loops":        loops_df,
            "queries":      queries,
            "final_answer": final_answer,
        }

    except Exception as e:
        logger.error("fetch_detail 실패: %s", e, exc_info=True)
        return {}


# ── 필터 옵션 조회 ────────────────────────────────────────────────────────────

def fetch_diseases() -> list[str]:
    """필터 드롭다운용 disease 목록 (NULL 제외, 알파벳순)."""
    sql = "SELECT DISTINCT disease FROM rag_audit_log_bak WHERE disease IS NOT NULL ORDER BY disease"
    try:
        conn = _conn()
        with conn.cursor() as cur:
            cur.execute(sql)
            values = [row[0] for row in cur.fetchall()]
        conn.close()
        return values
    except Exception as e:
        logger.error("fetch_diseases 실패: %s", e, exc_info=True)
        return []
