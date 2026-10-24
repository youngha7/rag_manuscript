import time
import uuid
import warnings
from typing import Callable, Literal, Optional

import langchain  # noqa: F401 — 아래 필터보다 먼저 import 되어야 함 (이유는 주석 참고)
import langchain_core  # noqa: F401

# langgraph의 체크포인트 직렬화기가 allowed_objects를 명시하지 않고 langchain_core
# Reviver를 생성하면서 나오는 라이브러리 내부 경고. langchain_core와 langchain 패키지는
# 각자 __init__ 시점에 surface_langchain_deprecation_warnings()를 호출해 이 카테고리를
# "default"로 (재)활성화하므로, 둘 다 먼저 import해서 그 등록을 끝낸 뒤에 필터를 걸어야
# 우리 필터가 우선순위 맨 앞에 남아 실제로 숨겨진다. langgraph가 자체적으로 값을 명시하기
# 전까지는 우리 쪽에서 근본적으로 고칠 수 없어 경고만 숨긴다.
warnings.filterwarnings("ignore", message=r".*allowed_objects.*")

from langgraph.graph import StateGraph, END
from langgraph.types import Command

import config.settings as settings
from models.state import GraphState
from tools.vector_search import initialize_vector_db
from agents.classifier import level_classifier
from agents.rewriter import adaptive_query_rewriter
from agents.rag_engine import rag_engine
from agents.critic import critic_agent, check_faithfulness, is_critically_low
from agents.output import output_agent
from core.llm_client import set_llm_provider, reset_llm_provider
from infra.audit_logger import save_audit_log, save_loop_log
from infra.evaluator import flesch_kincaid_grade_en, get_pure_fk_grade


# ── 노드 이름 → step_callback 이름 매핑 ─────────────────────────────────────
_NODE_TO_STEP = {
    "level_classifier": "level",
    "query_rewriter":   "rewriter",
    "rag_engine":       "rag",
    "critic":           "critic",
    "output":           "output",
    "fallback":         "fallback",
}


# ── Critic 노드 ──────────────────────────────────────────────────────────────

def _critic_node(
    state: GraphState,
) -> Command[Literal["query_rewriter", "output", "fallback"]]:
    """RAGAS 평가 후 Self-Corrective Loop 라우팅.

    조건별 동작:
    - A (Proposal System): 기본 동작 (자가 교정 + 멀티 티어)
    - E (Baseline)       : 항상 즉시 output (첫 번째 답변 그대로)

    루프 로그:
    - 매 평가 완료 후 무조건 save_loop_log() 호출 (is_final=False, goto 무관).
    - is_final=True 행은 output/fallback 노드에서 save_audit_log()가 별도 INSERT.
    """
    state = critic_agent(state)

    eval_count = state.get("eval_count", 0) + 1
    condition  = state.get("ablation_condition") or "A"
    tier  = state["search_tier"]
    loop  = state["loop_count"]
    ar    = state.get("answer_relevance_score", 0.0)
    f     = state.get("critic_score", 0.0)
    cp    = state.get("context_precision_score", 0.0)

    new_state = {**state, "log": list(state["log"]), "eval_count": eval_count}

    # Best answer 추적: Tier 0 / 2 에서만 (F·CP가 의미 있는 경우)
    if tier != 1:
        current_q = 0.4 * f + 0.4 * ar + 0.2 * cp
        if current_q > new_state.get("best_q_total", 0.0):
            new_state["best_answer"] = new_state["answer"]
            new_state["best_q_total"] = current_q
            new_state["log"].append(
                f"[Loop] Best answer updated: Q_total={current_q:.3f} "
                f"(F={f:.2f}, AR={ar:.2f}, CP={cp:.2f})."
            )

    goto: str = ""

    # ── 조건 E (Baseline): 평가 후 항상 즉시 output ──────────────────────────
    if condition == "E":
        new_state["log"].append(
            f"[Loop] Condition E (Baseline): immediate output "
            f"(F={f:.2f}, AR={ar:.2f}, CP={cp:.2f})."
        )
        goto = "output"

    # ── Tier 1: AR만 평가 ────────────────────────────────────────────────────
    elif tier == 1:
        if ar >= settings.AR_THRESHOLD:
            new_state["log"].append(
                f"[Loop] Tier 1 passed (AR={ar:.2f} ≥ {settings.AR_THRESHOLD}) → output."
            )
            goto = "output"
        else:
            new_state["search_tier"] = 2
            new_state["loop_count"]  = 0
            new_state["tier_path"]   = state.get("tier_path", "0→1") + "→2"
            new_state["log"].append(
                f"[Loop] Tier 1 below threshold (AR={ar:.2f} < {settings.AR_THRESHOLD}) "
                "→ Tier 2 escalation."
            )
            goto = "query_rewriter"

    # ── 성공 조건: F ∧ AR ∧ CP 모두 충족 ────────────────────────────────────
    elif check_faithfulness(state):
        new_state["log"].append(
            f"[Loop] Quality threshold met "
            f"(F={f:.2f}, AR={ar:.2f}, CP={cp:.2f}) → output."
        )
        goto = "output"

    # ── Tier 0 실패 라우팅 (Proposal System: 자가 교정 + 멀티 티어) ────────────
    elif tier == 0:
        if is_critically_low(state):
            new_state["search_tier"] = 1
            new_state["loop_count"]  = 0
            new_state["tier_path"]   = "0→1"
            new_state["log"].append(
                f"[Loop] RAGAS critically low "
                f"(AR={ar:.2f}, F={f:.2f}, CP={cp:.2f}) → immediate Tier 1 escalation."
            )
            goto = "query_rewriter"

        elif loop >= settings.MAX_LOOPS - 1:
            new_state["search_tier"] = 1
            new_state["loop_count"]  = 0
            new_state["tier_path"]   = "0→1"
            new_state["log"].append(
                f"[Loop] Tier 0 max retries ({settings.MAX_LOOPS}) exhausted "
                f"(F={f:.2f}, AR={ar:.2f}) → Tier 1 escalation."
            )
            goto = "query_rewriter"

        else:
            new_state["loop_count"]           = loop + 1
            new_state["self_correction_count"] = (
                state.get("self_correction_count", 0) + 1
            )
            reasons = []
            if f  < settings.FAITHFULNESS_THRESHOLD: reasons.append(f"F={f:.2f}<{settings.FAITHFULNESS_THRESHOLD}")
            if ar < settings.AR_THRESHOLD:            reasons.append(f"AR={ar:.2f}<{settings.AR_THRESHOLD}")
            if cp < settings.CP_THRESHOLD:            reasons.append(f"CP={cp:.2f}<{settings.CP_THRESHOLD}")
            new_state["log"].append(
                f"[Loop] Tier 0 retry {loop + 1}/{settings.MAX_LOOPS} — "
                f"{', '.join(reasons)} → query rewriting."
            )
            goto = "query_rewriter"

    # ── Tier 2: all tiers exhausted → fallback ───────────────────────────────
    else:
        new_state["log"].append(
            f"[Loop] All tiers exhausted (final F={f:.2f}, AR={ar:.2f}) → fallback."
        )
        goto = "fallback"

    # ── 매 평가마다 중간 로그 저장 (goto 무관) ──────────────────────────────
    # Tier 1 평가: 논문 설계상 AR만 사용하므로 F, CP는 null로 저장
    _log_state = new_state
    if tier == 1:
        _log_state = {**new_state, "critic_score": None, "context_precision_score": None}
    save_loop_log(_log_state, new_state.get("request_id", ""), eval_count)

    return Command(update=new_state, goto=goto)


# ── Output / Fallback 노드 ───────────────────────────────────────────────────

def _output_node(state: GraphState) -> GraphState:
    """한국어 번역 + 출처·면책 조항 추가 후 감사 로그 저장."""
    # FK Grade: Consumer는 의학 용어 마스킹 후 순수 문장 구조로 계산
    _fk_fn = get_pure_fk_grade if state.get("user_level") == "Consumer" else flesch_kincaid_grade_en
    fk = _fk_fn(state.get("answer") or "")
    result = output_agent(state)
    elapsed = int((time.time() - result.get("workflow_start_time", time.time())) * 1000)
    save_audit_log(result, result.get("request_id", ""), is_fallback=False,
                   execution_time_ms=elapsed, fk_grade=fk)
    return result


def _fallback_node(state: GraphState) -> GraphState:
    """모든 재시도 소진 후 최고 품질 답변을 사용하거나, 없으면 원문을 제시한다."""
    f = state.get("critic_score", 0.0)
    best_answer = state.get("best_answer", "")
    best_q = state.get("best_q_total", 0.0)

    if best_answer and best_q > 0:
        state["log"].append(
            f"[Final] Fallback: using best answer from loop "
            f"(Q_total={best_q:.3f}, final F={f:.2f})."
        )
        state["answer"] = best_answer
        _fk_fn = get_pure_fk_grade if state.get("user_level") == "Consumer" else flesch_kincaid_grade_en
        fk = _fk_fn(state["answer"])
    else:
        state["log"].append(
            f"[Final] All tiers exhausted (final F={f:.2f}) — "
            "no reliable source found."
        )
        raw_ctx = (
            "\n\n---\n".join(state["context"]) if state["context"] else "(no search results)"
        )
        state["answer"] = (
            "No reliable source could be found to answer this question. "
            "The following are the raw retrieved passages for your reference.\n\n"
            f"[Source Material]\n{raw_ctx}"
        )
        fk = None

    result = output_agent(state)
    elapsed = int((time.time() - result.get("workflow_start_time", time.time())) * 1000)
    save_audit_log(result, result.get("request_id", ""), is_fallback=True,
                   execution_time_ms=elapsed, fk_grade=fk)
    return result


# ── 그래프 빌드 ──────────────────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("level_classifier", level_classifier)
    graph.add_node("query_rewriter",   adaptive_query_rewriter)
    graph.add_node("rag_engine",       rag_engine)
    graph.add_node("critic",           _critic_node)
    graph.add_node("output",           _output_node)
    graph.add_node("fallback",         _fallback_node)

    graph.set_entry_point("level_classifier")
    graph.add_edge("level_classifier", "query_rewriter")
    graph.add_edge("query_rewriter",   "rag_engine")
    graph.add_edge("rag_engine",       "critic")
    graph.add_edge("output",           END)
    graph.add_edge("fallback",         END)

    return graph.compile()


_compiled_graph = None


def _get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


# ── 메인 실행 함수 ────────────────────────────────────────────────────────────

def run_medical_self_corrective_rag(
    question: str,
    forced_user_level: Optional[str] = None,
    step_callback: Optional[Callable[[str, GraphState], None]] = None,
    llm_provider: str = "openai",
    ablation_condition: str = "",
    query_index: int = 0,
    disease: str = "",
    query_level_label: str = "",
) -> GraphState:
    """Self-Corrective RAG 메인 실행 함수.

    Args:
        question:           사용자 질문
        forced_user_level:  수준 강제 지정 ("Professional"/"Consumer"). None이면 자동 분류.
        step_callback:      노드 실행마다 호출되는 스트리밍 콜백.
        llm_provider:       "openai" 또는 "gemini"
        ablation_condition: 시스템 조건 ("A"=Proposal System, "E"=Baseline). ""=일반 운영.
        query_index:        STQS 질문 순번 (1부터). 0=일반 운영.
        disease:            질환명 (STQS 메타데이터).
        query_level_label:  STQS 정답 레이블 ("P"/"C"). ""=일반 운영.
    """
    initialize_vector_db()

    prov = (llm_provider or "openai").strip().lower()
    if prov not in ("openai", "gemini"):
        prov = "openai"
    tok = set_llm_provider(prov)

    cond = (ablation_condition or "").strip().upper()

    # 조건 E: 수준 분류 없음 → Baseline (레벨 중립 응답)
    if cond == "E":
        forced_user_level = "Baseline"
    # 조건 D: 수준 분류기 없음 → Consumer 고정
    elif cond == "D" and not forced_user_level:
        forced_user_level = "Consumer"

    initial_state: GraphState = {
        # ── 요청 ──────────────────────────────────────────────────────────────
        "request_id":             str(uuid.uuid4()),
        "question":               question,
        "user_level":             forced_user_level or "",
        "queries":                [],
        "context":                [],
        "context_sources":        [],
        "answer":                 "",
        # ── RAGAS ─────────────────────────────────────────────────────────────
        "critic_score":           0.0,
        "answer_relevance_score": 0.0,
        "context_precision_score": 0.0,
        "critic_feedback":        "",
        # ── 성능평가 전용 지표 (disease 있는 STQS/ablation 행만 critic_agent에서 값 채움) ──
        "deepeval_faithfulness":         None,
        "deepeval_answer_relevancy":     None,
        # ── 티어 / 루프 ───────────────────────────────────────────────────────
        "search_tier":            0,
        "loop_count":             0,
        "tier_path":              "0",
        "self_correction_count":  0,
        "eval_count":             0,
        # ── Best Answer 추적 ──────────────────────────────────────────────────
        "best_answer":            "",
        "best_q_total":           0.0,
        # ── 시스템 ────────────────────────────────────────────────────────────
        "llm_provider":           prov,
        "workflow_start_time":    time.time(),
        "log": (
            [f"[Mode] 사용자 선택 레벨: {forced_user_level}."]
            if forced_user_level else []
        ),
        # ── Ablation Study 메타데이터 ──────────────────────────────────────────
        "ablation_condition":     cond,
        "query_index":            query_index,
        "disease":                disease,
        "query_level_label":      query_level_label,
    }

    try:
        graph = _get_graph()

        if step_callback is not None:
            current_state = dict(initial_state)
            last_rewriter_tier = 0
            for event in graph.stream(initial_state, stream_mode="updates"):
                for node_name, updates in event.items():
                    pre_update_queries = list(current_state.get("queries") or [])
                    if isinstance(updates, dict):
                        current_state = {**current_state, **updates}
                    step = _NODE_TO_STEP.get(node_name, node_name)
                    if node_name == "query_rewriter" and pre_update_queries and current_state.get("loop_count", 0) > 0:
                        new_tier = current_state.get("search_tier", 0)
                        extra = "tier_up" if new_tier > last_rewriter_tier else "retry"
                        try:
                            step_callback(extra, current_state)
                        except Exception:
                            pass
                    try:
                        step_callback(step, current_state)
                    except Exception:
                        pass
                    if node_name == "query_rewriter":
                        last_rewriter_tier = current_state.get("search_tier", 0)
            return current_state

        return graph.invoke(initial_state)

    finally:
        reset_llm_provider(tok)
