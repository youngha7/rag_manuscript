import re

import config.settings as settings
from models.state import GraphState
from infra.evaluator import (
    compute_official_ragas_scores,
    compute_ir_metrics,
    compute_deepeval_scores,
)


_SUMMARY_PREFIX = re.compile(r"^\[(Consumer|Professional) Summary\]\s*", re.IGNORECASE)


def _build_critic_feedback(f: float, ar: float, cp: float) -> str:
    """점수 저하 원인을 자연어 텍스트로 요약한다."""
    parts = []
    if ar < settings.AR_THRESHOLD:
        parts.append(f"AR={ar:.2f} (기준 {settings.AR_THRESHOLD}): 답변이 질문과 충분히 관련되지 않음")
    if f < settings.FAITHFULNESS_THRESHOLD:
        parts.append(f"F={f:.2f} (기준 {settings.FAITHFULNESS_THRESHOLD}): 답변이 컨텍스트에 근거하지 않음")
    if cp < settings.CP_THRESHOLD:
        parts.append(f"CP={cp:.2f} (기준 {settings.CP_THRESHOLD}): 검색된 청크 품질 불량")
    return " / ".join(parts) if parts else "품질 기준 충족"


def critic_agent(state: GraphState) -> GraphState:
    """RAGAS 공식 프레임워크 기반 3중 평가 에이전트."""
    # RAGAS 평가:
    # - 답변은 번역 전 영문 답변 사용 (prefix 제거)
    # - 질문은 영문 재작성 쿼리 사용 → 언어 불일치 방지 (영문 쿼리 ↔ 영문 답변)
    raw_answer = _SUMMARY_PREFIX.sub("", state["answer"]).strip()
    context_chunks = state["context"]
    eval_query = state["queries"][-1] if state["queries"] else state["question"]
    # AR은 첫 번째 쿼리로 고정: 재시도마다 쿼리가 바뀌어도 질문 의도는 동일
    ar_query = state["queries"][0] if state["queries"] else state["question"]

    official = compute_official_ragas_scores(eval_query, raw_answer, context_chunks, ar_query=ar_query)

    state["critic_score"] = official.faithfulness
    state["answer_relevance_score"] = official.answer_relevance
    state["context_precision_score"] = official.context_precision
    state["critic_feedback"] = _build_critic_feedback(
        official.faithfulness,
        official.answer_relevance,
        official.context_precision,
    )

    state["log"].append(
        f"[Critic] RAGAS 공식 평가: "
        f"F={official.faithfulness:.3f}, AR={official.answer_relevance:.3f}, CP={official.context_precision:.3f}"
    )
    state["log"].append(f"[Critic] 피드백: {state['critic_feedback']}")

    # 성능평가 전용 지표 (Self-Correction Loop 게이트와 무관, DB 기록·성능 시각화 전용).
    # disease(STQS/ablation 정답 라벨)가 있는 요청에서만 계산한다:
    # - IR 지표는 ground truth(disease)가 없는 일반 운영 쿼리에서는 애초에 계산 불가.
    # - DeepEval은 RAGAS와의 순환성 교차검증이 목적이라, 그 목적이 없는 일반 운영
    #   쿼리에까지 추가 LLM 호출(비용·지연)을 들일 이유가 없다.
    disease = state.get("disease") or ""
    if disease:
        ir = compute_ir_metrics(disease, state.get("context_sources", []))
        state["hit_rate_score"] = ir.hit_rate
        state["mrr_score"] = ir.mrr

        deepeval_scores = compute_deepeval_scores(eval_query, raw_answer, context_chunks)
        state["deepeval_faithfulness"] = deepeval_scores.faithfulness
        state["deepeval_answer_relevancy"] = deepeval_scores.answer_relevancy

        state["log"].append(
            f"[Critic] 성능평가 지표: Hit={ir.hit_rate} MRR={ir.mrr} | "
            f"DeepEval F={deepeval_scores.faithfulness} "
            f"AR={deepeval_scores.answer_relevancy}"
        )
    else:
        state["hit_rate_score"] = None
        state["mrr_score"] = None
        state["deepeval_faithfulness"] = None
        state["deepeval_answer_relevancy"] = None

    return state


def check_faithfulness(state: GraphState) -> bool:
    """F >= FAITHFULNESS_THRESHOLD AND AR >= AR_THRESHOLD AND CP >= CP_THRESHOLD 이면 Self-Correction Loop 종료.

    성공 기준:
      - F: 답변이 컨텍스트에 근거 (사실성)
      - AR: 답변이 질문에 충분히 관련 (관련성)
      - CP: 검색된 청크의 유효성 (정밀도)
    셋 다 충족해야 output으로 진행.
    """
    f_ok = state.get("critic_score", 0.0) >= settings.FAITHFULNESS_THRESHOLD
    ar_ok = state.get("answer_relevance_score", 0.0) >= settings.AR_THRESHOLD
    cp_ok = state.get("context_precision_score", 0.0) >= settings.CP_THRESHOLD
    return f_ok and ar_ok and cp_ok


def is_critically_low(state: GraphState) -> bool:
    """Tier 0 에서 RAGAS 지표가 현저히 낮아 query rewriting으로 개선 불가능한지 판단.

    즉시 에스컬레이션 조건 (OR):
    1. AR < CRITICAL_AR_THRESHOLD: VectorDB에 관련 내용이 아예 없음.
    2. F < CRITICAL_F_THRESHOLD AND CP < CRITICAL_CP_THRESHOLD: 검색 자체가 완전히 빗나감.
    """
    ar = state.get("answer_relevance_score", 0.0)
    f = state.get("critic_score", 0.0)
    cp = state.get("context_precision_score", 0.0)

    if ar < settings.CRITICAL_AR_THRESHOLD:
        return True
    if f < settings.CRITICAL_F_THRESHOLD and cp < settings.CRITICAL_CP_THRESHOLD:
        return True
    return False
