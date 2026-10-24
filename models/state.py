from typing import TypedDict, List


class GraphState(TypedDict):
    # ── 요청 식별 ──────────────────────────────────────────────────────────────
    request_id: str
    question: str
    user_level: str
    queries: List[str]
    context: List[str]
    context_sources: List[str]
    answer: str

    # ── RAGAS 평가 결과 ────────────────────────────────────────────────────────
    critic_score: float               # Faithfulness (α=0.4)
    answer_relevance_score: float     # Answer Relevance (α=0.4)
    context_precision_score: float    # Context Precision (α=0.2)
    critic_feedback: str

    # ── 성능평가 전용 지표 (critic 루프 게이트와 무관, disease 있는 STQS/ablation 행만 값 존재) ──
    deepeval_faithfulness: float          # DeepEval — RAGAS F 교차검증
    deepeval_answer_relevancy: float      # DeepEval — RAGAS AR 교차검증

    # ── 티어 및 루프 ───────────────────────────────────────────────────────────
    search_tier: int                  # 현재 검색 티어 (0/1/2)
    loop_count: int                   # 현재 티어 내 재시도 횟수
    tier_path: str                    # 에스컬레이션 경로: "0" / "0→1" / "0→1→2"
    self_correction_count: int        # Tier 0 자가 교정 누적 횟수
    eval_count: int                   # critic 평가 전체 누적 횟수 (루프 번호 추적용)

    # ── Best Answer 추적 ──────────────────────────────────────────────────────
    best_answer: str                  # 루프 전체에서 Q_total이 가장 높은 답변
    best_q_total: float               # 해당 답변의 Q_total (0.4*F + 0.4*AR + 0.2*CP)

    # ── 시스템 정보 ────────────────────────────────────────────────────────────
    llm_provider: str
    workflow_start_time: float        # time.time() 워크플로우 시작 시각
    log: List[str]

    # ── Ablation Study 메타데이터 ──────────────────────────────────────────────
    ablation_condition: str           # "A"(Proposal System)/"E"(Baseline), ""=일반 운영
    query_index: int                  # STQS 질문 번호 (1..240), 0=일반 운영
    disease: str                      # 질환명
    query_level_label: str            # "P"/"C" 정답 레이블, ""=일반 운영


TIER_LABELS = {0: "VectorDB(FAISS)", 1: "LLM 학습데이터", 2: "웹검색(DuckDuckGo)"}
