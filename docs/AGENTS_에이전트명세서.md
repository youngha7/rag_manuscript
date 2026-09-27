# 5-Agent 파이프라인 명세서

LangGraph `StateGraph` 위에서 순차 실행되는 5개 에이전트의 역할을 간단히 정리한다. 각 에이전트는 `GraphState`(`models/state.py`)를 입력받아 일부 필드를 갱신한 뒤 반환하는 순수 함수 형태다. 상세 티어 에스컬레이션·루프 로직은 [PIPELINE.md](PIPELINE.md) 참고.

```
level_classifier ─► adaptive_query_rewriter ─► rag_engine ─► critic_agent ─► output_agent
                            ▲                                     │
                            └─────────── 재시도/에스컬레이션 ───────┘
```

| 에이전트 | 파일 | 역할 한 줄 요약 |
|---------|------|----------------|
| level_classifier | `agents/classifier.py` | 질문의 문체·용어로 사용자 수준(Professional/Consumer) 판정 |
| adaptive_query_rewriter | `agents/rewriter.py` | 질문을 영문 검색 쿼리로 변환·재작성 |
| rag_engine | `agents/rag_engine.py` | Tier별 검색 실행 + 답변 합성 |
| critic_agent | `agents/critic.py` | RAGAS 지표(F/AR/CP)로 답변 품질 평가 |
| output_agent | `agents/output.py` | 출처·면책 조항을 붙여 최종 응답 완성 |

---

## 1. level_classifier — 수준 분류기

**함수**: `level_classifier(state) -> state`

- 사용자 질문을 분석해 `Professional`(의료 전문가) / `Consumer`(일반인) 중 하나로 분류한다.
- 판정 결과에 따라 이후 쿼리 최적화·답변 합성의 문체·난이도가 결정된다.
- 오분류 방지를 위해 LLM 호출 전 **정규식 규칙**을 먼저 적용한다.
  - `_PROFESSIONAL_RULE`: "~기술하시오/설명하시오/비교하시오" 형 서술형 지문 → 즉시 Professional
  - `_CONSUMER_RULE`: "왜 ~나요?", "~이유는 무엇인가요?" 등 일상 문체(한/영) → 즉시 Consumer
  - 두 규칙 모두 불일치 시에만 LLM(JSON 응답)으로 분류
- `state["user_level"]`이 이미 설정돼 있으면(수동 지정 또는 조건 E/D 강제) 스킵한다.
- 출력: `user_level`, `log`(분류 근거·신뢰도·의도 기록)

## 2. adaptive_query_rewriter — 적응형 쿼리 재작성기

**함수**: `adaptive_query_rewriter(state) -> state`

- 한국어 질문 → MSD 매뉴얼 검색에 최적화된 **영문 쿼리**를 생성한다.
- 두 가지 모드:
  - **최초/에스컬레이션**(`_optimize_query`): 질문 + 사용자 수준 + 감지된 의도로 쿼리 생성
  - **재시도**(`_refine_query`, `loop_count > 0`): 이전 쿼리 목록 + critic 평가 결과(F/AR/CP + `critic_feedback`)를 근거로 실패 원인을 피해 새 각도의 쿼리 생성
- Baseline(조건 E)은 레벨 중립(`General`) 쿼리를 생성한다.
- 출력: `queries`(리스트에 추가), `log`

## 3. rag_engine — RAG 엔진

**함수**: `rag_engine(state) -> state`

- `search_tier` 값에 따라 검색 도구를 선택하고, 검색 결과를 LLM으로 합성해 답변을 만든다.

| Tier | 검색 방식 | 비고 |
|------|----------|------|
| 0 | ReAct 에이전트 + `search_msd_manual`(FAISS) | temperature=0, 컨텍스트 외 지식 사용 금지 |
| 1 | LLM 사전 학습 지식 직접 생성 (도구 없음) | `_search_llm_knowledge` |
| 2 | ReAct 에이전트 + `search_web`(DuckDuckGo) | temperature=0.1 |

- 사용자 수준(Professional/Consumer/Baseline)별로 별도 시스템 프롬프트를 사용해 FK Grade 목표(Consumer ≤9, Professional ≥12)를 맞춘다.
- 출력: `context`, `context_sources`, `answer`, `log`

## 4. critic_agent — 평가 에이전트

**함수**: `critic_agent(state) -> state` (+ 라우팅 헬퍼 `check_faithfulness`, `is_critically_low`)

- RAGAS 공식 프레임워크로 3개 지표를 계산한다.
  - **Faithfulness(F)**: 답변이 검색된 context에 근거하는가
  - **Answer Relevance(AR)**: 답변이 질문과 관련 있는가
  - **Context Precision(CP)**: 검색된 청크가 질의에 정밀한가
- 평가 전 답변에서 `[Consumer/Professional Summary]` 접두어를 제거하고, 영문 쿼리 기준으로 평가해 언어 불일치를 방지한다.
- AR은 재시도로 쿼리가 바뀌어도 항상 **첫 번째 쿼리**를 기준으로 계산한다(질문 의도 고정).
- **판정 LLM은 Claude(Haiku 4.5)로 고정**되어 있으며, 답변 생성에 쓰이는 LLM(OpenAI/Gemini 토글)과 무관하다. 같은 모델이 답변 생성과 채점을 겸하면 생기는 순환성(circularity) 편향을 피하기 위함이다.
- `check_faithfulness`: F·AR·CP 모두 임계값(기본 0.8) 이상이면 성공 → output으로 라우팅(`graph.py`에서 사용)
- `is_critically_low`: AR이 매우 낮거나 F·CP가 동시에 매우 낮으면 재시도 대신 즉시 다음 Tier로 에스컬레이션
- 출력: `critic_score`(F), `answer_relevance_score`(AR), `context_precision_score`(CP), `critic_feedback`, `log`

**성능평가 전용 지표 (Self-Correction Loop 게이트와 무관)**

`disease`(STQS-240/ablation 정답 라벨)가 있는 요청에서만 추가로 계산해 DB에 기록한다. 일반 운영 쿼리는 ground truth가 없거나 교차검증 목적이 없어 계산 자체를 생략한다(불필요한 LLM 호출·비용 방지).

- **IR Hit Rate / MRR**: 검색된 청크의 출처 파일명(`context_sources`)에 `disease`명이 포함되는지로 정답 문서 적중 여부(Hit Rate)와 첫 적중 순위의 역수(MRR)를 계산하는 전통적 정보검색 지표. LLM 불필요, 순수 문자열 매칭.
- **DeepEval**: Faithfulness / Answer Relevancy 2종을 RAGAS와 **다른 프레임워크**(판정 LLM은 `DEEPEVAL_LLM_MODEL`, 기본 `gpt-4o-mini`)로 계산해 F/AR을 교차검증한다. RAGAS 단독 평가의 순환성 비판을 피하기 위한 독립 지표. Contextual Relevancy는 더 이상 사용하지 않는다.
- 출력: `hit_rate_score`, `mrr_score`, `deepeval_faithfulness`, `deepeval_answer_relevancy` (일반 운영 시 전부 `None`)

## 5. output_agent — 출력 에이전트

**함수**: `output_agent(state) -> state`

- 최종 답변에 **출처(Source)**와 **면책 조항(disclaimer)**을 덧붙여 사용자에게 보여줄 형태로 완성한다.
- Tier별 출처 표기가 다르다.
  - Tier 0: MSD 매뉴얼 파일명·페이지 번호 (`#p` 앵커 파싱)
  - Tier 1: LLM 학습데이터 (GPT)
  - Tier 2: 웹검색 출처 목록
- 답변 앞에 `[Professional Summary]` / `[Consumer Summary]` 라벨을 붙인다(Baseline 제외).
- 출력: `answer`(출처·면책 조항 포함 최종본), `log`
