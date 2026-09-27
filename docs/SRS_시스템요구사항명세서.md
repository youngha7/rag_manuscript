# 시스템 요구사항 명세서 (Software Requirements Specification)

**프로젝트명**: 의료 정보 자기교정 RAG 시스템  
**문서버전**: v3.0  
**작성일**: 2026-05-16  
**작성자**: 연구자  
**대상**: LangGraph 기반 Self-Corrective RAG 논문 시스템

---

## 1. 문서 개요

### 1.1 목적

본 문서는 LangGraph 기반 의료 정보 자기교정 RAG(Retrieval-Augmented Generation) 시스템의 기능적·비기능적 요구사항을 정의한다. 이 시스템은 사용자가 의료 관련 질문을 입력하면 신뢰성 있는 정보를 검색·생성하여 제공하며, RAGAS 평가 지표를 기반으로 답변 품질을 자동 교정하는 것을 목적으로 한다.

### 1.2 범위

본 시스템은 다음을 포함한다:

- MSD 매뉴얼 기반 PDF 문서 인덱싱 및 벡터 검색 (BAAI/bge-base-en-v1.5, 768차원)
- LLM을 이용한 사용자 수준(의료 전문가/일반인) 자동 분류
- 3단계 지식 계층(Tier 0: VectorDB, Tier 1: LLM 학습데이터, Tier 2: 웹검색) 기반 검색
- 사용자 수준별 맞춤 답변 생성 (Flesch-Kincaid Grade Level 목표 적용)
- RAGAS 자동 품질 평가 및 Self-Corrective Loop (Proposal System vs Baseline)
- 감사 로그 저장 (request_id당 N+1행: 평가마다 중간 행 + 최종 행, fk_grade 포함) 및 성능 시각화 대시보드 (8개 섹션)
- STQS-240 표준 질문 세트를 이용한 시스템 성능 평가 실험 (main.ipynb)

### 1.3 정의 및 약어

| 용어 | 정의 |
|------|------|
| RAG | Retrieval-Augmented Generation. 외부 문서를 검색하여 LLM 답변을 보강하는 기법 |
| RAGAS | RAG 시스템 평가 프레임워크. Faithfulness·Answer Relevance·Context Precision 측정 |
| Faithfulness (F) | 생성된 답변이 검색된 컨텍스트에 근거하는 정도 (0~1) |
| Answer Relevance (AR) | 답변이 질문과 관련된 정도 (0~1). Tier 1 평가에서는 단독으로 사용 |
| Context Precision (CP) | 검색된 청크의 유효성(노이즈 없이 관련 정보만 포함하는 정도) (0~1) |
| Q_total | 종합 품질 점수. 0.4·F + 0.4·AR + 0.2·CP |
| 판정 LLM 분리 | Quality Gate(RAGAS)와 답변 생성에 서로 다른 LLM을 사용하는 설계. RAGAS는 Claude, 답변 생성은 OpenAI/Gemini 토글로 고정해 순환성(circularity) 편향을 방지 |
| IR Hit Rate / MRR | 전통적 정보검색(IR) 지표. `disease` 정답 라벨과 검색 출처 파일명을 매칭해 계산하는 성능평가 전용 지표 (게이트 미사용) |
| DeepEval | RAGAS와 별개 프레임워크(`gpt-4o-mini` 판정)로 Faithfulness/Answer Relevancy를 계산해 F/AR을 교차검증하는 성능평가 전용 지표 (게이트 미사용). Contextual Relevancy는 더 이상 사용하지 않음 |
| FK Grade | Flesch-Kincaid Grade Level. 영어 텍스트 가독성 지표. 높을수록 어려운 글. Consumer 목표 ≤9, Professional 목표 ≥12 |
| Self-Corrective Loop | RAGAS 기준 미달 시 쿼리를 재최적화하여 재검색하는 반복 루프 |
| Tier | 지식 검색 계층. Tier 0(VectorDB) → Tier 1(LLM) → Tier 2(Web) |
| tier_path | 에스컬레이션 경로 문자열. "0" / "0→1" / "0→1→2" |
| Proposal System | 자가 교정 + 멀티 티어 + 수준 분류기를 모두 포함한 완성 시스템 (조건 A) |
| STQS-240 | Standard Test Query Set. 표준 테스트 질문 세트 (240건) |
| LangGraph | 상태 기반 LLM 워크플로우를 그래프 형태로 정의하는 프레임워크 |
| FAISS | Facebook AI 유사도 검색 라이브러리. 벡터 인덱싱 및 검색에 사용 |
| Professional (P) | 의료 전문가 사용자 수준 |
| Consumer (C) | 일반인 사용자 수준 |
| save_loop_log | critic 평가마다 is_final=FALSE 중간 행을 INSERT하는 함수 |
| save_audit_log | output/fallback 완료 시 is_final=TRUE 최종 행을 INSERT하는 함수 |

### 1.4 참고 문서

- LangGraph 공식 문서 (https://langchain-ai.github.io/langgraph)
- RAGAS 공식 문서 (https://docs.ragas.io)
- MSD 매뉴얼 (https://www.msdmanuals.com)

---

## 2. 전체 시스템 설명

### 2.1 시스템 개요

본 시스템은 MSD(Merck Sharp & Dohme) 매뉴얼 기반의 의료 정보를 활용하여 사용자 질문에 대한 신뢰성 있는 답변을 제공하는 한국어 의료 QA 시스템이다. LangGraph를 사용하여 Self-Corrective Loop를 구현하며, RAGAS 평가 기준(F ≥ 0.8, AR ≥ 0.8, CP ≥ 0.8)을 충족할 때까지 쿼리를 자동으로 개선한다. 2가지 Ablation Study 조건(A/E)을 통해 시스템 성능을 비교한다. 답변 생성 시 사용자 수준별 FK Grade 목표(Consumer ≤9, Professional ≥12)를 적용하여 가독성 적절성을 보장한다.

### 2.2 시스템 컨텍스트

```
사용자
  │
  ▼
[Streamlit UI]  ← app.py (항상 조건 A)
  │
  ▼
[LangGraph 워크플로우]
  ├─ 사용자 수준 분류 (LLM, 조건 E는 Consumer 고정)
  ├─ 쿼리 최적화 (LLM)
  ├─ Tier 0: FAISS VectorDB 검색 (BAAI/bge-base-en-v1.5)
  ├─ Tier 1: LLM 학습데이터 기반 생성 (AR만 평가)
  ├─ Tier 2: 웹검색 (DuckDuckGo)
  ├─ RAGAS 품질 평가 (F, AR, CP, Q_total)
  ├─ FK Grade 계산 (출처·면책 조항 추가 전 영어 원문, is_final=TRUE 행만)
  ├─ A/E 조건별 자기교정 루프 (Proposal System / Baseline)
  └─ 감사 로그 저장 (N+1행: save_loop_log + save_audit_log)
  │
  ▼
[Oracle Database] — rag_audit_log_bak (N+1행/요청, fk_grade 포함)
  │
  ▼
[성능 대시보드 — 8개 섹션 (Proposal System vs Baseline)]

연구자
  │
  ▼
[main.ipynb] — STQS-240 × 2조건 = 480건 자동 실험
```

### 2.3 사용자 특성

| 사용자 유형 | 설명 |
|-------------|------|
| 일반인 (Consumer) | 증상 설명, 복용 여부 등 일반적인 의료 정보를 요청하는 사용자 |
| 의료 전문가 (Professional) | 임상 용어, 약물 기전, 진단 기준 등 전문적인 의료 정보를 요청하는 사용자 |
| 시스템 관리자 | 인덱스 재빌드, 로그 조회, 성능 모니터링을 수행하는 사용자 |
| 연구자 | 시스템 성능 평가 실험 실행, STQS-240 평가, 논문 데이터 수집을 수행하는 사용자 |

### 2.4 제약 사항

- 본 시스템은 MSD 매뉴얼에 수록된 질환에 한해 Tier 0 정보를 제공한다.
- 실제 진단·처방·치료를 대체하지 않는다.
- OpenAI API 키가 필요하다.
- Oracle Database 연결이 없으면 감사 로그 저장 기능이 비활성화된다.
- FAISS 인덱스는 BAAI/bge-base-en-v1.5로 빌드된 것과 동일한 모델로 쿼리해야 한다.
- FK Grade는 영어 텍스트 기반 지표이므로 Tier 1(AR 단독 평가) 및 Fallback 행에는 적용되지 않는다.

---

## 3. 기능 요구사항

### 3.1 사용자 수준 자동 분류

| 요구사항 ID | FR-001 |
|-------------|--------|
| **요구사항명** | 사용자 수준 자동 분류 |
| **우선순위** | 필수 |
| **설명** | 시스템은 사용자의 질문 내용을 분석하여 자동으로 의료 전문가(Professional) 또는 일반인(Consumer)으로 분류해야 한다. |
| **입력** | 사용자 질문 텍스트 |
| **처리** | LLM이 질문을 분석하여 사용자 수준, 신뢰도(0~1), 분류 근거, 의도를 반환한다. |
| **출력** | user_level (Professional / Consumer), 신뢰도, 의도 분류, 근거 텍스트 |
| **예외 처리** | LLM 응답 파싱 실패 시 기본값 Consumer로 설정한다. |
| **Baseline 예외** | 조건 E(Baseline)에서는 run_medical_self_corrective_rag()가 forced_user_level="Baseline"을 설정하므로 LLM 분류를 우회한다. |

| 요구사항 ID | FR-002 |
|-------------|--------|
| **요구사항명** | 수동 사용자 수준 설정 |
| **우선순위** | 선택 |
| **설명** | 사용자는 사이드바에서 페르소나(자동 분류 / 의료 전문가 / 일반인)를 수동으로 선택할 수 있어야 한다. |
| **입력** | 사이드바 선택값 |
| **처리** | 수동 선택 시 LLM 분류를 생략하고 선택값을 강제 적용한다. |
| **출력** | forced_user_level 설정 |

---

### 3.2 쿼리 최적화

| 요구사항 ID | FR-003 |
|-------------|--------|
| **요구사항명** | 영문 쿼리 최적화 |
| **우선순위** | 필수 |
| **설명** | 시스템은 사용자의 한국어 질문을 영문 의료 학술 검색 쿼리로 변환하여 FAISS 검색 정확도를 높여야 한다. |
| **입력** | 사용자 질문, user_level, 이전 RAGAS 평가 결과 및 critic_feedback(재시도 시) |
| **처리** | LLM이 질문을 의료 도메인에 적합한 영문 검색어로 재작성한다. 재시도 시 critic_feedback을 참고하여 쿼리를 개선한다. |
| **출력** | 최적화된 영문 검색 쿼리 |

---

### 3.3 다중 계층 지식 검색

| 요구사항 ID | FR-004 |
|-------------|--------|
| **요구사항명** | Tier 0 VectorDB 검색 |
| **우선순위** | 필수 |
| **설명** | 시스템은 FAISS 인덱스에서 사용자 쿼리와 가장 유사한 문서 청크를 검색해야 한다. |
| **입력** | 최적화된 영문 쿼리 |
| **처리** | **BAAI/bge-base-en-v1.5** 임베딩 모델(768차원)로 쿼리를 벡터화 후 FAISS 코사인 유사도 검색 수행. Top-K 청크 반환. |
| **출력** | 검색된 청크 텍스트 목록, 출처 메타데이터 |

| 요구사항 ID | FR-005 |
|-------------|--------|
| **요구사항명** | Tier 1 LLM 학습데이터 기반 생성 |
| **우선순위** | 필수 |
| **설명** | Tier 0 검색 실패 시 LLM의 사전 학습 지식을 활용하여 답변을 생성해야 한다. |
| **입력** | 최적화된 쿼리, user_level |
| **처리** | LLM에 직접 질의하여 의료 지식 기반 답변 생성. 평가는 AR만 적용(컨텍스트 청크 없음) |
| **출력** | LLM 생성 답변 텍스트 |

| 요구사항 ID | FR-006 |
|-------------|--------|
| **요구사항명** | Tier 2 웹검색 |
| **우선순위** | 필수 |
| **설명** | Tier 1 기준 미달 시 DuckDuckGo 웹검색으로 최신 의료 정보를 보완해야 한다. |
| **입력** | 최적화된 쿼리 |
| **처리** | DuckDuckGo API를 통해 최대 3개 검색 결과 수집, 수집된 내용을 컨텍스트로 활용 |
| **출력** | 웹 검색 결과 기반 답변 텍스트 |

---

### 3.4 즉시 에스컬레이션

| 요구사항 ID | FR-007 |
|-------------|--------|
| **요구사항명** | 즉시 에스컬레이션 판단 |
| **우선순위** | 필수 |
| **설명** | 시스템은 RAGAS 점수가 현저히 낮은 경우 쿼리 재시도 없이 즉시 상위 Tier로 에스컬레이션해야 한다. |
| **조건** | ① AR < 0.3 (VectorDB에 관련 내용 없음) 또는 ② F < 0.3 AND CP < 0.2 (검색 완전 실패) |
| **처리** | 재시도 없이 search_tier를 1 증가시키고 tier_path에 "→1"을 추가하여 상위 Tier로 전환 |
| **목적** | 불필요한 재시도 루프를 방지하고 응답 효율성 향상 |

---

### 3.5 RAGAS 품질 평가 및 Self-Corrective Loop

| 요구사항 ID | FR-008 |
|-------------|--------|
| **요구사항명** | RAGAS 자동 품질 평가 |
| **우선순위** | 필수 |
| **설명** | 시스템은 답변 생성 후 자동으로 Faithfulness, Answer Relevance, Context Precision을 평가해야 한다. |
| **입력** | 질문, 생성된 답변, 검색된 컨텍스트 청크 |
| **처리** | RAGAS 프레임워크의 공식 메트릭 사용. Streamlit 이벤트 루프 충돌 방지를 위해 ThreadPoolExecutor 내 별도 이벤트 루프에서 실행. |
| **출력** | F, AR, CP 점수 (각 0~1), Q_total (0.4·F + 0.4·AR + 0.2·CP), 할루시네이션 플래그 목록, critic_feedback |
| **Tier 1 예외** | Tier 1은 컨텍스트 청크가 없으므로 AR만 평가한다. 중간 로그의 F, CP, q_total은 NULL로 저장. |
| **판정 LLM** | 답변 생성 LLM(OpenAI/Gemini 토글)과 무관하게 항상 **Claude**(`ANTHROPIC_MODEL`)로 고정. 동일 모델이 생성·채점을 겸할 때 발생하는 순환성(circularity) 편향을 방지하기 위함 |

| 요구사항 ID | FR-009 |
|-------------|--------|
| **요구사항명** | Self-Corrective Loop |
| **우선순위** | 필수 |
| **설명** | F ≥ 0.8 AND AR ≥ 0.8 AND CP ≥ 0.8 을 충족할 때까지 쿼리 재최적화 후 재검색을 반복해야 한다. |
| **최대 반복** | Tier당 최대 3회 (MAX_LOOPS = 3) |
| **성공 조건** | F ≥ 0.8 AND AR ≥ 0.8 AND CP ≥ 0.8 |
| **실패 처리** | 모든 Tier 소진 후에도 기준 미달 시 Fallback 노드로 라우팅 |
| **추적** | self_correction_count: Tier 0 내 자가 교정 누적 횟수를 GraphState에서 추적 |
| **중간 로그** | 매 critic 평가 완료 후 save_loop_log()로 is_final=FALSE 행을 INSERT하여 루프별 점수 변화를 추적 |

---

### 3.6 시스템 성능 비교

| 요구사항 ID | FR-010 |
|-------------|--------|
| **요구사항명** | Proposal System vs Baseline 2가지 조건 지원 |
| **우선순위** | 필수 (연구 목적) |
| **설명** | 시스템은 ablation_condition 파라미터에 따라 Proposal System(A)과 Baseline(E) 두 가지 라우팅 동작을 수행하여 시스템 성능을 비교할 수 있어야 한다. |

| 조건 | 이름 | 동작 |
|------|------|------|
| A | Proposal System | 자가 교정 + 멀티 티어 + 수준 분류기 (기본) |
| E | Baseline | RAGAS 후 즉시 출력, user_level="Baseline" 강제 |

| 요구사항 ID | FR-011 |
|-------------|--------|
| **요구사항명** | STQS-240 일괄 실험 (main.ipynb) |
| **우선순위** | 필수 (연구 목적) |
| **설명** | main.ipynb는 STQS-240 표준 질문 세트(240건)와 2가지 조건(A/E)을 교차 실험하여 결과를 Oracle DB에 자동 저장해야 한다. |
| **입력** | STQS-240 질문 목록 (disease, level, q_num, question) 4-tuple, 각 질문의 query_index (1-240) |
| **처리** | 2조건 × 240질문 = 480회 run_medical_self_corrective_rag() 호출 |
| **출력** | rag_audit_log_bak에 N+1행 × 480요청 INSERT (ablation_condition, query_index, disease 등 메타데이터 포함) |

---

### 3.7 할루시네이션 방지

| 요구사항 ID | FR-012 |
|-------------|--------|
| **요구사항명** | 의료 도메인 할루시네이션 방지 및 품질 평가 |
| **우선순위** | 필수 |
| **설명** | 시스템은 프롬프트 수준에서 답변이 검색된 컨텍스트에만 근거하도록 강제하고(`agents/rag_engine.py` 시스템 프롬프트: 용량·기전·권고사항을 컨텍스트 밖에서 추가하지 말 것), 사후적으로 RAGAS Faithfulness(F) 점수로 근거 없는 서술 비율을 정량 평가한다. |
| **현재 구현** | 패턴 매칭 기반의 별도 할루시네이션 탐지 모듈(`hallucination_flags`/`hallucination_count`)은 존재하지 않는다 — Faithfulness(F) 점수가 이 역할을 대신하며, F < 0.8이면 Self-Corrective Loop가 재시도를 유발한다. |
| **출력** | Faithfulness(F) 점수 (0~1), `critic_feedback`(기준 미달 원인 요약) |

---

### 3.8 최종 답변 생성

| 요구사항 ID | FR-013 |
|-------------|--------|
| **요구사항명** | 사용자 수준별 맞춤 답변 생성 (FK Grade 목표 적용) |
| **우선순위** | 필수 |
| **설명** | 시스템은 사용자 수준(Professional/Consumer)에 따라 답변의 전문성과 가독성을 조정해야 한다. |
| **Consumer 목표** | FK Grade ≤ 9 — 문장당 최대 15단어, 1~2음절 일상어, 의료 용어 시 괄호 설명, 불릿 포인트, 능동태 |
| **Professional 목표** | FK Grade ≥ 12 — 문장당 20단어 이상 복합 문장, 임상·약리 전문 용어, 라틴/그리스어 어근, Pathophysiology/Diagnostic Criteria/Therapeutic Approach/Clinical Considerations 구조화 |

| 요구사항 ID | FR-014 |
|-------------|--------|
| **요구사항명** | 출처·면책 조항 부가 |
| **우선순위** | 필수 |
| **설명** | 시스템은 영문으로 생성된 답변에 검색 출처(MSD 매뉴얼 파일·페이지 / LLM 학습데이터 / 웹 URL)와 표준 면책 조항을 덧붙여 최종 응답을 완성해야 한다. 번역은 수행하지 않으며 영어 원문을 그대로 사용자에게 제공한다. |
| **처리** | `agents/output.py`의 `output_agent`가 LLM 호출 없이 문자열 포맷팅으로 처리. FK Grade는 이 출처·면책 조항 추가 전 영어 원문으로 계산하여 저장한다. |

---

### 3.9 PDF 문서 관리

| 요구사항 ID | FR-015 |
|-------------|--------|
| **요구사항명** | PDF 업로드 및 인덱싱 |
| **우선순위** | 필수 |
| **설명** | 사용자는 UI를 통해 새로운 PDF 문서를 업로드하고 기존 인덱스에 추가할 수 있어야 한다. |
| **처리** | PyMuPDF로 텍스트 추출. 스캔 PDF의 경우 RapidOCR로 텍스트 인식 후 1000자 청크로 분할(`.env`의 `MEDICAL_RAG_CHUNK_MAX_CHARS` 기준, 코드 기본값은 500자), 60자 오버랩 적용. URL 청크 필터링 후 BAAI/bge-base-en-v1.5로 임베딩 |

| 요구사항 ID | FR-016 |
|-------------|--------|
| **요구사항명** | 전체 인덱스 재빌드 |
| **우선순위** | 선택 |
| **설명** | 관리자는 data 폴더의 모든 PDF를 재임베딩하여 FAISS 인덱스를 갱신할 수 있어야 한다. |
| **처리** | 백그라운드 스레드에서 재빌드. 진행률(%)을 실시간으로 UI에 표시 |

---

### 3.10 감사 로그

| 요구사항 ID | FR-017 |
|-------------|--------|
| **요구사항명** | 감사 로그 저장 (N+1행 설계) |
| **우선순위** | 필수 |
| **설명** | 시스템은 각 critic 평가마다 중간 행을 INSERT하고, output/fallback 완료 시 최종 행을 INSERT해야 한다. request_id당 총 N+1행 (N=critic 평가 횟수). |
| **중간 행** | save_loop_log() 호출. is_final=FALSE, final_answer=NULL, fk_grade=NULL |
| **최종 행 (output)** | save_audit_log(fk_grade=fk) 호출. is_final=TRUE, final_answer 포함, fk_grade 포함 |
| **최종 행 (fallback)** | save_audit_log(is_fallback=True, fk_grade=None). is_final=TRUE, fk_grade=NULL |
| **저장 항목** | request_id, loop_number, is_final, ablation_condition, query_index, disease, query_level_label, user_level, 원본/최적화 쿼리, final_tier, tier_path, is_escalated, is_fallback, self_correction_count, F/AR/CP/Q_total, retrieved_doc_count, llm_model, execution_time_ms, final_answer (CLOB), **fk_grade** |
| **제약** | UPDATE 없음. INSERT only. |

| 요구사항 ID | FR-018 |
|-------------|--------|
| **요구사항명** | 로그 조회 |
| **우선순위** | 선택 |
| **설명** | 관리자는 대시보드에서 감사 로그를 조회하고 상세 내용을 확인할 수 있어야 한다. 집계 쿼리는 반드시 WHERE is_final = TRUE 필터를 사용한다. |

---

### 3.11 성능 시각화 대시보드

| 요구사항 ID | FR-019 |
|-------------|--------|
| **요구사항명** | 성능 시각화 — 8개 섹션 |
| **우선순위** | 선택 |
| **설명** | 시스템은 rag_audit_log_bak 데이터를 기반으로 Proposal System vs Baseline 성능 비교를 위한 8개 섹션의 시각화를 제공해야 한다. 모든 matplotlib 차트 텍스트는 영어로 작성한다 (배포 환경 폰트 제한). |
| **섹션 1** | RAGAS 메트릭 비교: Proposal System(Tier 0) / Baseline(matched query_index) F / AR / CP / Q_total 평균 ± SEM 막대 차트 + paired t-test |
| **섹션 2** | 에스컬레이션 패턴: Proposal System Tier 0/1/2 분포 및 조건별 RAGAS 성능 표 |
| **섹션 3** | 자가 교정 루프 수렴: 루프 번호별 Mean Q_total + 95% CI + 수렴율(τ_Q=0.8) |
| **섹션 4** | FK Grade 간접 검증: user_level별 박스플롯 + 조건별 평균 막대차트 + 목표 달성률 표(is_final=1 only) |
| **섹션 5** | 계산 효율성: 조건별 평균 처리 시간 (초, 95% CI) |
| **섹션 6** | 수준 분류기 성능: Proposal System 기준 Accuracy / Precision / Recall / F1, TP/TN/FP/FN |
| **섹션 7** | DeepEval 메트릭 비교: Proposal System(Tier 0) / Baseline(matched query_index) Faithfulness / Answer Relevancy 독립 교차검증 + paired t-test |
| **섹션 8** | 검색 품질(IR): Tier 0, is_final=1 한정 Hit Rate / MRR — RAGAS CP와의 교차검증 |

---

### 3.12 FK Grade 측정

| 요구사항 ID | FR-020 |
|-------------|--------|
| **요구사항명** | Flesch-Kincaid Grade Level 측정 |
| **우선순위** | 필수 |
| **설명** | 시스템은 답변의 가독성을 Flesch-Kincaid Grade Level로 자동 측정하여 감사 로그에 저장해야 한다. |
| **계산 공식** | `0.39 × (단어수/문장수) + 11.8 × (음절수/단어수) − 15.59` |
| **계산 시점** | output_agent(출처·면책 조항 추가) 호출 전, 영어 원문 답변(state["answer"])을 기준으로 계산 |
| **저장 조건** | is_final=TRUE이고 is_fallback=FALSE인 행에만 저장. 나머지는 NULL |
| **목표 기준** | Consumer: fk_grade ≤ 9 (NIH 건강 정보 이해도 권고 수준) / Professional: fk_grade ≥ 12 (의학 저널 평균 수준) |
| **적용 예외** | Tier 1(fk_grade 저장은 하지만 AR 단독 평가이므로 영어 원문 품질이 낮을 수 있음), Fallback(fk_grade=NULL) |

| 요구사항 ID | FR-021 |
|-------------|--------|
| **요구사항명** | 성능평가 전용 지표 (IR Hit Rate/MRR, DeepEval) |
| **우선순위** | 선택 |
| **설명** | 시스템은 `disease`(STQS-240/ablation 정답 라벨)가 있는 요청에 한해, RAGAS(FR-008)와 독립적으로 전통적 정보검색 지표와 별도 프레임워크 기반 교차검증 지표를 계산해 감사 로그에 기록해야 한다. Self-Corrective Loop(FR-009)의 성공/에스컬레이션 판단에는 관여하지 않는다. |
| **배경** | Critic Agent가 RAGAS를 런타임 Quality Gate로 사용하면서 최종 성능평가까지 RAGAS 단독에 의존하면 "동일 지표로 최적화하고 동일 지표로 평가"하는 순환성(circularity) 편향 비판이 발생할 수 있어, 독립 지표로 교차검증한다. |
| **IR Hit Rate / MRR** | 검색된 청크의 출처 파일명(`context_sources`)에 `disease`명이 포함되는지로 정답 문서 적중 여부(Hit Rate, 0/1)와 첫 적중 순위의 역수(MRR)를 계산. LLM 불필요, 순수 문자열 매칭 |
| **DeepEval** | Faithfulness / Answer Relevancy 2종을 RAGAS와 다른 프레임워크(판정 LLM은 `DEEPEVAL_LLM_MODEL`, 기본 **gpt-4o-mini**)로 계산해 F/AR을 교차검증. Contextual Relevancy는 더 이상 사용하지 않음 |
| **생략 조건** | `disease`가 없는 일반 운영 쿼리는 ground truth가 없거나(IR) 교차검증 목적이 없어(DeepEval) 계산을 생략하고 NULL로 저장한다 (불필요한 LLM 호출 비용 방지) |
| **저장** | `hit_rate_score`, `mrr_score`, `deepeval_faithfulness`, `deepeval_answer_relevancy` → 감사 로그의 `ir_hit_rate`, `ir_mrr`, `deepeval_faithfulness`, `deepeval_answer_relevancy` 컬럼 |

---

## 4. 비기능 요구사항

### 4.1 성능 요구사항

| 요구사항 ID | NFR-001 |
|-------------|---------|
| **요구사항명** | 응답 시간 |
| **설명** | 일반적인 질의(Tier 0 성공, 루프 없음)의 경우 30초 이내에 답변을 제공해야 한다. RAGAS 평가 포함 시 최대 120초를 허용한다. |

| 요구사항 ID | NFR-002 |
|-------------|---------|
| **요구사항명** | RAGAS 평가 타임아웃 |
| **설명** | RAGAS 평가는 120초 이내에 완료되어야 하며, 초과 시 0.0 기본값을 반환하고 계속 진행한다. |

### 4.2 신뢰성 요구사항

| 요구사항 ID | NFR-003 |
|-------------|---------|
| **요구사항명** | Fallback 보장 |
| **설명** | 모든 Tier 소진 후에도 답변이 기준을 충족하지 못할 경우, 최선의 답변을 Fallback으로 제공하며 시스템이 오류 없이 종료되어야 한다. |

| 요구사항 ID | NFR-004 |
|-------------|---------|
| **요구사항명** | 이벤트 루프 안정성 |
| **설명** | Streamlit의 자체 이벤트 루프와 RAGAS의 비동기 평가 루프가 충돌하지 않도록 ThreadPoolExecutor를 통해 격리하여 실행해야 한다. |

### 4.3 보안 요구사항

| 요구사항 ID | NFR-005 |
|-------------|---------|
| **요구사항명** | API 키 보안 |
| **설명** | OpenAI API 키 및 Oracle DB 연결 정보는 .env 파일에만 저장하며 소스코드에 하드코딩하지 않는다. |

### 4.4 유지보수성 요구사항

| 요구사항 ID | NFR-006 |
|-------------|---------|
| **요구사항명** | 설정 중앙화 |
| **설명** | 모든 임계값(FAITHFULNESS_THRESHOLD, AR_THRESHOLD 등) 및 모델 설정은 config/settings.py에서 환경변수로 관리하여 코드 수정 없이 조정 가능해야 한다. |

| 요구사항 ID | NFR-007 |
|-------------|---------|
| **요구사항명** | 모듈 분리 |
| **설명** | 에이전트(agents/), 인프라(infra/), UI(ui/), 설정(config/)은 독립적으로 유지하여 각 컴포넌트를 개별적으로 교체 가능해야 한다. |

### 4.5 확장성 요구사항

| 요구사항 ID | NFR-008 |
|-------------|---------|
| **요구사항명** | LLM 백엔드 교체 가능성 |
| **설명** | OpenAI API를 LLM 백엔드로 사용하며, 추가 LLM 백엔드로 확장 가능한 구조로 설계되어야 한다. |

---

## 5. 시스템 인터페이스 요구사항

### 5.1 외부 인터페이스

| 인터페이스 | 유형 | 설명 |
|-----------|------|------|
| OpenAI API | REST API | GPT 모델을 통한 답변 생성, 쿼리 재작성, 분류 |
| Anthropic API | REST API | Claude 모델을 통한 RAGAS 판정 (F/AR/CP). 답변 생성 LLM과 무관하게 항상 사용 (필수) |
| Google Gemini API | REST API | 답변 생성 LLM 백엔드 토글 (`llm_provider="gemini"` 지정 시 선택적으로 사용). DeepEval 판정에는 더 이상 사용하지 않음(현재는 OpenAI gpt-4o-mini) |
| Oracle Database | oracledb 직접 연결 | 감사 로그 저장 및 조회 (N+1행 INSERT only) |
| DuckDuckGo Search | 라이브러리 | Tier 2 웹검색 |
| HuggingFace | 모델 다운로드 | **BAAI/bge-base-en-v1.5** 임베딩 모델 (768차원) |

### 5.2 사용자 인터페이스

| 화면 | 설명 |
|------|------|
| 메인 질의 화면 | 텍스트 입력, 실행 상태 표시, 점수 카드 (F/AR/CP/Q_total), 답변 출력, 로그 조회 |
| 사이드바 | 사용자 페르소나 선택, LLM 백엔드 선택, 인덱스 재빌더, 대시보드 메뉴 |
| 로그 조회 화면 | 감사 로그 목록 및 상세 조회 (tier_path, self_correction_count, fk_grade 포함) |
| 성능 시각화 화면 | Proposal System vs Baseline 8개 섹션 차트 및 요약 통계 카드 |

---

## 6. 데이터 요구사항

### 6.1 입력 데이터

- **PDF 문서**: MSD 매뉴얼 질환별 PDF (소비자용/전문가용)
- **사용자 질문**: 한국어 자연어 텍스트
- **STQS-240**: 표준 테스트 질문 세트 (main.ipynb에 정의, 240건, 40개 질환 × P3문항·C3문항)

### 6.2 저장 데이터

- **FAISS 인덱스**: `db/msd_faiss.index/` (LangChain FAISS, BAAI/bge-base-en-v1.5 768차원)
- **감사 로그**: Oracle DB의 rag_audit_log_bak 테이블 (request_id당 N+1행, fk_grade 포함, final_answer CLOB)

### 6.3 출력 데이터

- **최종 답변**: 한국어 의료 정보 텍스트
- **품질 점수**: F, AR, CP (0~1 실수), Q_total (0.4·F + 0.4·AR + 0.2·CP)
- **가독성 점수**: fk_grade (Flesch-Kincaid Grade Level, 영어 원문 기준)
- **출처 정보**: 검색된 PDF 파일명 및 페이지 번호

---

## 7. 품질 기준

### 7.1 시스템 임계값

| 지표 | 임계값 | 비고 |
|------|--------|------|
| Faithfulness | ≥ 0.8 | Self-Corrective Loop 성공 기준 |
| Answer Relevance | ≥ 0.8 | 모든 Tier 성공 기준 (Tier 1은 AR만 적용) |
| Context Precision | ≥ 0.8 | Self-Corrective Loop 성공 기준 |
| 즉시 에스컬레이션 AR | < 0.3 | VectorDB 미보유 판단 기준 |
| 즉시 에스컬레이션 F | < 0.3 | 검색 완전 실패 판단 기준 |
| 즉시 에스컬레이션 CP | < 0.2 | 검색 완전 실패 판단 기준 |
| 최대 재시도 횟수 | 3회/Tier | Tier당 최대 루프 횟수 |
| FK Grade (Consumer) | ≤ 9 | 일반인 가독성 목표 (NIH 건강 정보 이해도 기준) |
| FK Grade (Professional) | ≥ 12 | 전문가 가독성 목표 (의학 저널 평균 수준) |

### 7.2 연구 성과 지표 (STQS-240 기준)

| 지표 | 목표값 | Proposal System 달성값 | 비고 |
|------|--------|----------------------|------|
| Faithfulness (F) | ≥ 0.85 | 측정 중 | STQS-240 Proposal System 결과 |
| Answer Relevance (AR) | ≥ 0.80 | 측정 중 | STQS-240 Proposal System 결과 |
| Context Precision (CP) | ≥ 0.78 | 측정 중 | STQS-240 Proposal System 결과 |
| 할루시네이션 감소율 | ≥ 50% | 측정 중 | Proposal System vs Baseline 비교 |
| 사용자 수준 분류 정확도 | ≥ 90% | 측정 중 | Professional/Consumer 분류 |
| FK Grade Consumer 목표 달성률 | ≥ 70% | 측정 중 | fk_grade ≤ 9 비율 |
| FK Grade Professional 목표 달성률 | ≥ 70% | 측정 중 | fk_grade ≥ 12 비율 |

---

*본 문서는 논문 연구 목적의 시스템 산출물이며, 실제 임상 적용을 위한 의학적 검증은 포함하지 않습니다.*
