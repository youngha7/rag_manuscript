# 유스케이스 정의서 (Use Case Definition)

**프로젝트명**: 의료 정보 자기교정 RAG 시스템  
**문서버전**: v3.0  
**작성일**: 2026-05-16  
**작성자**: 연구자

---

## 1. 액터 정의

| 액터 | 유형 | 설명 |
|------|------|------|
| **일반 사용자** | 주 액터 | 의료 정보를 질의하는 사용자. 일반인 또는 의료 전문가 |
| **연구자 / 시스템 관리자** | 주 액터 | 인덱스 재빌드, 로그 조회, 성능 모니터링, 시스템 성능 평가 실험 수행 |
| **OpenAI API** | 보조 액터 | 사용자 분류, 쿼리 최적화, 답변 생성 수행 |
| **Anthropic API** | 보조 액터 | RAGAS 판정(F/AR/CP) 수행. 답변 생성 LLM과 무관하게 항상 사용 (순환성 방지) |
| **Google Gemini API** | 보조 액터 | 답변 생성 LLM 백엔드 토글 (`llm_provider="gemini"` 지정 시 선택적으로 사용) |
| **FAISS VectorDB** | 보조 액터 | 벡터 유사도 검색 수행 |
| **Oracle DB** | 보조 액터 | 감사 로그 저장 및 조회 (N+1행 INSERT) |
| **DuckDuckGo** | 보조 액터 | Tier 2 웹검색 수행 |

---

## 2. 유스케이스 목록

| UC ID | 유스케이스명 | 액터 | 우선순위 |
|-------|-------------|------|---------|
| UC-01 | 의료 정보 질의 | 일반 사용자 | 필수 |
| UC-02 | 사용자 수준 수동 설정 | 일반 사용자 | 선택 |
| UC-03 | LLM 백엔드 확인 | 일반 사용자 | 선택 |
| UC-04 | PDF 문서 업로드 | 시스템 관리자 | 선택 |
| UC-05 | 전체 인덱스 재빌드 | 시스템 관리자 | 선택 |
| UC-06 | 감사 로그 조회 | 시스템 관리자 | 선택 |
| UC-07 | 성능 시각화 조회 | 시스템 관리자 / 연구자 | 선택 |
| UC-08 | 시스템 성능 평가 일괄 실험 | 연구자 | 선택 |

---

## 3. 유스케이스 상세 정의

---

### UC-01: 의료 정보 질의

| 항목 | 내용 |
|------|------|
| **유스케이스 ID** | UC-01 |
| **유스케이스명** | 의료 정보 질의 |
| **액터** | 일반 사용자 |
| **목적** | 의료 관련 질문을 입력하여 신뢰성 있는 한국어 답변을 제공받는다 |
| **사전 조건** | 시스템이 정상 실행 중이며, FAISS 인덱스(`db/msd_faiss.index/`)가 존재한다 |
| **사후 조건** | RAGAS 기준(F≥0.8, AR≥0.8, CP≥0.8)을 충족하는 한국어 답변이 화면에 표시되고, Oracle DB에 **critic 평가 횟수 N개의 중간 행(is_final=FALSE) + 최종 행 1개(is_final=TRUE)**가 INSERT된다. 최종 행에는 fk_grade가 포함된다. |

**기본 흐름:**

```
1. 사용자가 질문을 텍스트 영역에 입력한다
2. 사용자가 "질문 제출" 버튼을 클릭한다
3. 시스템이 LLM으로 사용자 수준을 분류한다 (Professional / Consumer)
4. 시스템이 질문을 영문 의료 검색 쿼리로 최적화한다
5. 시스템이 FAISS VectorDB에서 관련 청크를 검색한다 (Tier 0)
6. 시스템이 RAGAS로 Faithfulness, Answer Relevance, Context Precision을 평가한다 (판정 LLM: Anthropic Claude, 답변 생성 LLM과 별도)
   → save_loop_log() : Oracle DB에 is_final=FALSE 중간 행 INSERT (eval_count=1)
7. [F≥0.8 AND AR≥0.8 AND CP≥0.8] → 8단계로 진행
8. 시스템이 사용자 수준별 FK Grade 목표를 적용하여 영문 답변을 생성한다
   (Consumer: ≤9, Professional: ≥12)
9. output_node에서 출처·면책 조항 추가 전 영어 원문으로 fk_grade를 계산한다
10. output_agent가 출처·면책 조항을 추가한다 (번역 없이 영어 원문 그대로 제공)
11. save_audit_log(fk_grade=fk) : is_final=TRUE 최종 행 INSERT
12. 화면에 점수 카드(F/AR/CP)와 최종 답변이 표시된다
```

**대안 흐름 A — Self-Corrective Loop (재시도):**
```
6a. RAGAS 점수가 기준 미달이며 즉시 에스컬레이션 조건이 아닌 경우
6b. save_loop_log() : is_final=FALSE 중간 행 INSERT
6c. 이전 평가 결과(F, AR, CP, critic_feedback)를 반영하여 쿼리를 재최적화한다
    (self_correction_count 1 증가)
6d. VectorDB 재검색 및 RAGAS 재평가를 수행한다
6e. 최대 3회까지 반복. 성공 시 8단계로 복귀
    tier_path는 "0" 유지, self_correction_count는 최대 3까지 누적
```

**대안 흐름 B — 즉시 에스컬레이션 (Tier 0 → Tier 1):**
```
6b-1. AR < 0.3 이거나 (F < 0.3 AND CP < 0.2) 인 경우
6b-2. save_loop_log() : is_final=FALSE 중간 행 INSERT
6b-3. 쿼리 재시도 없이 즉시 Tier 1(LLM 학습데이터)로 에스컬레이션
      tier_path = "0→1", is_escalated = true
6b-4. LLM 지식 기반 답변 생성 후 AR만으로 평가 (Tier 1은 AR 단독 평가)
      save_loop_log() : is_final=FALSE 중간 행 INSERT (F=NULL, CP=NULL)
6b-5. AR≥0.8 이면 8단계로 복귀. 미달 시 Tier 2로 에스컬레이션
      tier_path = "0→1→2"
```

**대안 흐름 C — Tier 2 웹검색 에스컬레이션:**
```
6c-1. Tier 1도 기준 미달인 경우
6c-2. DuckDuckGo 웹검색으로 최신 정보를 보완
6c-3. save_loop_log() : is_final=FALSE 중간 행 INSERT
6c-4. 웹 검색 결과 기반 답변 생성 후 F+AR+CP 재평가
6c-5. 기준 충족 시 8단계로 복귀. 미달 시 Fallback으로 전환
```

**예외 흐름 E — Fallback:**
```
E1. 모든 Tier 소진 후에도 기준 미달인 경우
E2. 검색된 원문 자료를 그대로 제시하며 신뢰도 부족 경고 메시지를 표시
E3. save_audit_log(is_fallback=True, fk_grade=None)
    is_final=TRUE, is_fallback=TRUE, fk_grade=NULL로 최종 행 INSERT
```

**예외 흐름 E2 — 빈 질문:**
```
E2-1. 사용자가 빈 텍스트로 제출한 경우
E2-2. "질문을 입력해주세요" 경고 메시지를 표시하고 처리를 중단
```

---

### UC-02: 사용자 수준 수동 설정

| 항목 | 내용 |
|------|------|
| **유스케이스 ID** | UC-02 |
| **유스케이스명** | 사용자 수준 수동 설정 |
| **액터** | 일반 사용자 |
| **목적** | LLM 자동 분류 대신 사용자가 직접 페르소나를 지정한다 |
| **사전 조건** | 사이드바가 표시되어 있다 |
| **사후 조건** | 선택된 페르소나가 다음 질의에 적용된다 |

**기본 흐름:**
```
1. 사용자가 사이드바의 "사용자 페르소나 선택" 드롭다운을 클릭한다
2. "자동 분류 / 의료 전문가 / 일반인" 중 하나를 선택한다
3. 선택값이 다음 질의 시 forced_user_level로 적용된다
4. "의료 전문가" 선택 시 → Professional 강제 적용 (FK Grade ≥12 목표)
5. "일반인" 선택 시 → Consumer 강제 적용 (FK Grade ≤9 목표)
6. "자동 분류" 선택 시 → LLM 자동 분류로 복귀
```

---

### UC-03: LLM 백엔드 확인

| 항목 | 내용 |
|------|------|
| **유스케이스 ID** | UC-03 |
| **유스케이스명** | LLM 백엔드 확인 |
| **액터** | 일반 사용자 |
| **목적** | 사이드바에서 현재 사용 중인 LLM 백엔드(OpenAI)를 확인한다 |
| **사전 조건** | 사이드바가 표시되어 있다 |
| **사후 조건** | 표시된 LLM이 다음 질의에 사용된다 |

**기본 흐름:**
```
1. 사이드바에 현재 LLM 백엔드(OpenAI)가 표시된다
2. 모든 질의에 OpenAI GPT 모델이 사용된다
```

---

### UC-04: PDF 문서 업로드

| 항목 | 내용 |
|------|------|
| **유스케이스 ID** | UC-04 |
| **유스케이스명** | PDF 문서 업로드 |
| **액터** | 시스템 관리자 |
| **목적** | 새로운 PDF 문서를 시스템에 추가하여 검색 대상을 확장한다 |
| **사전 조건** | PDF 파일이 준비되어 있다 |
| **사후 조건** | 업로드된 PDF가 인덱스에 추가되어 검색 가능 상태가 된다 |

**기본 흐름:**
```
1. 관리자가 메인 화면의 PDF 업로더를 클릭한다
2. PDF 파일을 선택하거나 드래그 앤 드롭한다
3. 시스템이 PyMuPDF로 텍스트를 추출한다
4. 스캔 PDF인 경우 RapidOCR로 텍스트를 인식한다
5. 텍스트를 1000자 청크(60자 오버랩)로 분할한다 (`.env`의 `MEDICAL_RAG_CHUNK_MAX_CHARS` 기준, 코드 기본값은 500자)
6. BAAI/bge-base-en-v1.5 모델로 청크를 벡터화하여 FAISS 인덱스에 추가한다
7. "N개 PDF 추가 완료" 메시지를 사이드바에 표시한다
```

**예외 흐름:**
```
E1. URL 청크(http://, https:// 포함) 등 유효하지 않은 청크는 필터링하여 인덱스에 추가하지 않는다
E2. 이미 인덱싱된 PDF는 중복 추가하지 않는다
```

---

### UC-05: 전체 인덱스 재빌드

| 항목 | 내용 |
|------|------|
| **유스케이스 ID** | UC-05 |
| **유스케이스명** | 전체 인덱스 재빌드 |
| **액터** | 시스템 관리자 |
| **목적** | data 폴더의 모든 PDF를 처음부터 재임베딩하여 FAISS 인덱스를 갱신한다 |
| **사전 조건** | data 폴더에 PDF 파일이 존재한다 |
| **사후 조건** | `db/msd_faiss.index/` 폴더가 최신 상태로 갱신된다 |

**기본 흐름:**
```
1. 관리자가 사이드바의 "인덱스 전체 재빌드" 버튼을 클릭한다
2. 시스템이 백그라운드 스레드에서 재빌드를 시작한다
3. 진행률(%)이 실시간으로 사이드바에 표시된다
4. 완료 시 "재빌드 완료: N개 PDF · M개 청크" 메시지가 표시된다
5. 관리자가 "확인" 버튼을 클릭하여 완료 상태를 종료한다
```

**예외 흐름:**
```
E1. 재빌드 중 오류 발생 시 오류 메시지를 표시하고 "닫기" 버튼을 제공한다
```

---

### UC-06: 감사 로그 조회

| 항목 | 내용 |
|------|------|
| **유스케이스 ID** | UC-06 |
| **유스케이스명** | 감사 로그 조회 |
| **액터** | 시스템 관리자 |
| **목적** | 시스템의 질의 이력과 RAGAS 평가 결과를 조회한다 |
| **사전 조건** | Oracle DB에 감사 로그가 저장되어 있다 |
| **사후 조건** | 로그 목록 및 선택한 로그의 상세 내용이 표시된다 |

**기본 흐름:**
```
1. 관리자가 사이드바의 "📊 RAG 성능 대시보드"를 클릭하여 펼친다
2. "📋 로그 조회" 버튼을 클릭한다
3. rag_audit_log에서 is_final=TRUE 행만 필터링하여 테이블로 표시한다
   (log_id, 생성일시, user_level, 원본질문, final_tier, tier_path, F/AR/CP, Q_total, fk_grade 등)
4. 관리자가 특정 로그 항목을 선택한다
5. 해당 요청의 전체 행(is_final=FALSE 중간 행 포함)의 상세 정보가 표시된다
   (질문, 답변, 점수, 루프별 변화, 티어 경로, 할루시네이션 정보, fk_grade 등)
```

---

### UC-07: 성능 시각화 조회

| 항목 | 내용 |
|------|------|
| **유스케이스 ID** | UC-07 |
| **유스케이스명** | 성능 시각화 조회 |
| **액터** | 시스템 관리자 / 연구자 |
| **목적** | Proposal System vs Baseline 성능 비교를 7개 섹션 차트로 시각화하여 분석한다 |
| **사전 조건** | rag_audit_log에 충분한 데이터가 존재한다 |
| **사후 조건** | 시스템 성능 시각화 차트와 요약 통계가 표시된다 |

**기본 흐름:**
```
1. 관리자/연구자가 사이드바의 "📊 RAG 성능 대시보드"를 클릭한다
2. "📈 성능 시각화" 버튼을 클릭한다
3. rag_audit_log에서 is_final=TRUE 행의 최신 데이터를 로드한다 (5분 캐시)
4. 요약 카드 표시: Proposal System / Baseline F 점수, 환각 비율, 건수
5. 섹션 1 — RAGAS 메트릭 비교
   : Proposal System / Baseline F / AR / CP 평균 ± 95% CI 막대 차트 + 데이터 테이블
6. 섹션 2 — 환각 감소 효과
   : 조건별 환각 감지 비율 및 Baseline 대비 감소율 + 테이블
7. 섹션 3 — 에스컬레이션 패턴 분석 (Tier 분포 — Proposal System)
   : Tier 분포 파이차트 + 막대차트 + 전문가/일반인/전체 Tier별 쿼리 건수 표
8. 섹션 4 — 수준 분류기 성능 (Proposal System)
   : Accuracy / Precision / Recall / F1 막대 차트 + 혼동 행렬 테이블
9. 섹션 4-b — FK Grade 간접 검증
   : user_level별 박스플롯 + 조건별 평균 막대차트 + 목표 달성률 상세 테이블
   (Consumer ≤9, Professional ≥12 목표 기준선 표시)
10. 섹션 5 — Self-Correction Loop 수렴 (Proposal System)
    : 루프 번호별 Mean Q_total 추이 + 95% CI + 수렴율 + 테이블
11. 섹션 6 — FK Grade 검증
    : Consumer/Professional 목표 달성률 시각화
12. 섹션 7 — 계산 효율성 (처리 시간)
    : 조건별 평균 처리 시간 (초, 95% CI) + 테이블
```

---

### UC-08: 시스템 성능 평가 일괄 실험

| 항목 | 내용 |
|------|------|
| **유스케이스 ID** | UC-08 |
| **유스케이스명** | 시스템 성능 평가 일괄 실험 |
| **액터** | 연구자 |
| **목적** | Proposal System(A)과 Baseline(E) 2가지 조건과 STQS-240 표준 질문 세트(240건)를 교차 실험하여 480건의 결과를 Oracle DB에 저장한다 |
| **사전 조건** | Jupyter Notebook(`main.ipynb`)이 실행 가능하고, FAISS 인덱스와 API 키가 설정되어 있다 |
| **사후 조건** | Oracle DB `rag_audit_log`에 480건의 요청 결과(각 요청당 N+1행)가 저장된다. is_final=TRUE 행에는 fk_grade가 포함된다. |

**기본 흐름:**
```
1. 연구자가 main.ipynb를 Jupyter Notebook에서 실행한다
2. STQS-240 질문 목록(240건)과 2가지 실험 조건(A/E)이 정의되어 있다
3. 외부 루프: 2가지 조건(A, E) 순회
4. 내부 루프: 240개 질문 순회 (disease, level, q_num, question) 4-tuple
   4a. run_medical_self_corrective_rag(
         question=question,
         ablation_condition=cond["key"],  # 'A' 또는 'E'
         query_index=q_num,               # 1-240
         disease=disease,                 # 질환명
         query_level_label=level,         # 'P' 또는 'C'
         forced_user_level=...,           # A: None, E: "Baseline"
       ) 호출
   4b. critic 평가마다 save_loop_log() → is_final=FALSE 중간 행 INSERT
   4c. output_node 또는 fallback_node에서 save_audit_log() → is_final=TRUE 최종 행 INSERT
       (output: fk_grade 포함, fallback: fk_grade=NULL)
5. 480건 완료 후 검증 쿼리(GROUP BY ablation_condition)로 건수 확인
```

**조건별 동작:**
```
조건 A (Proposal System): 자가 교정 + 멀티 티어, 사용자 분류기 실행
조건 E (Baseline)       : RAGAS 평가 후 즉시 출력, user_level="Baseline" 고정
```

**예외 흐름:**
```
E1. API 오류 발생 시 해당 질문 스킵 후 다음 질문으로 진행
E2. Supabase 연결 실패 시 로그에 오류 기록 후 계속 진행
```

---

## 4. 유스케이스 관계도

```
                    ┌────────────────────────────────────────┐
                    │         의료 정보 자기교정 RAG 시스템         │
                    │                                        │
  일반 사용자 ────────►│  UC-01: 의료 정보 질의                   │
                    │    └─ <<include>> 사용자 수준 자동 분류     │
                    │    └─ <<include>> 쿼리 최적화              │
                    │    └─ <<include>> Tier 0 VectorDB 검색   │
                    │    └─ <<include>> RAGAS 품질 평가         │
                    │    └─ <<include>> FK Grade 계산 (조항추가 전) │
                    │    └─ <<extend>>  Self-Corrective Loop  │
                    │    └─ <<extend>>  즉시 에스컬레이션 (Tier 1)│
                    │    └─ <<extend>>  웹검색 에스컬레이션 (Tier 2)│
                    │    └─ <<extend>>  Fallback               │
                    │    └─ <<include>> 감사 로그 저장 (N+1행)   │
                    │                                        │
                    │  UC-02: 사용자 수준 수동 설정              │
                    │    └─ <<extend>> UC-01 (분류 생략)        │
                    │                                        │
                    │  UC-03: LLM 백엔드 확인                  │
                    │                                        │
  시스템 관리자 ───────►│  UC-04: PDF 문서 업로드                 │
                    │  UC-05: 전체 인덱스 재빌드                 │
                    │  UC-06: 감사 로그 조회                   │
                    │  UC-07: 성능 시각화 조회 (7개 섹션)        │
                    │                                        │
  연구자      ────────►│  UC-08: 시스템 성능 평가 일괄 실험         │
                    │    └─ <<include>> UC-01 ×480회           │
                    └────────────────────────────────────────┘
```

---

## 5. 주요 시나리오

### 시나리오 1: 일반인의 감기 증상 질의 (Tier 0 성공)

```
사용자: "감기에 걸렸는데 어떤 증상이 나타나나요?"
→ LLM 분류: Consumer (신뢰도 0.92, 의도: 증상_설명)
→ 쿼리 최적화: "common cold symptoms consumer"
→ Tier 0 검색: MSD 매뉴얼 감기 챕터 청크 반환
→ RAGAS 평가: F=0.91, AR=0.88, CP=0.85 → Q_total=0.880
→ save_loop_log(): is_final=FALSE, loop_number=1
→ FK Grade 계산: flesch_kincaid_grade_en(영어 답변) → fk_grade=8.5 (≤9 목표 달성)
→ output_agent: 출처·면책 조항 추가 (번역 없음)
→ save_audit_log(fk_grade=8.5): is_final=TRUE
→ 감사 로그: tier_path="0", self_correction_count=0, is_fallback=false, fk_grade=8.5
```

### 시나리오 2: 전문가의 MDR-TB 질의 (Tier 2 에스컬레이션)

```
사용자: "MDR-TB 치료 프로토콜과 약물 레지멘은?"
→ LLM 분류: Professional (신뢰도 0.95, 의도: 처방_결정)
→ Tier 0: F=0.27, AR=0.19 → AR < 0.3 → 즉시 에스컬레이션
→ save_loop_log(): is_final=FALSE (Tier0 평가)
→ Tier 1 (LLM): AR=0.54 < 0.8 → Tier 2 에스컬레이션
→ save_loop_log(): is_final=FALSE (Tier1, F=NULL, CP=NULL)
→ Tier 2 (웹검색): F=0.84, AR=0.82, CP=0.81 → Q_total=0.828 → 성공
→ save_loop_log(): is_final=FALSE (Tier2 평가)
→ FK Grade: fk_grade=13.2 (≥12 목표 달성)
→ save_audit_log(fk_grade=13.2): is_final=TRUE
→ 감사 로그: tier_path="0→1→2", self_correction_count=0, is_escalated=true
```

### 시나리오 3: Baseline으로 실험

```
연구자: main.ipynb → 조건 E (Baseline), 질문 #15 (골관절염, Professional)
→ forced_user_level="Baseline" → LLM 분류 우회
→ Tier 0: F=0.52, AR=0.61 → 즉시 출력 (Baseline: 재시도 없음)
→ save_loop_log(): is_final=FALSE (Tier0 → 조건 E)
→ fk_grade 계산 → save_audit_log(): is_final=TRUE
→ 감사 로그: ablation_condition="E", query_index=15, disease="골관절염",
             tier_path="0", self_correction_count=0, fk_grade (값)
```

### 시나리오 4: 관리자의 성능 모니터링 (7개 섹션)

```
관리자: 사이드바 "📊 RAG 성능 대시보드" 클릭
→ "📈 성능 시각화" 선택
→ 요약 카드: Proposal System / Baseline F 점수, 환각 비율
→ 섹션 1: RAGAS 메트릭 비교 — Proposal System / Baseline 막대차트
→ 섹션 2: 환각 감소 — Proposal System이 Baseline 대비 52.3% 감소
→ 섹션 3: Tier 분포 — 파이차트 + 전문가/일반인 Tier별 표
           (Tier0: 38(76%)/42(84%), Tier1: 9(18%)/6(12%), Tier2: 3(6%)/2(4%))
→ 섹션 4: 수준 분류기 — Accuracy=94%, F1=0.93 확인
→ 섹션 4-b: FK Grade — Consumer 평균=8.5(≤9), Professional 평균=13.2(≥12) 확인
→ 섹션 5: Loop 수렴 — 루프 1→2→3 Q_total 상승 확인
→ 섹션 6: FK Grade 검증 — Consumer/Professional 목표 달성률 확인
→ 섹션 7: 처리 시간 — 조건별 평균 소요 시간 비교
```

---

*본 문서는 논문 연구 목적의 시스템 산출물이며, 실제 임상 적용을 위한 의학적 검증은 포함하지 않습니다.*
