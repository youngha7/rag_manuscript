import asyncio
import logging
import math
import re
from typing import List, NamedTuple, Optional, Sequence

import config.settings as settings
from core.llm_client import ragas_async_client, ragas_model
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections.answer_relevancy import AnswerRelevancy
from ragas.metrics.collections.context_precision import ContextPrecisionWithoutReference
from ragas.metrics.collections.faithfulness import Faithfulness

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Flesch-Kincaid Grade Level (영어 원문 기준)
# 공식: 0.39*(words/sentences) + 11.8*(syllables/words) - 15.59
# 번역 전 영어 RAG 답변에 대해 계산한다.
# ──────────────────────────────────────────────────────────────────────────────
def _syllables_en(word: str) -> int:
    """영어 단어의 음절 수를 근사 계산한다."""
    import re
    word = word.lower().rstrip(".,!?;:'\"")
    if not word:
        return 0
    word = re.sub(r"e$", "", word)          # 묵음 e 제거
    count = len(re.findall(r"[aeiou]+", word))
    return max(1, count)


def flesch_kincaid_grade_en(text: str) -> float:
    """영어 텍스트의 Flesch-Kincaid Grade Level을 계산한다.

    번역 전 영어 원문(state['answer'])을 인자로 받아야 한다.
    """
    import re
    if not text or not text.strip():
        return 0.0
    sentences = [s.strip() for s in re.split(r"[.!?]", text) if s.strip()]
    words     = re.findall(r"\b[a-zA-Z]+\b", text)

    n_sent = max(len(sentences), 1)
    n_word = max(len(words), 1)
    n_syl  = sum(_syllables_en(w) for w in words)

    grade = 0.39 * (n_word / n_sent) + 11.8 * (n_syl / n_word) - 15.59
    return round(max(0.0, grade), 3)


# ── Consumer FK Grade 의학 용어 마스킹 (음절 왜곡 제거) ───────────────────────
# 패턴 1: 그리스/라틴 의학 접미사 (-itis, -ology, -osis, -oma 등)
_PURE_FK_SUFFIX = re.compile(
    r"\b\w+(?:itis|ology|ectomy|plasty|oscopy|otomy|ostomy|algia|pathy"
    r"|emia|uria|graphy|trophy|genesis|lysis|osis|oma"
    r"|pnea|plasia|ectasis|rrhea)\b",
    re.IGNORECASE,
)
# 패턴 2: 소유격 병명 (Alzheimer's disease, Parkinson's syndrome 등)
_PURE_FK_EPONYM = re.compile(r"\b[A-Z][a-z]+'s\s+(?:disease|syndrome|disorder)\b")
# 패턴 3: 10자 이상 단어 (cardiovascular, bronchoconstriction 등)
# (2026-09-17: 8자 기준이었을 때 "pressure", "diagnosed", "resistance"처럼 흔한
#  단어까지 걸려 Consumer FK가 목표(4~6)보다 훨씬 낮게 측정됨 — 10자로 완화)
_PURE_FK_LONG = re.compile(r"\b[A-Za-z]{10,}\b")
# 패턴 3b: MSD 청크·출처 메타데이터 (FAISS 원문 유출 시 FK 왜곡)
_PURE_FK_BOILERPLATE = re.compile(
    r"\b(?:overview|manual|version|reviewed|modified|disorders|professional|consumer"
    r"|edition|msdmanuals|https?|www)\b",
    re.IGNORECASE,
)
# 패턴 3c: Professional 템플릿 섹션 제목 (답변 본문 FK 측정에서 제외)
_PURE_FK_SECTION = re.compile(
    r"(?im)\b(?:pathophysiology|diagnostic criteria|therapeutic approach|clinical considerations)\s*:\s*"
)
# 패턴 4: FAISS 원문 빈도 기반 명시 목록 (6-9자, 접미사·길이 패턴 미탐지)
_PURE_FK_EXPLICIT = frozenset({
    # Cardiovascular
    "hypertension", "coronary",    "myocardial",  "infarction",
    "arrhythmia",   "cholesterol", "angiotensin", "ischemic",
    "vascular",     "diastolic",   "cerebral",    "angina",
    # Respiratory
    "pulmonary",    "obstructive", "respiratory", "pneumonia",
    "bronchospasm", "emphysema",   "influenza",   "pleurisy",
    "pleuritic",    "sputum",      "exudate",     "atypical",
    # GI / Digestive
    "esophagus",    "stricture",   "pyloric",     "duodenal",
    "peptic",       "gastric",     "duodenum",    "mucosal",
    "mucosa",       "reflux",
    # Metabolic / Endocrine
    "diabetes",     "mellitus",    "metabolic",   "endocrine",
    "deficiency",   "potassium",   "estrogen",    "progesterone",
    "creatinine",   "metformin",   "thyroid",     "glycemic",
    # Oncology
    "carcinoma",    "colorectal",  "chemotherapy","tamoxifen",
    "metastatic",   "lymphoma",    "sarcoma",     "melanoma",
    "rectal",       "sigmoid",     "polyp",       "biopsy",
    # Neurology / Psychiatry
    "dementia",     "alzheimer",   "depression",  "depressive",
    "neurologic",   "cognitive",   "serotonin",   "dopamine",
    "confusion",
    # Musculoskeletal
    "synovial",     "cartilage",
    # Immunology / Inflammation
    "inflammatory", "inflammation","bacterial",   "antibiotics",
    "antibiotic",   "neutrophil",  "hormonal",    "antiviral",
    # Infectious disease / COVID
    "variant",      "quarantine",
    # General clinical symptoms
    "malaise",      "vomiting",
    # General medical
    "receptor",     "receptors",   "inhibitor",   "diagnostic",
    "impairment",   "mortality",   "hemorrhage",  "abdominal",
    # 6-9자 고빈도 의학 용어 (3음절 이상, 패턴 1·3 미탐지)
    "nausea",       "oxygen",
    "therapy",      "surgery",     "cardiac",     "calcium",
    "insulin",      "vitamin",     "hepatic",     "urinary",
    "aspirin",      "medicine",    "bacteria",
    "molecule",     "antibody",    "nutrient",    "cellular",
    "clinical",     "physical",    "surgical",
    "infection",    "condition",   "procedure",   "physician",
    "ibuprofen",
    "coffee",
    "caffeine",
    "alcohol",
    "vegetables",
    "vegetable",
    "fiber",
    "relatives",
    "daughter",
    "genetic",
    "bedtime",
    "nighttime",
    "breathing",
    "panic",
    "palpitations",
    "hyperglycemia",
    "hypoglycemia",
    "glycemic",
    "insomnia",     "anhedonia",   "dysthymia",
    # Anxiety / autonomic (불안장애 답변)
    "dizziness",    "trembling",
    # CKD (만성신장질환 답변)
    "nocturia",
    # Breast / oncology (유방암 답변)
    "mammogram",    "palpable",
    # General clinical (갑상선·빈혈·천식 답변)
    "fatigue",      "prognosis",   "inhaler",
    # Vascular / cardiovascular (고혈압·뇌졸중 답변)
    "arteries",     "arterial",    "venous",      "venules",
    "narrowing",    "narrowed",    "deposits",    "clotting",
    "elevated",     "ruptures",    "stenotic",
    # Hematology (빈혈 답변)
    "erythrocyte",  "ferritin",    "serum",       "marrow",
    "carries",      "carrying",    "endurance",
    # Immune / infection (폐렴·불안 답변)
    "triggers",     "mediates",    "activates",   "cascade",
    "pathogens",    "pathogen",    "microbes",
    # Anxiety disorder (불안장애 Consumer 답변 — FK 개선)
    "anxiety",      "excessive",   "somatic",     "disorder",
    "emotions",     "sadness",
    # Iron-deficiency anemia (빈혈 Consumer 답변 — FK 개선)
    "capacity",     "abnormal",    "produces",    "pallor",
    # Community-acquired pneumonia (폐렴 Consumer 답변 — FK 개선)
    "community",    "acquired",    "effusion",    "severity",
    "productive",
    # Cardiovascular / CAD·흡연 (관상동맥 Consumer FK≥9 대응)
    "smoking",      "tobacco",     "cigarette",   "nicotine",
    "atherosclerosis", "plaque",   "lipid",       "lipids",
    "triglyceride", "atheroma",    "ischemia",    "ischemic",
    "ventricle",    "ventricular", "myocardium",  "cardiovascular",
    "circulation",  "circulatory", "atherosclerotic",
    # STQS Consumer 공통 고음절 연결어
    # (2026-09-17: FK가 4~6 목표보다 너무 낮게 나와, 초등 수준의 흔한 단어(blood,
    #  heart, level, body, disease, chronic 등)는 마스킹 목록에서 제외함 —
    #  그런 단어까지 "it"으로 치환하면 실제 문장 복잡도보다 과도하게 낮게 측정됨)
    "demonstrates", "manifestation", "approximately", "significantly",
    "particularly", "especially",    "frequently",  "commonly",
    "typically",    "generally",     "therefore",
    "including",    "resulting",
    "damages",      "damaged",       "damaging",
    "promotes",     "promoting",     "prevents",    "preventing",
    "increases",    "increased",     "decreases",   "decreased",
    "affecting",    "develops",      "developed",   "developing",
    "occurred",      "symptoms",
    "syndrome",     "disorder",      "disorders",
    "treatment",    "treatments",
    "medication",   "medications",
    "intestine",    "intestinal",    "esophageal",  "colonic",
    "screening",    "endoscopy",     "colonoscopy", "mammography",
    "palpation",    "examination",   "findings",
    "prevention",   "preventive",    "lifestyle",   "exercise",
    "overweight",
    "pancreas",      "pancreatic",   "intestinal",
    "wheezing",      "breathing",
    "shortness",    "coughing",      "sputum",
    "abdominal",
    "initial",       "secondary",
    "abnormal",
    "proteins",
    "oxygen",       "dioxide",
})
_PURE_FK_EXPLICIT_PAT = re.compile(
    r"\b(?:" + "|".join(sorted(_PURE_FK_EXPLICIT, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def _prep_fk_measurement_text(text: str) -> str:
    if not text:
        return ""
    t = _PURE_FK_SECTION.sub("", text)
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"\b\d{1,2}/\d{1,2}\b", " ", t)
    return t


def get_pure_fk_grade(text: str) -> float:
    """의학 전문 용어를 'it'으로 마스킹한 순수 문장 구조의 FK Grade를 반환한다.

    RAGAS·번역·원문 답변에는 영향 없음 (graph._output_node에서만 호출).
    """
    masked = _prep_fk_measurement_text(text)
    masked = _PURE_FK_SUFFIX.sub("it", masked)
    masked = _PURE_FK_EPONYM.sub("it", masked)
    masked = _PURE_FK_BOILERPLATE.sub("it", masked)
    masked = _PURE_FK_LONG.sub("it", masked)
    masked = _PURE_FK_EXPLICIT_PAT.sub("it", masked)
    return flesch_kincaid_grade_en(masked)


class OfficialRagasScores(NamedTuple):
    faithfulness: float
    answer_relevance: float
    context_precision: float


def _safe_unit(v: float) -> float:
    if v is None:
        return 0.0
    x = float(v)
    if math.isnan(x):
        return 0.0
    return max(0.0, min(1.0, x))


def _prep_contexts(chunks: Sequence[str]) -> List[str]:
    out: List[str] = []
    for c in chunks:
        t = (c or "").strip()
        if t:
            out.append(t[:settings.RAGAS_CONTEXT_MAX_CHARS])
    if not out:
        out = ["(no retrieved context)"]
    return out


_ragas_embedder = None


def _get_ragas_embedder() -> "HuggingFaceEmbeddings":
    """AnswerRelevancy용 임베딩 모델을 지연 생성 후 재사용한다.

    쿼리마다 새로 생성하면 GPU에 모델 가중치가 매번 새로 올라가고 이전
    인스턴스가 즉시 해제되지 않아, 배치 실행(예: 480회 ablation study) 중
    CUDA 메모리가 누적되어 결국 OOM으로 이어진다.
    """
    global _ragas_embedder
    if _ragas_embedder is None:
        _ragas_embedder = HuggingFaceEmbeddings(
            model=settings.EMBEDDING_MODEL, device=settings.EMBEDDING_DEVICE
        )
    return _ragas_embedder


def compute_official_ragas_scores(
    question: str,
    answer_body: str,
    context_chunks: Sequence[str],
    ar_query: str = None,
) -> OfficialRagasScores:
    q = (question or "").strip()[:500]
    # AR은 질문 의도가 고정된 첫 번째 쿼리로 평가 (재시도마다 쿼리가 바뀌면 AR이 폭락)
    q_ar = (ar_query or question or "").strip()[:500]
    a = (answer_body or "").strip()[:settings.RAGAS_ANSWER_MAX_CHARS]
    ctx_list = _prep_contexts(context_chunks)

    eval_client = ragas_async_client()
    llm = llm_factory(
        ragas_model(),
        provider="openai",
        client=eval_client,
        max_tokens=settings.RAGAS_LLM_MAX_TOKENS,
    )
    embedder = _get_ragas_embedder()

    faith = Faithfulness(llm=llm)
    arel = AnswerRelevancy(llm=llm, embeddings=embedder, strictness=settings.RAGAS_STRICTNESS)
    cpre = ContextPrecisionWithoutReference(llm=llm)

    async def _run_all() -> OfficialRagasScores:
        async def _score_faith():
            try:
                r = await faith.ascore(user_input=q, response=a, retrieved_contexts=ctx_list)
                return _safe_unit(r.value)
            except Exception as e:
                logger.warning("Faithfulness 평가 실패: %s", e, exc_info=True)
                return 0.0

        async def _score_arel():
            try:
                r = await arel.ascore(user_input=q_ar, response=a)
                return _safe_unit(r.value)
            except Exception as e:
                logger.warning("AnswerRelevancy 평가 실패: %s", e, exc_info=True)
                return 0.0

        async def _score_cpre():
            try:
                r = await cpre.ascore(user_input=q, response=a, retrieved_contexts=ctx_list)
                return _safe_unit(r.value)
            except Exception as e:
                logger.warning("ContextPrecision 평가 실패: %s", e, exc_info=True)
                return 0.0

        try:
            ff, ar, cp = await asyncio.gather(_score_faith(), _score_arel(), _score_cpre())
        finally:
            # httpx AsyncClient를 루프 종료 전에 명시적으로 닫아
            # "Event loop is closed" 경고를 방지한다.
            try:
                await eval_client.close()
            except Exception:
                pass

        logger.warning("[RAGAS] scores F=%.3f AR=%.3f CP=%.3f | q=%r | ar_q=%r | a=%r | ctx=%d",
                       ff, ar, cp, q[:60], q_ar[:60], a[:60], len(ctx_list))

        return OfficialRagasScores(
            faithfulness=ff,
            answer_relevance=ar,
            context_precision=cp,
        )

    # Streamlit은 자체 이벤트 루프를 보유하므로 asyncio.run() 직접 호출 시
    # "This event loop is already running" 오류가 발생한다.
    # 별도 스레드에서 새 이벤트 루프를 생성해 실행하면 충돌을 피할 수 있다.
    import concurrent.futures

    def _run_in_thread() -> OfficialRagasScores:
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        try:
            return new_loop.run_until_complete(_run_all())
        finally:
            # pending 태스크를 모두 정리한 후 루프를 닫아
            # "Event loop is closed" 경고를 방지한다.
            try:
                pending = asyncio.all_tasks(new_loop)
                if pending:
                    new_loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
                new_loop.run_until_complete(new_loop.shutdown_asyncgens())
            except Exception:
                pass
            new_loop.close()

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(_run_in_thread)
            return future.result(timeout=120)
    except Exception as e:
        logger.error("RAGAS 평가 전체 실패: %s", e, exc_info=True)
        return OfficialRagasScores(
            faithfulness=0.0,
            answer_relevance=0.0,
            context_precision=0.0,
        )


# ──────────────────────────────────────────────────────────────────────────────
# 성능평가 전용 지표 (critic 루프 게이트와 무관, DB 기록·성능 시각화 전용)
#
# 아래 함수는 disease(STQS/ablation 정답 라벨)가 있는 요청에서만 호출한다
# (critic_agent에서 gating) — 순환성 교차검증이 의미가 없는 일반 운영 쿼리에
# 불필요한 LLM 호출·비용만 늘어난다.
# ──────────────────────────────────────────────────────────────────────────────
class DeepEvalScores(NamedTuple):
    faithfulness: Optional[float]
    answer_relevancy: Optional[float]


_deepeval_judge_model = None


def _get_deepeval_judge_model():
    """DeepEval용 판정 모델을 지연 생성 후 재사용한다 (settings.DEEPEVAL_LLM_MODEL,
    기본 gpt-4o-mini).

    원래는 RAGAS 판정 LLM(Claude)과 다른 provider(Gemini)를 써서 "같은 지표로
    최적화하고 같은 지표로 평가"하는 순환성 문제를 완화했으나, 현재는 비용/단순화를
    위해 RAGAS·DeepEval 판정 LLM을 모두 gpt-4o-mini로 통일했다 — 판정 LLM 간
    독립성은 더 이상 보장되지 않는다.
    """
    global _deepeval_judge_model
    if _deepeval_judge_model is None:
        import os
        os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")
        from deepeval.models import OpenAIModel
        _deepeval_judge_model = OpenAIModel(
            model=settings.DEEPEVAL_LLM_MODEL, api_key=settings.get_openai_api_key()
        )
    return _deepeval_judge_model


def _safe_unit_or_none(v) -> Optional[float]:
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(x):
        return None
    return max(0.0, min(1.0, x))


def compute_deepeval_scores(
    question: str,
    answer_body: str,
    context_chunks: Sequence[str],
) -> DeepEvalScores:
    """DeepEval 2종 지표 (Faithfulness / Answer Relevancy).

    RAGAS의 F/AR과 개념은 대응되지만 독립된 프레임워크로 채점하여 교차검증한다
    (판정 LLM은 settings.DEEPEVAL_LLM_MODEL). ContextualPrecision/Recall은
    reference(expected_output)가 필요해 제외했다 — 이 파이프라인은 정답 answer를
    따로 두지 않는다. Contextual Relevancy는 더 이상 사용하지 않는다.
    실패 시 0.0이 아닌 None을 반환해 DB에 NULL로 남긴다 (측정 실패를 낮은 점수로
    오인하지 않도록).
    """
    q = (question or "").strip()
    a = (answer_body or "").strip()
    chunks = [c.strip() for c in (context_chunks or []) if (c or "").strip()]
    if not q or not a or not chunks:
        return DeepEvalScores(None, None)

    try:
        model = _get_deepeval_judge_model()
    except Exception as e:
        logger.warning("DeepEval 판정 모델 생성 실패: %s", e, exc_info=True)
        return DeepEvalScores(None, None)

    from deepeval.test_case import LLMTestCase
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        FaithfulnessMetric,
    )

    test_case = LLMTestCase(input=q, actual_output=a, retrieval_context=chunks)

    def _faithfulness() -> Optional[float]:
        try:
            metric = FaithfulnessMetric(model=model, include_reason=False, async_mode=False)
            metric.measure(test_case)
            return metric.score
        except Exception as e:
            logger.warning("DeepEval Faithfulness 평가 실패: %s", e, exc_info=True)
            return None

    def _answer_relevancy() -> Optional[float]:
        try:
            metric = AnswerRelevancyMetric(model=model, include_reason=False, async_mode=False)
            metric.measure(test_case)
            return metric.score
        except Exception as e:
            logger.warning("DeepEval Answer Relevancy 평가 실패: %s", e, exc_info=True)
            return None

    import concurrent.futures

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            f_f = ex.submit(_faithfulness)
            f_ar = ex.submit(_answer_relevancy)
            f = f_f.result(timeout=90)
            ar = f_ar.result(timeout=90)
    except Exception as e:
        logger.error("DeepEval 2종 지표 평가 전체 실패: %s", e, exc_info=True)
        return DeepEvalScores(None, None)

    logger.info("[DeepEval] scores F=%s AR=%s", f, ar)

    return DeepEvalScores(
        faithfulness=_safe_unit_or_none(f),
        answer_relevancy=_safe_unit_or_none(ar),
    )
