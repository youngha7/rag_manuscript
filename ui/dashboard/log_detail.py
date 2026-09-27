"""Log viewer — detail screen."""
from __future__ import annotations

import streamlit as st

from ui.dashboard.log_query import TIER_LABEL, fetch_detail

_F_OK  = 0.8
_AR_OK = 0.8
_CP_OK = 0.8


def _score_badge(val: float, threshold: float, label: str) -> str:
    color = "#2ecc71" if val >= threshold else "#e74c3c"
    return (
        f"<span style='background:{color};color:#fff;"
        f"padding:2px 8px;border-radius:4px;font-size:0.85em'>"
        f"{label}={val:.3f}</span>"
    )


def render_detail(request_id: str) -> None:
    """Render the detail screen for a single request_id."""

    if st.button("← Back to List", type="secondary"):
        st.session_state["log_selected_id"] = None
        st.session_state.pop("log_table", None)
        st.rerun()

    st.subheader("Log Detail")

    detail = fetch_detail(request_id)
    if not detail:
        st.error("Could not load data.")
        return

    meta         = detail["meta"]
    loops_df     = detail["loops"]
    queries      = detail["queries"]
    final_answer = detail["final_answer"]

    st.markdown("#### Request Info")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Request ID",  meta["request_id"][:8] + "...")
    c2.metric("Timestamp",   meta["created_at"])
    c3.metric("User Level",  meta["user_level"])
    c4.metric("LLM",         meta["llm_model"] or "—")

    if meta.get("disease") or meta.get("ablation_condition"):
        c5, c6, c7 = st.columns(3)
        c5.metric("Ablation Condition", meta.get("ablation_condition") or "—")
        c6.metric("Disease", meta.get("disease") or "—")
        query_index = meta.get("query_index")
        qll = meta.get("query_level_label") or ""
        c7.metric(
            "STQS Query",
            f"#{int(query_index)} ({qll})" if query_index is not None else "—",
        )

    st.markdown(
        f"**Original Query** &nbsp; `{meta['original_query']}`",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    st.markdown("#### Loop History")

    show_cols = {
        "tier_label":                "Tier",
        "loop_number":               "Loop",
        "ragas_f":                   "F",
        "ragas_ar":                  "AR",
        "ragas_cp":                  "CP",
        "deepeval_faithfulness":     "DE-F",
        "deepeval_answer_relevancy": "DE-AR",
        "fk_grade":                  "FK",
        "is_escalated":              "Escalated",
        "is_fallback":               "Fallback",
        "retrieved_doc_count":       "Chunks",
        "execution_time_ms":         "Time (ms)",
        "created_at":                "Timestamp",
    }
    loops_view = loops_df[list(show_cols.keys())].rename(columns=show_cols).copy()

    styled = (
        loops_view.style
        .applymap(lambda v: "color:#2ecc71" if isinstance(v, float) and v >= _F_OK  else
                            "color:#e74c3c" if isinstance(v, float) else "",
                  subset=["F"])
        .applymap(lambda v: "color:#2ecc71" if isinstance(v, float) and v >= _AR_OK else
                            "color:#e74c3c" if isinstance(v, float) else "",
                  subset=["AR"])
        .applymap(lambda v: "color:#2ecc71" if isinstance(v, float) and v >= _CP_OK else
                            "color:#e74c3c" if isinstance(v, float) else "",
                  subset=["CP"])
        .format(
            {
                "F": "{:.3f}", "AR": "{:.3f}", "CP": "{:.3f}",
                "DE-F": "{:.3f}", "DE-AR": "{:.3f}", "FK": "{:.1f}",
            },
            na_rep="—",
        )
    )
    st.dataframe(styled, width="stretch", hide_index=True)

    _render_escalation_summary(loops_df)

    st.markdown("---")

    st.markdown("#### Optimized Query History")
    if queries:
        for i, q in enumerate(queries, 1):
            st.markdown(f"**Attempt {i}** &nbsp; `{q}`", unsafe_allow_html=True)
    else:
        st.caption("No query history.")

    st.markdown("---")

    st.markdown("#### Final Answer")
    if final_answer:
        st.text_area(
            label="final_answer",
            value=final_answer,
            height=300,
            disabled=True,
            label_visibility="collapsed",
        )
    else:
        st.caption("Final answer not yet saved.")


def _render_escalation_summary(loops_df) -> None:
    """Automatically analyze escalation causes from loop history and display them."""
    msgs: list[str] = []
    for _, row in loops_df.iterrows():
        if not row.get("is_escalated"):
            continue
        tier  = row["final_tier"]
        ar    = row.get("ragas_ar", 0.0) or 0.0
        f_val = row.get("ragas_f",  0.0) or 0.0
        cp    = row.get("ragas_cp", 0.0) or 0.0
        loop  = row["loop_number"]

        if tier == 0 and ar < 0.3:
            msgs.append(f"Tier 0 · Loop {loop} — AR={ar:.3f} < 0.3 → immediate escalation")
        elif tier == 0:
            msgs.append(f"Tier 0 · Loop {loop} — max loops exhausted (F={f_val:.3f}) → move to Tier 1")
        elif tier == 1:
            msgs.append(f"Tier 1 → F={f_val:.3f} < threshold → move to Tier 2")
        elif tier == 2:
            msgs.append(f"Tier 2 → F={f_val:.3f} below threshold → Fallback")

    if msgs:
        with st.expander("Escalation Analysis", expanded=True):
            for m in msgs:
                st.markdown(f"- {m}")
