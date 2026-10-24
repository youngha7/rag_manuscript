"""System Performance Visualization Dashboard.

Oracle rag_audit_log_bak-based visualization:
  1. RAGAS Metric Comparison (Proposal System vs Baseline)
  2. Escalation Pattern Analysis (Tier Distribution)
  3. Self-Correction Loop Convergence
  4. FK Grade Validation
  5. DeepEval Metric Comparison (Independent Cross-Validation)
  6. Level Classifier Performance
"""

from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
_COND_ORDER = ["A", "E"]
_COND_LABELS = {
    "A": "Proposal System",
    "E": "Baseline",
}
_COND_SHORT = {
    "A": "Proposal\nSystem",
    "E": "Baseline",
}
_COND_COLORS = {
    "A": "#2ecc71",
    "E": "#95a5a6",
}
_THRESHOLD = 0.80
_AR_THRESHOLD = 0.80

matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["axes.unicode_minus"] = False


# ── Data Load ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def _load_data() -> pd.DataFrame:
    try:
        import oracledb
    except ModuleNotFoundError:
        logger.warning("oracledb not installed — returning empty DataFrame")
        return pd.DataFrame()
    import config.settings as s

    sql = """
        SELECT
            request_id, ablation_condition, user_level, query_level_label,
            query_index, final_tier, loop_number, is_final, self_correction_count,
            ragas_f, ragas_ar, ragas_cp, q_total,
            is_escalated, is_fallback,
            execution_time_ms, fk_grade, created_at,
            deepeval_faithfulness, deepeval_answer_relevancy
        FROM rag_audit_log_bak
        WHERE ablation_condition IS NOT NULL
        ORDER BY created_at, loop_number
    """
    conn = oracledb.connect(
        user=s.ORACLE_USER, password=s.ORACLE_PASSWORD, dsn=s.ORACLE_DSN
    )
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            cols = [d[0].lower() for d in cur.description]
            rows = cur.fetchall()
    finally:
        conn.close()
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows, columns=cols)

    for col in [
        "ragas_f", "ragas_ar", "ragas_cp", "q_total", "fk_grade", "query_index",
        "deepeval_faithfulness", "deepeval_answer_relevancy",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["query_index"] = df["query_index"].astype("Int64")

    valid_q = df["ragas_f"].notna() & df["ragas_ar"].notna() & df["ragas_cp"].notna()
    df.loc[valid_q, "q_total"] = (
        0.4 * df.loc[valid_q, "ragas_f"]
        + 0.4 * df.loc[valid_q, "ragas_ar"]
        + 0.2 * df.loc[valid_q, "ragas_cp"]
    ).round(6)
    # Tier 1: no F/CP -> Q_total = AR (also corrects legacy NULL rows)
    tier1 = df["final_tier"] == 1
    df.loc[tier1, "q_total"] = df.loc[tier1, "ragas_ar"].round(6)

    df["loop_number"] = df["loop_number"].fillna(1).astype(int)
    df["self_correction_count"] = df["self_correction_count"].fillna(0).astype(int)
    df["final_tier"] = df["final_tier"].fillna(0).astype(int)
    # Oracle stores BOOLEAN as NUMBER(1)
    df["is_final"] = df["is_final"].fillna(0).astype(int).astype(bool)
    df["is_escalated"] = df["is_escalated"].fillna(0).astype(int).astype(bool)
    df["is_fallback"] = df["is_fallback"].fillna(0).astype(int).astype(bool)
    return df


# ── Common Helpers ────────────────────────────────────────────────────────────
def _fig(w=10, h=5.5):
    sns.set_theme(style="whitegrid", font_scale=1.05)
    return plt.subplots(figsize=(w, h), dpi=150)


def _savebuf(fig):
    import io

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf


def _final(df):
    return df[df["is_final"]].copy()


def _present_conds(df):
    fin = _final(df)
    return [c for c in _COND_ORDER if c in fin["ablation_condition"].values]


def _tier0_qidx(fin: pd.DataFrame) -> set:
    """query_index set (int) where condition A reached Tier 0."""
    a_tier0 = fin[(fin["ablation_condition"] == "A") & (fin["final_tier"] == 0)]
    return set(a_tier0["query_index"].dropna().astype(int).tolist())


# ══════════════════════════════════════════════════════════════════════════════
# 1. RAGAS Metric Comparison
# ══════════════════════════════════════════════════════════════════════════════
def _cond_subset(fin: pd.DataFrame, c: str) -> pd.DataFrame:
    """Return final rows for condition c.
    Condition A: Tier 0 successes only.
    Condition E: counterpart rows matching A's Tier 0 query_index (fair comparison).
    """
    sub = fin[fin["ablation_condition"] == c]
    if c == "A":
        sub = sub[sub["final_tier"] == 0]
    else:
        tier0_idx = _tier0_qidx(fin)
        if tier0_idx:
            sub = sub[sub["query_index"].dropna().astype(int).isin(tier0_idx)]
    return sub


def _plot_ragas_comparison(df):
    fin = _final(df)
    conds = _present_conds(df)
    metrics = [
        ("ragas_f", "Faithfulness (F)", "#2980b9"),
        ("ragas_ar", "Answer Relevance (AR)", "#e67e22"),
        ("ragas_cp", "Context Precision (CP)", "#27ae60"),
        ("q_total", "Q_total", "#8e44ad"),
    ]
    x = np.arange(len(conds))
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(metrics))

    fig, ax = _fig(11, 6)
    for i, (col, lbl, color) in enumerate(metrics):
        means, errs = [], []
        for c in conds:
            vals = _cond_subset(fin, c)[col].dropna().values
            means.append(vals.mean() if len(vals) else 0)
            errs.append(vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0)
        bars = ax.bar(
            x + offsets[i],
            means,
            width,
            label=lbl,
            color=color,
            alpha=0.85,
            edgecolor="white",
            linewidth=1.2,
            zorder=3,
        )
        ax.errorbar(
            x + offsets[i],
            means,
            yerr=errs,
            fmt="none",
            color="#2c3e50",
            capsize=4,
            linewidth=1.3,
            zorder=4,
        )
        for bar, mean in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f"{mean:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    ax.axhline(
        _THRESHOLD,
        color="#e74c3c",
        linestyle="--",
        linewidth=1.5,
        alpha=0.75,
        label=f"Threshold ({_THRESHOLD})",
        zorder=2,
    )
    ax.set_title(
        "Table 1. RAGAS Metric Comparison — Proposal System (Tier 0, n=A) vs Baseline (matched query_index, Mean ± SEM)",
        fontsize=12,
        fontweight="bold",
        pad=14,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([_COND_SHORT[c] for c in conds], fontsize=10)
    ax.set_ylabel("Mean Score", fontsize=11)
    ax.set_ylim(0.45, 1.15)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, axis="y", alpha=0.4)
    fig.tight_layout()
    return _savebuf(fig)


def _table_ragas(df):
    fin = _final(df)
    rows = []
    for c in _COND_ORDER:
        sub = _cond_subset(fin, c)

        def ms(col, _sub=sub):
            v = _sub[col].dropna()
            return f"{v.mean():.3f} ± {v.std(ddof=1) / np.sqrt(len(v)):.3f}" if len(v) > 1 else "—"

        n_tier0 = len(_cond_subset(fin, "A"))
        note = f" (Tier 0 only, n={n_tier0})" if c == "A" else f" (matched query_index, n={len(sub)})"
        rows.append(
            {
                "Condition": _COND_LABELS.get(c, c) + note,
                "F (Faithfulness)": ms("ragas_f"),
                "AR (Answer Relevance)": ms("ragas_ar"),
                "CP (Context Precision)": ms("ragas_cp"),
                "Q_total": ms("q_total"),
                "n": len(sub),
            }
        )
    return pd.DataFrame(rows)


def _table_ttest(df) -> pd.DataFrame:
    """Paired t-test: A (Tier 0) vs E (matched query_index counterpart)."""
    try:
        from scipy import stats as _stats
    except ImportError:
        return pd.DataFrame()

    fin = _final(df)
    tier0_idx = _tier0_qidx(fin)
    if not tier0_idx:
        return pd.DataFrame()

    a_sub = fin[(fin["ablation_condition"] == "A") & (fin["final_tier"] == 0)].copy()
    e_sub = fin[
        (fin["ablation_condition"] == "E")
        & fin["query_index"].dropna().astype(int).isin(tier0_idx)
    ].copy()

    merged = a_sub.merge(
        e_sub[["query_index", "ragas_f", "ragas_ar", "ragas_cp", "q_total"]],
        on="query_index",
        suffixes=("_a", "_e"),
        how="inner",
    )

    metric_pairs = [
        ("ragas_f_a", "ragas_f_e", "Faithfulness (F)"),
        ("ragas_ar_a", "ragas_ar_e", "Answer Relevance (AR)"),
        ("ragas_cp_a", "ragas_cp_e", "Context Precision (CP)"),
        ("q_total_a", "q_total_e", "Q-total"),
    ]

    rows = []
    for col_a, col_e, label in metric_pairs:
        pairs = merged[[col_a, col_e]].dropna()
        n = len(pairs)
        if n < 2:
            rows.append(
                {"Metric": label, "N": n, "A mean": "—", "E mean": "—",
                 "Δ (A−E)": "—", "t": "—", "p-value": "—", "sig": "—"}
            )
            continue
        a_vals = pairs[col_a].values
        e_vals = pairs[col_e].values
        t_stat, p_val = _stats.ttest_rel(a_vals, e_vals)
        sig = (
            "***" if p_val < 0.001 else
            "**"  if p_val < 0.01  else
            "*"   if p_val < 0.05  else
            "†"   if p_val < 0.10  else "ns"
        )
        rows.append(
            {
                "Metric": label,
                "N": n,
                "A mean": f"{a_vals.mean():.4f}",
                "E mean": f"{e_vals.mean():.4f}",
                "Δ (A−E)": f"{a_vals.mean() - e_vals.mean():+.4f}",
                "t": f"{t_stat:.3f}",
                "p-value": f"{p_val:.4f}",
                "sig": sig,
            }
        )
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Escalation Pattern (Tier Distribution — Proposal System)
# ══════════════════════════════════════════════════════════════════════════════
def _plot_tier_distribution(df):
    fin = _final(df)
    sub = fin[fin["ablation_condition"] == "A"]
    if sub.empty:
        return None

    tier_counts = sub["final_tier"].value_counts().sort_index()
    tier_labels = {0: "Tier 0\n(VectorDB)", 1: "Tier 1\n(LLM)", 2: "Tier 2\n(Web)"}
    labels = [tier_labels.get(t, f"Tier {t}") for t in tier_counts.index]
    colors = ["#27ae60", "#f39c12", "#e74c3c"][: len(tier_counts)]
    total = tier_counts.sum()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5), dpi=150)

    wedges, texts, autotexts = ax1.pie(
        tier_counts.values,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 10},
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight("bold")
    ax1.set_title(
        "Tier Distribution (Proposal System)", fontsize=12, fontweight="bold", pad=12
    )

    bars = ax2.bar(
        labels,
        tier_counts.values,
        color=colors,
        alpha=0.85,
        edgecolor="white",
        linewidth=1.2,
    )
    for bar, cnt in zip(bars, tier_counts.values):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{cnt}\n({cnt/total*100:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax2.set_title("Query Count by Tier", fontsize=12, fontweight="bold", pad=12)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.set_ylim(0, tier_counts.max() * 1.35)
    ax2.grid(True, axis="y", alpha=0.4)

    fig.suptitle(
        "Table 2. Tier Distribution — Escalation Pattern Analysis (Proposal System)",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    return _savebuf(fig)


def _table_tier_distribution(df):
    fin = _final(df)
    sub = fin[fin["ablation_condition"] == "A"]
    if sub.empty:
        return pd.DataFrame()

    tier_names = {
        0: "Tier 0 (MSD Manual)",
        1: "Tier 1 (LLM Knowledge)",
        2: "Tier 2 (Web Search)",
    }
    pro = sub[sub["user_level"] == "Professional"]
    cons = sub[sub["user_level"] == "Consumer"]
    total = sub

    n_pro = max(len(pro), 1)
    n_cons = max(len(cons), 1)
    n_tot = max(len(total), 1)

    rows = []
    for tier in sorted(sub["final_tier"].unique()):
        cp = len(pro[pro["final_tier"] == tier])
        cc = len(cons[cons["final_tier"] == tier])
        ct = len(total[total["final_tier"] == tier])
        rows.append(
            {
                "Tier": tier_names.get(tier, f"Tier {tier}"),
                "Professional": f"{cp} ({cp/n_pro*100:.0f}%)",
                "Consumer": f"{cc} ({cc/n_cons*100:.0f}%)",
                "Total": f"{ct} ({ct/n_tot*100:.0f}%)",
            }
        )
    return pd.DataFrame(rows)


def _table_tier_performance(df) -> pd.DataFrame:
    """조건 A의 티어별 최종 RAGAS 성능 요약 (비교 불가 주석 포함용 데이터)."""
    fin = _final(df)
    sub = fin[fin["ablation_condition"] == "A"]
    if sub.empty:
        return pd.DataFrame()

    def ms(series):
        v = series.dropna()
        if len(v) == 0:
            return "—"
        if len(v) == 1:
            return f"{v.iloc[0]:.3f}"
        return f"{v.mean():.3f} ± {v.std(ddof=1):.3f}"

    tier_meta = {
        0: "Tier 0 — MSD Manual (VectorDB)",
        1: "Tier 1 — LLM Knowledge",
        2: "Tier 2 — Web Search",
    }
    rows = []
    for tier in sorted(sub["final_tier"].unique()):
        t_sub = sub[sub["final_tier"] == tier]
        n = len(t_sub)
        is_tier1 = (tier == 1)
        avg_sc = (
            f"{t_sub['self_correction_count'].mean():.1f}"
            if tier == 0 else "—"
        )
        rows.append({
            "Tier": tier_meta.get(tier, f"Tier {tier}"),
            "n": n,
            "F (Faithfulness)":      "N/A *" if is_tier1 else ms(t_sub["ragas_f"]),
            "AR (Answer Relevance)": ms(t_sub["ragas_ar"]),
            "CP (Context Precision)": "N/A *" if is_tier1 else ms(t_sub["ragas_cp"]),
            "Q_total":               ms(t_sub["q_total"]),
            "Avg Self-Correction":   avg_sc,
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# 3. Self-Correction Loop Convergence
# ══════════════════════════════════════════════════════════════════════════════
def _plot_loop_convergence(df):
    sub = df[
        (df["ablation_condition"] == "A")
        & (df["final_tier"] == 0)
        & (df["is_final"] == True)  # final rows only — questions converged at loop N
    ].copy()
    if sub.empty:
        return None

    grp = (
        sub[sub["q_total"].notna()]
        .groupby("loop_number")["q_total"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    if grp.empty:
        return None
    grp["ci"] = 1.96 * grp["std"] / np.sqrt(grp["count"])

    fig, ax = _fig(8, 5.5)
    ax.plot(
        grp["loop_number"],
        grp["mean"],
        marker="o",
        color="#2980b9",
        linewidth=2.5,
        markersize=9,
        zorder=3,
        label="Mean Q_total",
    )
    ax.fill_between(
        grp["loop_number"],
        grp["mean"] - grp["ci"],
        grp["mean"] + grp["ci"],
        alpha=0.18,
        color="#2980b9",
        label="95% CI",
    )
    for _, row in grp.iterrows():
        ax.annotate(
            f"{row['mean']:.3f}",
            (row["loop_number"], row["mean"]),
            textcoords="offset points",
            xytext=(0, 12),
            ha="center",
            fontsize=10,
            fontweight="bold",
            color="#2c3e50",
        )

    for _, row in grp.iterrows():
        loop_sub = sub[sub["loop_number"] == row["loop_number"]]["q_total"].dropna()
        if len(loop_sub):
            conv = (loop_sub >= _THRESHOLD).mean() * 100
            ax.annotate(
                f"Conv. {conv:.0f}%",
                (row["loop_number"], row["mean"] - grp["ci"].max() - 0.04),
                ha="center",
                fontsize=8.5,
                color="#7f8c8d",
            )

    ax.axhline(
        _THRESHOLD,
        color="#e74c3c",
        linestyle="--",
        linewidth=1.5,
        alpha=0.8,
        label=f"τ_Q = {_THRESHOLD}",
    )
    ax.set_title(
        "Table 3. Self-Correction Loop Convergence — Tier0 Q_total (Proposal System)",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel("Loop Number (Tier0 Evaluations)", fontsize=11)
    ax.set_ylabel("Mean Q_total Score", fontsize=11)
    ax.set_xticks(grp["loop_number"].tolist())
    ax.set_ylim(0.45, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    return _savebuf(fig)


def _table_loop(df):
    sub = df[
        (df["ablation_condition"] == "A")
        & (df["final_tier"] == 0)
        & (df["is_final"] == True)  # final rows only — questions converged at loop N
    ].copy()
    rows = []
    for lc in sorted(sub["loop_number"].unique()):
        vals = sub[sub["loop_number"] == lc]["q_total"].dropna().values
        if len(vals) == 0:
            continue
        conv = (vals >= _THRESHOLD).mean() * 100
        rows.append(
            {
                "Loop": int(lc),
                "Mean Q_total": round(float(vals.mean()), 3),
                "Std Dev": round(float(vals.std(ddof=1)) if len(vals) > 1 else 0, 3),
                "Conv. Rate (τ≥0.8)": f"{conv:.1f}%",
                "n": len(vals),
            }
        )
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# 4. FK Grade — Level Classifier Indirect Validation
# ══════════════════════════════════════════════════════════════════════════════
_FK_CONSUMER_MAX = 9.0
_FK_PROFESSIONAL_MIN = 12.0


def _plot_fk_grade(df):
    fin = _final(df)
    sub = fin[
        fin["fk_grade"].notna() & fin["user_level"].isin(["Professional", "Consumer"])
    ]
    if sub.empty:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=150)
    sns.set_theme(style="whitegrid", font_scale=1.05)

    ax1 = axes[0]
    palette = {"Consumer": "#3498db", "Professional": "#e74c3c"}
    sns.boxplot(
        data=sub,
        x="user_level",
        y="fk_grade",
        palette=palette,
        width=0.45,
        linewidth=1.5,
        order=["Consumer", "Professional"],
        ax=ax1,
    )
    ax1.axhline(
        _FK_CONSUMER_MAX,
        color="#3498db",
        linestyle="--",
        linewidth=1.4,
        alpha=0.8,
        label=f"Consumer target (≤{_FK_CONSUMER_MAX})",
    )
    ax1.axhline(
        _FK_PROFESSIONAL_MIN,
        color="#e74c3c",
        linestyle="--",
        linewidth=1.4,
        alpha=0.8,
        label=f"Professional target (≥{_FK_PROFESSIONAL_MIN})",
    )
    for level, color in palette.items():
        mean_val = sub[sub["user_level"] == level]["fk_grade"].mean()
        if not pd.isna(mean_val):
            ax1.text(
                0 if level == "Consumer" else 1,
                mean_val + 0.3,
                f"mean={mean_val:.2f}",
                ha="center",
                fontsize=10,
                fontweight="bold",
                color=color,
            )
    ax1.set_title("FK Grade by User Level", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xlabel("User Level", fontsize=11)
    ax1.set_ylabel("FK Grade", fontsize=11)
    ax1.legend(fontsize=9, loc="upper left")

    ax2 = axes[1]
    conds = _present_conds(df)
    x = np.arange(len(conds))
    width = 0.32
    for i, (level, color) in enumerate(palette.items()):
        means = []
        for c in conds:
            vals = (
                fin[(fin["ablation_condition"] == c) & (fin["user_level"] == level)][
                    "fk_grade"
                ]
                .dropna()
                .values
            )
            means.append(vals.mean() if len(vals) else 0)
        offset = -width / 2 if i == 0 else width / 2
        bars = ax2.bar(
            x + offset,
            means,
            width,
            label=level,
            color=color,
            alpha=0.8,
            edgecolor="white",
            linewidth=1.2,
        )
        for bar, val in zip(bars, means):
            if val > 0:
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.15,
                    f"{val:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

    ax2.axhline(
        _FK_CONSUMER_MAX, color="#3498db", linestyle="--", linewidth=1.3, alpha=0.7
    )
    ax2.axhline(
        _FK_PROFESSIONAL_MIN, color="#e74c3c", linestyle="--", linewidth=1.3, alpha=0.7
    )
    ax2.set_title(
        "Mean FK Grade by Condition & Level", fontsize=12, fontweight="bold", pad=10
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels([_COND_SHORT[c] for c in conds], fontsize=10)
    ax2.set_ylabel("Mean FK Grade", fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, axis="y", alpha=0.4)

    fig.suptitle(
        "Table 4. FK Grade — Level Classifier Indirect Validation (is_final=1 only)",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    return _savebuf(fig)


def _table_fk_grade(df):
    fin = _final(df)
    rows = []
    for c in _COND_ORDER:
        sub = fin[fin["ablation_condition"] == c]
        for level in ["Consumer", "Professional"]:
            vals = sub[sub["user_level"] == level]["fk_grade"].dropna()
            target = (
                f"≤ {_FK_CONSUMER_MAX}"
                if level == "Consumer"
                else f"≥ {_FK_PROFESSIONAL_MIN}"
            )
            if len(vals) == 0:
                rows.append(
                    {
                        "Condition": _COND_LABELS.get(c, c),
                        "User Level": level,
                        "Mean FK Grade": "—",
                        "Std Dev": "—",
                        "Target": target,
                        "Within Target": "—",
                        "n": 0,
                    }
                )
                continue
            mean_v = vals.mean()
            within = (
                (vals <= _FK_CONSUMER_MAX).mean() * 100
                if level == "Consumer"
                else (vals >= _FK_PROFESSIONAL_MIN).mean() * 100
            )
            rows.append(
                {
                    "Condition": _COND_LABELS.get(c, c),
                    "User Level": level,
                    "Mean FK Grade": f"{mean_v:.2f}",
                    "Std Dev": f"{vals.std(ddof=1):.2f}" if len(vals) > 1 else "—",
                    "Target": target,
                    "Within Target": f"{within:.1f}%",
                    "n": len(vals),
                }
            )
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# 5. DeepEval Metric Comparison — Proposal System vs Baseline (independent judge)
# ══════════════════════════════════════════════════════════════════════════════
# RAGAS (F/AR/CP) is the metric the Self-Correction Loop directly optimizes, so it
# can be self-referential. DeepEval is scored by an independent framework (judge
# LLM: gpt-4o-mini), so comparing the two conditions on DeepEval alone, without
# mixing in RAGAS, gives an "independent" validation.
_DEEPEVAL_METRICS = [
    ("deepeval_faithfulness", "Faithfulness (F)", "#2980b9"),
    ("deepeval_answer_relevancy", "Answer Relevancy (AR)", "#e67e22"),
]


def _plot_deepeval_comparison(df):
    fin = _final(df)
    conds = _present_conds(df)
    if not conds:
        return None

    x = np.arange(len(conds))
    width = 0.22
    offsets = np.linspace(-width, width, len(_DEEPEVAL_METRICS))

    fig, ax = _fig(10, 6)
    any_data = False
    for i, (col, lbl, color) in enumerate(_DEEPEVAL_METRICS):
        means, errs = [], []
        for c in conds:
            vals = _cond_subset(fin, c)[col].dropna().values
            if len(vals):
                any_data = True
            means.append(vals.mean() if len(vals) else 0)
            errs.append(vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0)
        bars = ax.bar(
            x + offsets[i], means, width, label=lbl, color=color,
            alpha=0.85, edgecolor="white", linewidth=1.2, zorder=3,
        )
        ax.errorbar(
            x + offsets[i], means, yerr=errs, fmt="none", color="#2c3e50",
            capsize=4, linewidth=1.3, zorder=4,
        )
        for bar, mean in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                f"{mean:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold",
            )

    if not any_data:
        plt.close(fig)
        return None

    ax.axhline(
        _THRESHOLD, color="#e74c3c", linestyle="--", linewidth=1.5, alpha=0.75,
        label=f"Reference ({_THRESHOLD})", zorder=2,
    )
    ax.set_title(
        "Table 5. DeepEval Metric Comparison — Proposal System (Tier 0, n=A) vs Baseline "
        "(matched query_index, Mean ± SEM)",
        fontsize=12, fontweight="bold", pad=14,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([_COND_SHORT[c] for c in conds], fontsize=10)
    ax.set_ylabel("Mean DeepEval Score", fontsize=11)
    ax.set_ylim(0.45, 1.15)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, axis="y", alpha=0.4)
    fig.tight_layout()
    return _savebuf(fig)


def _table_deepeval_comparison(df):
    fin = _final(df)
    rows = []
    for c in _COND_ORDER:
        sub = _cond_subset(fin, c)

        def ms(col, _sub=sub):
            v = _sub[col].dropna()
            return f"{v.mean():.3f} ± {v.std(ddof=1) / np.sqrt(len(v)):.3f}" if len(v) > 1 else "—"

        n_tier0 = len(_cond_subset(fin, "A"))
        note = f" (Tier 0 only, n={n_tier0})" if c == "A" else f" (matched query_index, n={len(sub)})"
        rows.append(
            {
                "Condition": _COND_LABELS.get(c, c) + note,
                "Faithfulness (F)": ms("deepeval_faithfulness"),
                "Answer Relevancy (AR)": ms("deepeval_answer_relevancy"),
                "n": len(sub),
            }
        )
    return pd.DataFrame(rows)


def _table_deepeval_ttest(df) -> pd.DataFrame:
    """Paired t-test (DeepEval only): A (Tier 0) vs E (matched query_index counterpart)."""
    try:
        from scipy import stats as _stats
    except ImportError:
        return pd.DataFrame()

    fin = _final(df)
    tier0_idx = _tier0_qidx(fin)
    if not tier0_idx:
        return pd.DataFrame()

    a_sub = fin[(fin["ablation_condition"] == "A") & (fin["final_tier"] == 0)].copy()
    e_sub = fin[
        (fin["ablation_condition"] == "E")
        & fin["query_index"].dropna().astype(int).isin(tier0_idx)
    ].copy()

    deepeval_cols = ["deepeval_faithfulness", "deepeval_answer_relevancy"]
    merged = a_sub.merge(
        e_sub[["query_index"] + deepeval_cols],
        on="query_index",
        suffixes=("_a", "_e"),
        how="inner",
    )

    metric_pairs = [
        ("deepeval_faithfulness_a", "deepeval_faithfulness_e", "Faithfulness (F)"),
        ("deepeval_answer_relevancy_a", "deepeval_answer_relevancy_e", "Answer Relevancy (AR)"),
    ]

    rows = []
    for col_a, col_e, label in metric_pairs:
        pairs = merged[[col_a, col_e]].dropna()
        n = len(pairs)
        if n < 2:
            rows.append(
                {"Metric": label, "N": n, "A mean": "—", "E mean": "—",
                 "Δ (A−E)": "—", "t": "—", "p-value": "—", "sig": "—"}
            )
            continue
        a_vals = pairs[col_a].values
        e_vals = pairs[col_e].values
        t_stat, p_val = _stats.ttest_rel(a_vals, e_vals)
        sig = (
            "***" if p_val < 0.001 else
            "**"  if p_val < 0.01  else
            "*"   if p_val < 0.05  else
            "†"   if p_val < 0.10  else "ns"
        )
        rows.append(
            {
                "Metric": label,
                "N": n,
                "A mean": f"{a_vals.mean():.4f}",
                "E mean": f"{e_vals.mean():.4f}",
                "Δ (A−E)": f"{a_vals.mean() - e_vals.mean():+.4f}",
                "t": f"{t_stat:.3f}",
                "p-value": f"{p_val:.4f}",
                "sig": sig,
            }
        )
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# 6. Level Classifier Performance
# ══════════════════════════════════════════════════════════════════════════════
def _compute_classifier_metrics(df):
    sub = (
        df[
            df["is_final"]
            & df["ablation_condition"].isin(["A"])
            & df["query_level_label"].notna()
            & df["user_level"].notna()
        ]
        .sort_values("created_at")
        .drop_duplicates(subset=["query_index"], keep="last")
        .copy()
    )
    if sub.empty:
        return None

    label_map = {"P": "Professional", "C": "Consumer"}
    sub["expected"] = sub["query_level_label"].map(label_map)
    sub["predicted"] = sub["user_level"]
    valid = sub[
        sub["expected"].isin(["Professional", "Consumer"])
        & sub["predicted"].isin(["Professional", "Consumer"])
    ]
    if len(valid) == 0:
        return None

    y_true = (valid["expected"] == "Professional").astype(int)
    y_pred = (valid["predicted"] == "Professional").astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "n": len(valid),
    }


def _plot_classifier(m):
    keys = ["Accuracy", "Precision", "Recall", "F1"]
    values = [m[k] for k in keys]
    colors = ["#2980b9", "#27ae60", "#e67e22", "#9b59b6"]

    fig, ax = _fig(8, 5)
    bars = ax.bar(
        keys,
        values,
        color=colors,
        alpha=0.85,
        edgecolor="white",
        linewidth=1.2,
        width=0.5,
        zorder=3,
    )
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.008,
            f"{val*100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
        )

    ax.axhline(
        0.9,
        color="#e74c3c",
        linestyle="--",
        linewidth=1.5,
        alpha=0.75,
        label="90% Threshold",
        zorder=2,
    )
    ax.set_title(
        "Table 6. Level Classifier Performance (Proposal System)",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0.0, 1.15)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.4)
    fig.tight_layout()
    return _savebuf(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Summary Cards
# ══════════════════════════════════════════════════════════════════════════════
def _render_summary_cards(df):
    fin = _final(df)
    cols = st.columns(2)
    names = ["Proposal System (Tier 0 only)", "Baseline (matched subset)"]
    for col, c, name in zip(cols, _COND_ORDER, names):
        sub = _cond_subset(fin, c)
        avg_q = sub["q_total"].mean()
        col.metric(
            label=name,
            value=f"Q_total = {avg_q:.3f}" if not pd.isna(avg_q) else "—",
            delta=f"n = {len(sub)}",
        )


# ══════════════════════════════════════════════════════════════════════════════
# Main Render
# ══════════════════════════════════════════════════════════════════════════════
def render_performance_viz() -> None:
    matplotlib.rcParams["font.family"] = "DejaVu Sans"
    matplotlib.rcParams["axes.unicode_minus"] = False

    st.title("Performance Visualization")
    st.caption("System performance visualization — Oracle `rag_audit_log_bak` (Proposal System vs Baseline)")

    try:
        with st.spinner("Loading data..."):
            df = _load_data()
    except Exception as e:
        logger.error("Data load failed: %s", e)
        st.error(f"Failed to load data. Please check the Oracle connection.\n\n`{e}`")
        if st.button("Retry", type="secondary"):
            st.cache_data.clear()
            st.rerun()
        return

    if df.empty:
        st.warning("Query succeeded but returned no rows.")
        if st.button("Refresh", type="secondary"):
            st.cache_data.clear()
            st.rerun()
        return

    if st.button("Refresh", type="secondary"):
        st.cache_data.clear()
        st.rerun()

    st.markdown("### Summary by Condition")
    _render_summary_cards(df)
    st.markdown("---")

    st.markdown("### Table 1. RAGAS Metric Comparison")
    st.caption(
        "**Fair subset comparison** — Proposal System: Tier 0 questions only (self-correction applied)  |  "
        "Baseline: same query_index counterparts only (matched set)  |  Mean ± SEM"
    )
    buf = _plot_ragas_comparison(df)
    if buf:
        st.image(buf, use_container_width=True)
    st.dataframe(_table_ragas(df), hide_index=True, use_container_width=True)

    st.markdown("#### Paired t-test (A Tier 0 vs E matched counterpart)")
    st.caption("Two-tailed paired t-test per metric  |  \\*\\*\\* p<.001  \\*\\* p<.01  \\* p<.05  † p<.10  ns p≥.10")
    ttest_tbl = _table_ttest(df)
    if not ttest_tbl.empty:
        st.dataframe(
            ttest_tbl.style.applymap(
                lambda v: "color: #e74c3c; font-weight: bold"
                if v in ("***", "**", "*")
                else ("color: #e67e22" if v == "†" else ""),
                subset=["sig"],
            ),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("scipy not available or insufficient data for t-test.")
    st.markdown("---")

    st.markdown("### Table 2. Escalation Pattern Analysis (Tier Distribution — Proposal System)")
    st.caption("Tier 0 → 1 → 2 distribution in the Proposal System condition")
    buf = _plot_tier_distribution(df)
    if buf:
        st.image(buf, use_container_width=True)
    else:
        st.info("No data for Condition A.")
    tier_tbl = _table_tier_distribution(df)
    if not tier_tbl.empty:
        st.dataframe(tier_tbl, hide_index=True, use_container_width=True)

    st.markdown("#### Condition A — RAGAS Performance by Tier")
    st.info(
        "**Caution — Tier 1 and Tier 2 results cannot be directly compared with Condition E (Baseline).**\n\n"
        "Condition E always operates on Tier 0 (VectorDB) only. "
        "Questions that reach Tier 1 or Tier 2 are either outside MSD Manual coverage or cases where Tier 0 retrieval repeatedly failed, "
        "making a fair comparison with Condition E results for the same question impossible.\n\n"
        "The figures below represent **the degree of quality recovery after escalation** and should be interpreted "
        "solely as a reference for internal system behavior — not as a performance comparison against Condition E.\n\n"
        "\\* Tier 1 relies exclusively on LLM training data without any retrieved context, so F (Faithfulness) and CP (Context Precision) "
        "are NULL by design and excluded from evaluation. Q_total is computed from AR alone."
    )
    try:
        tier_perf_tbl = _table_tier_performance(df)
    except Exception as _e:
        tier_perf_tbl = pd.DataFrame()
        st.exception(_e)
    if not tier_perf_tbl.empty:
        st.dataframe(tier_perf_tbl, hide_index=True, use_container_width=True)
    else:
        st.caption("No data — run Condition A batch experiment first, then click Refresh.")
    st.markdown("---")

    st.markdown("### Table 3. Self-Correction Loop Convergence (Proposal System)")
    st.caption("Mean Q_total and convergence rate per loop iteration (τ_Q = 0.8)")
    buf = _plot_loop_convergence(df)
    if buf:
        st.image(buf, use_container_width=True)
    loop_tbl = _table_loop(df)
    if not loop_tbl.empty:
        st.dataframe(loop_tbl, hide_index=True, use_container_width=True)
    else:
        st.info("No data for Condition A.")
    st.markdown("---")

    st.markdown("### Table 4. FK Grade — Level Classifier Indirect Validation")
    st.caption(
        "FK Grade computed on the English RAG answer  |  "
        "Consumer target ≤ 9  |  Professional target ≥ 12  |  is_final=1 only"
    )
    buf = _plot_fk_grade(df)
    if buf:
        st.image(buf, use_container_width=True)
    else:
        st.info("No fk_grade data available.")
    fk_tbl = _table_fk_grade(df)
    if not fk_tbl.empty:
        st.dataframe(fk_tbl, hide_index=True, use_container_width=True)
    st.markdown("---")

    st.markdown("### Table 5. DeepEval Metric Comparison (Independent Cross-Validation)")
    st.caption(
        "**Fair subset comparison** — Proposal System: Tier 0 questions only (self-correction applied)  |  "
        "Baseline: same query_index counterparts only (matched set)  |  Mean ± SEM  |  "
        "RAGAS (F/AR/CP) is the metric the Self-Correction Loop directly optimizes, so it can be self-referential. "
        "Here the two conditions are compared using scores from an independent framework (DeepEval, judged by gpt-4o-mini) instead, without mixing in RAGAS."
    )
    buf = _plot_deepeval_comparison(df)
    if buf:
        st.image(buf, use_container_width=True)
    else:
        st.info("No DeepEval data available. (disease 라벨이 있는 STQS/ablation 요청 필요)")
    st.dataframe(_table_deepeval_comparison(df), hide_index=True, use_container_width=True)

    st.markdown("#### Paired t-test (A Tier 0 vs E matched counterpart) — DeepEval only")
    st.caption("Two-tailed paired t-test per metric  |  \\*\\*\\* p<.001  \\*\\* p<.01  \\* p<.05  † p<.10  ns p≥.10")
    deepeval_ttest_tbl = _table_deepeval_ttest(df)
    if not deepeval_ttest_tbl.empty:
        st.dataframe(
            deepeval_ttest_tbl.style.applymap(
                lambda v: "color: #e74c3c; font-weight: bold"
                if v in ("***", "**", "*")
                else ("color: #e67e22" if v == "†" else ""),
                subset=["sig"],
            ),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("scipy not available or insufficient data for t-test.")
    st.markdown("---")

    st.markdown("### Table 6. Level Classifier Performance (Proposal System)")
    st.caption(
        "query_level_label (P/C) vs user_level — Accuracy / Precision / Recall / F1"
    )
    m = _compute_classifier_metrics(df)
    if m:
        buf = _plot_classifier(m)
        if buf:
            st.image(buf, use_container_width=True)
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Accuracy": f"{m['Accuracy']*100:.1f}%",
                        "Precision": f"{m['Precision']*100:.1f}%",
                        "Recall": f"{m['Recall']*100:.1f}%",
                        "F1": f"{m['F1']*100:.1f}%",
                        "TP": m["TP"],
                        "TN": m["TN"],
                        "FP": m["FP"],
                        "FN": m["FN"],
                        "n": m["n"],
                    }
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("No classifier evaluation data. (query_level_label column required)")
    st.markdown("---")

