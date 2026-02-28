"""
SmartTender AI — Streamlit Dashboard
======================================
Professional tender intelligence platform for Inetum Tunisie.

Connects to the FastAPI backend at http://localhost:8000

Author: SmartTender AI Team
"""

import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# ================================================================
# CONFIG
# ================================================================

API_BASE = "http://localhost:8000"
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"

# Refined color palette
COLORS = {
    "primary": "#4F46E5",
    "primary_light": "#818CF8",
    "primary_dark": "#3730A3",
    "success": "#059669",
    "success_light": "#D1FAE5",
    "success_dark": "#065F46",
    "warning": "#D97706",
    "warning_light": "#FEF3C7",
    "warning_dark": "#92400E",
    "danger": "#DC2626",
    "danger_light": "#FEE2E2",
    "danger_dark": "#991B1B",
    "neutral_50": "#F9FAFB",
    "neutral_100": "#F3F4F6",
    "neutral_200": "#E5E7EB",
    "neutral_300": "#D1D5DB",
    "neutral_500": "#4B5563",
    "neutral_700": "#1F2937",
    "neutral_900": "#111827",
    "surface": "#FFFFFF",
}

DECISION_COLORS = {
    "RELEVANT": COLORS["success"],
    "LOW_RELEVANCE": COLORS["warning"],
    "IRRELEVANT": COLORS["danger"],
}
DECISION_BG = {
    "RELEVANT": COLORS["success_light"],
    "LOW_RELEVANCE": COLORS["warning_light"],
    "IRRELEVANT": COLORS["danger_light"],
}
DECISION_TEXT = {
    "RELEVANT": COLORS["success_dark"],
    "LOW_RELEVANCE": COLORS["warning_dark"],
    "IRRELEVANT": COLORS["danger_dark"],
}

RISK_COLORS = {"LOW": COLORS["success"], "MEDIUM": COLORS["warning"], "HIGH": COLORS["danger"]}
LEVEL_COLORS = {
    "LOW": COLORS["success"], "MODERATE": COLORS["primary"],
    "HIGH": COLORS["warning"], "VERY_HIGH": COLORS["danger"],
}

SOURCE_OPTIONS = ["SAM.GOV", "TED", "UNGM", "TUNEPS", "CONTRACTS_FINDER"]

PLOTLY_TEMPLATE = "plotly_white"
CHART_COLORS = ["#4F46E5", "#059669", "#D97706", "#DC2626", "#7C3AED", "#0891B2"]

# ================================================================
# PAGE CONFIG
# ================================================================

st.set_page_config(
    page_title="SmartTender AI",
    page_icon="ST",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ================================================================
# GLOBAL CSS
# ================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #1F2937;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }

    /* Force all main content text to be dark and readable */
    .main .block-container, .main .block-container * {
        color: #1F2937;
    }
    .main p, .main span, .main div, .main label, .main li {
        color: #1F2937;
    }
    .main .stMarkdown p {
        color: #1F2937;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    .main .stCaption, .main .stCaption p {
        color: #6B7280 !important;
        font-size: 0.85rem;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] { background: transparent; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1E1B4B 0%, #312E81 100%);
    }
    section[data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }
    section[data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(255,255,255,0.08);
        border-radius: 6px;
    }
    section[data-testid="stSidebar"] .stRadio label {
        font-weight: 500;
        padding: 6px 0;
    }
    section[data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.12);
    }

    /* Page headers */
    .page-title {
        font-size: 1.75rem;
        font-weight: 700;
        color: #111827 !important;
        margin-bottom: 2px;
        letter-spacing: -0.02em;
    }
    .page-subtitle {
        font-size: 0.95rem;
        color: #4B5563 !important;
        margin-top: 0;
        margin-bottom: 24px;
        font-weight: 400;
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        font-weight: 700;
        font-size: 1.5rem;
        color: #111827 !important;
    }
    [data-testid="stMetricLabel"] {
        font-weight: 600;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #4B5563 !important;
    }

    /* Section headers */
    .card-header {
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #4B5563 !important;
        margin-bottom: 12px;
    }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }

    hr { border-color: #E5E7EB !important; }

    /* Expanders */
    .streamlit-expanderHeader {
        font-weight: 600;
        font-size: 0.95rem;
        color: #1F2937 !important;
    }
    .streamlit-expanderContent p, .streamlit-expanderContent div {
        color: #1F2937;
    }

    .stDataFrame { border-radius: 8px; overflow: hidden; }

    /* Selectbox, multiselect, inputs */
    .stSelectbox label, .stMultiSelect label, .stTextInput label, .stSlider label {
        color: #1F2937 !important;
        font-weight: 500;
    }

    /* Info, success, warning, error boxes */
    .stAlert p { font-size: 0.9rem; }

    .stButton > button {
        font-weight: 600;
        border-radius: 8px;
        font-size: 0.88rem;
        letter-spacing: 0.01em;
    }

    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 500;
        color: #374151;
    }
</style>
""", unsafe_allow_html=True)


# ================================================================
# HELPERS
# ================================================================

def api_call(method, endpoint, payload=None, timeout=300):
    url = f"{API_BASE}{endpoint}"
    try:
        if method == "GET":
            resp = requests.get(url, timeout=timeout)
        else:
            resp = requests.post(url, json=payload or {}, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        st.error("Cannot connect to API. Please start the backend server.")
        return None
    except requests.HTTPError as e:
        st.error(f"API error: {e.response.status_code}")
        return None
    except Exception as e:
        st.error(f"Error: {e}")
        return None


def load_from_disk(filename):
    filepath = OUTPUT_DIR / filename
    if filepath.exists():
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def decision_badge_html(decision):
    bg = DECISION_BG.get(decision, COLORS["neutral_100"])
    fg = DECISION_TEXT.get(decision, COLORS["neutral_700"])
    label = decision.replace("_", " ")
    return f'<span class="badge" style="background:{bg};color:{fg}">{label}</span>'


# ================================================================
# SIDEBAR
# ================================================================

with st.sidebar:
    st.markdown(
        '<div style="margin:12px 0 4px 0;font-size:1.3rem;font-weight:700;color:#FFFFFF !important;letter-spacing:-0.01em">'
        'SmartTender AI</div>'
        '<div style="font-size:0.82rem;color:#C7D2FE !important;margin-bottom:8px">'
        'Intelligent Tender Detection</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["Dashboard", "Scrape Tenders", "Analysis Results", "Tender Details"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    try:
        requests.get(f"{API_BASE}/health", timeout=3).json()
        st.markdown(
            '<div style="display:flex;align-items:center;gap:6px;font-size:0.82rem;color:#FFFFFF !important">'
            '<div style="width:8px;height:8px;background:#34D399;border-radius:50%"></div>'
            'API Connected</div>',
            unsafe_allow_html=True,
        )
    except Exception:
        st.markdown(
            '<div style="display:flex;align-items:center;gap:6px;font-size:0.82rem;color:#FFFFFF !important">'
            '<div style="width:8px;height:8px;background:#F87171;border-radius:50%"></div>'
            'API Offline</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.72rem;color:#A5B4FC !important">'
        '&copy; 2026 Inetum Tunisie &middot; v1.0</div>',
        unsafe_allow_html=True,
    )


# ================================================================
# PAGE: DASHBOARD
# ================================================================

if page == "Dashboard":
    st.markdown('<p class="page-title">Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Real-time tender intelligence overview</p>', unsafe_allow_html=True)

    data = api_call("GET", "/results/latest")
    results = data.get("results", []) if data else load_from_disk("analysis_results.json") or []

    if not results:
        st.info("No analysis results yet. Navigate to **Scrape Tenders** to start.")
        st.stop()

    total = len(results)
    relevant = sum(1 for r in results if r.get("decision") == "RELEVANT")
    low_rel = sum(1 for r in results if r.get("decision") == "LOW_RELEVANCE")
    irrelevant = sum(1 for r in results if r.get("decision") == "IRRELEVANT")
    avg_score = sum(r.get("relevance_score", 0) for r in results) / total if total else 0
    avg_win = sum(r.get("win_probability", 0) for r in results) / total if total else 0

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Total Tenders", total)
    k2.metric("Relevant", relevant)
    k3.metric("Low Relevance", low_rel)
    k4.metric("Irrelevant", irrelevant)
    k5.metric("Avg Score", f"{avg_score:.1f}%")
    k6.metric("Avg Win Prob", f"{avg_win:.0f}%")

    st.markdown("---")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown('<div class="card-header">Decision Distribution</div>', unsafe_allow_html=True)
        fig = px.pie(
            names=["Relevant", "Low Relevance", "Irrelevant"],
            values=[relevant, low_rel, irrelevant],
            color_discrete_sequence=[COLORS["success"], COLORS["warning"], COLORS["danger"]],
            hole=0.45,
        )
        fig.update_traces(textinfo="percent+label", textfont_size=12)
        fig.update_layout(
            template=PLOTLY_TEMPLATE, height=300, showlegend=False,
            margin=dict(t=10, b=10, l=10, r=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="card-header">Platform Breakdown</div>', unsafe_allow_html=True)
        plats = {}
        for r in results:
            p = r.get("platform", "Unknown")
            plats[p] = plats.get(p, 0) + 1
        fig2 = px.bar(
            x=list(plats.keys()), y=list(plats.values()),
            color=list(plats.keys()),
            labels={"x": "Platform", "y": "Count"},
            color_discrete_sequence=CHART_COLORS,
        )
        fig2.update_layout(
            template=PLOTLY_TEMPLATE, height=300, showlegend=False,
            margin=dict(t=10, b=10, l=10, r=10),
        )
        st.plotly_chart(fig2, use_container_width=True)

    with c3:
        st.markdown('<div class="card-header">Score Distribution</div>', unsafe_allow_html=True)
        fig3 = px.histogram(
            x=[r.get("relevance_score", 0) for r in results], nbins=20,
            labels={"x": "Relevance Score (%)", "y": "Count"},
            color_discrete_sequence=[COLORS["primary"]],
        )
        fig3.add_vline(x=55, line_dash="dash", line_color=COLORS["success"], annotation_text="Relevant")
        fig3.add_vline(x=30, line_dash="dash", line_color=COLORS["warning"], annotation_text="Low")
        fig3.update_layout(
            template=PLOTLY_TEMPLATE, height=300,
            margin=dict(t=10, b=10, l=10, r=10),
        )
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")

    st.markdown('<div class="card-header">Relevant Tenders</div>', unsafe_allow_html=True)
    top_relevant = [r for r in results if r.get("decision") == "RELEVANT"]
    if top_relevant:
        df_top = pd.DataFrame([
            {
                "Title": r.get("title", "")[:90],
                "Score": f"{r.get('relevance_score', 0):.1f}%",
                "Win Prob": f"{r.get('win_probability', 0)}%",
                "Domain": r.get("best_matching_domain", ""),
                "Platform": r.get("platform", ""),
                "Deadline": r.get("deadline", "")[:10] if r.get("deadline") else "",
                "Risk": r.get("deadline_risk", ""),
                "Difficulty": r.get("difficulty_level", ""),
                "Link": r.get("url", ""),
            }
            for r in top_relevant
        ])
        st.dataframe(
            df_top, use_container_width=True, hide_index=True,
            column_config={
                "Link": st.column_config.LinkColumn("Link", display_text="Open"),
            },
        )
    else:
        st.caption("No relevant tenders found in current results.")


# ================================================================
# PAGE: SCRAPE TENDERS
# ================================================================

elif page == "Scrape Tenders":
    st.markdown('<p class="page-title">Scrape Tenders</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Collect tenders from public procurement platforms</p>', unsafe_allow_html=True)

    col_src, col_q = st.columns([3, 1])
    with col_src:
        selected_sources = st.multiselect(
            "Platforms", SOURCE_OPTIONS, default=SOURCE_OPTIONS,
            help="Select procurement platforms to scrape",
        )
    with col_q:
        query = st.text_input("Search query", value="IT services")

    with st.expander("Source Details"):
        st.dataframe(
            pd.DataFrame({
                "Source": ["SAM.GOV", "TED", "UNGM", "TUNEPS", "CONTRACTS_FINDER"],
                "Method": ["REST API", "RSS Feed", "HTML Scraping", "JS Browser", "REST API"],
                "Region": ["United States", "European Union", "UN (Global)", "Tunisia", "United Kingdom"],
                "Speed": ["Fast", "Fast", "Medium", "Slow", "Fast"],
            }),
            use_container_width=True, hide_index=True,
        )

    st.markdown("---")

    b1, b2, b3 = st.columns(3)
    scrape_only = b1.button("Scrape Only", use_container_width=True, type="secondary")
    full_pipeline = b2.button("Full Pipeline", use_container_width=True, type="primary")
    analyze_existing = b3.button("Analyze Latest", use_container_width=True, type="secondary")

    if scrape_only:
        with st.spinner("Scraping tenders..."):
            t0 = time.time()
            data = api_call("POST", "/scrape", {
                "query": query,
                "sources": selected_sources if len(selected_sources) < 5 else None,
            })
            elapsed = time.time() - t0
        if data:
            st.success(f"Scraped **{data['total']}** tenders in **{elapsed:.1f}s**")
            cols = st.columns(len(data.get("by_platform", {})) or 1)
            for i, (plat, cnt) in enumerate(data.get("by_platform", {}).items()):
                cols[i % len(cols)].metric(plat, cnt)
            st.session_state["scraped_tenders"] = data["tenders"]

            st.markdown('<div class="card-header">Scraped Tenders</div>', unsafe_allow_html=True)
            df = pd.DataFrame([
                {
                    "Title": t.get("title", "")[:80],
                    "Platform": t.get("platform", ""),
                    "Organization": t.get("organization", "")[:50],
                    "Deadline": t.get("deadline", "")[:10] if t.get("deadline") else "",
                    "Country": t.get("country", ""),
                }
                for t in data["tenders"]
            ])
            st.dataframe(df, use_container_width=True, hide_index=True)

    if full_pipeline:
        with st.spinner("Running full pipeline ..."):
            t0 = time.time()
            data = api_call("POST", "/pipeline", {
                "query": query,
                "sources": selected_sources if len(selected_sources) < 5 else None,
            }, timeout=600)
            elapsed = time.time() - t0
        if data:
            s = data.get("summary", {})
            st.success(
                f"Pipeline complete in **{elapsed:.1f}s** — "
                f"Scraped {s.get('total_scraped', 0)}, Analyzed {s.get('total_analyzed', 0)}"
            )
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Relevant", s.get("relevant", 0))
            m2.metric("Low Relevance", s.get("low_relevance", 0))
            m3.metric("Irrelevant", s.get("irrelevant", 0))
            m4.metric("Duration", f"{elapsed:.1f}s")
            st.session_state["analysis_results"] = data["results"]

            if s.get("by_platform"):
                fig = px.bar(
                    x=list(s["by_platform"].keys()), y=list(s["by_platform"].values()),
                    color=list(s["by_platform"].keys()), labels={"x": "Source", "y": "Tenders"},
                    color_discrete_sequence=CHART_COLORS,
                )
                fig.update_layout(template=PLOTLY_TEMPLATE, height=250, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

    if analyze_existing:
        with st.spinner("Loading and analyzing latest scraped tenders..."):
            tenders_data = api_call("GET", "/tenders/latest")
            if tenders_data and tenders_data.get("tenders"):
                tenders = tenders_data["tenders"]
                legacy = [{
                    "id": t.get("source_id", ""),
                    "title": t.get("title", ""),
                    "url": t.get("url", "") or t.get("source_url", ""),
                    "platform": t.get("platform", ""),
                    "description": t.get("description", ""),
                    "deadline": t.get("deadline", ""),
                    "budget": t.get("budget", ""),
                    "budget_amount": t.get("budget_amount"),
                    "location": t.get("location", "") or t.get("country", ""),
                    "required_skills": t.get("required_skills", []),
                    "category": t.get("category", ""),
                    "organization": t.get("organization", ""),
                } for t in tenders]

                data = api_call("POST", "/analyze", {"tenders": legacy}, timeout=600)
                if data:
                    s = data["summary"]
                    st.success(
                        f"Analyzed **{s['total']}** tenders — "
                        f"{s['relevant']} relevant, {s['low_relevance']} low, {s['irrelevant']} irrelevant"
                    )
                    st.session_state["analysis_results"] = data["results"]
            else:
                st.warning("No scraped tenders found. Run Scrape first.")


# ================================================================
# PAGE: ANALYSIS RESULTS
# ================================================================

elif page == "Analysis Results":
    st.markdown('<p class="page-title">Analysis Results</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">NLP extraction, relevance scoring and strategic evaluation</p>', unsafe_allow_html=True)

    results = st.session_state.get("analysis_results")
    if not results:
        data = api_call("GET", "/results/latest")
        if data:
            results = data.get("results", [])
            st.session_state["analysis_results"] = results

    if not results:
        st.info("No results yet. Navigate to **Scrape Tenders** to run the pipeline.")
        st.stop()

    st.markdown('<div class="card-header">Filters</div>', unsafe_allow_html=True)
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        filter_decision = st.multiselect(
            "Decision", ["RELEVANT", "LOW_RELEVANCE", "IRRELEVANT"],
            default=["RELEVANT", "LOW_RELEVANCE", "IRRELEVANT"],
        )
    with f2:
        platforms = sorted(set(r.get("platform", "Unknown") for r in results))
        filter_platform = st.multiselect("Platform", platforms, default=platforms)
    with f3:
        domains = sorted(set(r.get("best_matching_domain", "General") for r in results))
        filter_domain = st.multiselect("Domain", domains, default=domains)
    with f4:
        min_score = st.slider("Min Score (%)", 0, 100, 0)

    filtered = [
        r for r in results
        if r.get("decision", "") in filter_decision
        and r.get("platform", "Unknown") in filter_platform
        and r.get("best_matching_domain", "General") in filter_domain
        and r.get("relevance_score", 0) >= min_score
    ]

    st.caption(f"Showing {len(filtered)} of {len(results)} tenders")
    st.markdown("---")

    if filtered:
        s1, s2, s3, s4, s5 = st.columns(5)
        avg_s = sum(r.get("relevance_score", 0) for r in filtered) / len(filtered)
        avg_w = sum(r.get("win_probability", 0) for r in filtered) / len(filtered)
        high_risk = sum(1 for r in filtered if r.get("deadline_risk") == "HIGH")
        best = max(filtered, key=lambda r: r.get("relevance_score", 0))
        s1.metric("Avg Score", f"{avg_s:.1f}%")
        s2.metric("Avg Win Prob", f"{avg_w:.0f}%")
        s3.metric("High Risk", high_risk)
        s4.metric("Filtered", len(filtered))
        s5.metric("Best Score", f"{best.get('relevance_score', 0):.1f}%")

    st.markdown('<div class="card-header">Results</div>', unsafe_allow_html=True)
    if filtered:
        df = pd.DataFrame([
            {
                "Title": r.get("title", "")[:75],
                "Score (%)": round(r.get("relevance_score", 0), 1),
                "Decision": r.get("decision", ""),
                "Win (%)": r.get("win_probability", 0),
                "Domain": r.get("best_matching_domain", ""),
                "Platform": r.get("platform", ""),
                "Risk": r.get("deadline_risk", ""),
                "Difficulty": r.get("difficulty_level", ""),
                "Deadline": str(r.get("deadline", ""))[:10],
                "Days Left": r.get("days_remaining", ""),
                "Link": r.get("url", ""),
            }
            for r in filtered
        ])

        def _color_decision(val):
            m = {
                "RELEVANT": f"background-color: {COLORS['success_light']}; color: {COLORS['success_dark']}",
                "LOW_RELEVANCE": f"background-color: {COLORS['warning_light']}; color: {COLORS['warning_dark']}",
                "IRRELEVANT": f"background-color: {COLORS['danger_light']}; color: {COLORS['danger_dark']}",
            }
            return m.get(val, "")

        try:
            styled = df.style.map(_color_decision, subset=["Decision"])
        except AttributeError:
            styled = df.style.applymap(_color_decision, subset=["Decision"])

        st.dataframe(
            styled, use_container_width=True, hide_index=True, height=500,
            column_config={"Link": st.column_config.LinkColumn("Link", display_text="Open")},
        )

    if filtered:
        st.markdown("---")
        st.markdown('<div class="card-header">AI Score Explanations</div>', unsafe_allow_html=True)
        st.caption("Expand any tender to understand its scoring.")

        for i, r in enumerate(filtered):
            det = r.get("score_explanation_detail")
            if not det:
                txt = r.get("score_explanation", "")
                if not txt:
                    continue
                with st.expander(f"{r.get('decision','')} — {r.get('title','')[:75]}  |  {r.get('relevance_score',0):.0f}%"):
                    st.markdown(txt)
                continue

            v = det.get("verdict", {})
            rec = det.get("recommendation", {})
            sk = det.get("skills", {})
            score_val = r.get("relevance_score", 0)
            dec = v.get("label", "")

            with st.expander(f"{dec} — {r.get('title','')[:75]}  |  {score_val:.0f}%"):
                cl, cr = st.columns([3, 2])
                with cl:
                    st.markdown(
                        f'<div style="font-size:0.88rem;line-height:1.65;color:{COLORS["neutral_700"]}">'
                        f'<strong>{v.get("summary","")}</strong><br>'
                        f'<span style="color:{COLORS["neutral_500"]}">Domain:</span> {det.get("domain",{}).get("text","")}<br>'
                        f'<span style="color:{COLORS["neutral_500"]}">Semantic:</span> {det.get("semantic",{}).get("text","")}<br>'
                        f'<span style="color:{COLORS["neutral_500"]}">Skills:</span> {sk.get("text","")}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                with cr:
                    rec_bg = rec.get("color", COLORS["neutral_500"])
                    st.markdown(
                        f'<div style="background:{rec_bg};color:white;padding:12px 16px;'
                        f'border-radius:10px;text-align:center">'
                        f'<div style="font-size:1rem;font-weight:700">{rec.get("action","")}</div>'
                        f'<div style="font-size:0.78rem;margin-top:4px;opacity:0.9">{rec.get("text","")}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                matched = sk.get("matched", [])
                missing = sk.get("missing", [])
                if matched or missing:
                    pills = '<div style="display:flex;flex-wrap:wrap;gap:5px;margin-top:10px">'
                    for s in matched:
                        pills += (
                            f'<span style="background:{COLORS["success_light"]};color:{COLORS["success_dark"]};'
                            f'padding:3px 12px;border-radius:16px;font-size:0.76rem;font-weight:500">{s}</span>'
                        )
                    for s in missing:
                        pills += (
                            f'<span style="background:{COLORS["danger_light"]};color:{COLORS["danger_dark"]};'
                            f'padding:3px 12px;border-radius:16px;font-size:0.76rem;font-weight:500">{s}</span>'
                        )
                    pills += '</div>'
                    st.markdown(pills, unsafe_allow_html=True)

    st.markdown("---")

    if filtered:
        ch1, ch2 = st.columns(2)

        with ch1:
            st.markdown('<div class="card-header">Relevance vs Win Probability</div>', unsafe_allow_html=True)
            fig = px.scatter(
                pd.DataFrame([{
                    "Relevance Score": r.get("relevance_score", 0),
                    "Win Probability": r.get("win_probability", 0),
                    "Decision": r.get("decision", ""),
                    "Title": r.get("title", "")[:50],
                } for r in filtered]),
                x="Relevance Score", y="Win Probability", color="Decision",
                color_discrete_map=DECISION_COLORS, hover_data=["Title"],
            )
            fig.update_layout(template=PLOTLY_TEMPLATE, height=400)
            st.plotly_chart(fig, use_container_width=True)

        with ch2:
            st.markdown('<div class="card-header">Average Score by Domain</div>', unsafe_allow_html=True)
            ds = {}
            for r in filtered:
                d = r.get("best_matching_domain", "General")
                ds.setdefault(d, []).append(r.get("relevance_score", 0))
            davg = {d: sum(s) / len(s) for d, s in ds.items()}
            fig2 = px.bar(
                x=list(davg.keys()), y=list(davg.values()),
                labels={"x": "Domain", "y": "Avg Score (%)"},
                color=list(davg.values()), color_continuous_scale="Tealgrn",
            )
            fig2.update_layout(template=PLOTLY_TEMPLATE, height=400, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)


# ================================================================
# PAGE: TENDER DETAILS
# ================================================================

elif page == "Tender Details":
    st.markdown('<p class="page-title">Tender Details</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">In-depth analysis of individual tenders</p>', unsafe_allow_html=True)

    results = st.session_state.get("analysis_results")
    if not results:
        data = api_call("GET", "/results/latest")
        if data:
            results = data.get("results", [])
            st.session_state["analysis_results"] = results

    if not results:
        st.info("No results yet. Navigate to **Scrape Tenders** to run the pipeline.")
        st.stop()

    titles = [f"{r.get('title', 'Untitled')[:85]}  [{r.get('decision', '')}]" for r in results]
    selected_idx = st.selectbox("Select a tender", range(len(titles)), format_func=lambda i: titles[i])
    tender = results[selected_idx]

    st.markdown("---")

    decision = tender.get("decision", "IRRELEVANT")
    st.markdown(
        f'<div style="margin-bottom:16px">'
        f'<h3 style="margin:0 0 8px 0;color:{COLORS["neutral_900"]};font-weight:700;line-height:1.3">'
        f'{tender.get("title", "Untitled")}</h3>'
        f'{decision_badge_html(decision)}</div>',
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Relevance", f"{tender.get('relevance_score', 0):.1f}%")
    m2.metric("Strategic", f"{tender.get('strategic_relevance_score', 0)}%")
    m3.metric("Win Prob", f"{tender.get('win_probability', 0)}%")
    m4.metric("Risk", tender.get("deadline_risk", ""))
    m5.metric("Difficulty", tender.get("difficulty_level", ""))
    m6.metric("Competition", tender.get("competition_intensity", ""))

    detail = tender.get("score_explanation_detail")
    explanation_text = tender.get("score_explanation", "")

    if detail:
        verdict = detail.get("verdict", {})
        domain_info = detail.get("domain", {})
        semantic_info = detail.get("semantic", {})
        skills_info = detail.get("skills", {})
        strat_info = detail.get("strategic", {})
        rec = detail.get("recommendation", {})

        verdict_color = verdict.get("color", COLORS["neutral_500"])
        rec_color = rec.get("color", COLORS["neutral_500"])

        st.markdown(
            f'<div style="background:{verdict_color};color:white;padding:14px 24px;'
            f'border-radius:10px 10px 0 0;margin-top:20px;font-size:1rem;font-weight:600">'
            f'{verdict.get("label","")} &mdash; {verdict.get("summary","")}</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div style="background:{COLORS["neutral_50"]};border:1px solid {COLORS["neutral_200"]};'
            f'border-top:none;border-radius:0 0 10px 10px;padding:24px">',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="card-header">Score Components</div>', unsafe_allow_html=True)

        def _bar(label, value, desc):
            pct = min(int(value * 100), 100)
            c = COLORS["success"] if pct >= 50 else (COLORS["warning"] if pct >= 35 else COLORS["danger"])
            return (
                f'<div style="margin-bottom:16px">'
                f'<div style="display:flex;justify-content:space-between;margin-bottom:4px">'
                f'<span style="font-weight:700;font-size:0.88rem;color:{COLORS["neutral_700"]}">{label}</span>'
                f'<span style="font-weight:700;font-size:0.88rem;color:{c}">{pct}%</span></div>'
                f'<div style="background:{COLORS["neutral_200"]};border-radius:6px;height:8px;overflow:hidden">'
                f'<div style="background:{c};height:100%;width:{pct}%;border-radius:6px"></div></div>'
                f'<div style="font-size:0.82rem;color:{COLORS["neutral_500"]};margin-top:3px">{desc}</div></div>'
            )

        bars = ""
        bars += _bar("Semantic Similarity", semantic_info.get("value", 0), semantic_info.get("text", ""))
        bars += _bar("Skill Overlap", skills_info.get("value", 0), skills_info.get("text", ""))
        bars += _bar("Domain Match", domain_info.get("value", 0), domain_info.get("text", ""))
        st.markdown(bars, unsafe_allow_html=True)

        matched_list = skills_info.get("matched", [])
        missing_list = skills_info.get("missing", [])
        if matched_list or missing_list:
            st.markdown('<div class="card-header" style="margin-top:8px">Skills</div>', unsafe_allow_html=True)
            pills = '<div style="display:flex;flex-wrap:wrap;gap:6px">'
            for s in matched_list:
                pills += (
                    f'<span style="background:{COLORS["success_light"]};color:{COLORS["success_dark"]};'
                    f'padding:4px 14px;border-radius:20px;font-size:0.78rem;font-weight:500">{s}</span>'
                )
            for s in missing_list:
                pills += (
                    f'<span style="background:{COLORS["danger_light"]};color:{COLORS["danger_dark"]};'
                    f'padding:4px 14px;border-radius:20px;font-size:0.78rem;font-weight:500">{s}</span>'
                )
            pills += '</div>'
            st.markdown(pills, unsafe_allow_html=True)

        st.markdown('<div class="card-header" style="margin-top:16px">Strategic Assessment</div>', unsafe_allow_html=True)
        risk = strat_info.get("deadline_risk", "")
        days = strat_info.get("days_remaining", "")
        difficulty = strat_info.get("difficulty", "")
        competition = strat_info.get("competition", "")

        risk_c = RISK_COLORS.get(risk, COLORS["neutral_500"])
        diff_c = LEVEL_COLORS.get(difficulty, COLORS["neutral_500"])
        comp_c = LEVEL_COLORS.get(competition, COLORS["neutral_500"])
        win_c = COLORS["success"] if strat_info.get("win_probability", 0) >= 70 else (
            COLORS["warning"] if strat_info.get("win_probability", 0) >= 50 else COLORS["danger"]
        )

        grid = (
            f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:16px">'
            f'<div style="background:white;border-radius:8px;padding:14px;text-align:center;'
            f'border:1px solid {COLORS["neutral_200"]}">'
            f'<div style="font-size:0.7rem;font-weight:600;text-transform:uppercase;'
            f'letter-spacing:0.05em;color:{COLORS["neutral_500"]};margin-bottom:4px">Win Probability</div>'
            f'<div style="font-size:1.4rem;font-weight:700;color:{win_c}">{strat_info.get("win_probability", 0)}%</div>'
            f'</div>'
            f'<div style="background:white;border-radius:8px;padding:14px;text-align:center;'
            f'border:1px solid {COLORS["neutral_200"]}">'
            f'<div style="font-size:0.7rem;font-weight:600;text-transform:uppercase;'
            f'letter-spacing:0.05em;color:{COLORS["neutral_500"]};margin-bottom:4px">Deadline Risk</div>'
            f'<div style="font-size:1.15rem;font-weight:700;color:{risk_c}">{risk}</div>'
            f'<div style="font-size:0.72rem;color:{COLORS["neutral_500"]}">{days} days left</div>'
            f'</div>'
            f'<div style="background:white;border-radius:8px;padding:14px;text-align:center;'
            f'border:1px solid {COLORS["neutral_200"]}">'
            f'<div style="font-size:0.7rem;font-weight:600;text-transform:uppercase;'
            f'letter-spacing:0.05em;color:{COLORS["neutral_500"]};margin-bottom:4px">Difficulty</div>'
            f'<div style="font-size:1.15rem;font-weight:700;color:{diff_c}">{difficulty.replace("_"," ")}</div>'
            f'</div>'
            f'<div style="background:white;border-radius:8px;padding:14px;text-align:center;'
            f'border:1px solid {COLORS["neutral_200"]}">'
            f'<div style="font-size:0.7rem;font-weight:600;text-transform:uppercase;'
            f'letter-spacing:0.05em;color:{COLORS["neutral_500"]};margin-bottom:4px">Competition</div>'
            f'<div style="font-size:1.15rem;font-weight:700;color:{comp_c}">{competition.replace("_"," ")}</div>'
            f'</div></div>'
        )
        st.markdown(grid, unsafe_allow_html=True)

        st.markdown(
            f'<div style="background:{rec_color};color:white;padding:14px 20px;'
            f'border-radius:8px;font-weight:600;font-size:0.92rem">'
            f'Recommendation: {rec.get("action","")} &mdash; {rec.get("text","")}</div>',
            unsafe_allow_html=True,
        )

        st.markdown('</div>', unsafe_allow_html=True)

    elif explanation_text:
        bg = DECISION_BG.get(decision, COLORS["neutral_100"])
        fg = DECISION_TEXT.get(decision, COLORS["neutral_700"])
        st.markdown(
            f'<div style="background:{bg};color:{fg};padding:16px 20px;border-radius:10px;'
            f'margin:16px 0;line-height:1.7;font-size:0.92rem">{explanation_text}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    left, right = st.columns(2)

    with left:
        st.markdown('<div class="card-header">Tender Information</div>', unsafe_allow_html=True)
        tender_url = tender.get("url", "")
        if tender_url:
            st.markdown(
                f'<a href="{tender_url}" target="_blank" style="display:inline-block;'
                f'background:{COLORS["primary"]};color:white;padding:8px 20px;border-radius:8px;'
                f'text-decoration:none;font-weight:600;font-size:0.85rem;margin-bottom:14px">'
                f'View Original Tender</a>',
                unsafe_allow_html=True,
            )

        info_rows = [
            ("Platform", tender.get("platform", "")),
            ("Organization", tender.get("organization", "")),
            ("Location", tender.get("location", "")),
            ("Budget", tender.get("budget", "")),
            ("Deadline", tender.get("deadline", "")),
            ("Days Remaining", tender.get("days_remaining", "")),
            ("Domain", tender.get("best_matching_domain", "")),
            ("Detected Domain", tender.get("detected_domain", "")),
            ("Category", tender.get("category", "")),
            ("Complexity", tender.get("complexity_score", "")),
        ]
        info_html = '<div style="line-height:2.1">'
        for label, val in info_rows:
            info_html += (
                f'<div><span style="font-size:0.85rem;color:{COLORS["neutral_500"]};'
                f'font-weight:600;min-width:130px;display:inline-block">{label}</span> '
                f'<span style="font-size:0.9rem;color:{COLORS["neutral_900"]};font-weight:600">{val}</span></div>'
            )
        info_html += '</div>'
        st.markdown(info_html, unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card-header">Scoring Breakdown</div>', unsafe_allow_html=True)

        categories = ["Semantic Similarity", "Skill Overlap", "Domain Similarity"]
        values = [
            tender.get("semantic_similarity", 0),
            tender.get("skill_overlap", 0),
            tender.get("domain_similarity", 0),
        ]

        fig_radar = go.Figure(data=go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            fillcolor="rgba(79, 70, 229, 0.15)",
            line=dict(color=COLORS["primary"], width=2),
        ))
        fig_radar.update_layout(
            template=PLOTLY_TEMPLATE,
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            height=300, margin=dict(t=30, b=30, l=60, r=60),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        breakdown = tender.get("score_breakdown", {})
        if breakdown:
            comps = {
                "Final Score": breakdown.get("final_score_component", 0),
                "Skill Coverage": breakdown.get("skill_coverage_component", 0),
                "Domain Weight": breakdown.get("domain_weight_component", 0),
                "Budget Compat": breakdown.get("budget_compat_component", 0),
                "Geo Match": breakdown.get("geographic_match_component", 0),
            }
            fig_bd = px.bar(
                x=list(comps.values()), y=list(comps.keys()), orientation="h",
                labels={"x": "Contribution", "y": ""},
                color_discrete_sequence=[COLORS["primary"]],
            )
            fig_bd.update_layout(template=PLOTLY_TEMPLATE, height=220, margin=dict(t=10, b=10, l=0, r=0))
            st.plotly_chart(fig_bd, use_container_width=True)

    st.markdown("---")

    sk1, sk2 = st.columns(2)
    with sk1:
        st.markdown('<div class="card-header">Matched Skills</div>', unsafe_allow_html=True)
        matched = tender.get("matched_skills", [])
        if matched:
            pills = '<div style="display:flex;flex-wrap:wrap;gap:6px">'
            for s in matched:
                pills += (
                    f'<span style="background:{COLORS["success_light"]};color:{COLORS["success_dark"]};'
                    f'padding:5px 14px;border-radius:20px;font-size:0.82rem;font-weight:500">{s}</span>'
                )
            pills += '</div>'
            st.markdown(pills, unsafe_allow_html=True)
        else:
            st.caption("No matched skills")

    with sk2:
        st.markdown('<div class="card-header">Missing Skills</div>', unsafe_allow_html=True)
        missing = tender.get("missing_skills", [])
        if missing:
            pills = '<div style="display:flex;flex-wrap:wrap;gap:6px">'
            for s in missing:
                pills += (
                    f'<span style="background:{COLORS["danger_light"]};color:{COLORS["danger_dark"]};'
                    f'padding:5px 14px;border-radius:20px;font-size:0.82rem;font-weight:500">{s}</span>'
                )
            pills += '</div>'
            st.markdown(pills, unsafe_allow_html=True)
        else:
            st.caption("No missing skills")

    kw1, kw2 = st.columns(2)
    with kw1:
        st.markdown('<div class="card-header">Top Keywords</div>', unsafe_allow_html=True)
        keywords = tender.get("top_keywords", [])
        if not keywords:
            nlp = tender.get("nlp_extraction", {})
            if nlp:
                for key in ("top_keywords", "noun_chunks"):
                    items = nlp.get(key, [])
                    if items:
                        keywords = items if isinstance(items, list) else []
                        break
        if isinstance(keywords, dict):
            flat = []
            for v in keywords.values():
                if isinstance(v, list):
                    flat.extend(v)
            keywords = flat
        if keywords:
            pills = '<div style="display:flex;flex-wrap:wrap;gap:5px">'
            for kw in keywords[:20]:
                pills += (
                    f'<span style="background:{COLORS["neutral_100"]};color:{COLORS["neutral_700"]};'
                    f'padding:4px 12px;border-radius:16px;font-size:0.78rem;font-weight:500">{kw}</span>'
                )
            pills += '</div>'
            st.markdown(pills, unsafe_allow_html=True)
        else:
            st.caption("No keywords extracted")

    with kw2:
        st.markdown('<div class="card-header">Certifications</div>', unsafe_allow_html=True)
        certs = tender.get("detected_certifications", [])
        if certs:
            pills = '<div style="display:flex;flex-wrap:wrap;gap:5px">'
            for c in certs:
                pills += (
                    f'<span style="background:#EDE9FE;color:#5B21B6;'
                    f'padding:4px 12px;border-radius:16px;font-size:0.78rem;font-weight:500">{c}</span>'
                )
            pills += '</div>'
            st.markdown(pills, unsafe_allow_html=True)
        else:
            st.caption("No certifications detected")

    with st.expander("Full NLP Extraction"):
        nlp_data = tender.get("nlp_extraction", {})
        if nlp_data:
            st.json(nlp_data)
        else:
            st.caption("No NLP extraction data")

    with st.expander("Raw JSON"):
        st.json(tender)
