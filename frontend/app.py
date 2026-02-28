"""
SmartTender AI — Streamlit Dashboard
======================================
Interactive UI for Inetum Tunisie to:
    1. Launch web scraping across 5 sources
    2. View scraped tenders with filters
    3. Run NLP keyword extraction + relevance scoring
    4. View strategic evaluation (win probability, deadline risk)
    5. Drill into individual tender details

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

DECISION_COLORS = {
    "RELEVANT": "#28a745",
    "LOW_RELEVANCE": "#ffc107",
    "IRRELEVANT": "#dc3545",
}

RISK_COLORS = {
    "LOW": "#28a745",
    "MEDIUM": "#ffc107",
    "HIGH": "#dc3545",
}

SOURCE_OPTIONS = ["SAM.GOV", "TED", "UNGM", "TUNEPS", "CONTRACTS_FINDER"]

# ================================================================
# PAGE CONFIG
# ================================================================

st.set_page_config(
    page_title="SmartTender AI — Inetum Tunisie",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ================================================================
# CUSTOM CSS
# ================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #666;
        margin-top: -10px;
        margin-bottom: 20px;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 20px;
        color: white;
        text-align: center;
    }
    .metric-card h3 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
    }
    .metric-card p {
        margin: 5px 0 0 0;
        font-size: 0.85rem;
        opacity: 0.9;
    }
    .status-relevant { color: #28a745; font-weight: 700; }
    .status-low { color: #ffc107; font-weight: 700; }
    .status-irrelevant { color: #dc3545; font-weight: 700; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)


# ================================================================
# HELPERS
# ================================================================

def api_call(method: str, endpoint: str, payload: dict = None, timeout: int = 300):
    """Make an API call to the FastAPI backend."""
    url = f"{API_BASE}{endpoint}"
    try:
        if method == "GET":
            resp = requests.get(url, timeout=timeout)
        else:
            resp = requests.post(url, json=payload or {}, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        st.error("❌ Cannot connect to API. Start the backend: `uvicorn backend.api:app --reload`")
        return None
    except requests.HTTPError as e:
        st.error(f"❌ API error: {e.response.status_code} — {e.response.text[:300]}")
        return None
    except Exception as e:
        st.error(f"❌ Error: {e}")
        return None


def load_from_disk(filename: str):
    """Load JSON from output directory (fallback if API is down)."""
    filepath = OUTPUT_DIR / filename
    if filepath.exists():
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def render_decision_badge(decision: str) -> str:
    """Return HTML badge for decision."""
    color = DECISION_COLORS.get(decision, "#888")
    return f'<span style="background:{color};color:white;padding:3px 10px;border-radius:12px;font-size:0.8rem;font-weight:600">{decision}</span>'


def render_risk_badge(risk: str) -> str:
    """Return HTML badge for risk level."""
    color = RISK_COLORS.get(risk, "#888")
    return f'<span style="background:{color};color:white;padding:3px 10px;border-radius:12px;font-size:0.8rem;font-weight:600">{risk}</span>'


# ================================================================
# SIDEBAR
# ================================================================

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/31/Inetum_logo.svg/1200px-Inetum_logo.svg.png", width=180)
    st.markdown("### 🎯 SmartTender AI")
    st.markdown("*Intelligent Tender Detection*")
    st.markdown("---")

    # Navigation
    page = st.radio(
        "Navigation",
        ["🏠 Dashboard", "🔍 Scrape Tenders", "📊 Analysis Results", "📋 Tender Details"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # API status
    try:
        health = requests.get(f"{API_BASE}/health", timeout=3).json()
        st.success("🟢 API Connected")
    except Exception:
        st.warning("🔴 API Offline")
        st.caption("Start with: `uvicorn backend.api:app`")

    st.markdown("---")
    st.caption("© 2026 Inetum Tunisie")
    st.caption("SmartTender AI v1.0")


# ================================================================
# PAGE: DASHBOARD
# ================================================================

if page == "🏠 Dashboard":
    st.markdown('<p class="main-header">🏠 SmartTender AI Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Real-time tender intelligence for Inetum Tunisie</p>', unsafe_allow_html=True)

    # Try to load latest results
    data = api_call("GET", "/results/latest") if True else None
    results = data.get("results", []) if data else load_from_disk("analysis_results.json") or []

    if not results:
        st.info("No analysis results yet. Go to **🔍 Scrape Tenders** to start the pipeline.")
        st.stop()

    # Top metrics
    total = len(results)
    relevant = sum(1 for r in results if r.get("decision") == "RELEVANT")
    low_rel = sum(1 for r in results if r.get("decision") == "LOW_RELEVANCE")
    irrelevant = sum(1 for r in results if r.get("decision") == "IRRELEVANT")
    avg_score = sum(r.get("relevance_score", 0) for r in results) / total if total else 0
    avg_win = sum(r.get("win_probability", 0) for r in results) / total if total else 0

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("📦 Total Tenders", total)
    col2.metric("✅ Relevant", relevant)
    col3.metric("🔶 Low Relevance", low_rel)
    col4.metric("⬜ Irrelevant", irrelevant)
    col5.metric("📈 Avg Score", f"{avg_score:.1f}%")
    col6.metric("🏆 Avg Win Prob", f"{avg_win:.0f}%")

    st.markdown("---")

    # Charts row
    chart_col1, chart_col2, chart_col3 = st.columns(3)

    with chart_col1:
        st.markdown("#### Decision Distribution")
        fig_pie = px.pie(
            names=["Relevant", "Low Relevance", "Irrelevant"],
            values=[relevant, low_rel, irrelevant],
            color_discrete_sequence=["#28a745", "#ffc107", "#dc3545"],
            hole=0.4,
        )
        fig_pie.update_layout(height=300, margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_pie, width="stretch")

    with chart_col2:
        st.markdown("#### Platform Breakdown")
        platforms = {}
        for r in results:
            p = r.get("platform", "Unknown")
            platforms[p] = platforms.get(p, 0) + 1
        fig_bar = px.bar(
            x=list(platforms.keys()),
            y=list(platforms.values()),
            color=list(platforms.keys()),
            labels={"x": "Platform", "y": "Count"},
        )
        fig_bar.update_layout(height=300, margin=dict(t=20, b=20, l=20, r=20), showlegend=False)
        st.plotly_chart(fig_bar, width="stretch")

    with chart_col3:
        st.markdown("#### Score Distribution")
        scores = [r.get("relevance_score", 0) for r in results]
        fig_hist = px.histogram(
            x=scores,
            nbins=20,
            labels={"x": "Relevance Score (%)", "y": "Count"},
            color_discrete_sequence=["#667eea"],
        )
        fig_hist.add_vline(x=65, line_dash="dash", line_color="green", annotation_text="Relevant ≥65%")
        fig_hist.add_vline(x=40, line_dash="dash", line_color="orange", annotation_text="Low ≥40%")
        fig_hist.update_layout(height=300, margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_hist, width="stretch")

    st.markdown("---")

    # Top relevant tenders table
    st.markdown("#### 🏆 Top Relevant Tenders")
    top_relevant = [r for r in results if r.get("decision") == "RELEVANT"]
    if top_relevant:
        df_top = pd.DataFrame([
            {
                "Title": r.get("title", "")[:80],
                "Score": f"{r.get('relevance_score', 0):.1f}%",
                "Win Prob": f"{r.get('win_probability', 0)}%",
                "Domain": r.get("best_matching_domain", "N/A"),
                "Platform": r.get("platform", "N/A"),
                "Deadline": r.get("deadline", "N/A")[:10] if r.get("deadline") else "N/A",
                "Risk": r.get("deadline_risk", "N/A"),
                "Difficulty": r.get("difficulty_level", "N/A"),
            }
            for r in top_relevant[:15]
        ])
        st.dataframe(df_top, width="stretch", hide_index=True)
    else:
        st.info("No relevant tenders found.")


# ================================================================
# PAGE: SCRAPE TENDERS
# ================================================================

elif page == "🔍 Scrape Tenders":
    st.markdown('<p class="main-header">🔍 Scrape Tenders</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Launch web scraping across public procurement platforms</p>', unsafe_allow_html=True)

    # Source selection
    st.markdown("#### Select Sources")
    col_src1, col_src2 = st.columns([3, 1])

    with col_src1:
        selected_sources = st.multiselect(
            "Choose platforms to scrape",
            SOURCE_OPTIONS,
            default=SOURCE_OPTIONS,
            help="Select one or more procurement platforms",
        )

    with col_src2:
        query = st.text_input("Search query", value="IT services")

    # Source info
    with st.expander("ℹ️ Source Details"):
        source_data = {
            "Source": ["SAM.GOV", "TED", "UNGM", "TUNEPS", "CONTRACTS_FINDER"],
            "Method": ["REST API", "RSS Feed", "HTML Scraping", "JS Browser", "REST API"],
            "Region": ["United States", "European Union", "UN (Global)", "Tunisia", "United Kingdom"],
            "Speed": ["⚡ Fast", "⚡ Fast", "🐢 Medium", "🐢 Slow (JS)", "⚡ Fast"],
        }
        st.dataframe(pd.DataFrame(source_data), width="stretch", hide_index=True)

    st.markdown("---")

    # Action buttons
    col_btn1, col_btn2, col_btn3 = st.columns(3)

    with col_btn1:
        scrape_only = st.button("🔍 Scrape Only", width="stretch", type="secondary")
    with col_btn2:
        full_pipeline = st.button("🚀 Full Pipeline (Scrape + Analyze)", width="stretch", type="primary")
    with col_btn3:
        analyze_existing = st.button("📊 Analyze Latest Scraped", width="stretch", type="secondary")

    # --- Scrape Only ---
    if scrape_only:
        with st.spinner("🔍 Scraping tenders... This may take 30-60 seconds."):
            start = time.time()
            data = api_call("POST", "/scrape", {
                "query": query,
                "sources": selected_sources if len(selected_sources) < 5 else None,
            })
            elapsed = time.time() - start

        if data:
            st.success(f"✅ Scraped **{data['total']}** tenders in **{elapsed:.1f}s**")

            # Platform breakdown
            cols = st.columns(len(data.get("by_platform", {})) or 1)
            for i, (platform, count) in enumerate(data.get("by_platform", {}).items()):
                cols[i % len(cols)].metric(platform, count)

            # Store in session for later analysis
            st.session_state["scraped_tenders"] = data["tenders"]

            # Show preview
            st.markdown("#### Preview (first 10)")
            df = pd.DataFrame([
                {
                    "Title": t.get("title", "")[:70],
                    "Platform": t.get("platform", ""),
                    "Organization": t.get("organization", "")[:40],
                    "Deadline": t.get("deadline", "N/A")[:10] if t.get("deadline") else "N/A",
                    "Country": t.get("country", "N/A"),
                }
                for t in data["tenders"][:10]
            ])
            st.dataframe(df, width="stretch", hide_index=True)

    # --- Full Pipeline ---
    if full_pipeline:
        with st.spinner("🚀 Running full pipeline: Scrape → NLP → Relevance → Strategic..."):
            start = time.time()
            data = api_call("POST", "/pipeline", {
                "query": query,
                "sources": selected_sources if len(selected_sources) < 5 else None,
            }, timeout=600)
            elapsed = time.time() - start

        if data:
            summary = data.get("summary", {})
            st.success(
                f"✅ Pipeline complete in **{elapsed:.1f}s** — "
                f"Scraped: {summary.get('total_scraped', 0)}, "
                f"Analyzed: {summary.get('total_analyzed', 0)}"
            )

            # Summary metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("✅ Relevant", summary.get("relevant", 0))
            m2.metric("🔶 Low Relevance", summary.get("low_relevance", 0))
            m3.metric("⬜ Irrelevant", summary.get("irrelevant", 0))
            m4.metric("⏱️ Time", f"{elapsed:.1f}s")

            # Store results
            st.session_state["analysis_results"] = data["results"]

            # Show platform breakdown
            if summary.get("by_platform"):
                st.markdown("#### Sources Breakdown")
                bp = summary["by_platform"]
                fig = px.bar(x=list(bp.keys()), y=list(bp.values()),
                             color=list(bp.keys()),
                             labels={"x": "Source", "y": "Tenders"})
                fig.update_layout(height=250, showlegend=False)
                st.plotly_chart(fig, width="stretch")

    # --- Analyze Existing ---
    if analyze_existing:
        with st.spinner("📊 Loading latest scraped tenders and analyzing..."):
            # Load latest
            tenders_data = api_call("GET", "/tenders/latest")
            if tenders_data and tenders_data.get("tenders"):
                tenders = tenders_data["tenders"]
                # Convert to legacy format for the analyze endpoint
                legacy = []
                for t in tenders:
                    legacy.append({
                        "id": t.get("source_id", ""),
                        "title": t.get("title", ""),
                        "platform": t.get("platform", ""),
                        "description": t.get("description", ""),
                        "deadline": t.get("deadline", ""),
                        "budget": t.get("budget", ""),
                        "budget_amount": t.get("budget_amount"),
                        "location": t.get("location", "") or t.get("country", ""),
                        "required_skills": t.get("required_skills", []),
                        "category": t.get("category", ""),
                        "organization": t.get("organization", ""),
                    })

                data = api_call("POST", "/analyze", {"tenders": legacy}, timeout=600)
                if data:
                    s = data["summary"]
                    st.success(
                        f"✅ Analyzed **{s['total']}** tenders — "
                        f"{s['relevant']} relevant, {s['low_relevance']} low, {s['irrelevant']} irrelevant"
                    )
                    st.session_state["analysis_results"] = data["results"]
            else:
                st.warning("No scraped tenders found. Run **Scrape Only** first.")


# ================================================================
# PAGE: ANALYSIS RESULTS
# ================================================================

elif page == "📊 Analysis Results":
    st.markdown('<p class="main-header">📊 Analysis Results</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">NLP extraction, relevance scoring & strategic evaluation</p>', unsafe_allow_html=True)

    # Load results from session or API
    results = st.session_state.get("analysis_results")
    if not results:
        data = api_call("GET", "/results/latest")
        if data:
            results = data.get("results", [])
            st.session_state["analysis_results"] = results

    if not results:
        st.info("No results yet. Go to **🔍 Scrape Tenders** and run the pipeline.")
        st.stop()

    # Filters
    st.markdown("#### 🔎 Filters")
    f1, f2, f3, f4 = st.columns(4)

    with f1:
        filter_decision = st.multiselect(
            "Decision",
            ["RELEVANT", "LOW_RELEVANCE", "IRRELEVANT"],
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

    # Apply filters
    filtered = [
        r for r in results
        if r.get("decision", "") in filter_decision
        and r.get("platform", "Unknown") in filter_platform
        and r.get("best_matching_domain", "General") in filter_domain
        and r.get("relevance_score", 0) >= min_score
    ]

    st.markdown(f"**Showing {len(filtered)} of {len(results)} tenders**")
    st.markdown("---")

    # Summary stats for filtered results
    if filtered:
        s1, s2, s3, s4, s5 = st.columns(5)
        avg_score = sum(r.get("relevance_score", 0) for r in filtered) / len(filtered)
        avg_win = sum(r.get("win_probability", 0) for r in filtered) / len(filtered)
        high_risk = sum(1 for r in filtered if r.get("deadline_risk") == "HIGH")
        s1.metric("Avg Score", f"{avg_score:.1f}%")
        s2.metric("Avg Win Prob", f"{avg_win:.0f}%")
        s3.metric("⚠️ High Risk", high_risk)
        s4.metric("Total Filtered", len(filtered))
        best = max(filtered, key=lambda r: r.get("relevance_score", 0))
        s5.metric("Best Score", f"{best.get('relevance_score', 0):.1f}%")

    # Results table
    st.markdown("#### 📋 Results Table")
    if filtered:
        df = pd.DataFrame([
            {
                "Title": r.get("title", "")[:65],
                "Score (%)": round(r.get("relevance_score", 0), 1),
                "Decision": r.get("decision", ""),
                "Win Prob (%)": r.get("win_probability", 0),
                "Strategic (%)": r.get("strategic_relevance_score", 0),
                "Domain": r.get("best_matching_domain", "N/A"),
                "Platform": r.get("platform", "N/A"),
                "Risk": r.get("deadline_risk", "N/A"),
                "Difficulty": r.get("difficulty_level", "N/A"),
                "Competition": r.get("competition_intensity", "N/A"),
                "Deadline": str(r.get("deadline", "N/A"))[:10],
                "Days Left": r.get("days_remaining", "N/A"),
            }
            for r in filtered
        ])

        # Color the dataframe
        def color_decision(val):
            colors = {"RELEVANT": "background-color: #d4edda", "LOW_RELEVANCE": "background-color: #fff3cd", "IRRELEVANT": "background-color: #f8d7da"}
            return colors.get(val, "")

        try:
            styled = df.style.map(color_decision, subset=["Decision"])
        except AttributeError:
            styled = df.style.applymap(color_decision, subset=["Decision"])
        st.dataframe(styled, width="stretch", hide_index=True, height=500)

    st.markdown("---")

    # Charts
    if filtered:
        ch1, ch2 = st.columns(2)

        with ch1:
            st.markdown("#### Relevance Score vs Win Probability")
            df_scatter = pd.DataFrame([
                {
                    "Relevance Score": r.get("relevance_score", 0),
                    "Win Probability": r.get("win_probability", 0),
                    "Decision": r.get("decision", ""),
                    "Title": r.get("title", "")[:50],
                }
                for r in filtered
            ])
            fig = px.scatter(
                df_scatter,
                x="Relevance Score",
                y="Win Probability",
                color="Decision",
                color_discrete_map=DECISION_COLORS,
                hover_data=["Title"],
                size_max=10,
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, width="stretch")

        with ch2:
            st.markdown("#### Score Breakdown by Domain")
            domain_scores = {}
            for r in filtered:
                d = r.get("best_matching_domain", "General")
                if d not in domain_scores:
                    domain_scores[d] = []
                domain_scores[d].append(r.get("relevance_score", 0))

            domain_avg = {d: sum(s) / len(s) for d, s in domain_scores.items()}
            fig2 = px.bar(
                x=list(domain_avg.keys()),
                y=list(domain_avg.values()),
                labels={"x": "Domain", "y": "Avg Relevance Score (%)"},
                color=list(domain_avg.values()),
                color_continuous_scale="RdYlGn",
            )
            fig2.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig2, width="stretch")


# ================================================================
# PAGE: TENDER DETAILS
# ================================================================

elif page == "📋 Tender Details":
    st.markdown('<p class="main-header">📋 Tender Details</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Deep dive into individual tender analysis</p>', unsafe_allow_html=True)

    # Load results
    results = st.session_state.get("analysis_results")
    if not results:
        data = api_call("GET", "/results/latest")
        if data:
            results = data.get("results", [])
            st.session_state["analysis_results"] = results

    if not results:
        st.info("No results yet. Go to **🔍 Scrape Tenders** and run the pipeline.")
        st.stop()

    # Tender selector
    titles = [f"{r.get('title', 'Untitled')[:80]} [{r.get('decision', '')}]" for r in results]
    selected_idx = st.selectbox("Select a tender", range(len(titles)), format_func=lambda i: titles[i])
    tender = results[selected_idx]

    st.markdown("---")

    # Header with decision badge
    decision = tender.get("decision", "IRRELEVANT")
    decision_color = DECISION_COLORS.get(decision, "#888")
    st.markdown(
        f'### {tender.get("title", "Untitled")}'
        f'<br><span style="background:{decision_color};color:white;padding:4px 14px;border-radius:16px;font-size:0.9rem">'
        f'{decision}</span>',
        unsafe_allow_html=True,
    )

    st.markdown("")

    # Key metrics row
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Relevance Score", f"{tender.get('relevance_score', 0):.1f}%")
    m2.metric("Strategic Score", f"{tender.get('strategic_relevance_score', 0)}%")
    m3.metric("Win Probability", f"{tender.get('win_probability', 0)}%")
    m4.metric("Deadline Risk", tender.get("deadline_risk", "N/A"))
    m5.metric("Difficulty", tender.get("difficulty_level", "N/A"))
    m6.metric("Competition", tender.get("competition_intensity", "N/A"))

    st.markdown("---")

    # Two columns: info + scoring
    left, right = st.columns(2)

    with left:
        st.markdown("#### 📄 Tender Information")
        st.markdown(f"**Platform:** {tender.get('platform', 'N/A')}")
        st.markdown(f"**Organization:** {tender.get('organization', 'N/A')}")
        st.markdown(f"**Location:** {tender.get('location', 'N/A')}")
        st.markdown(f"**Budget:** {tender.get('budget', 'N/A')}")
        st.markdown(f"**Deadline:** {tender.get('deadline', 'N/A')}")
        st.markdown(f"**Days Remaining:** {tender.get('days_remaining', 'N/A')}")
        st.markdown(f"**Domain:** {tender.get('best_matching_domain', 'N/A')}")
        st.markdown(f"**Detected Domain:** {tender.get('detected_domain', 'N/A')}")
        st.markdown(f"**Category:** {tender.get('category', 'N/A')}")
        st.markdown(f"**Complexity Score:** {tender.get('complexity_score', 'N/A')}")

    with right:
        st.markdown("#### 📊 Scoring Breakdown")

        # Radar chart for 3 similarity components
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
            fillcolor="rgba(102, 126, 234, 0.3)",
            line=dict(color="#667eea", width=2),
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            height=300,
            margin=dict(t=30, b=30, l=60, r=60),
        )
        st.plotly_chart(fig_radar, width="stretch")

        # Score breakdown bar
        breakdown = tender.get("score_breakdown", {})
        if breakdown:
            components = {
                "Final Score": breakdown.get("final_score_component", 0),
                "Skill Coverage": breakdown.get("skill_coverage_component", 0),
                "Domain Weight": breakdown.get("domain_weight_component", 0),
                "Budget Compat": breakdown.get("budget_compat_component", 0),
                "Geo Match": breakdown.get("geographic_match_component", 0),
            }
            fig_bd = px.bar(
                x=list(components.values()),
                y=list(components.keys()),
                orientation="h",
                labels={"x": "Contribution", "y": "Component"},
                color_discrete_sequence=["#667eea"],
            )
            fig_bd.update_layout(height=250, margin=dict(t=10, b=10))
            st.plotly_chart(fig_bd, width="stretch")

    st.markdown("---")

    # Skills section
    sk1, sk2 = st.columns(2)
    with sk1:
        st.markdown("#### ✅ Matched Skills")
        matched = tender.get("matched_skills", [])
        if matched:
            for skill in matched:
                st.markdown(f"- 🟢 `{skill}`")
        else:
            st.caption("No matched skills")

    with sk2:
        st.markdown("#### ❌ Missing Skills")
        missing = tender.get("missing_skills", [])
        if missing:
            for skill in missing:
                st.markdown(f"- 🔴 `{skill}`")
        else:
            st.caption("No missing skills")

    # Keywords and certifications
    kw1, kw2 = st.columns(2)
    with kw1:
        st.markdown("#### 🔑 Top Keywords")
        keywords = tender.get("top_keywords", [])
        # Fallback: extract from nlp_extraction
        if not keywords:
            nlp = tender.get("nlp_extraction", {})
            if nlp:
                for key in ("top_keywords", "noun_chunks"):
                    items = nlp.get(key, [])
                    if items:
                        keywords = items if isinstance(items, list) else []
                        break
        # Flatten if keywords is a dict (legacy format)
        if isinstance(keywords, dict):
            flat = []
            for v in keywords.values():
                if isinstance(v, list):
                    flat.extend(v)
            keywords = flat
        if keywords:
            st.markdown(", ".join([f"`{kw}`" for kw in keywords[:20]]))
        else:
            st.caption("No keywords extracted")

    with kw2:
        st.markdown("#### 📜 Certifications")
        certs = tender.get("detected_certifications", [])
        if certs:
            for c in certs:
                st.markdown(f"- 🏅 `{c}`")
        else:
            st.caption("No certifications detected")

    # NLP extraction details (expandable)
    with st.expander("🔬 Full NLP Extraction"):
        nlp_data = tender.get("nlp_extraction", {})
        if nlp_data:
            st.json(nlp_data)
        else:
            st.caption("No NLP extraction data")

    # Raw JSON (expandable)
    with st.expander("📦 Raw JSON"):
        st.json(tender)
