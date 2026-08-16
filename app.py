import os
import sys
import glob
import pickle
import streamlit as st
import numpy as np

from books_recommender.log_details.log import logging
from books_recommender.config.configuration import AppConfiguration
from books_recommender.pipeline.training_pipeline import TrainingPipeline
from books_recommender.exception.exception_handler import AppException


class Recommendation:

    def __init__(self, app_config=AppConfiguration()):
        try:
            self.recommendation_config = (
                app_config.get_recommendation_config()
            )
        except Exception as e:
            raise AppException(e, sys) from e


    def fetch_poster(self, suggestion):
        try:
            book_name = []
            ids_index = []
            poster_url = []

            book_pivot = pickle.load(
                open(self.recommendation_config.book_pivot_file, "rb")
            )

            final_rating = pickle.load(
                open(self.recommendation_config.final_rating_file, "rb")
            )

            for book_id in suggestion.flatten():
                name = book_pivot.index[book_id]
                ids = np.where(final_rating["title"] == name)[0][0]
                ids_index.append(ids)

            for idx in ids_index:
                poster_url.append(
                    final_rating.iloc[idx]["image_url"]
                )

            return poster_url

        except Exception as e:
            raise AppException(e, sys) from e


    def recommend_book(self, book_name):
        try:
            books_list = []

            model = pickle.load(
                open(self.recommendation_config.trained_model_path, "rb")
            )

            book_pivot = pickle.load(
                open(self.recommendation_config.book_pivot_file, "rb")
            )

            book_id = np.where(
                book_pivot.index == book_name
            )[0][0]

            distance, suggestion = model.kneighbors(
                book_pivot.iloc[book_id].values.reshape(1, -1),
                n_neighbors=6
            )

            poster_url = self.fetch_poster(suggestion)

            for idx in suggestion.flatten():
                books_list.append(book_pivot.index[idx])

            return books_list, poster_url

        except Exception as e:
            raise AppException(e, sys) from e


    def train_engine(self):
        try:
            obj = TrainingPipeline()
            obj.start_training_pipeline()

            st.success("Training Completed!")

            logging.info("Training completed successfully.")

        except Exception as e:
            raise AppException(e, sys) from e


    def recommendations_engine(self, selected_books):
        try:
            recommended_books, poster_url = (
                self.recommend_book(selected_books)
            )

            cols = st.columns(5)

            for i in range(1, 6):
                with cols[i - 1]:
                    st.text(recommended_books[i])
                    st.image(poster_url[i])

        except Exception as e:
            raise AppException(e, sys) from e


# ==============================================================================
# PRESENTATION LAYER ONLY — everything below only arranges/styles/navigates the
# page. Nothing above this line has been changed. No backend, pipeline,
# config, or recommendation logic is touched anywhere below.
# ==============================================================================

def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        #MainMenu, footer, header { visibility: hidden; }

        .stApp {
            background: radial-gradient(circle at 15% 0%, #170f2b 0%, #0a0912 45%, #08070f 100%);
            color: #e8e6f0;
        }

        .block-container {
            padding-top: 1.6rem;
            padding-bottom: 2rem;
            max-width: 1500px;
        }

        /* ---------------- Sidebar ---------------- */
        section[data-testid="stSidebar"] {
            width: 248px !important;
            min-width: 248px !important;
            background: #0c0a16;
            border-right: 1px solid rgba(148,120,255,0.12);
        }
        section[data-testid="stSidebar"] > div:first-child {
            width: 248px !important;
            padding-top: 1.2rem;
        }

        .sb-logo {
            display: flex;
            align-items: center;
            gap: 0.6rem;
            padding: 0 1.1rem 1.2rem 1.1rem;
            border-bottom: 1px solid rgba(148,120,255,0.12);
            margin-bottom: 0.6rem;
        }
        .sb-logo-icon {
            width: 34px; height: 34px;
            border-radius: 9px;
            background: linear-gradient(135deg,#7c3aed,#a855f7);
            display: flex; align-items: center; justify-content: center;
            font-size: 1.1rem;
        }
        .sb-logo-text { font-weight: 800; font-size: 1.05rem; color: #f4f2fb; line-height:1.1rem; }
        .sb-logo-sub { font-size: 0.68rem; color: #8a80ab; font-weight: 600; letter-spacing: 0.04em; }

        .sb-section-title {
            font-size: 0.68rem;
            color: #6f6690;
            font-weight: 700;
            letter-spacing: 0.08em;
            padding: 0.7rem 1.1rem 0.3rem 1.1rem;
        }

        .sb-item {
            display: grid;
            grid-template-columns: minmax(0, 1fr) auto;
            align-items: center;
            column-gap: 0.5rem;
            padding: 0.55rem 1.1rem;
            margin: 0.1rem 0.6rem;
            border-radius: 9px;
            font-size: 0.87rem;
            font-weight: 500;
            color: #b6afd1;
        }
        .sb-item-label {
            display: flex;
            align-items: center;
            gap: 0.55rem;
            min-width: 0;
            overflow-wrap: break-word;
            word-break: break-word;
            line-height: 1.25;
        }
        .sb-item.active {
            background: linear-gradient(90deg, rgba(124,58,237,0.35), rgba(124,58,237,0.12));
            color: #ffffff;
            font-weight: 700;
            border: 1px solid rgba(168,140,255,0.35);
        }
        .sb-item .dot { font-size: 0.65rem; flex-shrink: 0; }
        .sb-badge-done {
            margin-left: auto;
            font-size: 0.6rem;
            color: #4ade80;
            background: rgba(34,197,94,0.12);
            border: 1px solid rgba(34,197,94,0.3);
            padding: 0.05rem 0.35rem;
            border-radius: 999px;
            white-space: nowrap;
            flex-shrink: 0;
        }

        /* Sidebar nav buttons styled to look like sb-item */
        section[data-testid="stSidebar"] div[data-testid="stButton"] {
            margin: 0.1rem 0.6rem;
        }
        section[data-testid="stSidebar"] div[data-testid="stButton"] > button {
            background: transparent !important;
            border: 1px solid transparent !important;
            color: #b6afd1 !important;
            font-weight: 500 !important;
            font-size: 0.87rem !important;
            text-align: left !important;
            justify-content: flex-start !important;
            box-shadow: none !important;
            padding: 0.55rem 0.6rem !important;
            border-radius: 9px !important;
            width: 100%;
        }
        section[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover {
            background: rgba(124,58,237,0.15) !important;
            color: #ffffff !important;
            transform: none !important;
            box-shadow: none !important;
        }
        section[data-testid="stSidebar"] div[data-testid="stButton"] > button div,
        section[data-testid="stSidebar"] div[data-testid="stButton"] > button p,
        section[data-testid="stSidebar"] div[data-testid="stButton"] > button span {
            text-align: left !important;
            justify-content: flex-start !important;
            width: 100%;
        }

        /* ---------------- Top header ---------------- */
        .top-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.3rem;
            flex-wrap: wrap;
            gap: 0.8rem;
        }
        .top-header h1 {
            font-size: 1.7rem;
            font-weight: 800;
            margin: 0;
            color: #f4f2fb;
        }
        .top-header p {
            margin: 0.15rem 0 0 0;
            color: #9a90bd;
            font-size: 0.88rem;
        }
        .badge-active {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.4rem 0.85rem;
            border-radius: 999px;
            background: rgba(34,197,94,0.12);
            border: 1px solid rgba(34,197,94,0.35);
            color: #4ade80;
            font-size: 0.8rem;
            font-weight: 700;
        }
        .badge-active::before { content: "●"; font-size: 0.6rem; }

        /* ---------------- KPI cards ---------------- */
        .kpi-card {
            background: linear-gradient(180deg, rgba(30,26,50,0.85), rgba(16,14,26,0.85));
            border: 1px solid rgba(148,120,255,0.18);
            border-radius: 16px;
            padding: 1.1rem 1.2rem;
            height: 100%;
        }
        .kpi-label { font-size: 0.8rem; color: #9a90bd; font-weight: 600; }
        .kpi-value { font-size: 1.7rem; font-weight: 800; color: #ffffff; margin: 0.2rem 0 0.35rem 0; }
        .kpi-sub { font-size: 0.72rem; color: #7fd99a; font-weight: 600; }

        /* ---------------- Pipeline status ---------------- */
        .pipe-card {
            background: linear-gradient(180deg, rgba(30,26,50,0.85), rgba(16,14,26,0.85));
            border: 1px solid rgba(148,120,255,0.18);
            border-radius: 18px;
            padding: 1.3rem 1.4rem;
            margin: 1.1rem 0;
        }
        .pipe-title { font-weight: 800; font-size: 1.05rem; color: #f4f2fb; margin-bottom: 0.15rem; }
        .pipe-sub { font-size: 0.82rem; color: #9a90bd; margin-bottom: 1rem; }
        .pipe-step { text-align: center; }
        .pipe-icon {
            width: 46px; height: 46px;
            border-radius: 12px;
            background: rgba(34,197,94,0.14);
            border: 1px solid rgba(34,197,94,0.35);
            display: flex; align-items: center; justify-content: center;
            font-size: 1.25rem;
            margin: 0 auto 0.55rem auto;
        }
        .pipe-name { font-weight: 700; font-size: 0.85rem; color: #f0edf9; }
        .pipe-desc { font-size: 0.7rem; color: #837aa3; margin: 0.15rem 0 0.4rem 0; }
        .pipe-status { font-size: 0.72rem; color: #4ade80; font-weight: 700; }

        /* ---------------- Recommendation engine card ---------------- */
        .rec-card {
            background: linear-gradient(180deg, rgba(30,26,50,0.85), rgba(16,14,26,0.85));
            border: 1px solid rgba(148,120,255,0.22);
            border-radius: 20px;
            padding: 1.3rem 1.4rem;
            margin-bottom: 0.9rem;
        }
        .rec-title { font-weight: 800; font-size: 1.1rem; color: #f4f2fb; display:flex; align-items:center; gap:0.5rem; }
        .rec-sub { font-size: 0.83rem; color: #9a90bd; margin: 0.1rem 0 1rem 0; }

        .results-wrap {
            background: linear-gradient(180deg, rgba(30,26,50,0.6), rgba(16,14,26,0.6));
            border: 1px solid rgba(148,120,255,0.18);
            border-radius: 18px;
            padding: 1rem 1.1rem 0.4rem 1.1rem;
            margin-bottom: 0.8rem;
        }
        .results-title {
            font-weight: 700;
            font-size: 0.95rem;
            color: #f4f2fb;
            margin-bottom: 0.4rem;
        }

        /* Selectbox styling */
        div[data-baseweb="select"] > div {
            background-color: rgba(20,17,34,0.9) !important;
            border: 1px solid rgba(148,120,255,0.3) !important;
            border-radius: 10px !important;
        }
        label { color: #c8c1e8 !important; font-weight: 600 !important; }

        /* Buttons (main content area only, sidebar overridden above) */
        .stButton > button {
            background: linear-gradient(90deg, #7c3aed, #a855f7);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.6rem 1.3rem;
            font-weight: 700;
            font-size: 0.9rem;
            box-shadow: 0 6px 18px rgba(124,58,237,0.35);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 26px rgba(124,58,237,0.5);
            color: white;
        }
        section[data-testid="stSidebar"] .stButton > button:hover {
            transform: none;
        }

        /* Recommended book cards (produced by recommendations_engine's st.columns) */
        [data-testid="stImage"] img {
            border-radius: 12px;
            border: 1px solid rgba(148,120,255,0.18);
            transition: transform 0.25s ease, box-shadow 0.25s ease;
        }
        [data-testid="stImage"] img:hover {
            transform: translateY(-6px) scale(1.02);
            box-shadow: 0 14px 30px rgba(124,58,237,0.35);
        }
        [data-testid="stText"] {
            font-weight: 700 !important;
            color: #f1eefc !important;
            text-align: center;
            font-size: 0.85rem !important;
        }
        [data-testid="column"] { text-align: center; }

        /* ---------------- Logs (dropdown + full page) ---------------- */
        .log-line {
            font-family: 'Courier New', monospace;
            font-size: 0.76rem;
            color: #b6afd1;
            padding: 0.32rem 0;
            border-bottom: 1px solid rgba(148,120,255,0.08);
            word-break: break-word;
        }
        .log-line .lvl-info { color: #60a5fa; font-weight: 700; }
        .log-line .lvl-warn { color: #fbbf24; font-weight: 700; }
        .log-line .lvl-error { color: #f87171; font-weight: 700; }
        .log-empty { color: #6f6690; font-size: 0.8rem; font-style: italic; }

        .log-page-card {
            background: linear-gradient(180deg, rgba(30,26,50,0.85), rgba(16,14,26,0.85));
            border: 1px solid rgba(148,120,255,0.18);
            border-radius: 18px;
            padding: 1.2rem 1.3rem;
        }

        div[data-testid="stExpander"] {
            background: linear-gradient(180deg, rgba(30,26,50,0.85), rgba(16,14,26,0.85));
            border: 1px solid rgba(148,120,255,0.18) !important;
            border-radius: 16px !important;
            margin: 0.9rem 0;
        }
        div[data-testid="stExpander"] summary {
            font-weight: 700 !important;
            color: #f1eefc !important;
        }

        /* ---------------- Footer / tech stack ---------------- */
        .footer-wrap {
            margin-top: 1.6rem;
            padding-top: 1rem;
            border-top: 1px solid rgba(148,120,255,0.12);
        }
        .footer-label {
            font-size: 0.72rem;
            color: #6f6690;
            font-weight: 700;
            letter-spacing: 0.06em;
            margin-bottom: 0.5rem;
        }
        .tech-pill {
            display: inline-block;
            padding: 0.3rem 0.7rem;
            margin: 0.2rem 0.25rem 0 0;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 600;
            background: rgba(124,58,237,0.14);
            border: 1px solid rgba(148,120,255,0.3);
            color: #c9b6ff;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


# ---------------- Sidebar navigation ----------------

def sidebar_nav_item(label, icon, key, page_id):
    is_active = st.session_state.page == page_id
    if is_active:
        st.markdown(
            f"<div class='sb-item active'><span class='sb-item-label'><span class='dot'>{icon}</span> {label}</span></div>",
            unsafe_allow_html=True
        )
    else:
        if st.button(f"{icon}   {label}", key=key, use_container_width=True):
            st.session_state.page = page_id
            st.rerun()


def render_sidebar():
    with st.sidebar:
        st.markdown(
            """
            <div class="sb-logo">
                <div class="sb-logo-icon">📚</div>
                <div>
                    <div class="sb-logo-text">Book-Reco</div>
                    <div class="sb-logo-sub">ML SYSTEM V1.0</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<div class='sb-section-title'>MAIN</div>", unsafe_allow_html=True)
        sidebar_nav_item("Dashboard", "●", "nav_dashboard", "dashboard")
        sidebar_nav_item("Recommendations", "◇", "nav_recommendations", "recommendations")

        st.markdown(
            """
            <div class="sb-section-title">PIPELINE</div>
            <div class="sb-item"><span class="sb-item-label"><span class="dot">⇩</span> Data Ingestion</span><span class="sb-badge-done">Done</span></div>
            <div class="sb-item"><span class="sb-item-label"><span class="dot">◈</span> Validation</span><span class="sb-badge-done">Done</span></div>
            <div class="sb-item"><span class="sb-item-label"><span class="dot">⇄</span> Transformation</span><span class="sb-badge-done">Done</span></div>
            <div class="sb-item"><span class="sb-item-label"><span class="dot">◆</span> Model Training</span><span class="sb-badge-done">Done</span></div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<div class='sb-section-title'>SYSTEM</div>", unsafe_allow_html=True)
        sidebar_nav_item("Logs", "▥", "nav_logs", "logs")


def render_top_header(title, subtitle, show_badge=True):
    left, right = st.columns([3, 1])
    with left:
        st.markdown(
            f"""
            <div class="top-header">
                <div>
                    <h1>{title}</h1>
                    <p>{subtitle}</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with right:
        if show_badge:
            st.markdown(
                "<div style='display:flex; justify-content:flex-end; padding-top:0.4rem;'>"
                "<span class='badge-active'>Model Active</span></div>",
                unsafe_allow_html=True
            )


# ---------------- Shared helpers ----------------

def safe_load_pickle(path):
    try:
        if path and os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
    except Exception:
        pass
    return None


def compute_kpis(recommendation_config):
    total_books = "—"
    unique_users = "—"
    total_ratings = "—"
    knn_k = "5"

    book_pivot = safe_load_pickle(getattr(recommendation_config, "book_pivot_file", None))
    final_rating = safe_load_pickle(getattr(recommendation_config, "final_rating_file", None))
    book_names = safe_load_pickle(getattr(recommendation_config, "book_names_file", None))

    try:
        if book_names is not None:
            total_books = f"{len(book_names):,}"
        elif book_pivot is not None:
            total_books = f"{book_pivot.shape[0]:,}"
    except Exception:
        pass

    try:
        if book_pivot is not None:
            unique_users = f"{book_pivot.shape[1]:,}"
    except Exception:
        pass

    try:
        if final_rating is not None:
            total_ratings = f"{len(final_rating):,}"
    except Exception:
        pass

    return total_books, unique_users, total_ratings, knn_k


def render_kpis(recommendation_config):
    total_books, unique_users, total_ratings, knn_k = compute_kpis(recommendation_config)

    cards = [
        ("Total Books", total_books, "From trained artifacts"),
        ("Unique Users", unique_users, "From ratings matrix"),
        ("Total Ratings", total_ratings, "From ratings dataset"),
        ("KNN Neighbors", f"K = {knn_k}", "Cosine similarity"),
    ]

    cols = st.columns(4)
    for col, (label, value, sub) in zip(cols, cards):
        with col:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value">{value}</div>
                    <div class="kpi-sub">{sub}</div>
                </div>
                """,
                unsafe_allow_html=True
            )


def render_pipeline_status():
    steps = [
        ("⇩", "Data Ingestion", "Books · Users · Ratings"),
        ("◈", "Validation", "Schema · Null checks"),
        ("⇄", "Transformation", "Pivot table · Sparse matrix"),
        ("◆", "Model Training", "KNN · Cosine · K=5"),
        ("✨", "Recommendation", "Streamlit UI · Covers"),
    ]

    st.markdown(
        """
        <div class="pipe-card">
            <div class="pipe-title">ML Pipeline Status</div>
            <div class="pipe-sub">Modular pipeline — each component runs independently</div>
        """,
        unsafe_allow_html=True
    )

    cols = st.columns(5)
    for col, (icon, name, desc) in zip(cols, steps):
        with col:
            st.markdown(
                f"""
                <div class="pipe-step">
                    <div class="pipe-icon">{icon}</div>
                    <div class="pipe-name">{name}</div>
                    <div class="pipe-desc">{desc}</div>
                    <div class="pipe-status">✓ Completed</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("</div>", unsafe_allow_html=True)


def find_log_file():
    try:
        for handler in logging.root.handlers:
            base_filename = getattr(handler, "baseFilename", None)
            if base_filename and os.path.exists(base_filename):
                return base_filename
    except Exception:
        pass

    for pattern in ("logs/*.log", "*.log", "logs/**/*.log"):
        try:
            matches = glob.glob(pattern, recursive=True)
            if matches:
                return max(matches, key=os.path.getmtime)
        except Exception:
            continue

    return None


def render_log_line(line):
    escaped = line.replace("<", "&lt;").replace(">", "&gt;")
    lvl_class = ""
    if "ERROR" in line:
        lvl_class = "lvl-error"
    elif "WARN" in line:
        lvl_class = "lvl-warn"
    elif "INFO" in line:
        lvl_class = "lvl-info"
    return f"<div class='log-line'><span class='{lvl_class}'>{escaped}</span></div>"


def get_recent_log_lines(n):
    log_path = find_log_file()
    if not log_path:
        return None
    try:
        with open(log_path, "r", errors="ignore") as f:
            lines = f.readlines()
        return [ln.strip() for ln in lines[-n:] if ln.strip()]
    except Exception:
        return None


def render_footer():
    pills = [
        "Python", "Pandas", "Scikit-learn", "Streamlit",
        "KNN Algorithm", "Pickle Artifacts", "Collaborative Filtering"
    ]
    st.markdown(
        "<div class='footer-wrap'>"
        "<div class='footer-label'>TECH STACK</div>"
        + "".join(f"<span class='tech-pill'>{p}</span>" for p in pills)
        + "</div>",
        unsafe_allow_html=True
    )


# ---------------- Pages ----------------

def page_dashboard(obj):
    render_top_header(
        "Book Recommendation Dashboard",
        "Collaborative Filtering · KNN Algorithm · Scikit-learn"
    )

    render_kpis(obj.recommendation_config)
    render_pipeline_status()

    with st.expander("📋  System Logs", expanded=False):
        recent = get_recent_log_lines(10)
        if recent:
            st.markdown(
                "".join(render_log_line(ln) for ln in reversed(recent)),
                unsafe_allow_html=True
            )
        elif recent is None:
            st.markdown("<div class='log-empty'>No log file found yet.</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='log-empty'>Log file is empty.</div>", unsafe_allow_html=True)

    render_footer()


def page_recommendations(obj):
    render_top_header(
        "Recommendation Engine",
        "Search a book title to get KNN-based similar recommendations"
    )

    st.markdown("<div class='rec-card'>", unsafe_allow_html=True)

    top_row = st.columns([3, 1])
    with top_row[0]:
        if st.button("🔁 Train Recommender System"):
            obj.train_engine()

    book_names = pickle.load(
        open(
            obj.recommendation_config.book_names_file,
            "rb"
        )
    )

    selected_books = st.selectbox(
        "Type or select a book",
        book_names
    )

    show_rec = st.button("✨ Show Recommendation")

    st.markdown("</div>", unsafe_allow_html=True)

    if show_rec:
        st.markdown("<div class='results-wrap'>", unsafe_allow_html=True)
        st.markdown("<div class='results-title'>Top 5 Recommended Books</div>", unsafe_allow_html=True)
        obj.recommendations_engine(selected_books)
        st.markdown("</div>", unsafe_allow_html=True)


def page_logs():
    render_top_header("System Logs", "Live tail of the application log file", show_badge=False)

    st.markdown("<div class='log-page-card'>", unsafe_allow_html=True)
    recent = get_recent_log_lines(60)
    if recent:
        st.markdown(
            "".join(render_log_line(ln) for ln in reversed(recent)),
            unsafe_allow_html=True
        )
    elif recent is None:
        st.markdown("<div class='log-empty'>No log file found yet.</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='log-empty'>Log file is empty.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":

    st.set_page_config(
        page_title="Book-Reco",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    if "page" not in st.session_state:
        st.session_state.page = "dashboard"

    inject_css()
    render_sidebar()

    obj = Recommendation()

    if st.session_state.page == "dashboard":
        page_dashboard(obj)
    elif st.session_state.page == "recommendations":
        page_recommendations(obj)
    elif st.session_state.page == "logs":
        page_logs()