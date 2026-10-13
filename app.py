"""
app.py - EmergencyNLP dashboard (Streamlit)
===========================================
Run with:   streamlit run app.py

Pages: Home | Emergency Analyzer | Analytics | Model Performance | History

This file only contains the user interface. The NLP work is done in
nlp_utils.py, the model training in train_model.py and the database code in
database.py.
"""

import html
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageDraw

from database import (count_messages, delete_all_messages,
                      get_history, init_db, save_message)
from nlp_utils import (CATEGORIES, PRIORITY_LEVELS, SAMPLE_MESSAGES,
                       EmergencyAnalyzer, entities_to_rows)
from preprocessing import spacy_available
from train_model import load_metrics, train_and_save

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(BASE_DIR, "assets", "logo.png")

# ---- Colours and icons -------------------------------------------------------
NAVY = "#0E1A2B"
RED = "#C62839"
PRIORITY_COLORS = {"Critical": "#C62839", "High": "#E4761B", "Medium": "#C99700", "Low": "#2E8B6A"}
CATEGORY_ICONS = {
    "Accident": "🚗", "Fire": "🔥", "Flood": "🌊", "Medical Emergency": "🩺",
    "Earthquake": "🏚️", "Crime": "🚨", "Rescue": "🛟", "Other": "📋",
}
PIPELINE_STEPS = [
    "Input text", "Text cleaning", "Lowercase", "Tokenization", "Stop-word removal",
    "Lemmatization", "TF-IDF features", "ML classification", "Named entity recognition",
    "Keyword extraction", "Priority detection", "Emergency response",
]


# ==========================================================================
# Setup helpers
# ==========================================================================
def ensure_logo() -> None:
    """Create assets/logo.png with Pillow if the file is missing."""
    if os.path.exists(LOGO_PATH):
        return
    try:
        os.makedirs(os.path.dirname(LOGO_PATH), exist_ok=True)
        size = 256
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.rounded_rectangle([8, 8, size - 8, size - 8], radius=56, fill=(14, 26, 43, 255))
        draw.ellipse([46, 46, size - 46, size - 46], fill=(198, 40, 57, 255))
        draw.rectangle([108, 70, 148, 186], fill=(255, 255, 255, 255))   # cross
        draw.rectangle([70, 108, 186, 148], fill=(255, 255, 255, 255))
        img.save(LOGO_PATH)
    except Exception:
        pass  # the app also works without a logo


# ---- Friendly message if the file is started with "python app.py" ----------------
from streamlit.runtime.scriptrunner import get_script_run_ctx
if get_script_run_ctx() is None:
    print("\nThis is a Streamlit app. Start it with:\n\n    streamlit run app.py\n")
    raise SystemExit(0)

ensure_logo()
st.set_page_config(
    page_title="EmergencyNLP",
    page_icon=Image.open(LOGO_PATH) if os.path.exists(LOGO_PATH) else "🚨",
    layout="wide",
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
html, body, .stApp { font-family: 'IBM Plex Sans', system-ui, sans-serif; }
.stApp { background: radial-gradient(1200px 500px at 85% -10%, #FBE3E6 0%, transparent 60%),
                     radial-gradient(900px 400px at -10% 10%, #E3ECF8 0%, transparent 55%), #F5F7FA;
         color: #14202E; }
h1, h2, h3, h4 { font-family: 'Barlow Condensed', 'IBM Plex Sans', sans-serif !important; letter-spacing: .3px; color: __NAVY__; }
.block-container { padding-top: 1.6rem; max-width: 1200px; }
header[data-testid="stHeader"] { background: transparent; }

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0B1626 0%, #13233B 60%, #2A1220 100%); }
[data-testid="stSidebar"] * { color: #E7ECF3; }
[data-testid="stSidebar"] hr { border-color: #2A3B55; }
[data-testid="stSidebar"] [role="radiogroup"] label { padding: 8px 10px; border-radius: 8px; transition: background .2s; }
[data-testid="stSidebar"] [role="radiogroup"] label:hover { background: rgba(255,255,255,.08); }

/* ---------- hero / page header ---------- */
.hero { position: relative; overflow: hidden; border-radius: 16px; padding: 28px 32px; margin-bottom: 22px;
        background: linear-gradient(120deg, #0B1626 0%, #1B2F52 55%, #8E1B29 100%);
        box-shadow: 0 14px 34px rgba(11,22,38,.28); }
.hero::before { content:""; position:absolute; right:-60px; top:-60px; width:260px; height:260px; border-radius:50%;
        background: radial-gradient(circle, rgba(255,90,105,.45), transparent 65%); }
.hero::after { content:""; position:absolute; left:0; bottom:0; height:4px; width:100%;
        background: linear-gradient(90deg, #C62839, #E4761B, #C99700, #2E8B6A); }
.hero h1 { color:#fff !important; margin:0; font-size:2.6rem; line-height:1.05; position:relative; }
.hero p { color:#C5D2E4; margin:8px 0 0 0; font-size:1.02rem; position:relative; }
.live { display:inline-flex; align-items:center; gap:8px; background:rgba(255,255,255,.12);
        border:1px solid rgba(255,255,255,.25); color:#fff; border-radius:20px; padding:3px 12px;
        font-size:.78rem; letter-spacing:1.2px; margin-bottom:10px; position:relative; }
.live i { width:9px; height:9px; border-radius:50%; background:#FF4D5E; display:inline-block;
          animation: pulse 1.4s infinite; }
@keyframes pulse { 0% { box-shadow:0 0 0 0 rgba(255,77,94,.75); } 70% { box-shadow:0 0 0 10px rgba(255,77,94,0); }
                   100% { box-shadow:0 0 0 0 rgba(255,77,94,0); } }

/* ---------- cards ---------- */
.card { background:#fff; border:1px solid #E1E6EE; border-radius:14px; padding:14px 18px; margin-bottom:14px;
        min-height:108px; box-shadow:0 2px 10px rgba(20,32,46,.06); transition: transform .18s, box-shadow .18s; }
.card:hover { transform: translateY(-3px); box-shadow:0 12px 26px rgba(20,32,46,.13); }
.card-top { display:flex; align-items:center; gap:10px; margin-bottom:6px; }
.card-icon { width:34px; height:34px; border-radius:10px; display:flex; align-items:center; justify-content:center;
             font-size:1.1rem; background:#EEF2F8; }
.card-label { color:#5B6B7F; font-size:.8rem; text-transform:uppercase; letter-spacing:.8px; font-weight:600; }
.card-value { font-family:'Barlow Condensed',sans-serif; font-weight:700; font-size:1.85rem; line-height:1.15;
              color:__NAVY__; word-break:break-word; }
.card-sub { color:#6B7A8E; font-size:.82rem; margin-top:4px; }
.bar-outer { background:#E8ECF2; border-radius:4px; height:8px; margin-top:10px; }
.bar-inner { height:8px; border-radius:4px; }

/* ---------- result banner ---------- */
.banner { display:flex; align-items:center; gap:20px; border-radius:16px; padding:18px 24px; margin:6px 0 18px 0;
          color:#fff; box-shadow:0 12px 28px rgba(0,0,0,.18); }
.banner.critical { animation: glow 1.8s infinite; }
@keyframes glow { 0%,100% { box-shadow:0 0 0 0 rgba(198,40,57,.55), 0 12px 28px rgba(0,0,0,.18); }
                  50% { box-shadow:0 0 0 12px rgba(198,40,57,0), 0 12px 28px rgba(0,0,0,.18); } }
.banner .b-icon { font-size:2.6rem; background:rgba(255,255,255,.18); width:74px; height:74px; border-radius:50%;
                  display:flex; align-items:center; justify-content:center; flex-shrink:0; }
.banner .b-badge { display:inline-block; background:rgba(255,255,255,.22); border:1px solid rgba(255,255,255,.4);
                   border-radius:20px; padding:2px 12px; font-size:.78rem; letter-spacing:1.3px; font-weight:600; }
.banner .b-title { font-family:'Barlow Condensed',sans-serif; font-weight:700; font-size:2rem; line-height:1.1; margin-top:4px; }
.banner .b-sub { opacity:.94; font-size:.95rem; margin-top:2px; }

/* ---------- chips, steps, panels ---------- */
.chip { display:inline-block; background:linear-gradient(180deg,#FFF,#FBE9EB); color:#8E1B29; border:1px solid #F1C4CA;
        border-radius:16px; padding:4px 14px; margin:0 6px 8px 0; font-size:.88rem; font-weight:500; }
.chip.blue { background:linear-gradient(180deg,#FFF,#E8F0FB); color:#1B3F7A; border-color:#C6D8F1; }
.flow { display:flex; flex-wrap:wrap; gap:6px 0; margin-bottom:8px; }
.step { background:#fff; border:1px solid #DCE2EA; border-radius:10px; padding:8px 12px; font-size:.86rem;
        position:relative; margin-right:22px; box-shadow:0 2px 6px rgba(20,32,46,.05); }
.step b { color:#C62839; margin-right:6px; }
.step::after { content:"\u279C"; position:absolute; right:-18px; top:50%; transform:translateY(-50%); color:#9AA8BA; }
.step:last-child::after { content:""; }
.panel { background:#fff; border:1px solid #E1E6EE; border-left:5px solid #C62839; border-radius:12px;
         padding:16px 20px; margin-bottom:14px; box-shadow:0 2px 10px rgba(20,32,46,.05); line-height:1.55; }
.notice { background:#FFF6E0; border:1px solid #F0D48A; border-radius:12px; padding:12px 18px; font-size:.9rem; color:#5C4600; }
.stat { background:#fff; border:1px solid #E1E6EE; border-radius:14px; padding:16px 10px; text-align:center;
        box-shadow:0 2px 10px rgba(20,32,46,.06); }
.stat .n { font-family:'Barlow Condensed',sans-serif; font-weight:700; font-size:2.4rem; color:#C62839; line-height:1; }
.stat .t { color:#5B6B7F; font-size:.82rem; margin-top:6px; text-transform:uppercase; letter-spacing:.8px; }
.feature { background:#fff; border:1px solid #E1E6EE; border-radius:14px; padding:18px 20px; height:100%;
           box-shadow:0 2px 10px rgba(20,32,46,.06); }
.feature .fi { font-size:1.8rem; }
.feature h4 { margin:.3rem 0 .3rem 0; }
.prio-row { display:flex; align-items:center; gap:10px; background:#fff; border:1px solid #E1E6EE; border-radius:10px;
            padding:8px 14px; margin-bottom:8px; }
.dot { display:inline-block; width:12px; height:12px; border-radius:50%; }

/* ---------- buttons ---------- */
.stButton > button[kind="primary"], button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #C62839, #E4761B); border: 0; color:#fff; font-weight:600;
    border-radius:10px; padding:.55rem 1.4rem; box-shadow:0 6px 16px rgba(198,40,57,.35); transition: transform .15s; }
.stButton > button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover { transform: translateY(-2px); }
.stButton > button { border-radius:10px; }
textarea { border-radius:12px !important; }
.footer { color:#8391A5; font-size:.8rem; margin-top:28px; }

/* ---------- extra polish ---------- */
::-webkit-scrollbar { width:10px; height:10px; }
::-webkit-scrollbar-track { background:#EEF1F6; }
::-webkit-scrollbar-thumb { background:linear-gradient(180deg,#C62839,#8E1B29); border-radius:6px; }
* { scrollbar-color:#C62839 #EEF1F6; }

/* glass select / text area / inputs */
.stTextArea textarea, .stSelectbox [data-baseweb="select"] > div, .stTextInput input {
    background:#fff !important; border:1.5px solid #E1E6EE !important; border-radius:12px !important;
    transition: border-color .2s, box-shadow .2s; }
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color:#C62839 !important; box-shadow:0 0 0 3px rgba(198,40,57,.12) !important; }
.stSelectbox [data-baseweb="select"]:focus-within > div { border-color:#C62839 !important; }

/* secondary buttons */
.stButton > button:not([kind="primary"]) { background:#fff; border:1.5px solid #DCE2EA; color:__NAVY__; font-weight:600; }
.stButton > button:not([kind="primary"]):hover { border-color:#C62839; color:#C62839; }

/* active sidebar radio item */
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
    background:linear-gradient(90deg, rgba(198,40,57,.35), rgba(198,40,57,0));
    border-left:3px solid #E4761B; }

/* metrics / dataframe / expander polish */
[data-testid="stDataFrame"] { border-radius:12px; overflow:hidden; border:1px solid #E1E6EE; }
.streamlit-expanderHeader { border-radius:10px !important; font-weight:600; }
[data-testid="stExpander"] { border:1px solid #E1E6EE !important; border-radius:12px !important;
    box-shadow:0 2px 8px rgba(20,32,46,.05); overflow:hidden; }

/* tabs / radio pills for future use */
.stRadio [role="radiogroup"] label { border-radius:8px; }

/* subtle fade-in for main content */
.block-container { animation: fadein .35s ease; }
@keyframes fadein { from { opacity:0; transform:translateY(6px);} to { opacity:1; transform:translateY(0);} }

/* divider glow */
hr { border-top:1px solid #E1E6EE !important; }

/* headings get a small red tick */
h2, h3 { position:relative; padding-left:14px; }
h2::before, h3::before { content:""; position:absolute; left:0; top:.25em; bottom:.25em; width:5px;
    border-radius:3px; background:linear-gradient(180deg,#C62839,#E4761B); }

/* caption + info/warning boxes */
[data-testid="stCaptionContainer"] { color:#6B7A8E; }
div[data-testid="stAlert"] { border-radius:12px; }
</style>
""".replace("__NAVY__", NAVY)
st.markdown(CSS, unsafe_allow_html=True)


# ==========================================================================
# Cached resources
# ==========================================================================
@st.cache_resource(show_spinner="Loading the NLP model (trains automatically on first run)...")
def get_analyzer() -> EmergencyAnalyzer:
    """Load (or train) the model once and reuse it for every request."""
    return EmergencyAnalyzer()


# ==========================================================================
# Small HTML helpers
# ==========================================================================
def esc(value) -> str:
    """Escape user text before it is placed inside HTML."""
    return html.escape(str(value))


def page_header(title: str, subtitle: str) -> None:
    """Gradient hero header with a pulsing LIVE badge."""
    st.markdown(f'<div class="hero"><div class="live"><i></i>LIVE &nbsp;|&nbsp; EMERGENCY INTELLIGENCE</div>'
                f'<h1>{esc(title)}</h1><p>{esc(subtitle)}</p></div>', unsafe_allow_html=True)


def card(label: str, value: str, sub: str = "", accent: str = "", bar: float = None, icon: str = "") -> None:
    """A dashboard card with an optional icon. accent = colour of the top border."""
    border = f"border-top: 5px solid {accent};" if accent else ""
    icon_html = f'<div class="card-icon">{icon}</div>' if icon else ""
    bar_html = ""
    if bar is not None:
        bar_html = (f'<div class="bar-outer"><div class="bar-inner" '
                    f'style="width:{max(2, min(bar, 100)):.0f}%; background:{accent or NAVY};"></div></div>')
    sub_html = f'<div class="card-sub">{esc(sub)}</div>' if sub else ""
    st.markdown(f'<div class="card" style="{border}"><div class="card-top">{icon_html}'
                f'<div class="card-label">{esc(label)}</div></div>'
                f'<div class="card-value">{esc(value)}</div>{sub_html}{bar_html}</div>',
                unsafe_allow_html=True)


def confidence_gauge(value: float, color: str):
    """Half-circle gauge for the confidence score."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value, number={"suffix": "%", "font": {"size": 34, "family": "Barlow Condensed"}},
        title={"text": "CONFIDENCE", "font": {"size": 13, "family": "IBM Plex Sans"}},
        gauge={"axis": {"range": [0, 100], "tickwidth": 1}, "bar": {"color": color, "thickness": 0.28},
               "steps": [{"range": [0, 40], "color": "#FBE9EB"}, {"range": [40, 70], "color": "#FFF3D6"},
                         {"range": [70, 100], "color": "#E2F4EC"}], "borderwidth": 0}))
    fig.update_layout(height=190, margin=dict(l=25, r=25, t=45, b=0), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def style_fig(fig):
    """Common look for all Plotly charts."""
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font_family="IBM Plex Sans",
                      title_font_family="Barlow Condensed", title_font_size=20)
    return fig


# ==========================================================================
# Page: Home
# ==========================================================================
def page_home() -> None:
    page_header("EmergencyNLP", "Real-Time Emergency Message Intelligence System")

    st.markdown(
        "EmergencyNLP reads an emergency message written in plain English and turns it into a "
        "structured response ticket: **what** happened, **how urgent** it is, **where**, "
        "**how many people** are affected and **which service** should be sent. "
        "Everything runs locally with Python, spaCy and scikit-learn - no paid API is needed.")

    s1, s2, s3, s4 = st.columns(4)
    for col, num, text in [(s1, "8", "Emergency categories"), (s2, "4", "Priority levels"),
                           (s3, "12", "Pipeline steps"), (s4, "100%", "Offline & free")]:
        col.markdown(f'<div class="stat"><div class="n">{num}</div><div class="t">{text}</div></div>',
                     unsafe_allow_html=True)

    st.subheader("How the system works")
    steps = "".join(f'<span class="step"><b>{i}</b>{esc(name)}</span>' for i, name in enumerate(PIPELINE_STEPS, 1))
    st.markdown(f'<div class="flow">{steps}</div>', unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3)
    for col, icon, title, text in [
        (f1, "🧹", "1. Understand the text", "The message is cleaned, lowercased, tokenized, stripped of stop words "
                                             "and lemmatized (injured becomes injure)."),
        (f2, "🤖", "2. Classify with ML", "TF-IDF turns words into numbers and Logistic Regression picks one of "
                                          "eight emergency categories with a confidence score."),
        (f3, "📍", "3. Extract the details", "spaCy finds locations and entities. Rules find people affected, "
                                             "the required service and a priority score.")]:
        col.markdown(f'<div class="feature"><div class="fi">{icon}</div><h4>{title}</h4><div>{text}</div></div>',
                     unsafe_allow_html=True)

    st.markdown("")
    left, right = st.columns(2)
    with left:
        st.subheader("Emergency categories")
        st.markdown("".join(f'<span class="chip">{CATEGORY_ICONS[c]} {esc(c)}</span>' for c in CATEGORIES),
                    unsafe_allow_html=True)
    with right:
        st.subheader("Priority levels")
        st.markdown("".join(
            f'<div class="prio-row"><span class="dot" style="background:{PRIORITY_COLORS[p]}"></span>'
            f'<b>{p}</b></div>' for p in PRIORITY_LEVELS), unsafe_allow_html=True)

    st.markdown('<div class="notice"><b>Academic prototype.</b> This project is for learning and '
                'demonstration. In a real emergency, call your local emergency number '
                '(112 in India) immediately.</div>', unsafe_allow_html=True)
    st.info("Open **Emergency Analyzer** in the sidebar and try one of the sample messages.")


# ==========================================================================
# Page: Emergency Analyzer
# ==========================================================================
SAMPLE_PLACEHOLDER = "Choose a sample message..."


def _load_sample() -> None:
    """Callback: copy the chosen sample message into the text box."""
    choice = st.session_state.get("sample_choice")
    if choice in SAMPLE_MESSAGES:
        st.session_state["message_input"] = SAMPLE_MESSAGES[choice]


def _clear_message() -> None:
    st.session_state["message_input"] = ""
    st.session_state["sample_choice"] = SAMPLE_PLACEHOLDER
    st.session_state["last_result"] = None


def render_result(result: dict) -> None:
    """Show the complete analysis: banner, six cards, keywords, details."""
    priority = result["priority"]
    color = PRIORITY_COLORS.get(priority, NAVY)
    icon = CATEGORY_ICONS.get(result["emergency_type"], "📋")

    css_class = "banner critical" if priority == "Critical" else "banner"
    st.markdown(
        f'<div class="{css_class}" style="background:linear-gradient(120deg,{color},{NAVY})">'
        f'<div class="b-icon">{icon}</div><div>'
        f'<span class="b-badge">{esc(priority.upper())} PRIORITY</span>'
        f'<div class="b-title">{esc(result["emergency_type"])}</div>'
        f'<div class="b-sub">📍 {esc(result["location"])} &nbsp;|&nbsp; 🚑 {esc(result["required_service"])}'
        f' &nbsp;|&nbsp; 🕒 {esc(result["timestamp"])}</div></div></div>', unsafe_allow_html=True)

    for warning in result.get("warnings", []):
        st.warning(warning)

    c1, c2, c3 = st.columns(3)
    with c1:
        card("Emergency type", result["emergency_type"], "Predicted by the ML model", icon=icon)
    with c2:
        card("Priority", priority, f'Score {result["priority_score"]}', accent=color, icon="⚠️")
    with c3:
        card("Location", result["location"], f'Found by: {result["location_method"]}', icon="📍")
    c4, c5, c6 = st.columns(3)
    with c4:
        card("People affected", str(result["people_affected"]), icon="👥")
    with c5:
        card("Required service", result["required_service"], icon="🚑")
    with c6:
        st.plotly_chart(confidence_gauge(result["confidence"], color))

    st.markdown("##### Extracted keywords")
    if result["keywords"]:
        st.markdown("".join(f'<span class="chip">{esc(k)}</span>' for k in result["keywords"]),
                    unsafe_allow_html=True)
    else:
        st.caption("No keywords found.")

    left, right = st.columns(2)
    with left:
        st.markdown("##### Recommended response")
        st.markdown(f'<div class="panel">{esc(result["recommended_response"])}</div>', unsafe_allow_html=True)
        with st.expander("Why this priority?"):
            for reason in result["priority_reasons"]:
                st.write(f"- {reason}")
            st.write(f"**Total score: {result['priority_score']}**  "
                     "(Critical >= 14, High >= 5, Medium >= 2, otherwise Low)")
    with right:
        st.markdown("##### NLP entities (spaCy NER)")
        rows = entities_to_rows(result["entities"])
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True)
        else:
            st.caption("No named entities were found in this message.")
        if not spacy_available():
            st.caption("The spaCy model is not installed, so a simple fallback was used.")

    with st.expander("See every NLP pipeline step"):
        steps = result["pipeline_steps"]
        for name, value in steps.items():
            st.write(f"**{name}:** `{value}`")
    if result["probabilities"]:
        with st.expander("Category probabilities"):
            prob = (pd.DataFrame({"Category": list(result["probabilities"]),
                                  "Probability (%)": [round(p * 100, 1) for p in result["probabilities"].values()]})
                    .sort_values("Probability (%)"))
            fig = px.bar(prob, x="Probability (%)", y="Category", orientation="h",
                         color_discrete_sequence=[NAVY])
            st.plotly_chart(style_fig(fig))


def page_analyzer() -> None:
    page_header("Emergency Analyzer", "Paste an emergency message and get an instant response ticket")
    analyzer = get_analyzer()

    st.session_state.setdefault("message_input", "")
    st.session_state.setdefault("last_result", None)
    st.session_state.setdefault("sample_choice", SAMPLE_PLACEHOLDER)

    st.selectbox("Try a sample message", [SAMPLE_PLACEHOLDER] + list(SAMPLE_MESSAGES),
                 key="sample_choice", on_change=_load_sample)
    st.text_area("Emergency message", key="message_input", height=130,
                 placeholder="Example: There is a major accident near the railway station. "
                             "Two people are seriously injured. Please send an ambulance immediately.")

    col_a, col_b, _ = st.columns([1, 1, 4])
    analyze = col_a.button("Analyze message", type="primary")
    col_b.button("Clear", on_click=_clear_message)

    if analyze:
        result = analyzer.analyze(st.session_state["message_input"])
        if result.get("success"):
            if save_message(result) is None:
                st.warning("The analysis worked, but it could not be saved to the database.")
            st.session_state["last_result"] = result
        else:
            st.session_state["last_result"] = None
            st.error(result.get("error", "The message could not be analysed."))

    if st.session_state.get("last_result"):
        render_result(st.session_state["last_result"])

    st.subheader("Recent analysis history")
    history = get_history(limit=5)
    if history.empty:
        st.caption("No messages analysed yet.")
    else:
        st.dataframe(history[["timestamp", "message", "emergency_type", "priority", "location",
                              "required_service"]], hide_index=True)


# ==========================================================================
# Page: Analytics
# ==========================================================================
def _add_demo_messages() -> None:
    """Analyse the sample messages so the charts have data (for demos)."""
    analyzer = get_analyzer()
    for message in SAMPLE_MESSAGES.values():
        result = analyzer.analyze(message)
        if result.get("success"):
            save_message(result)


def page_analytics() -> None:
    page_header("Analytics", "What kinds of emergencies have been analysed so far?")
    df = get_history()
    if df.empty:
        st.info("No data yet. Analyze some messages first, or load the sample messages below.")
        if st.button("Analyze the 10 sample messages"):
            with st.spinner("Analyzing..."):
                _add_demo_messages()
            st.rerun()
        return

    critical_share = (df["priority"].isin(["Critical", "High"])).mean() * 100
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        card("Messages analysed", str(len(df)))
    with c2:
        card("Critical / High", f"{critical_share:.0f}%", "Share of all messages")
    with c3:
        card("Most common type", df["emergency_type"].mode().iloc[0])
    with c4:
        card("Average confidence", f'{df["confidence"].mean():.1f}%')

    left, right = st.columns(2)
    with left:
        counts = df["emergency_type"].value_counts().reset_index()
        counts.columns = ["Category", "Messages"]
        fig = px.bar(counts, x="Messages", y="Category", orientation="h",
                     title="Emergency category distribution", color_discrete_sequence=[NAVY])
        fig.update_yaxes(categoryorder="total ascending")
        st.plotly_chart(style_fig(fig))
    with right:
        counts = df["priority"].value_counts().reindex(PRIORITY_LEVELS).fillna(0).reset_index()
        counts.columns = ["Priority", "Messages"]
        fig = px.pie(counts, names="Priority", values="Messages", hole=0.55, title="Priority distribution",
                     color="Priority", color_discrete_map=PRIORITY_COLORS, category_orders={"Priority": PRIORITY_LEVELS})
        st.plotly_chart(style_fig(fig))

    left, right = st.columns(2)
    with left:
        cross = pd.crosstab(df["emergency_type"], df["priority"]).reindex(columns=PRIORITY_LEVELS, fill_value=0)
        cross = cross.reset_index().melt(id_vars="emergency_type", var_name="Priority", value_name="Messages")
        fig = px.bar(cross, x="emergency_type", y="Messages", color="Priority", title="Priority within each category",
                     color_discrete_map=PRIORITY_COLORS, category_orders={"Priority": PRIORITY_LEVELS})
        fig.update_xaxes(title="")
        st.plotly_chart(style_fig(fig))
    with right:
        df["date"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.date
        per_day = df.groupby("date").size().reset_index(name="Messages")
        fig = px.bar(per_day, x="date", y="Messages", title="Messages per day", color_discrete_sequence=[RED])
        fig.update_xaxes(title="")
        st.plotly_chart(style_fig(fig))


# ==========================================================================
# Page: Model Performance
# ==========================================================================
def _heatmap(matrix, labels, title):
    fig = px.imshow(matrix, x=labels, y=labels, text_auto=True, aspect="auto", title=title,
                    color_continuous_scale=[[0, "#F3F5F8"], [1, RED]],
                    labels=dict(x="Predicted", y="Actual", color="Messages"))
    fig.update_layout(coloraxis_showscale=False)
    return style_fig(fig)


def page_model() -> None:
    page_header("Model Performance", "How well does the emergency-category classifier work?")
    metrics = load_metrics()

    if metrics is None:
        st.warning("No saved metrics were found. Click the button below to train the model.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            card("Accuracy", f'{metrics["accuracy"] * 100:.1f}%', "On the unseen test set")
        with c2:
            card("Precision", f'{metrics["precision"] * 100:.1f}%', "Weighted average")
        with c3:
            card("Recall", f'{metrics["recall"] * 100:.1f}%', "Weighted average")
        with c4:
            card("F1-score", f'{metrics["f1_score"] * 100:.1f}%', "Weighted average")

        st.caption(
            f'Algorithm: TF-IDF + Logistic Regression | {metrics["dataset_size"]} messages '
            f'({metrics["train_size"]} train / {metrics["test_size"]} test) | '
            f'{metrics["num_features"]} features | split: {metrics.get("split_strategy", "random")} | '
            f'5-fold CV accuracy: {metrics["cv_accuracy_mean"] * 100:.1f}% '
            f'(+/- {metrics["cv_accuracy_std"] * 100:.1f}) | trained {metrics["trained_at"]}')

        left, right = st.columns([3, 2])
        with left:
            st.plotly_chart(_heatmap(metrics["confusion_matrix"], metrics["labels"], "Confusion matrix (test set)"))
        with right:
            st.markdown("##### Per-category results")
            report = metrics["classification_report"]
            table = pd.DataFrame({label: report[label] for label in metrics["labels"]}).T
            table = table[["precision", "recall", "f1-score", "support"]].round(2)
            table["support"] = table["support"].astype(int)
            st.dataframe(table)

        st.markdown("##### Priority detection (rule based)")
        pr = metrics.get("priority_rules")
        if pr:
            left, right = st.columns([2, 3])
            with left:
                card("Agreement with dataset labels", f'{pr["accuracy"] * 100:.1f}%',
                     f'{pr["samples"]} messages, true categories used')
                st.caption("The priority scorer is not trained; this only shows how often its result "
                           "matches the priority column of the dataset.")
            with right:
                st.plotly_chart(_heatmap(pr["confusion_matrix"], pr["labels"], "Priority confusion matrix"))

        st.info("The sample dataset is small and was written from message templates, so treat these "
                "numbers as a demonstration. The test set only contains templates the model never saw "
                "during training. Real messages will be harder - see the README for how to improve this.")

    if st.button("Retrain the model now"):
        with st.spinner("Training..."):
            try:
                train_and_save(verbose=False)
                st.cache_resource.clear()
                st.success("Training finished.")
                st.rerun()
            except Exception as exc:
                st.error(f"Training failed: {exc}")


# ==========================================================================
# Page: History
# ==========================================================================
def page_history() -> None:
    page_header("History", "Every analysed message is stored in the SQLite database")
    df = get_history()
    if df.empty:
        st.info("The history is empty. Analyze a message first.")
        return

    f1, f2, f3 = st.columns([2, 2, 3])
    types = f1.multiselect("Emergency type", sorted(df["emergency_type"].unique()))
    priorities = f2.multiselect("Priority", [p for p in PRIORITY_LEVELS if p in set(df["priority"])])
    search = f3.text_input("Search in messages or locations")

    view = df
    if types:
        view = view[view["emergency_type"].isin(types)]
    if priorities:
        view = view[view["priority"].isin(priorities)]
    if search.strip():
        needle = search.strip().lower()
        view = view[view["message"].str.lower().str.contains(needle, regex=False)
                    | view["location"].str.lower().str.contains(needle, regex=False)]

    st.caption(f"Showing {len(view)} of {len(df)} messages")
    st.dataframe(view[["id", "timestamp", "emergency_type", "priority", "location", "people_affected",
                       "required_service", "confidence", "message"]], hide_index=True)
    st.download_button("Download as CSV", view.to_csv(index=False).encode("utf-8"),
                       file_name="emergency_history.csv", mime="text/csv")

    with st.expander("Delete history"):
        confirm = st.checkbox("I understand that this deletes all saved messages")
        if st.button("Delete all messages", disabled=not confirm):
            if delete_all_messages():
                st.success("History deleted.")
                st.rerun()
            else:
                st.error("The history could not be deleted.")


# ==========================================================================
# Main: sidebar navigation
# ==========================================================================
def main() -> None:
    if not init_db():
        st.sidebar.error("The database could not be created. Check the folder permissions.")

    with st.sidebar:
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, width=84)
        st.markdown("### EmergencyNLP")
        st.caption("Real-Time Emergency Message Intelligence System")
        st.divider()
        page = st.radio("Navigate", ["Home", "Emergency Analyzer", "Analytics", "Model Performance", "History"],
                        label_visibility="collapsed")
        st.divider()
        st.caption(f"Messages analysed: {count_messages()}")
        st.caption("Developed by Janani")

    try:
        {"Home": page_home, "Emergency Analyzer": page_analyzer, "Analytics": page_analytics,
         "Model Performance": page_model, "History": page_history}[page]()
    except FileNotFoundError as exc:
        st.error(str(exc))
    except Exception as exc:  # last safety net: show a message instead of a crash
        st.error(f"Something went wrong: {exc}")


main()
