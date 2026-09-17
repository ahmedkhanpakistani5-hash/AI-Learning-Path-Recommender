import os
import io
import numpy as np
import pandas as pd
import streamlit as st

# Optional AI layer
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

from sklearn.preprocessing import MinMaxScaler


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="LearnPath AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# THEME / CSS
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 78% 3%, rgba(255, 91, 38, .12), transparent 25%),
        radial-gradient(circle at 30% 30%, rgba(94, 53, 177, .08), transparent 28%),
        #060d1b;
    color: #eef4ff;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #071120 0%, #081426 55%, #050b16 100%);
    border-right: 1px solid rgba(125, 151, 190, .18);
}

[data-testid="stSidebar"] * {
    color: #dce8fb;
}

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}

.hero {
    min-height: 150px;
    border-radius: 24px;
    padding: 28px 34px;
    position: relative;
    overflow: hidden;
    background:
        radial-gradient(circle at 85% 30%, rgba(255, 170, 55, .35), transparent 24%),
        radial-gradient(circle at 60% 85%, rgba(255, 56, 102, .28), transparent 35%),
        linear-gradient(110deg, #9f092d 0%, #dc263b 34%, #ff6929 72%, #ff9c2e 100%);
    box-shadow: 0 18px 45px rgba(255, 74, 42, .16);
    border: 1px solid rgba(255,255,255,.10);
}

.hero:after {
    content: "";
    position: absolute;
    width: 520px;
    height: 520px;
    right: -180px;
    top: -290px;
    border-radius: 50%;
    background: rgba(255,255,255,.10);
}

.hero-title {
    font-size: 36px;
    font-weight: 800;
    letter-spacing: -1px;
    position: relative;
    z-index: 2;
}

.hero-title span {
    color: #ffd22e;
}

.hero-subtitle {
    font-size: 15px;
    color: rgba(255,255,255,.90);
    margin-top: 5px;
    position: relative;
    z-index: 2;
}

.hero-small {
    font-size: 13px;
    color: rgba(255,255,255,.72);
    margin-top: 13px;
    position: relative;
    z-index: 2;
}

.section-title {
    font-size: 20px;
    font-weight: 800;
    margin: 20px 0 6px 0;
    color: #f2f7ff;
}

.section-caption {
    color: #7fa5d1;
    font-size: 12px;
    margin-bottom: 12px;
}

.metric-card {
    background: linear-gradient(145deg, rgba(16,29,52,.95), rgba(8,18,34,.96));
    border: 1px solid rgba(116, 143, 183, .14);
    border-radius: 16px;
    padding: 17px 18px;
    min-height: 100px;
    box-shadow: 0 10px 28px rgba(0,0,0,.18);
}

.metric-icon {
    font-size: 22px;
}

.metric-value {
    font-size: 26px;
    font-weight: 800;
    color: #f5f8ff;
    margin-top: 3px;
}

.metric-label {
    font-size: 11px;
    color: #86a8cf;
    margin-top: 2px;
}

.glass {
    background: linear-gradient(145deg, rgba(14,27,49,.94), rgba(7,16,30,.95));
    border: 1px solid rgba(115, 145, 188, .14);
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0 12px 35px rgba(0,0,0,.16);
}

.rec-card {
    border-radius: 16px;
    padding: 17px 20px;
    margin: 9px 0;
    background: linear-gradient(105deg, rgba(30,18,43,.96), rgba(28,19,30,.92));
    border: 1px solid rgba(255, 84, 78, .55);
    box-shadow: 0 8px 22px rgba(0,0,0,.16);
}

.rec-card:nth-child(3n) {
    border-color: rgba(255, 176, 33, .65);
    background: linear-gradient(105deg, rgba(38,29,20,.96), rgba(27,25,17,.94));
}

.rec-card:nth-child(4n) {
    border-color: rgba(255, 104, 45, .62);
}

.rec-row {
    display: flex;
    align-items: center;
    gap: 15px;
}

.course-icon {
    width: 48px;
    height: 48px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(145deg, #263b5c, #12243d);
    font-size: 24px;
    flex-shrink: 0;
}

.course-name {
    font-size: 16px;
    font-weight: 800;
    color: #f2f7ff;
}

.course-meta {
    font-size: 11px;
    color: #8eb0d6;
    margin-top: 4px;
}

.badge {
    display: inline-block;
    padding: 5px 9px;
    border-radius: 9px;
    background: linear-gradient(90deg, #ed173f, #ff7040);
    color: white;
    font-size: 10px;
    font-weight: 800;
    margin-right: 7px;
}

.score {
    margin-left: auto;
    text-align: right;
}

.score-number {
    font-size: 19px;
    font-weight: 800;
    color: #fff;
}

.score-label {
    color: #8b8fa2;
    font-size: 9px;
}

.progress-shell {
    height: 7px;
    width: 100%;
    background: rgba(255,255,255,.08);
    border-radius: 20px;
    overflow: hidden;
    margin-top: 9px;
}

.progress-fill {
    height: 100%;
    background: linear-gradient(90deg, #ff3d58, #ff9b29);
    border-radius: 20px;
}

.ai-box {
    background:
        radial-gradient(circle at 100% 0%, rgba(120, 78, 255, .20), transparent 35%),
        linear-gradient(145deg, rgba(23,20,49,.96), rgba(9,16,31,.97));
    border: 1px solid rgba(134, 103, 255, .34);
    border-radius: 18px;
    padding: 20px;
    box-shadow: 0 14px 38px rgba(51, 30, 128, .12);
}

.ai-title {
    font-size: 18px;
    font-weight: 800;
    color: #f2edff;
}

.ai-caption {
    font-size: 12px;
    color: #a6a3cf;
}

.pill {
    display: inline-block;
    border: 1px solid rgba(120, 161, 211, .22);
    color: #9bc4ef;
    padding: 5px 9px;
    border-radius: 999px;
    font-size: 10px;
    margin: 3px;
    background: rgba(22, 42, 70, .55);
}

div[data-testid="stButton"] > button {
    border-radius: 11px;
    border: 1px solid rgba(255,255,255,.10);
    background: linear-gradient(90deg, #ef2946, #ff772c);
    color: white;
    font-weight: 700;
    min-height: 42px;
}

div[data-testid="stButton"] > button:hover {
    border-color: rgba(255,255,255,.35);
    box-shadow: 0 8px 22px rgba(255,80,50,.20);
}

div[data-testid="stDownloadButton"] > button {
    border-radius: 11px;
    font-weight: 700;
}

[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: rgba(15,27,47,.78);
    border-radius: 12px;
    padding: 5px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 9px;
    color: #91a9c8;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, rgba(255,77,70,.30), rgba(255,135,43,.18));
    color: #fff !important;
}

div[data-testid="stMetric"] {
    background: rgba(12,23,41,.82);
    border: 1px solid rgba(125,151,190,.12);
    border-radius: 14px;
    padding: 12px;
}

hr {
    border-color: rgba(118, 143, 180, .12);
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# DATA
# ============================================================
COURSES = pd.DataFrame([
    ["Python Fundamentals", "Python", "Beginner", 4, "🐍"],
    ["Advanced Python", "Python", "Intermediate", 6, "🐍"],
    ["SQL Foundations", "Database", "Beginner", 4, "🗄️"],
    ["Advanced SQL", "Database", "Advanced", 6, "🗄️"],
    ["Machine Learning Basics", "Machine Learning", "Intermediate", 6, "🧠"],
    ["Applied Machine Learning", "Machine Learning", "Advanced", 8, "🤖"],
    ["Data Structures & Algorithms", "Computer Science", "Intermediate", 6, "🌳"],
    ["Git & GitHub", "Developer Tools", "Beginner", 3, "🔗"],
    ["REST API Development", "Backend", "Intermediate", 5, "⚡"],
    ["Streamlit App Development", "AI Apps", "Intermediate", 4, "🚀"],
    ["Prompt Engineering", "Generative AI", "Intermediate", 3, "✨"],
    ["LLM Application Development", "Generative AI", "Advanced", 7, "🧩"],
    ["Computer Vision", "AI", "Advanced", 7, "👁️"],
    ["Cloud Fundamentals", "Cloud", "Beginner", 4, "☁️"],
    ["AWS for Developers", "Cloud", "Intermediate", 6, "☁️"],
    ["Data Visualization", "Data Science", "Intermediate", 4, "📊"],
    ["Statistics for ML", "Data Science", "Intermediate", 5, "📐"],
    ["Deep Learning", "AI", "Advanced", 8, "🔥"],
    ["NLP Fundamentals", "AI", "Intermediate", 6, "💬"],
    ["MLOps Foundations", "MLOps", "Advanced", 7, "⚙️"],
], columns=["course", "category", "level", "weeks", "icon"])


def demo_interactions():
    """Synthetic demo data so the app works immediately on Streamlit Cloud."""
    rng = np.random.default_rng(42)
    interns = [f"Intern {i:02d}" for i in range(1, 13)]
    rows = []

    # Each intern has a slightly different learning profile.
    preferences = {
        "Intern 01": ["Python", "AI", "Generative AI"],
        "Intern 02": ["Database", "Backend", "Cloud"],
        "Intern 03": ["Machine Learning", "Data Science", "Python"],
        "Intern 04": ["Developer Tools", "Backend", "AI Apps"],
        "Intern 05": ["AI", "Machine Learning", "Data Science"],
        "Intern 06": ["Cloud", "Backend", "Database"],
        "Intern 07": ["Python", "Generative AI", "AI"],
        "Intern 08": ["Computer Science", "Python", "Backend"],
        "Intern 09": ["Data Science", "Machine Learning", "AI"],
        "Intern 10": ["Cloud", "Developer Tools", "MLOps"],
        "Intern 11": ["Generative AI", "AI Apps", "Python"],
        "Intern 12": ["Database", "Data Science", "Computer Science"],
    }

    for intern in interns:
        liked = preferences[intern]
        for _, c in COURSES.iterrows():
            base = 1.7
            if c["category"] in liked:
                base += 1.6
            if c["level"] == "Intermediate":
                base += 0.25
            rating = np.clip(base + rng.normal(0, 0.75), 1, 5)
            completed = int(rng.random() > 0.34)
            engagement = np.clip(rating + rng.normal(0, .45), 1, 5)
            rows.append([
                intern, c["course"], round(float(rating), 2),
                round(float(engagement), 2), completed
            ])

    return pd.DataFrame(
        rows,
        columns=["intern", "course", "rating", "engagement", "completed"]
    )


def make_interaction_matrix(df):
    matrix = df.pivot_table(
        index="intern",
        columns="course",
        values="rating",
        aggfunc="mean"
    ).fillna(0)
    return matrix


# ============================================================
# MATRIX FACTORIZATION
# ============================================================
def matrix_factorization(R, factors=4, steps=650, lr=0.008, reg=0.025):
    """
    Lightweight SGD matrix factorization.
    This avoids extra recommender-system packages and works well on Streamlit Cloud.
    """
    R = np.asarray(R, dtype=float)
    mask = R > 0

    rng = np.random.default_rng(7)
    P = rng.normal(0, 0.12, (R.shape[0], factors))
    Q = rng.normal(0, 0.12, (R.shape[1], factors))

    rows, cols = np.where(mask)

    for _ in range(steps):
        order = rng.permutation(len(rows))
        for idx in order:
            i, j = rows[idx], cols[idx]
            prediction = float(np.dot(P[i], Q[j]))
            error = R[i, j] - prediction

            P[i] += lr * (error * Q[j] - reg * P[i])
            Q[j] += lr * (error * P[i] - reg * Q[j])

    predicted = np.dot(P, Q.T)
    return predicted


def recommend_for_intern(interactions, courses, intern, top_n=5):
    matrix = make_interaction_matrix(interactions)
    predicted = matrix_factorization(matrix.values)

    if intern not in matrix.index:
        return pd.DataFrame()

    i = list(matrix.index).index(intern)
    scores = predicted[i]

    # Prefer modules the intern has not completed.
    history = interactions[interactions["intern"] == intern]
    completed = set(history.loc[history["completed"] == 1, "course"].tolist())

    recs = pd.DataFrame({
        "course": matrix.columns,
        "score": scores
    })

    recs = recs[~recs["course"].isin(completed)].copy()

    if recs.empty:
        recs = pd.DataFrame({"course": matrix.columns, "score": scores})

    # Convert raw MF scores into an attractive 70-98 recommendation percentage.
    scaler = MinMaxScaler(feature_range=(72, 98))
    recs["recommendation"] = scaler.fit_transform(
        recs[["score"]]
    ).ravel()

    recs = recs.merge(courses, on="course", how="left")
    recs = recs.sort_values("recommendation", ascending=False).head(top_n)
    recs["recommendation"] = recs["recommendation"].round(0).astype(int)

    return recs


# ============================================================
# GROQ
# ============================================================
def get_groq_client(api_key=None):
    if not GROQ_AVAILABLE:
        return None

    key = api_key or os.getenv("GROQ_API_KEY")
    if not key:
        try:
            key = st.secrets.get("GROQ_API_KEY", "")
        except Exception:
            key = ""

    if not key:
        return None

    return Groq(api_key=key)


def ask_groq(prompt, api_key=None):
    client = get_groq_client(api_key)
    if client is None:
        return (
            "Groq is not connected yet. Add your GROQ_API_KEY in Streamlit "
            "Secrets or the environment, then try again."
        )

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are LearnPath AI, an expert learning advisor for interns. "
                        "Give concise, practical, personalized learning guidance. "
                        "Use the provided recommendation data; do not invent learner "
                        "history. Format answers with short headings and bullets when useful."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.45,
            max_tokens=900,
        )
        return response.choices[0].message.content
    except Exception as e:
        return (
            f"AI request could not be completed: {type(e).__name__}. "
            "Check that GROQ_API_KEY is valid and that the Groq model is available "
            "for your account."
        )


# ============================================================
# SESSION STATE
# ============================================================
if "page" not in st.session_state:
    st.session_state.page = "Home"

if "interactions" not in st.session_state:
    st.session_state.interactions = demo_interactions()

if "selected_intern" not in st.session_state:
    st.session_state.selected_intern = "Intern 01"

if "uploaded_name" not in st.session_state:
    st.session_state.uploaded_name = "Demo learning history"


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style="padding:10px 5px 18px 5px;">
        <div style="font-size:28px;font-weight:800;">
            🎓 <span style="color:#f6f8ff;">LearnPath</span>
            <span style="color:#ff8b2b;">AI</span>
        </div>
        <div style="font-size:11px;color:#78a1ce;margin-left:38px;">
            Personalized Learning Paths
        </div>
    </div>
    """, unsafe_allow_html=True)

    pages = {
        "🏠  Home": "Home",
        "👤  Intern Profile": "Intern Profile",
        "📊  Learning Analytics": "Learning Analytics",
        "🧠  AI Advisor": "AI Advisor",
        "": "",
        "🔗  AI Pipeline": "AI Pipeline",
        "①  Learning history": "Learning history",
        "②  Interaction matrix": "Interaction matrix",
        "③  Matrix Factorization": "Matrix Factorization",
        "④  Course prediction": "Course prediction",
        "⑤  Personalized path": "Personalized path",
        "⑥  Groq AI explanation": "Groq AI explanation",
    }

    for label, value in pages.items():
        if not value:
            st.markdown("<hr>", unsafe_allow_html=True)
            continue
        if st.button(label, use_container_width=True, key=f"nav_{value}"):
            st.session_state.page = value
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        "<div style='font-size:10px;color:#6487ae;'>MODEL</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='pill'>Matrix Factorization</div>"
        "<div class='pill'>Collaborative Filtering</div>"
        "<div class='pill'>Groq AI</div>",
        unsafe_allow_html=True,
    )


# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="hero">
    <div class="hero-title">🎓 LearnPath <span>AI</span></div>
    <div class="hero-subtitle">
        Personalized learning paths powered by <b>Matrix Factorization + Generative AI</b>
    </div>
    <div class="hero-small">
        Turn previous intern learning behavior into a customized training journey.
    </div>
</div>
""", unsafe_allow_html=True)

st.write("")


# ============================================================
# SELECTOR
# ============================================================
left, right = st.columns([2.5, 1])
with left:
    interns = sorted(st.session_state.interactions["intern"].unique().tolist())
    selected = st.selectbox(
        "Active intern profile",
        interns,
        index=interns.index(st.session_state.selected_intern)
        if st.session_state.selected_intern in interns else 0,
    )
    st.session_state.selected_intern = selected

with right:
    uploaded = st.file_uploader(
        "Optional learning-history CSV",
        type=["csv"],
        help=(
            "Columns supported: intern, course, rating, engagement, completed. "
            "Extra columns are ignored."
        ),
    )

if uploaded is not None:
    try:
        new_df = pd.read_csv(uploaded)
        required = {"intern", "course", "rating", "engagement", "completed"}
        if required.issubset(new_df.columns):
            st.session_state.interactions = new_df.copy()
            st.session_state.uploaded_name = uploaded.name
            st.success(f"Loaded {uploaded.name}")
        else:
            st.error(
                "CSV needs these columns: intern, course, rating, engagement, completed"
            )
    except Exception as e:
        st.error(f"Could not read CSV: {e}")


interactions = st.session_state.interactions
active = st.session_state.selected_intern
history = interactions[interactions["intern"] == active].copy()
recommendations = recommend_for_intern(interactions, COURSES, active, top_n=6)


# ============================================================
# METRICS
# ============================================================
avg_engagement = history["engagement"].mean() if not history.empty else 0
completed_count = int(history["completed"].sum()) if not history.empty else 0
categories = int(
    history.merge(COURSES[["course", "category"]], on="course", how="left")["category"]
    .nunique()
) if not history.empty else 0

metric_cols = st.columns(4)

metrics = [
    ("📚", len(history), "Learning Interactions"),
    ("⭐", f"{avg_engagement:.1f}/5", "Average Engagement"),
    ("🧩", categories, "Skill Categories"),
    ("🗄️", len(COURSES), "Available Modules"),
]

for col, (icon, value, label) in zip(metric_cols, metrics):
    with col:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-icon">{icon}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-label">{label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# HOME
# ============================================================
if st.session_state.page == "Home":
    st.markdown(
        '<div class="section-title">✨ Your Personalized Learning Path</div>'
        '<div class="section-caption">'
        'Recommendations are generated from historical learning behavior using Matrix Factorization.'
        '</div>',
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs([
        "✨ Recommended Path",
        "📊 Learning Analytics",
        "🤖 AI Learning Advisor",
    ])

    with tab1:
        if recommendations.empty:
            st.warning("Not enough data to generate recommendations.")
        else:
            for rank, (_, row) in enumerate(recommendations.iterrows(), start=1):
                score = int(row["recommendation"])
                st.markdown(
                    f"""
                    <div class="rec-card">
                        <div class="rec-row">
                            <div class="course-icon">{row['icon']}</div>
                            <div>
                                <span class="badge">#{rank} RECOMMENDED</span>
                                <div class="course-name">{row['course']}</div>
                                <div class="course-meta">
                                    {row['category']} &nbsp;|&nbsp;
                                    {row['level']} &nbsp;|&nbsp;
                                    {row['weeks']} weeks
                                </div>
                            </div>
                            <div class="score">
                                <div class="score-number">{score}%</div>
                                <div class="score-label">Recommendation<br>Score</div>
                            </div>
                        </div>
                        <div class="progress-shell">
                            <div class="progress-fill" style="width:{score}%"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.download_button(
                "⬇️ Download Learning Path",
                data=recommendations[
                    ["course", "category", "level", "weeks", "recommendation"]
                ].to_csv(index=False),
                file_name=f"{active.lower().replace(' ', '_')}_learning_path.csv",
                mime="text/csv",
                use_container_width=False,
            )

    with tab2:
        c1, c2 = st.columns(2)

        with c1:
            st.markdown(
                '<div class="glass"><b>📈 Engagement by Course</b></div>',
                unsafe_allow_html=True,
            )
            chart_df = history[["course", "engagement"]].sort_values(
                "engagement", ascending=True
            )
            st.bar_chart(chart_df.set_index("course"), height=330)

        with c2:
            st.markdown(
                '<div class="glass"><b>🏷️ Category Activity</b></div>',
                unsafe_allow_html=True,
            )
            category_df = (
                history.merge(
                    COURSES[["course", "category"]],
                    on="course",
                    how="left"
                )
                .groupby("category")["engagement"]
                .mean()
                .sort_values(ascending=True)
            )
            st.bar_chart(category_df, height=330)

    with tab3:
        st.markdown(
            """
            <div class="ai-box">
                <div class="ai-title">🧠 AI Learning Advisor</div>
                <div class="ai-caption">
                    Ask Groq to explain why these modules fit the intern's learning behavior.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        question = st.text_input(
            "Ask a learning question",
            placeholder="Why are these courses recommended for me?",
            key="home_ai_question",
        )
        if st.button("✨ Ask Groq AI", key="home_ask"):
            if question.strip():
                rec_text = recommendations[
                    ["course", "category", "level", "weeks", "recommendation"]
                ].to_string(index=False)
                prompt = f"""
Intern: {active}
Average engagement: {avg_engagement:.2f}/5
Completed modules: {completed_count}
Recommended modules:
{rec_text}

User question:
{question}

Explain the recommendation using the supplied data and give a practical next step.
"""
                with st.spinner("Groq is analyzing the learning path..."):
                    st.markdown(
                        f"<div class='ai-box'>{ask_groq(prompt)}</div>",
                        unsafe_allow_html=True,
                    )


# ============================================================
# INTERN PROFILE
# ============================================================
elif st.session_state.page == "Intern Profile":
    st.markdown('<div class="section-title">👤 Intern Learning Profile</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">Insights based on historical learning behavior.</div>',
        unsafe_allow_html=True,
    )

    p1, p2, p3 = st.columns(3)
    with p1:
        st.metric("Intern", active)
    with p2:
        st.metric("Completed", completed_count)
    with p3:
        st.metric("Avg. Engagement", f"{avg_engagement:.2f}/5")

    st.markdown("### 🧬 Learning fingerprint")

    category_profile = (
        history.merge(COURSES[["course", "category"]], on="course", how="left")
        .groupby("category")
        .agg(
            engagement=("engagement", "mean"),
            interactions=("course", "count"),
        )
        .sort_values("engagement", ascending=False)
    )

    for category, row in category_profile.iterrows():
        value = int(np.clip(row["engagement"] / 5 * 100, 0, 100))
        st.markdown(
            f"""
            <div class="glass" style="margin:8px 0;">
                <b>{category}</b>
                <span style="float:right;color:#ffad35;">
                    {row['engagement']:.1f}/5
                </span>
                <div class="progress-shell">
                    <div class="progress-fill" style="width:{value}%"></div>
                </div>
                <div style="font-size:10px;color:#7898bd;margin-top:6px;">
                    {int(row['interactions'])} historical interactions
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# LEARNING ANALYTICS
# ============================================================
elif st.session_state.page == "Learning Analytics":
    st.markdown('<div class="section-title">📊 Learning Analytics</div>', unsafe_allow_html=True)

    a1, a2 = st.columns(2)
    with a1:
        st.markdown("### Engagement distribution")
        st.bar_chart(history["engagement"].round(1).value_counts().sort_index(), height=320)
    with a2:
        st.markdown("### Rating distribution")
        st.bar_chart(history["rating"].round(1).value_counts().sort_index(), height=320)

    st.markdown("### Detailed learning history")
    display = history.merge(COURSES, on="course", how="left")
    st.dataframe(
        display[
            ["course", "category", "level", "rating", "engagement", "completed"]
        ].sort_values("engagement", ascending=False),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# AI ADVISOR
# ============================================================
elif st.session_state.page == "AI Advisor":
    st.markdown('<div class="section-title">🧠 Groq AI Learning Advisor</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="ai-box">
            <div class="ai-title">Personalized coaching layer</div>
            <div class="ai-caption">
                Matrix Factorization finds learning patterns. Groq turns those patterns
                into an understandable learning strategy.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    q = st.text_area(
        "What would you like the advisor to analyze?",
        placeholder=(
            "Example: Build a 6-week plan for this intern and explain which skills "
            "should be learned first."
        ),
        height=130,
    )

    if st.button("🚀 Generate AI Learning Plan"):
        if not q.strip():
            st.warning("Write a question first.")
        else:
            rec_text = recommendations[
                ["course", "category", "level", "weeks", "recommendation"]
            ].to_string(index=False)
            prompt = f"""
Intern: {active}
Historical average engagement: {avg_engagement:.2f}/5
Completed modules: {completed_count}

Matrix-factorization recommendations:
{rec_text}

Advisor request:
{q}

Create a concise practical plan. Explain the reasoning from the supplied data.
"""
            with st.spinner("Building personalized plan..."):
                answer = ask_groq(prompt)
            st.markdown(f"<div class='ai-box'>{answer}</div>", unsafe_allow_html=True)


# ============================================================
# AI PIPELINE
# ============================================================
elif st.session_state.page == "AI Pipeline":
    st.markdown('<div class="section-title">🔗 AI Recommendation Pipeline</div>', unsafe_allow_html=True)
    steps = [
        ("01", "Learning History", "Collect ratings, engagement and completion behavior."),
        ("02", "Interaction Matrix", "Convert intern-course activity into a user-item matrix."),
        ("03", "Matrix Factorization", "Learn hidden skill preferences from the interaction matrix."),
        ("04", "Course Prediction", "Predict affinity for unseen learning modules."),
        ("05", "Personalized Path", "Rank modules into an individual learning journey."),
        ("06", "Groq AI Explanation", "Generate a human-friendly explanation and study plan."),
    ]

    for num, title, desc in steps:
        st.markdown(
            f"""
            <div class="glass" style="margin:10px 0;">
                <span style="display:inline-flex;width:38px;height:38px;border-radius:50%;
                    align-items:center;justify-content:center;background:linear-gradient(135deg,#e52242,#ff8c2b);
                    font-weight:800;">{num}</span>
                <span style="font-size:16px;font-weight:800;margin-left:10px;">{title}</span>
                <div style="color:#87a9cf;font-size:12px;margin:8px 0 0 49px;">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# LEARNING HISTORY
# ============================================================
elif st.session_state.page == "Learning history":
    st.markdown('<div class="section-title">① Learning History</div>', unsafe_allow_html=True)
    st.dataframe(
        interactions.sort_values(["intern", "course"]),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "⬇️ Download Current Learning History",
        interactions.to_csv(index=False),
        "learning_history.csv",
        "text/csv",
    )


# ============================================================
# INTERACTION MATRIX
# ============================================================
elif st.session_state.page == "Interaction matrix":
    st.markdown('<div class="section-title">② Interaction Matrix</div>', unsafe_allow_html=True)
    matrix = make_interaction_matrix(interactions)
    st.markdown(
        '<div class="section-caption">'
        'Rows represent interns and columns represent courses. Values are historical ratings.'
        '</div>',
        unsafe_allow_html=True,
    )
    st.dataframe(
        matrix.round(2),
        use_container_width=True,
    )


# ============================================================
# MATRIX FACTORIZATION
# ============================================================
elif st.session_state.page == "Matrix Factorization":
    st.markdown('<div class="section-title">③ Matrix Factorization</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="glass">
        <b>How the recommender works</b><br><br>
        Collaborative filtering learns hidden preferences from historical
        intern-course interactions. The interaction matrix is decomposed into
        lower-dimensional intern and course representations. Their dot product
        produces predicted affinity scores for courses the intern has not completed.
        </div>
        """,
        unsafe_allow_html=True,
    )

    matrix = make_interaction_matrix(interactions)
    predicted = matrix_factorization(matrix.values)

    mf1, mf2, mf3 = st.columns(3)
    mf1.metric("Intern vectors", matrix.shape[0])
    mf2.metric("Course vectors", matrix.shape[1])
    mf3.metric("Latent factors", 4)

    st.markdown("### Predicted affinity matrix")
    predicted_df = pd.DataFrame(
        predicted,
        index=matrix.index,
        columns=matrix.columns,
    )
    st.dataframe(predicted_df.round(2), use_container_width=True)


# ============================================================
# COURSE PREDICTION
# ============================================================
elif st.session_state.page == "Course prediction":
    st.markdown('<div class="section-title">④ Course Prediction</div>', unsafe_allow_html=True)

    if recommendations.empty:
        st.warning("Not enough historical data.")
    else:
        prediction_df = recommendations[
            ["course", "category", "level", "weeks", "recommendation"]
        ].copy()
        prediction_df.columns = [
            "Course", "Category", "Level", "Weeks", "Predicted Recommendation %"
        ]
        st.dataframe(prediction_df, use_container_width=True, hide_index=True)

        st.markdown("### Top predicted modules")
        st.bar_chart(
            recommendations.set_index("course")["recommendation"],
            height=350,
        )


# ============================================================
# PERSONALIZED PATH
# ============================================================
elif st.session_state.page == "Personalized path":
    st.markdown('<div class="section-title">⑤ Personalized Learning Path</div>', unsafe_allow_html=True)

    if recommendations.empty:
        st.warning("Not enough data to create a learning path.")
    else:
        path = recommendations.copy()
        path["week_start"] = range(1, len(path) + 1)

        for _, row in path.iterrows():
            st.markdown(
                f"""
                <div class="glass" style="margin:10px 0;border-left:4px solid #ff5c35;">
                    <div style="font-size:11px;color:#ff9c39;font-weight:800;">
                        WEEK {int(row['week_start'])}
                    </div>
                    <div style="font-size:18px;font-weight:800;margin-top:4px;">
                        {row['icon']} {row['course']}
                    </div>
                    <div style="font-size:12px;color:#89a9cc;margin-top:5px;">
                        {row['category']} · {row['level']} · {int(row['weeks'])} weeks
                    </div>
                    <div style="font-size:12px;color:#d9e6f8;margin-top:8px;">
                        Recommendation confidence: <b>{int(row['recommendation'])}%</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# GROQ EXPLANATION
# ============================================================
elif st.session_state.page == "Groq AI explanation":
    st.markdown('<div class="section-title">⑥ Groq AI Explanation</div>', unsafe_allow_html=True)

    if recommendations.empty:
        st.warning("Generate recommendations first.")
    else:
        rec_text = recommendations[
            ["course", "category", "level", "weeks", "recommendation"]
        ].to_string(index=False)

        if st.button("✨ Explain My Learning Path"):
            prompt = f"""
Intern: {active}
Historical interactions: {len(history)}
Average engagement: {avg_engagement:.2f}/5
Completed modules: {completed_count}

Recommended path:
{rec_text}

Explain:
1. Why these modules are relevant.
2. Which module should be started first and why.
3. What skills the intern is likely to build.
4. A practical weekly study routine.

Keep the answer concise and grounded only in the supplied data.
"""
            with st.spinner("Groq is explaining the recommendation..."):
                answer = ask_groq(prompt)
            st.markdown(f"<div class='ai-box'>{answer}</div>", unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="text-align:center;color:#5f7fa6;font-size:10px;padding:8px;">
        LearnPath AI · Collaborative Filtering · Matrix Factorization · Groq AI
    </div>
    """,
    unsafe_allow_html=True,
)
