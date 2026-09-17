import streamlit as st
import pandas as pd
import numpy as np
import random
import textwrap
from sklearn.decomposition import TruncatedSVD

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="LearnPath AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# HTML HELPER
# =========================================================

def render_html(html):
    st.markdown(
        textwrap.dedent(html),
        unsafe_allow_html=True
    )

# =========================================================
# CSS
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(
            circle at 80% 0%,
            rgba(239, 68, 68, 0.10),
            transparent 28%
        ),
        #050b18;
    color: #f8fafc;
}

.block-container {
    max-width: 1500px;
    padding-top: 1.4rem;
    padding-bottom: 3rem;
}

/* ================= SIDEBAR ================= */

[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #07101f 0%,
        #050b17 100%
    );
    border-right: 1px solid rgba(255,255,255,0.08);
}

.sidebar-logo {
    text-align: center;
    padding: 8px 0 22px 0;
}

.sidebar-logo-icon {
    font-size: 38px;
}

.sidebar-logo-title {
    font-size: 22px;
    font-weight: 800;
    color: #f8fafc;
    margin-top: 5px;
}

.sidebar-logo-title span {
    color: #ff7a18;
}

.sidebar-subtitle {
    color: #7dd3fc;
    font-size: 11px;
    margin-top: 3px;
}

.sidebar-section {
    color: #94a3b8;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 700;
    margin-top: 25px;
    margin-bottom: 12px;
}

/* ================= HERO ================= */

.hero {
    min-height: 175px;
    border-radius: 22px;

    background:
        radial-gradient(
            circle at 88% 35%,
            rgba(255, 137, 41, 0.85),
            transparent 24%
        ),
        radial-gradient(
            circle at 70% 100%,
            rgba(220, 38, 38, 0.55),
            transparent 35%
        ),
        linear-gradient(
            120deg,
            #9f1239 0%,
            #dc2626 38%,
            #f97316 72%,
            #c2410c 100%
        );

    border: 1px solid rgba(255,255,255,0.15);

    box-shadow:
        0 20px 60px rgba(0,0,0,0.35),
        inset 0 1px 0 rgba(255,255,255,0.12);

    padding: 30px 38px;
    margin-bottom: 22px;
}

.hero-title {
    font-size: 40px;
    font-weight: 800;
    color: white;
}

.hero-title span {
    color: #fed7aa;
}

.hero-subtitle {
    font-size: 16px;
    color: white;
    margin-top: 6px;
}

.hero-description {
    font-size: 13px;
    color: rgba(255,255,255,0.82);
    margin-top: 13px;
}

/* ================= PROFILE ================= */

.profile-wrapper {
    background:
        linear-gradient(
            90deg,
            rgba(15,23,42,0.96),
            rgba(15,23,42,0.75)
        );

    border: 1px solid rgba(148,163,184,0.14);
    border-radius: 16px;

    padding: 17px 20px;
    margin-bottom: 12px;
}

.profile-title {
    font-size: 19px;
    font-weight: 700;
}

.profile-subtitle {
    color: #67c8f5;
    font-size: 12px;
    margin-top: 4px;
}

/* ================= METRICS ================= */

.metric-card {
    background:
        linear-gradient(
            145deg,
            rgba(19,30,52,0.95),
            rgba(12,20,37,0.92)
        );

    border: 1px solid rgba(148,163,184,0.10);
    border-radius: 13px;

    padding: 14px 18px;
    min-height: 82px;

    box-shadow:
        inset 0 1px 0 rgba(255,255,255,0.03),
        0 8px 20px rgba(0,0,0,0.15);
}

.metric-icon {
    font-size: 19px;
}

.metric-number {
    font-size: 25px;
    font-weight: 800;
    color: #f8fafc;
    margin-top: 3px;
}

.metric-label {
    color: #94a3b8;
    font-size: 10px;
}

/* ================= SECTION ================= */

.section-title {
    font-size: 21px;
    font-weight: 750;
    margin-top: 25px;
    margin-bottom: 3px;
}

.section-title span {
    color: #67c8f5;
}

.section-caption {
    color: #64748b;
    font-size: 11px;
    margin-bottom: 14px;
}

/* ================= RECOMMENDATIONS ================= */

.recommendation {
    display: flex;
    align-items: center;

    min-height: 78px;

    padding: 11px 17px;

    border-radius: 12px;

    margin-bottom: 9px;

    background:
        linear-gradient(
            90deg,
            rgba(88,16,35,0.85),
            rgba(35,17,29,0.82)
        );

    border: 1px solid rgba(244,63,94,0.65);

    box-shadow:
        inset 0 1px 0 rgba(255,255,255,0.03),
        0 8px 25px rgba(0,0,0,0.18);
}

.recommendation.orange {
    background:
        linear-gradient(
            90deg,
            rgba(84,39,18,0.85),
            rgba(34,23,20,0.82)
        );

    border-color: rgba(249,115,22,0.70);
}

.recommendation.gold {
    background:
        linear-gradient(
            90deg,
            rgba(69,52,17,0.85),
            rgba(34,30,18,0.82)
        );

    border-color: rgba(234,179,8,0.70);
}

.course-icon {
    width: 47px;
    height: 47px;

    border-radius: 10px;

    display: flex;
    align-items: center;
    justify-content: center;

    background: rgba(30,41,59,0.95);

    font-size: 24px;

    margin-right: 15px;

    flex-shrink: 0;
}

.course-content {
    flex: 1;
}

.course-name {
    font-size: 15px;
    font-weight: 700;
    color: #f8fafc;
    margin-top: 4px;
}

.course-meta {
    color: #67c8f5;
    font-size: 10px;
    margin-top: 3px;
}

.recommend-badge {
    display: inline-block;

    padding: 4px 9px;

    border-radius: 20px;

    background:
        linear-gradient(
            90deg,
            #dc2626,
            #f43f5e
        );

    color: white;

    font-size: 9px;
    font-weight: 800;
}

.score {
    min-width: 58px;
    height: 37px;

    display: flex;
    align-items: center;
    justify-content: center;

    border-radius: 20px;

    background: rgba(255,255,255,0.10);

    color: white;

    font-weight: 800;
    font-size: 15px;
}

/* ================= DOWNLOAD ================= */

.stDownloadButton {
    margin-top: 14px;
}

.stDownloadButton > button {
    width: 330px !important;
    height: 58px !important;

    border-radius: 13px !important;

    background: #ffffff !important;

    color: #263449 !important;

    border: 1px solid #e5e7eb !important;

    font-size: 16px !important;

    font-weight: 600 !important;

    box-shadow:
        0 10px 25px rgba(0,0,0,0.20) !important;

    transition: 0.2s ease !important;
}

.stDownloadButton > button:hover {
    background: #fff7ed !important;

    color: #ea580c !important;

    border-color: #fb923c !important;

    transform: translateY(-2px);
}

/* ================= TABS ================= */

.stTabs [data-baseweb="tab-list"] {
    gap: 3px;

    background: rgba(20,29,48,0.85);

    border-radius: 12px;

    padding: 3px;
}

.stTabs [data-baseweb="tab"] {
    height: 40px;

    border-radius: 10px;

    color: #cbd5e1;

    font-size: 13px;

    padding-left: 20px;
    padding-right: 20px;
}

.stTabs [aria-selected="true"] {
    background:
        linear-gradient(
            90deg,
            rgba(234,88,12,0.30),
            rgba(220,38,38,0.25)
        ) !important;

    color: #fed7aa !important;
}

/* ================= SELECTBOX ================= */

div[data-baseweb="select"] > div {
    background-color: #101a2e !important;
    border-color: rgba(148,163,184,0.20) !important;
    color: white !important;
}

/* ================= BUTTON ================= */

.stButton > button {
    border-radius: 10px;

    background:
        linear-gradient(
            90deg,
            #dc2626,
            #f97316
        );

    color: white;

    border: none;

    font-weight: 700;
}

.stButton > button:hover {
    background:
        linear-gradient(
            90deg,
            #ef4444,
            #fb923c
        );

    color: white;
}

/* ================= AI CARD ================= */

.ai-card {
    background:
        linear-gradient(
            135deg,
            rgba(127,29,29,0.35),
            rgba(67,20,7,0.35)
        );

    border: 1px solid rgba(249,115,22,0.25);

    border-radius: 17px;

    padding: 25px;

    box-shadow: 0 15px 40px rgba(0,0,0,0.20);
}

/* ================= FOOTER ================= */

.footer {
    text-align: center;

    margin-top: 40px;

    padding: 20px;

    color: #475569;

    font-size: 11px;

    border-top: 1px solid rgba(255,255,255,0.05);
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# DEMO DATA
# =========================================================

@st.cache_data
def create_data():

    random.seed(42)
    np.random.seed(42)

    interns = [
        "Ali",
        "Ahmed",
        "Sara",
        "Hamza",
        "Ayesha",
        "Usman",
        "Fatima",
        "Bilal",
        "Hassan",
        "Zainab",
        "Daniyal",
        "Maham"
    ]

    courses = [
        ("Python Fundamentals", "Python", "Beginner", 4),
        ("Advanced Python", "Python", "Intermediate", 6),
        ("Machine Learning Basics", "Machine Learning", "Intermediate", 6),
        ("Deep Learning", "AI", "Advanced", 8),
        ("Data Analysis with Pandas", "Data Science", "Intermediate", 5),
        ("SQL Fundamentals", "Database", "Beginner", 4),
        ("Advanced SQL", "Database", "Advanced", 6),
        ("Git & GitHub", "Development", "Beginner", 3),
        ("REST API Development", "Backend", "Intermediate", 5),
        ("FastAPI Development", "Backend", "Advanced", 6),
        ("Streamlit Development", "Web Development", "Intermediate", 4),
        ("Cloud Fundamentals", "Cloud", "Beginner", 4),
        ("AWS Essentials", "Cloud", "Intermediate", 6),
        ("Computer Vision", "AI", "Advanced", 7),
        ("Natural Language Processing", "AI", "Advanced", 7),
        ("Data Structures & Algorithms", "Programming", "Intermediate", 7),
        ("Object Oriented Programming", "Programming", "Intermediate", 5),
        ("Software Engineering Basics", "Software Engineering", "Beginner", 4),
        ("Agile & Scrum", "Software Engineering", "Beginner", 3),
        ("Cybersecurity Fundamentals", "Security", "Beginner", 5)
    ]

    course_df = pd.DataFrame(
        courses,
        columns=[
            "Course",
            "Category",
            "Level",
            "Duration"
        ]
    )

    rows = []

    for intern in interns:

        preferred_categories = random.sample(
            list(course_df["Category"].unique()),
            4
        )

        for _, course in course_df.iterrows():

            if course["Category"] in preferred_categories:
                probability = 0.75
            else:
                probability = 0.28

            if random.random() < probability:

                rows.append({
                    "Intern": intern,
                    "Course": course["Course"],
                    "Rating": random.choice(
                        [1, 2, 3, 4, 5]
                    )
                })

    interactions = pd.DataFrame(rows)

    return interns, course_df, interactions


interns, course_df, interactions = create_data()


# =========================================================
# MATRIX FACTORIZATION
# =========================================================

@st.cache_resource
def train_model(data):

    matrix = data.pivot_table(
        index="Intern",
        columns="Course",
        values="Rating",
        fill_value=0
    )

    components = max(
        1,
        min(8, min(matrix.shape) - 1)
    )

    model = TruncatedSVD(
        n_components=components,
        random_state=42
    )

    latent_matrix = model.fit_transform(matrix)

    reconstructed_matrix = np.dot(
        latent_matrix,
        model.components_
    )

    predictions = pd.DataFrame(
        reconstructed_matrix,
        index=matrix.index,
        columns=matrix.columns
    )

    return matrix, predictions


interaction_matrix, prediction_matrix = train_model(
    interactions
)


# =========================================================
# RECOMMENDATION ENGINE
# =========================================================

def get_recommendations(intern, count=6):

    scores = prediction_matrix.loc[intern].copy()

    completed = interaction_matrix.loc[intern]

    scores[completed > 0] = -999

    top_courses = scores.sort_values(
        ascending=False
    ).head(count)

    result = []

    for course_name, raw_score in top_courses.items():

        info = course_df[
            course_df["Course"] == course_name
        ].iloc[0]

        recommendation_score = int(
            min(
                98,
                max(
                    72,
                    72 + float(raw_score) * 8
                )
            )
        )

        result.append({
            "Course": course_name,
            "Category": info["Category"],
            "Level": info["Level"],
            "Duration": info["Duration"],
            "Score": recommendation_score
        })

    return pd.DataFrame(result)


# =========================================================
# GROQ AI
# =========================================================

def generate_ai_plan(intern, recommendations):

    try:

        from groq import Groq

        api_key = st.secrets.get(
            "GROQ_API_KEY",
            ""
        )

        if not api_key:

            return """
### 🔑 Groq API Key Required

The Matrix Factorization recommendation engine is working.

Add your `GROQ_API_KEY` inside Streamlit Secrets to
enable the AI Learning Advisor.
"""

        client = Groq(
            api_key=api_key
        )

        course_list = "\n".join(
            [
                f"- {row['Course']} | "
                f"{row['Category']} | "
                f"{row['Level']}"
                for _, row in recommendations.iterrows()
            ]
        )

        prompt = f"""
You are an AI learning advisor for an internship program.

Intern:
{intern}

The recommendation engine generated these learning modules:

{course_list}

Create a concise personalized learning strategy.

Include:

1. Why these modules fit the intern's learning pattern.
2. Which course should be started first.
3. Skills the intern can gain.
4. A practical mini-project.
5. Suggested learning order.

Keep the response professional, practical and concise.
"""

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional AI "
                        "learning advisor."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.5,
            max_completion_tokens=800
        )

        return response.choices[0].message.content

    except Exception as error:

        return f"""
### ⚠️ AI Advisor Temporarily Unavailable

Your Matrix Factorization recommendation system is
still working.

Please check your Groq API key and Streamlit Secrets.

Technical error type:
`{type(error).__name__}`
"""


# =========================================================
# HERO
# =========================================================

render_html("""
<div class="hero">

    <div class="hero-title">
        🎓 LearnPath <span>AI</span>
    </div>

    <div class="hero-subtitle">
        Personalized learning paths powered by
        <strong>Matrix Factorization + Generative AI</strong>
    </div>

    <div class="hero-description">
        Turn previous intern learning behavior into a customized
        training journey.
    </div>

</div>
""")


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    render_html("""
    <div class="sidebar-logo">

        <div class="sidebar-logo-icon">
            🎓
        </div>

        <div class="sidebar-logo-title">
            LearnPath <span>AI</span>
        </div>

        <div class="sidebar-subtitle">
            Personalized Learning Paths
        </div>

    </div>
    """)

    st.markdown("### 🏠 Home")

    st.markdown("👤 &nbsp; Intern Profile")

    st.markdown("📊 &nbsp; Learning Analytics")

    st.markdown("🧠 &nbsp; AI Advisor")

    render_html("""
    <div class="sidebar-section">
        AI Pipeline
    </div>

    <div style="
        line-height:2.5;
        color:#94a3b8;
        font-size:13px;
    ">

        <b style="color:#f97316;">1</b>
        &nbsp; Learning history<br>

        <b style="color:#f97316;">2</b>
        &nbsp; Interaction matrix<br>

        <b style="color:#f97316;">3</b>
        &nbsp; Matrix Factorization<br>

        <b style="color:#f97316;">4</b>
        &nbsp; Course prediction<br>

        <b style="color:#f97316;">5</b>
        &nbsp; Personalized path<br>

        <b style="color:#f97316;">6</b>
        &nbsp; Groq AI explanation

    </div>
    """)

    st.markdown("---")

    selected_intern = st.selectbox(
        "👤 Select Intern",
        interns
    )

    recommendation_count = st.slider(
        "📚 Recommendations",
        min_value=3,
        max_value=10,
        value=6
    )


# =========================================================
# PROFILE DATA
# =========================================================

recommendations = get_recommendations(
    selected_intern,
    recommendation_count
)

history = interactions[
    interactions["Intern"] == selected_intern
]

average_rating = round(
    history["Rating"].mean(),
    1
)

interaction_count = len(history)

category_count = history["Course"].map(
    course_df.set_index("Course")["Category"]
).nunique()


# =========================================================
# PROFILE HEADER
# =========================================================

render_html("""
<div class="profile-wrapper">

    <div class="profile-title">
        👤 Intern Learning Profile
    </div>

    <div class="profile-subtitle">
        Insights based on your past learning behavior
    </div>

</div>
""")


# =========================================================
# METRICS
# =========================================================

m1, m2, m3, m4 = st.columns(4)

with m1:

    render_html(f"""
    <div class="metric-card">

        <div class="metric-icon">
            📖
        </div>

        <div class="metric-number">
            {interaction_count}
        </div>

        <div class="metric-label">
            Learning Interactions
        </div>

    </div>
    """)

with m2:

    render_html(f"""
    <div class="metric-card">

        <div class="metric-icon">
            ⭐
        </div>

        <div class="metric-number">
            {average_rating}/5
        </div>

        <div class="metric-label">
            Average Engagement
        </div>

    </div>
    """)

with m3:

    render_html(f"""
    <div class="metric-card">

        <div class="metric-icon">
            🧩
        </div>

        <div class="metric-number">
            {category_count}
        </div>

        <div class="metric-label">
            Skill Categories
        </div>

    </div>
    """)

with m4:

    render_html(f"""
    <div class="metric-card">

        <div class="metric-icon">
            📚
        </div>

        <div class="metric-number">
            {len(course_df)}
        </div>

        <div class="metric-label">
            Available Modules
        </div>

    </div>
    """)


# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3 = st.tabs(
    [
        "✨ Recommended Path",
        "📊 Learning Analytics",
        "🧠 AI Learning Advisor"
    ]
)


# =========================================================
# TAB 1
# =========================================================

with tab1:

    render_html("""
    <div class="section-title">
        ✨ Your <span>Personalized Learning Path</span>
    </div>

    <div class="section-caption">
        Recommendations are generated from historical learning
        patterns using Matrix Factorization.
    </div>
    """)

    icons = [
        "🐍",
        "🧠",
        "🗄️",
        "☁️",
        "📊",
        "💻",
        "🔐",
        "⚙️"
    ]

    card_styles = [
        "",
        "orange",
        "gold",
        "orange",
        "",
        "gold",
        "orange",
        ""
    ]

    for index, (_, row) in enumerate(
        recommendations.iterrows()
    ):

        icon = icons[
            index % len(icons)
        ]

        style = card_styles[
            index % len(card_styles)
        ]

        render_html(f"""
        <div class="recommendation {style}">

            <div class="course-icon">
                {icon}
            </div>

            <div class="course-content">

                <div>
                    <span class="recommend-badge">
                        #{index + 1} RECOMMENDED
                    </span>
                </div>

                <div class="course-name">
                    {row['Course']}
                </div>

                <div class="course-meta">
                    {row['Category']}
                    &nbsp; | &nbsp;
                    {row['Level']}
                    &nbsp; | &nbsp;
                    {row['Duration']} weeks
                </div>

            </div>

            <div class="score">
                {row['Score']}%
            </div>

        </div>
        """)

    # =====================================================
    # DOWNLOAD BUTTON
    # =====================================================

    csv_data = recommendations.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="📥  Download Learning Path",
        data=csv_data,
        file_name=(
            f"{selected_intern}_"
            "personalized_learning_path.csv"
        ),
        mime="text/csv",
        key="download_learning_path"
    )


# =========================================================
# TAB 2
# =========================================================

with tab2:

    render_html("""
    <div class="section-title">
        📊 Learning <span>Analytics</span>
    </div>
    """)

    left, right = st.columns(2)

    with left:

        st.markdown(
            "### 📚 Previous Learning Activity"
        )

        history_table = history.merge(
            course_df,
            on="Course"
        )

        st.dataframe(
            history_table[
                [
                    "Course",
                    "Category",
                    "Level",
                    "Rating"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

    with right:

        st.markdown(
            "### 🧠 Category Engagement"
        )

        category_data = history_table.groupby(
            "Category"
        )["Rating"].mean()

        st.bar_chart(
            category_data
        )


# =========================================================
# TAB 3
# =========================================================

with tab3:

    render_html("""
    <div class="section-title">
        🧠 AI <span>Learning Advisor</span>
    </div>

    <div class="ai-card">

        <h3>
            🤖 Personalized AI Strategy
        </h3>

        <p style="color:#94a3b8;">
            Groq converts the recommendation engine's
            results into a practical learning strategy.
        </p>

    </div>
    """)

    st.write("")

    if st.button(
        "✨ Generate My AI Learning Plan",
        key="generate_ai"
    ):

        with st.spinner(
            "AI advisor is preparing your learning strategy..."
        ):

            ai_result = generate_ai_plan(
                selected_intern,
                recommendations
            )

        st.markdown(ai_result)


# =========================================================
# FOOTER
# =========================================================

render_html("""
<div class="footer">

    🎓 LearnPath AI
    &nbsp; • &nbsp;
    Matrix Factorization
    &nbsp; • &nbsp;
    Collaborative Filtering
    &nbsp; • &nbsp;
    Groq AI

</div>
""")
