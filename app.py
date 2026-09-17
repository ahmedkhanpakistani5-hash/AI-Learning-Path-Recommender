import streamlit as st
import pandas as pd
import numpy as np
import random
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
# CUSTOM CSS
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
            circle at 75% 5%,
            rgba(239, 68, 68, 0.10),
            transparent 25%
        ),
        radial-gradient(
            circle at 30% 60%,
            rgba(249, 115, 22, 0.06),
            transparent 25%
        ),
        #050b18;
    color: #f8fafc;
}

.block-container {
    max-width: 1500px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

/* =========================================================
   SIDEBAR
   ========================================================= */

[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            #07101f 0%,
            #050b17 100%
        );
    border-right: 1px solid rgba(255,255,255,0.08);
}

[data-testid="stSidebar"] > div {
    padding-top: 1.5rem;
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
    margin-top: 4px;
    color: #f8fafc;
}

.sidebar-logo-title span {
    color: #ff7a18;
}

.sidebar-subtitle {
    color: #7dd3fc;
    font-size: 11px;
    margin-top: 2px;
}

.sidebar-section {
    color: #94a3b8;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 700;
    margin-top: 25px;
    margin-bottom: 10px;
}

/* =========================================================
   HERO
   ========================================================= */

.hero {
    min-height: 190px;
    border-radius: 22px;

    background:
        radial-gradient(
            circle at 88% 35%,
            rgba(255, 137, 41, 0.80),
            transparent 24%
        ),
        radial-gradient(
            circle at 72% 100%,
            rgba(220, 38, 38, 0.55),
            transparent 32%
        ),
        linear-gradient(
            120deg,
            #9f1239 0%,
            #dc2626 35%,
            #f97316 68%,
            #c2410c 100%
        );

    border: 1px solid rgba(255,255,255,0.14);

    box-shadow:
        0 20px 60px rgba(0,0,0,0.35),
        inset 0 1px 0 rgba(255,255,255,0.10);

    padding: 30px 38px;
    margin-bottom: 22px;
    position: relative;
    overflow: hidden;
}

.hero:after {
    content: "";
    position: absolute;
    width: 450px;
    height: 450px;
    right: -100px;
    top: -220px;
    background: rgba(255,255,255,0.08);
    border-radius: 50%;
}

.hero-title {
    font-size: 39px;
    font-weight: 800;
    color: white;
    margin: 0;
}

.hero-title span {
    color: #fed7aa;
}

.hero-subtitle {
    font-size: 16px;
    color: rgba(255,255,255,0.92);
    margin-top: 7px;
}

.hero-description {
    font-size: 13px;
    color: rgba(255,255,255,0.78);
    margin-top: 14px;
}

/* =========================================================
   PROFILE BAR
   ========================================================= */

.profile-wrapper {
    background:
        linear-gradient(
            90deg,
            rgba(15,23,42,0.96),
            rgba(15,23,42,0.72)
        );

    border: 1px solid rgba(148,163,184,0.14);
    border-radius: 16px;
    padding: 18px;
    margin-bottom: 20px;

    box-shadow: 0 12px 35px rgba(0,0,0,0.18);
}

.profile-title {
    font-size: 19px;
    font-weight: 700;
    color: #f8fafc;
}

.profile-subtitle {
    color: #67c8f5;
    font-size: 12px;
    margin-top: 4px;
}

/* =========================================================
   METRIC CARDS
   ========================================================= */

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

    min-height: 72px;

    box-shadow:
        inset 0 1px 0 rgba(255,255,255,0.03),
        0 8px 20px rgba(0,0,0,0.15);
}

.metric-number {
    font-size: 24px;
    font-weight: 800;
    color: #f8fafc;
}

.metric-label {
    color: #94a3b8;
    font-size: 10px;
    margin-top: 3px;
}

.metric-icon {
    font-size: 19px;
}

/* =========================================================
   SECTION TITLE
   ========================================================= */

.section-title {
    font-size: 20px;
    font-weight: 750;
    color: #f8fafc;
    margin-top: 22px;
    margin-bottom: 2px;
}

.section-title span {
    color: #67c8f5;
}

.section-caption {
    color: #64748b;
    font-size: 11px;
    margin-bottom: 14px;
}

/* =========================================================
   RECOMMENDATION CARDS
   ========================================================= */

.recommendation {
    display: flex;
    align-items: center;

    min-height: 82px;

    padding: 12px 18px;

    border-radius: 12px;

    margin-bottom: 9px;

    background:
        linear-gradient(
            90deg,
            rgba(88, 16, 35, 0.82),
            rgba(35, 17, 29, 0.82)
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
            rgba(84, 39, 18, 0.82),
            rgba(34, 23, 20, 0.82)
        );

    border-color: rgba(249,115,22,0.70);
}

.recommendation.gold {
    background:
        linear-gradient(
            90deg,
            rgba(69, 52, 17, 0.82),
            rgba(34, 30, 18, 0.82)
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

    font-size: 25px;

    margin-right: 16px;

    flex-shrink: 0;
}

.course-content {
    flex: 1;
}

.course-name {
    font-size: 15px;
    font-weight: 700;
    color: #f8fafc;
}

.course-meta {
    color: #67c8f5;
    font-size: 10px;
    margin-top: 3px;
}

.recommend-badge {
    display: inline-block;

    padding: 5px 10px;

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

    margin-right: 12px;
}

.score {
    min-width: 60px;
    height: 38px;

    display: flex;
    align-items: center;
    justify-content: center;

    border-radius: 18px;

    background: rgba(255,255,255,0.09);

    color: white;

    font-weight: 800;
    font-size: 16px;
}

/* =========================================================
   DOWNLOAD BUTTON
   ========================================================= */

.stDownloadButton {
    margin-top: 13px;
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

    transition: all 0.2s ease !important;
}

.stDownloadButton > button:hover {
    background: #fff7ed !important;

    color: #ea580c !important;

    border-color: #fb923c !important;

    transform: translateY(-2px);

    box-shadow:
        0 14px 30px rgba(249,115,22,0.18) !important;
}

/* =========================================================
   TABS
   ========================================================= */

.stTabs [data-baseweb="tab-list"] {
    gap: 3px;

    background:
        rgba(20,29,48,0.85);

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

/* =========================================================
   SELECT BOX
   ========================================================= */

div[data-baseweb="select"] > div {
    background-color: #101a2e !important;
    border-color: rgba(148,163,184,0.20) !important;
    color: white !important;
}

/* =========================================================
   BUTTONS
   ========================================================= */

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

/* =========================================================
   AI CARD
   ========================================================= */

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

/* =========================================================
   FOOTER
   ========================================================= */

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
# DATA
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

        preferred = random.sample(
            list(course_df["Category"].unique()),
            4
        )

        for _, course in course_df.iterrows():

            probability = (
                0.75
                if course["Category"] in preferred
                else 0.28
            )

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

    latent = model.fit_transform(matrix)

    reconstructed = np.dot(
        latent,
        model.components_
    )

    predictions = pd.DataFrame(
        reconstructed,
        index=matrix.index,
        columns=matrix.columns
    )

    return matrix, predictions


interaction_matrix, prediction_matrix = train_model(
    interactions
)


# =========================================================
# RECOMMENDATIONS
# =========================================================

def get_recommendations(
    intern,
    count=6
):

    scores = prediction_matrix.loc[
        intern
    ].copy()

    completed = interaction_matrix.loc[
        intern
    ]

    scores[completed > 0] = -999

    top = scores.sort_values(
        ascending=False
    ).head(count)

    result = []

    for course, raw_score in top.items():

        info = course_df[
            course_df["Course"] == course
        ].iloc[0]

        score = int(
            min(
                98,
                max(
                    72,
                    72 + float(raw_score) * 8
                )
            )
        )

        result.append({
            "Course": course,
            "Category": info["Category"],
            "Level": info["Level"],
            "Duration": info["Duration"],
            "Score": score
        })

    return pd.DataFrame(result)


# =========================================================
# GROQ
# =========================================================

def generate_ai_plan(
    intern,
    recommendations
):

    try:

        from groq import Groq

        api_key = st.secrets.get(
            "GROQ_API_KEY",
            ""
        )

        if not api_key:

            return """
### 🔑 Groq API Key Required

The recommendation engine is working correctly.

Add your `GROQ_API_KEY` inside Streamlit Cloud:

**App → Settings → Secrets**
"""

        client = Groq(
            api_key=api_key
        )

        courses = "\n".join(
            [
                f"- {row['Course']} "
                f"({row['Category']}, "
                f"{row['Level']})"
                for _, row in recommendations.iterrows()
            ]
        )

        prompt = f"""
You are an AI learning advisor.

Intern:
{intern}

The Matrix Factorization recommendation system generated
the following personalized courses:

{courses}

Create a concise professional learning strategy.

Include:

1. Why these courses fit the intern.
2. Recommended starting course.
3. Skills the intern should gain.
4. A practical mini-project.
5. A suggested learning order.

Do not mention that you are guessing.
Keep it practical and concise.
"""

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional internship "
                        "learning advisor."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.5,
            max_completion_tokens=800,
            include_reasoning=False
        )

        return response.choices[
            0
        ].message.content

    except Exception as error:

        return f"""
### ⚠️ AI Advisor Temporarily Unavailable

The Matrix Factorization recommendations are still
working correctly.

Please check your Groq API key and Streamlit Secrets.

Technical status: `{type(error).__name__}`
"""


# =========================================================
# HERO
# =========================================================

st.markdown("""
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
""", unsafe_allow_html=True)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("""
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
    """, unsafe_allow_html=True)

    st.markdown("### 🏠 Home")

    st.markdown(
        "👤 &nbsp; Intern Profile"
    )

    st.markdown(
        "📊 &nbsp; Learning Analytics"
    )

    st.markdown(
        "🧠 &nbsp; AI Advisor"
    )

    st.markdown("""
    <div class="sidebar-section">
        AI Pipeline
    </div>

    <div style="line-height:2.4; color:#94a3b8; font-size:13px;">

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
    """, unsafe_allow_html=True)

    st.markdown("---")

    selected_intern = st.selectbox(
        "👤 Select Intern",
        interns
    )

    recommendation_count = st.slider(
        "📚 Recommendations",
        3,
        10,
        6
    )


# =========================================================
# PROFILE
# =========================================================

recommendations = get_recommendations(
    selected_intern,
    recommendation_count
)

history = interactions[
    interactions["Intern"] == selected_intern
]

avg_rating = round(
    history["Rating"].mean(),
    1
)

interaction_count = len(history)

category_count = history["Course"].map(
    course_df.set_index("Course")["Category"]
).nunique()


st.markdown("""
<div class="profile-wrapper">

<div class="profile-title">
👤 Intern Learning Profile
</div>

<div class="profile-subtitle">
Insights based on your past learning behavior
</div>

</div>
""", unsafe_allow_html=True)


m1, m2, m3, m4 = st.columns(4)

with m1:

    st.markdown(
        f"""
        <div class="metric-card">

        <div class="metric-icon">📖</div>

        <div class="metric-number">
        {interaction_count}
        </div>

        <div class="metric-label">
        Learning Interactions
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )

with m2:

    st.markdown(
        f"""
        <div class="metric-card">

        <div class="metric-icon">⭐</div>

        <div class="metric-number">
        {avg_rating}/5
        </div>

        <div class="metric-label">
        Average Engagement
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )

with m3:

    st.markdown(
        f"""
        <div class="metric-card">

        <div class="metric-icon">🧩</div>

        <div class="metric-number">
        {category_count}
        </div>

        <div class="metric-label">
        Skill Categories
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )

with m4:

    st.markdown(
        f"""
        <div class="metric-card">

        <div class="metric-icon">📚</div>

        <div class="metric-number">
        {len(course_df)}
        </div>

        <div class="metric-label">
        Available Modules
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


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
# RECOMMENDED PATH
# =========================================================

with tab1:

    st.markdown(
        """
        <div class="section-title">
        ✨ Your <span>Personalized Learning Path</span>
        </div>

        <div class="section-caption">
        Recommendations are generated from historical learning
        patterns using Matrix Factorization.
        </div>
        """,
        unsafe_allow_html=True
    )

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

    styles = [
        "",
        "orange",
        "gold",
        "orange",
        "",
        "gold",
        "orange",
        ""
    ]

    for i, (_, row) in enumerate(
        recommendations.iterrows()
    ):

        icon = icons[
            i % len(icons)
        ]

        style = styles[
            i % len(styles)
        ]

        st.markdown(
            f"""
            <div class="recommendation {style}">

                <div class="course-icon">
                    {icon}
                </div>

                <div class="course-content">

                    <div>

                        <span class="recommend-badge">
                        #{i + 1} RECOMMENDED
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
            """,
            unsafe_allow_html=True
        )

    csv = recommendations.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="📥  Download Learning Path",
        data=csv,
        file_name=(
            f"{selected_intern}_"
            "personalized_learning_path.csv"
        ),
        mime="text/csv",
        key="download_learning_path"
    )


# =========================================================
# ANALYTICS
# =========================================================

with tab2:

    st.markdown(
        """
        <div class="section-title">
        📊 Learning <span>Analytics</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    left, right = st.columns(2)

    with left:

        st.markdown("### 📚 Previous Learning Activity")

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

        st.markdown("### 🧠 Category Engagement")

        category_data = history_table.groupby(
            "Category"
        )["Rating"].mean()

        st.bar_chart(
            category_data
        )


# =========================================================
# AI ADVISOR
# =========================================================

with tab3:

    st.markdown(
        """
        <div class="section-title">
        🧠 AI <span>Learning Advisor</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="ai-card">

        <h3>🤖 Personalized AI Strategy</h3>

        <p style="color:#94a3b8;">
        Groq analyzes the recommended learning path and
        converts it into a practical internship strategy.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    if st.button(
        "✨ Generate My AI Learning Plan",
        key="generate_ai"
    ):

        with st.spinner(
            "AI advisor is preparing your learning strategy..."
        ):

            result = generate_ai_plan(
                selected_intern,
                recommendations
            )

        st.markdown(result)


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">

    🎓 LearnPath AI
    &nbsp; • &nbsp;
    Matrix Factorization
    &nbsp; • &nbsp;
    Collaborative Filtering
    &nbsp; • &nbsp;
    Groq AI

    </div>
    """,
    unsafe_allow_html=True
)
