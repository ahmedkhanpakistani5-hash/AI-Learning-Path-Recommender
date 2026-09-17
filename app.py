import streamlit as st
import pandas as pd
import numpy as np
import random
import os
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
# DARK GLASS UI
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
            circle at 85% 5%,
            rgba(249,115,22,0.10),
            transparent 25%
        ),
        radial-gradient(
            circle at 5% 80%,
            rgba(168,85,247,0.08),
            transparent 25%
        ),
        #060b16;
    color: #f8fafc;
}

.block-container {
    max-width: 1500px;
    padding: 25px 35px 50px 35px;
}

/* SIDEBAR */

[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            #080f1e 0%,
            #050914 100%
        );
    border-right: 1px solid rgba(255,255,255,0.06);
}

[data-testid="stSidebar"] * {
    color: #cbd5e1;
}

/* HERO */

.hero-box {
    background:
        linear-gradient(
            110deg,
            #9f1239 0%,
            #dc2626 38%,
            #f97316 72%,
            #ea580c 100%
        );

    border-radius: 24px;
    padding: 34px 40px;
    margin-bottom: 22px;

    box-shadow:
        0 25px 60px rgba(0,0,0,0.35),
        inset 0 1px 0 rgba(255,255,255,0.15);
}

.hero-title {
    font-size: 42px;
    font-weight: 800;
    color: white;
    margin-bottom: 7px;
}

.hero-title span {
    color: #fed7aa;
}

.hero-subtitle {
    font-size: 17px;
    color: white;
}

.hero-description {
    font-size: 13px;
    color: rgba(255,255,255,0.78);
    margin-top: 12px;
}

/* GLASS CARD */

.glass-card {
    background:
        linear-gradient(
            145deg,
            rgba(17,27,48,0.92),
            rgba(10,18,34,0.88)
        );

    border: 1px solid rgba(148,163,184,0.12);
    border-radius: 18px;

    padding: 20px;

    box-shadow:
        0 12px 35px rgba(0,0,0,0.20),
        inset 0 1px 0 rgba(255,255,255,0.03);
}

/* PROFILE */

.profile-title {
    font-size: 20px;
    font-weight: 750;
    color: white;
}

.profile-subtitle {
    color: #67c8f5;
    font-size: 12px;
    margin-top: 5px;
}

/* METRICS */

.metric-box {
    background:
        linear-gradient(
            145deg,
            rgba(18,29,51,0.96),
            rgba(11,19,35,0.94)
        );

    border: 1px solid rgba(148,163,184,0.10);

    border-radius: 15px;

    padding: 16px 18px;

    min-height: 105px;

    box-shadow:
        0 10px 30px rgba(0,0,0,0.18);
}

.metric-icon {
    font-size: 20px;
}

.metric-value {
    font-size: 27px;
    font-weight: 800;
    color: white;
    margin-top: 3px;
}

.metric-label {
    color: #94a3b8;
    font-size: 10px;
}

/* SECTION */

.section-title {
    font-size: 22px;
    font-weight: 800;
    color: white;
    margin-top: 25px;
}

.section-title span {
    color: #67c8f5;
}

.section-caption {
    color: #64748b;
    font-size: 12px;
    margin-bottom: 15px;
}

/* COURSE CARDS */

.course-card {
    background:
        linear-gradient(
            100deg,
            rgba(55,16,32,0.94),
            rgba(22,19,34,0.94)
        );

    border: 1px solid rgba(244,63,94,0.40);

    border-radius: 15px;

    padding: 15px 18px;

    margin-bottom: 10px;

    box-shadow:
        0 8px 25px rgba(0,0,0,0.18);
}

.course-card.orange {
    background:
        linear-gradient(
            100deg,
            rgba(67,30,13,0.94),
            rgba(23,20,30,0.94)
        );

    border-color: rgba(249,115,22,0.40);
}

.course-card.purple {
    background:
        linear-gradient(
            100deg,
            rgba(47,22,70,0.94),
            rgba(18,20,34,0.94)
        );

    border-color: rgba(168,85,247,0.40);
}

.course-name {
    font-size: 15px;
    font-weight: 750;
    color: white;
}

.course-meta {
    color: #7dd3fc;
    font-size: 10px;
    margin-top: 5px;
}

.course-icon {
    font-size: 26px;
}

/* BADGE */

.badge {
    display: inline-block;

    background:
        linear-gradient(
            90deg,
            #dc2626,
            #f97316
        );

    color: white;

    border-radius: 20px;

    padding: 4px 9px;

    font-size: 9px;
    font-weight: 800;

    margin-bottom: 5px;
}

.score-badge {
    background: rgba(255,255,255,0.09);

    border: 1px solid rgba(255,255,255,0.08);

    border-radius: 20px;

    padding: 7px 12px;

    color: white;

    font-size: 13px;
    font-weight: 800;
}

/* DOWNLOAD */

.stDownloadButton button {
    width: 320px !important;

    height: 52px !important;

    background: white !important;

    color: #172033 !important;

    border: none !important;

    border-radius: 12px !important;

    font-weight: 700 !important;

    box-shadow:
        0 10px 25px rgba(0,0,0,0.25) !important;
}

.stDownloadButton button:hover {
    background: #fff7ed !important;
    color: #ea580c !important;
}

/* NORMAL BUTTON */

.stButton button {
    border: none !important;

    border-radius: 11px !important;

    background:
        linear-gradient(
            90deg,
            #dc2626,
            #f97316
        ) !important;

    color: white !important;

    font-weight: 700 !important;
}

.stButton button:hover {
    background:
        linear-gradient(
            90deg,
            #ef4444,
            #fb923c
        ) !important;
}

/* TABS */

.stTabs [data-baseweb="tab-list"] {
    gap: 5px;

    background: rgba(15,23,42,0.75);

    padding: 5px;

    border-radius: 13px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 9px;

    color: #94a3b8;

    padding: 10px 18px;
}

.stTabs [aria-selected="true"] {
    background:
        linear-gradient(
            90deg,
            rgba(220,38,38,0.28),
            rgba(249,115,22,0.20)
        ) !important;

    color: #fed7aa !important;
}

/* SELECTBOX */

div[data-baseweb="select"] > div {
    background: #10192b !important;

    border-color: rgba(148,163,184,0.15) !important;

    color: white !important;
}

/* DATAFRAME */

[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}

/* FOOTER */

.footer {
    text-align: center;

    color: #475569;

    font-size: 11px;

    padding: 35px 0 10px;
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

    return (
        interns,
        course_df,
        pd.DataFrame(rows)
    )


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

    n_components = max(
        1,
        min(8, min(matrix.shape) - 1)
    )

    model = TruncatedSVD(
        n_components=n_components,
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

    scores = prediction_matrix.loc[intern].copy()

    completed = interaction_matrix.loc[intern]

    scores[completed > 0] = -999

    top = scores.sort_values(
        ascending=False
    ).head(count)

    results = []

    for course_name, score in top.items():

        info = course_df[
            course_df["Course"] == course_name
        ].iloc[0]

        percentage = int(
            np.clip(
                70 + float(score) * 8,
                72,
                98
            )
        )

        results.append({
            "Course": course_name,
            "Category": info["Category"],
            "Level": info["Level"],
            "Duration": info["Duration"],
            "Score": percentage
        })

    return pd.DataFrame(results)


# =========================================================
# GROQ
# =========================================================

def generate_ai_plan(
    intern,
    recommendations
):

    try:

        from groq import Groq

        api_key = None

        try:
            api_key = st.secrets["GROQ_API_KEY"]
        except Exception:
            api_key = os.getenv("GROQ_API_KEY")

        if not api_key:

            return (
                "Add your GROQ_API_KEY to Streamlit Secrets "
                "to activate the AI Learning Advisor."
            )

        client = Groq(
            api_key=api_key
        )

        courses_text = "\n".join(
            [
                f"- {row['Course']} "
                f"({row['Category']}, {row['Level']})"
                for _, row in recommendations.iterrows()
            ]
        )

        prompt = f"""
You are an AI learning advisor.

Intern: {intern}

Personalized recommendations:

{courses_text}

Create a practical personalized learning plan.

Include:
- Recommended starting point
- Learning sequence
- Skills to develop
- One practical project
- Expected outcome

Keep it concise and professional.
"""

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional "
                        "learning-path advisor."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.5,
            max_completion_tokens=700,
            include_reasoning=False
        )

        return response.choices[0].message.content

    except Exception as e:

        return (
            "The AI Advisor could not connect right now.\n\n"
            f"Error: `{type(e).__name__}`"
        )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        "<h1 style='text-align:center;'>🎓</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <h2 style="
            text-align:center;
            color:white;
            margin-top:-10px;
        ">
        LearnPath <span style="color:#f97316;">AI</span>
        </h2>
        """,
        unsafe_allow_html=True
    )

    st.caption(
        "Personalized Learning Paths"
    )

    st.divider()

    st.markdown("### 🧭 Navigation")

    st.markdown(
        "🏠 **Home**"
    )

    st.markdown(
        "👤 **Intern Profile**"
    )

    st.markdown(
        "📊 **Learning Analytics**"
    )

    st.markdown(
        "🧠 **AI Advisor**"
    )

    st.divider()

    st.markdown("### ⚙️ Personalization")

    selected_intern = st.selectbox(
        "Select Intern",
        interns
    )

    recommendation_count = st.slider(
        "Number of Recommendations",
        3,
        10,
        6
    )

    st.divider()

    st.markdown("### 🔬 AI Pipeline")

    st.caption(
        "1️⃣ Learning history"
    )

    st.caption(
        "2️⃣ Interaction matrix"
    )

    st.caption(
        "3️⃣ Matrix Factorization"
    )

    st.caption(
        "4️⃣ Course prediction"
    )

    st.caption(
        "5️⃣ Personalized path"
    )

    st.caption(
        "6️⃣ Groq AI Advisor"
    )


# =========================================================
# HERO
# =========================================================

st.markdown(
    """
    <div class="hero-box">

        <div class="hero-title">
            🎓 LearnPath <span>AI</span>
        </div>

        <div class="hero-subtitle">
            Personalized learning paths powered by
            Matrix Factorization + Generative AI
        </div>

        <div class="hero-description">
            Turn previous intern learning behavior into
            a customized training journey.
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# PROFILE
# =========================================================

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

recommendations = get_recommendations(
    selected_intern,
    recommendation_count
)


st.markdown(
    """
    <div class="glass-card">

        <div class="profile-title">
            👤 Intern Learning Profile
        </div>

        <div class="profile-subtitle">
            Insights based on previous learning behavior
        </div>

    </div>
    """,
    unsafe_allow_html=True
)

st.write("")


# =========================================================
# METRICS
# =========================================================

c1, c2, c3, c4 = st.columns(4)

with c1:

    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-icon">📚</div>
            <div class="metric-value">
                {interaction_count}
            </div>
            <div class="metric-label">
                Learning Interactions
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c2:

    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-icon">⭐</div>
            <div class="metric-value">
                {average_rating}/5
            </div>
            <div class="metric-label">
                Average Engagement
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c3:

    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-icon">🧩</div>
            <div class="metric-value">
                {category_count}
            </div>
            <div class="metric-label">
                Skill Categories
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c4:

    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-icon">🎯</div>
            <div class="metric-value">
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

st.write("")

tab1, tab2, tab3 = st.tabs(
    [
        "✨ Recommended Path",
        "📊 Learning Analytics",
        "🧠 AI Advisor"
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
            Recommendations generated using
            Collaborative Filtering + Matrix Factorization.
        </div>
        """,
        unsafe_allow_html=True
    )

    icons = [
        "🐍",
        "🤖",
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
        "purple"
    ]

    for i, (_, row) in enumerate(
        recommendations.iterrows()
    ):

        style = styles[
            i % len(styles)
        ]

        left, center, right = st.columns(
            [0.7, 6, 1]
        )

        with left:

            st.markdown(
                f"""
                <div class="course-icon">
                    {icons[i % len(icons)]}
                </div>
                """,
                unsafe_allow_html=True
            )

        with center:

            st.markdown(
                f"""
                <div class="course-card {style}">

                    <div class="badge">
                        #{i + 1} RECOMMENDED
                    </div>

                    <div class="course-name">
                        {row["Course"]}
                    </div>

                    <div class="course-meta">
                        {row["Category"]}
                        &nbsp; • &nbsp;
                        {row["Level"]}
                        &nbsp; • &nbsp;
                        {row["Duration"]} weeks
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        with right:

            st.markdown(
                f"""
                <div style="
                    margin-top:20px;
                    text-align:center;
                ">
                    <span class="score-badge">
                        {row["Score"]}%
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.write("")

    csv_data = recommendations.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "📥  Download Learning Path",
        data=csv_data,
        file_name=(
            f"{selected_intern}_"
            "learning_path.csv"
        ),
        mime="text/csv"
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

    col1, col2 = st.columns(2)

    with col1:

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

    with col2:

        st.markdown(
            "### 📈 Category Engagement"
        )

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

        <div class="section-caption">
            Groq analyzes the personalized recommendations
            and converts them into a practical learning strategy.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="glass-card">

            <h3 style="color:white;">
                🤖 Personalized AI Strategy
            </h3>

            <p style="color:#94a3b8;">
                Generate a customized roadmap based on
                the intern's historical learning behavior.
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    if st.button(
        "✨ Generate AI Learning Plan",
        use_container_width=True
    ):

        with st.spinner(
            "Creating personalized learning strategy..."
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
        Collaborative Filtering
        &nbsp; • &nbsp;
        Matrix Factorization
        &nbsp; • &nbsp;
        Groq AI
    </div>
    """,
    unsafe_allow_html=True
)
