import streamlit as st
import pandas as pd
import numpy as np
import random
from sklearn.decomposition import TruncatedSVD

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="LearnPath AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# CUSTOM CSS
# =========================

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(99,102,241,0.20), transparent 25%),
        radial-gradient(circle at 90% 20%, rgba(168,85,247,0.18), transparent 25%),
        linear-gradient(135deg, #080b18 0%, #11152b 50%, #0b1020 100%);
    color: #f8fafc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

.hero {
    padding: 35px;
    border-radius: 25px;
    background: linear-gradient(
        135deg,
        rgba(99,102,241,0.22),
        rgba(168,85,247,0.16),
        rgba(15,23,42,0.75)
    );
    border: 1px solid rgba(255,255,255,0.10);
    box-shadow: 0 15px 50px rgba(0,0,0,0.30);
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 45px;
    margin-bottom: 8px;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero p {
    color: #cbd5e1;
    font-size: 17px;
}

.card {
    background: rgba(15,23,42,0.72);
    border: 1px solid rgba(148,163,184,0.15);
    border-radius: 18px;
    padding: 22px;
    margin-bottom: 18px;
    box-shadow: 0 10px 35px rgba(0,0,0,0.18);
}

.recommendation {
    background: linear-gradient(
        135deg,
        rgba(79,70,229,0.20),
        rgba(30,41,59,0.75)
    );
    border: 1px solid rgba(129,140,248,0.25);
    border-radius: 18px;
    padding: 20px;
    margin: 12px 0;
}

.badge {
    display: inline-block;
    padding: 5px 11px;
    border-radius: 20px;
    background: rgba(99,102,241,0.20);
    color: #c4b5fd;
    font-size: 12px;
    font-weight: 600;
}

.metric-card {
    background: rgba(15,23,42,0.70);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 17px;
    padding: 20px;
    text-align: center;
}

.metric-number {
    font-size: 30px;
    font-weight: 700;
    color: #a78bfa;
}

.metric-label {
    color: #94a3b8;
    font-size: 13px;
}

.stButton > button {
    width: 100%;
    border-radius: 12px;
    border: none;
    padding: 10px 20px;
    font-weight: 600;
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
    color: white;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #7c3aed, #6366f1);
    color: white;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b1020, #111827);
}

h2, h3 {
    color: #f8fafc;
}

</style>
""", unsafe_allow_html=True)


# =========================
# DEMO DATA
# =========================

@st.cache_data
def create_demo_data():

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
        columns=["Course", "Category", "Level", "Duration_Weeks"]
    )

    rows = []

    for intern in interns:

        # Create different learning preferences for each intern
        preferred_categories = random.sample(
            list(course_df["Category"].unique()),
            4
        )

        for _, course in course_df.iterrows():

            if course["Category"] in preferred_categories:
                probability = 0.72
            else:
                probability = 0.25

            if random.random() < probability:

                interaction = random.choice([1, 2, 3, 4, 5])

                rows.append({
                    "Intern": intern,
                    "Course": course["Course"],
                    "Rating": interaction
                })

    interactions = pd.DataFrame(rows)

    return interns, course_df, interactions


interns, course_df, interactions = create_demo_data()


# =========================
# MATRIX FACTORIZATION
# =========================

@st.cache_resource
def train_model(interaction_data):

    matrix = interaction_data.pivot_table(
        index="Intern",
        columns="Course",
        values="Rating",
        fill_value=0
    )

    # Make sure the matrix is large enough
    max_components = min(matrix.shape) - 1

    if max_components < 1:
        max_components = 1

    components = min(8, max_components)

    model = TruncatedSVD(
        n_components=components,
        random_state=42
    )

    latent_matrix = model.fit_transform(matrix)

    reconstructed = np.dot(
        latent_matrix,
        model.components_
    )

    predicted_matrix = pd.DataFrame(
        reconstructed,
        index=matrix.index,
        columns=matrix.columns
    )

    return matrix, predicted_matrix


interaction_matrix, prediction_matrix = train_model(
    interactions
)


# =========================
# RECOMMENDATION FUNCTION
# =========================

def get_recommendations(intern_name, number=6):

    if intern_name not in prediction_matrix.index:
        return pd.DataFrame()

    predictions = prediction_matrix.loc[intern_name].copy()

    already_taken = interaction_matrix.loc[intern_name]
    predictions[already_taken > 0] = -999

    top_courses = predictions.sort_values(
        ascending=False
    ).head(number)

    results = []

    for course_name, score in top_courses.items():

        course_info = course_df[
            course_df["Course"] == course_name
        ].iloc[0]

        # Convert model score into a simple percentage
        recommendation_score = min(
            99,
            max(
                65,
                int(65 + (float(score) * 7))
            )
        )

        results.append({
            "Course": course_name,
            "Category": course_info["Category"],
            "Level": course_info["Level"],
            "Duration": course_info["Duration_Weeks"],
            "Score": recommendation_score
        })

    return pd.DataFrame(results)


# =========================
# GROQ AI
# =========================

def generate_ai_explanation(intern_name, recommendations):

    try:

        from groq import Groq

        api_key = st.secrets.get("GROQ_API_KEY", "")

        if not api_key:
            return (
                "Add your GROQ_API_KEY in Streamlit Secrets to "
                "enable the AI learning advisor."
            )

        client = Groq(api_key=api_key)

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

Intern: {intern_name}

Recommended learning modules:
{course_list}

Explain the recommended learning path in simple professional language.

Include:
1. Why these modules fit the intern's learning pattern.
2. Which module they should start with.
3. What skills they can gain.
4. A short practical project idea.

Keep the response concise and useful.
"""

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional AI learning advisor."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.5,
            max_tokens=700
        )

        return response.choices[0].message.content

    except Exception as e:

        return (
            "AI explanation is temporarily unavailable. "
            "The recommendations are still generated by the "
            "Matrix Factorization model."
        )


# =========================
# HERO
# =========================

st.markdown("""
<div class="hero">

<h1>🎓 LearnPath AI</h1>

<p>
Personalized learning paths powered by
<strong>Matrix Factorization + Generative AI</strong>
</p>

<p>
Turn previous intern learning behavior into a customized
training journey.
</p>

</div>
""", unsafe_allow_html=True)


# =========================
# SIDEBAR
# =========================

st.sidebar.markdown("## 🎓 LearnPath AI")

st.sidebar.markdown(
    "### Personalized Learning Engine"
)

selected_intern = st.sidebar.selectbox(
    "👤 Select Intern",
    interns
)

recommendation_count = st.sidebar.slider(
    "📚 Number of Recommendations",
    min_value=3,
    max_value=10,
    value=6
)

st.sidebar.markdown("---")

st.sidebar.markdown("""
**🤖 AI Pipeline**

1. Learning history
2. Interaction matrix
3. Matrix Factorization
4. Course prediction
5. Personalized path
6. Groq AI explanation
""")


# =========================
# INTERN PROFILE
# =========================

recommendations = get_recommendations(
    selected_intern,
    recommendation_count
)

user_history = interactions[
    interactions["Intern"] == selected_intern
]

average_rating = round(
    user_history["Rating"].mean(),
    2
)

courses_completed = len(user_history)

unique_categories = user_history["Course"].map(
    course_df.set_index("Course")["Category"]
).nunique()


st.markdown("## 👤 Intern Learning Profile")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""
        <div class="metric-card">
        <div class="metric-number">{courses_completed}</div>
        <div class="metric-label">Learning Interactions</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"""
        <div class="metric-card">
        <div class="metric-number">{average_rating}/5</div>
        <div class="metric-label">Average Engagement</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        f"""
        <div class="metric-card">
        <div class="metric-number">{unique_categories}</div>
        <div class="metric-label">Skill Categories</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col4:
    st.markdown(
        f"""
        <div class="metric-card">
        <div class="metric-number">{len(course_df)}</div>
        <div class="metric-label">Available Modules</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================
# TABS
# =========================

tab1, tab2, tab3 = st.tabs(
    [
        "✨ Recommended Path",
        "📊 Learning Analytics",
        "🤖 AI Learning Advisor"
    ]
)


# =========================
# TAB 1
# =========================

with tab1:

    st.markdown("## ✨ Your Personalized Learning Path")

    st.caption(
        "Recommendations are generated from historical learning "
        "patterns using Matrix Factorization."
    )

    if recommendations.empty:

        st.warning(
            "Not enough learning history for recommendations."
        )

    else:

        for index, row in recommendations.iterrows():

            st.markdown(
                f"""
                <div class="recommendation">

                <span class="badge">
                #{index + 1} RECOMMENDED
                </span>

                <h3>📘 {row['Course']}</h3>

                <p>
                <strong>Category:</strong> {row['Category']}
                &nbsp;&nbsp; | &nbsp;&nbsp;
                <strong>Level:</strong> {row['Level']}
                &nbsp;&nbsp; | &nbsp;&nbsp;
                <strong>Duration:</strong> {row['Duration']} weeks
                </p>

                <p>
                🎯 Recommendation Score:
                <strong>{row['Score']}%</strong>
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )

        csv_data = recommendations.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "📥 Download Learning Path",
            data=csv_data,
            file_name=f"{selected_intern}_learning_path.csv",
            mime="text/csv"
        )


# =========================
# TAB 2
# =========================

with tab2:

    st.markdown("## 📊 Learning Analytics")

    left, right = st.columns(2)

    with left:

        st.markdown("### 📚 Previous Learning Activity")

        history_display = user_history.merge(
            course_df,
            on="Course"
        )[
            [
                "Course",
                "Category",
                "Level",
                "Rating"
            ]
        ]

        st.dataframe(
            history_display,
            use_container_width=True,
            hide_index=True
        )

    with right:

        st.markdown("### 🧠 Category Engagement")

        category_data = history_display.groupby(
            "Category"
        )["Rating"].mean().sort_values(
            ascending=False
        )

        st.bar_chart(category_data)


# =========================
# TAB 3
# =========================

with tab3:

    st.markdown("## 🤖 AI Learning Advisor")

    st.write(
        "Use Groq AI to turn the model's recommendations "
        "into a human-friendly learning strategy."
    )

    if st.button("✨ Generate My AI Learning Plan"):

        with st.spinner(
            "AI advisor is analyzing the recommended path..."
        ):

            explanation = generate_ai_explanation(
                selected_intern,
                recommendations
            )

        st.markdown(
            f"""
            <div class="card">

            <h3>🧠 AI Learning Strategy for {selected_intern}</h3>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(explanation)


# =========================
# FOOTER
# =========================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; color:#64748b;">
    🎓 LearnPath AI &nbsp;•&nbsp;
    Collaborative Filtering &nbsp;•&nbsp;
    Matrix Factorization &nbsp;•&nbsp;
    Groq AI
    </div>
    """,
    unsafe_allow_html=True
)
