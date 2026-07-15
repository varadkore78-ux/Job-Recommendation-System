import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title=" Job Recommendation System",
   
    layout="wide"
)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    jobs = pd.read_csv("clean_job_small.csv")
    resumes = pd.read_csv("clean_resume_small.csv")
    return jobs, resumes


job_df, resume_df = load_data()

# ---------------- MODEL ----------------

@st.cache_resource
def build_model():

    tfidf = TfidfVectorizer(max_features=5000)

    job_vectors = tfidf.fit_transform(
        job_df["combined_text"]
    )

    return tfidf, job_vectors


tfidf, job_vectors = build_model()

# ---------------- CUSTOM CSS ----------------

st.markdown(
    """
    <style>

    .main-title{
        text-align:center;
        font-size:45px;
        font-weight:bold;
        color:#4CAF50;
    }

    .subtitle{
        text-align:center;
        font-size:20px;
        color:gray;
        margin-bottom:20px;
    }

    .card{
        background-color:#262730;
        padding:20px;
        border-radius:15px;
        border:1px solid #444;
        margin-bottom:20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- HEADER ----------------

st.markdown(
    '<p class="main-title">  Job Recommendation System</p>',
    unsafe_allow_html=True
)



# ---------------- SIDEBAR ----------------

st.sidebar.title(" Settings")

resume_index = st.sidebar.number_input(
    "Enter Resume Index",
    min_value=0,
    max_value=len(resume_df) - 1,
    value=0
)

st.sidebar.success(
    f"Selected Resume: {resume_index}"
)

# ---------------- BUTTON ----------------

if st.button(" Recommend Jobs", use_container_width=True):

    with st.spinner("Finding jobs..."):

        selected_resume = resume_df.loc[
            resume_index,
            "combined_resume"
        ]

        resume_vector = tfidf.transform(
            [selected_resume]
        )

        similarity = cosine_similarity(
            resume_vector,
            job_vectors
        )

        top_jobs = similarity[0].argsort()[-5:][::-1]

        recommendations = job_df.iloc[top_jobs]

    st.success("Top 5 Recommended Jobs")

    col1, col2 = st.columns(2)

    for i, (_, row) in enumerate(
        recommendations.iterrows(),
        start=1
    ):

        card = f"""
        <div class="card">

        <h3> {row['job_title']}</h3>

        <p><b> Company:</b> {row['organization']}</p>

        <p><b> Location:</b> {row['location']}</p>

        <p><b> Sector:</b> {row['sector']}</p>

        <p><b> Salary:</b> {row['salary']}</p>

        <p><b> Job Type:</b> {row['job_type']}</p>

        </div>
        """

        if i % 2 == 0:
            col2.markdown(card, unsafe_allow_html=True)
        else:
            col1.markdown(card, unsafe_allow_html=True)


