import streamlit as st
import pdfplumber
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_text_from_pdf(pdf_file):
    """Extracts raw text from an uploaded PDF file."""
    with pdfplumber.open(pdf_file) as pdf:
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()

def calculate_match_score(resume_text, job_description):
    """Calculates the similarity percentage between a resume and a job description."""
    text_corpus = [resume_text, job_description]
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(text_corpus)
    similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    match_percentage = round(float(similarity_matrix[0][0]) * 100, 2)
    return match_percentage

# --- STREAMLIT DASHBOARD UI ---
st.set_page_config(page_title="AI Resume Screener", page_icon="💼", layout="wide")

st.title("💼 AI-Powered Resume Screener & Recruiter Dashboard")
st.markdown("Rank applicant resumes instantly using Natural Language Processing (NLP).")
st.markdown("---")

# Sidebar: Job Description Input
st.sidebar.header("📋 Job Specification")
job_desc = st.sidebar.text_area(
    "Paste the Job Description (JD) here:", 
    placeholder="e.g., Seeking a Python Developer with experience in Streamlit, Pandas, and Machine Learning...",
    height=300
)

# Main Screen: File Upload
st.header("📥 Upload Applicant Resumes")
uploaded_files = st.file_uploader(
    "Upload one or more candidate resumes (PDF format only):", 
    type=["pdf"], 
    accept_multiple_files=True
)

# --- PROCESSING PIPELINE ---
if uploaded_files and job_desc.strip():
    st.subheader("📊 Screened Candidates Leaderboard")
    
    results = []
    
    for file in uploaded_files:
        with st.spinner(f"Processing {file.name}..."):
            resume_text = extract_text_from_pdf(file)
            score = calculate_match_score(resume_text, job_desc)
            
            results.append({
                "Candidate Name": file.name.replace(".pdf", "").replace(".PDF", ""),
                "Match Score (%)": score
            })
            
    df = pd.DataFrame(results)
    df = df.sort_values(by="Match Score (%)", ascending=False).reset_index(drop=True)
    
    top_candidate = df.iloc[0]["Candidate Name"]
    top_score = df.iloc[0]["Match Score (%)"]
    
    col1, col2 = st.columns(2)
    col1.metric(label="Total Resumes Evaluated", value=len(df))
    col2.metric(label="Top Match Candidate", value=top_candidate, delta=f"{top_score}% Match")
    
    st.markdown("### 🏆 Final Rankings")
    st.dataframe(df, use_container_width=True)

elif uploaded_files and not job_desc.strip():
    st.warning("⚠️ Please paste a Job Description in the sidebar to begin screening.")
