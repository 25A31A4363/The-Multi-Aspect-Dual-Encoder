import streamlit as st
import os
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime

# Import modules from src
from main import parse_jd_input, prepare_query_aspects, load_config
from encoder import EmbeddingCache, embed_aspect, load_model
from pipeline import process_dataset
from scoring import (
    compute_scores,
    cosine_similarity_matrix,
    calculate_behavioral_multiplier,
    is_honeypot
)

# Page configuration
st.set_page_config(
    page_title="SkyAid AI — Candidate Discovery Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium styling
st.markdown("""
<style>
    /* Premium Theme Styles */
    .stApp {
        background: linear-gradient(135deg, #0e1117 0%, #151922 100%);
        color: #f0f2f6;
    }
    
    /* Header Card */
    .header-card {
        background: rgba(25, 30, 42, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(8px);
    }
    .header-title {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 8px;
    }
    .header-subtitle {
        color: #8f9cae;
        font-size: 1.1rem;
        margin-bottom: 0px;
    }

    /* Candidate Card */
    .candidate-card {
        background: rgba(30, 36, 50, 0.7);
        border-left: 5px solid #00f2fe;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 0px 8px 8px 0px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .candidate-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 242, 254, 0.15);
        background: rgba(35, 42, 58, 0.8);
    }

    /* Metric Badges */
    .badge {
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
        margin-right: 8px;
    }
    .badge-score {
        background: rgba(0, 242, 254, 0.15);
        color: #00f2fe;
        border: 1px solid rgba(0, 242, 254, 0.3);
    }
    .badge-signal {
        background: rgba(79, 172, 254, 0.15);
        color: #4facfe;
        border: 1px solid rgba(79, 172, 254, 0.3);
    }
    .badge-warning {
        background: rgba(255, 75, 75, 0.15);
        color: #ff4b4b;
        border: 1px solid rgba(255, 75, 75, 0.3);
    }

    /* Similarity Meters */
    .sim-bar-container {
        width: 100%;
        background-color: #222a36;
        border-radius: 4px;
        margin-top: 4px;
        margin-bottom: 8px;
        height: 8px;
    }
    .sim-bar-fill {
        height: 100%;
        border-radius: 4px;
    }
    .sim-text {
        font-size: 0.85rem;
        color: #a0aec0;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to get default JD text
def get_default_jd():
    # Try reading the job description file from the workspace
    for filename in ["job_description.txt", "job_description.docx"]:
        path = os.path.join(os.path.dirname(__file__), "dataset", filename)
        if os.path.exists(path):
            return parse_jd_input(path)
    return """Job Description: Senior AI Engineer — Founding Team
Company: Redrob AI (Series A AI-talent intelligence platform)
Location: Pune/Noida, India (Hybrid)
Experience Required: 5–9 years

We are looking for a Founding AI Engineer to own the candidate search and match systems.
Required Skills:
- Production experience with embeddings-based retrieval systems (sentence-transformers, BGE, E5)
- Production experience with vector databases (Pinecone, FAISS, Milvus)
- Strong Python skills and building ranking evaluation frameworks (NDCG, MAP)

Disqualifiers:
- Pure research roles without production deployments.
- Candidates who have spent their entire career only at consulting firms (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini).
- Computer vision/speech/robotics engineers without NLP exposure."""

# Load config defaults
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
config = load_config(config_path) if os.path.exists(config_path) else {}

# App Header
st.markdown("""
<div class="header-card">
    <div class="header-title">SkyAid AI — Intelligent Candidate Discovery Engine</div>
    <div class="header-subtitle">Multi-Aspect Dual Encoder candidate retrieval with semantic title, project, and skill indexing & behavioral feedback loops.</div>
</div>
""", unsafe_allow_html=True)

# Sidebar Options
st.sidebar.title("🎛️ Engine Controls")

model_name = st.sidebar.text_input("Embedding Model", value=config.get("model_name", "BAAI/bge-large-en-v1.5"))
cache_path = st.sidebar.text_input("Cache Path", value=config.get("cache_path", "outputs/embeddings_cache.sqlite"))

st.sidebar.markdown("---")
st.sidebar.subheader("⚖️ Aspect Weights")
w1 = st.sidebar.slider("Title Relevance (W1)", 0.0, 1.0, config.get("default_weights", {}).get("w1", 0.4), 0.05)
w2 = st.sidebar.slider("Project Depth (W2)", 0.0, 1.0, config.get("default_weights", {}).get("w2", 0.4), 0.05)
w3 = st.sidebar.slider("Skill Match (W3)", 0.0, 1.0, config.get("default_weights", {}).get("w3", 0.2), 0.05)

# Weight validation
total_w = w1 + w2 + w3
if abs(total_w - 1.0) > 1e-4:
    st.sidebar.warning(f"⚠️ Weights do not sum to 1.0 (Sum: {total_w:.2f}). Scores will be scaled.")
else:
    st.sidebar.info("✨ Weights normalized perfectly (Sum: 1.0).")

st.sidebar.markdown("---")
st.sidebar.subheader("🔒 Defensive Engineering")
exclude_honeypots = st.sidebar.checkbox("Activate Honeypot Filters", value=True, help="Deterministic filters to block profiles with impossible dates, durations or skills.")
only_india = st.sidebar.checkbox("Require India/Relocation", value=False, help="Downweight profiles outside India who are unwilling to relocate.")

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Speed & Scale")
limit = st.sidebar.selectbox(
    "Dataset Limit", 
    [100, 500, 1000, 5000, 10000, 50000, 100000],
    index=2, 
    help="Limit records parsed to speed up CPU inference. The full dataset contains 100,000 records."
)
top_n = st.sidebar.slider("Candidates to Output (Top-N)", 5, 100, config.get("top_n", 10), 5)

# Main Workspace
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 Ingest Job Description (JD)")
    
    # Template loader
    if st.button("🔌 Load Released Senior AI Engineer JD Template"):
        st.session_state.jd_text = get_default_jd()
        
    jd_input = st.text_area(
        "Paste Job Description text here:", 
        value=st.session_state.get("jd_text", get_default_jd()), 
        height=320,
        key="jd_area"
    )

    # File uploader
    uploaded_file = st.file_uploader("Or upload a JD file (.txt or .docx)", type=["txt", "docx"])
    if uploaded_file is not None:
        # Save temp file
        temp_path = os.path.join(os.path.dirname(__file__), "dataset", "uploaded_jd" + os.path.splitext(uploaded_file.name)[1])
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        jd_input = parse_jd_input(temp_path)
        st.session_state.jd_text = jd_input
        st.success("Successfully loaded JD from file!")

with col2:
    st.subheader("📊 Engine Execution Status")
    
    dataset_file = os.path.join(os.path.dirname(__file__), "dataset", "candidates.jsonl")
    
    st.markdown(f"**Dataset Source:** `{os.path.basename(dataset_file)}` (100,000 Candidates Pool)")
    st.markdown(f"**Aspects Analyzed:** `job_titles` (Titles), `projects` (Experiences), `skills` (Skills)")
    
    if st.button("🚀 Search & Rank Candidates", type="primary", use_container_width=True):
        if not jd_input.strip():
            st.error("Please provide Job Description input.")
        else:
            with st.spinner("Executing Candidate Discovery Pipeline..."):
                t_start = time.time()
                
                # 1. Parse query
                query_aspects = prepare_query_aspects(jd_input)
                
                # 2. Process dataset (retrieve cached embeddings/compute misses)
                st.info("Parsing profiles and loading embeddings cache...")
                dataset_result = process_dataset(
                    dataset_file,
                    model_name,
                    cache_path,
                    force_recompute=False,
                    limit=limit
                )
                
                # 3. Load model to embed query
                st.info("Generating query embeddings...")
                model = load_model(model_name)
                db_cache = EmbeddingCache(cache_path)
                query_title_vector = embed_aspect(model, db_cache, "query", "job_titles", query_aspects["job_titles"])
                query_project_vector = embed_aspect(model, db_cache, "query", "projects", query_aspects["projects"])
                query_skill_vector = embed_aspect(model, db_cache, "query", "skills", query_aspects["skills"])
                db_cache.close()
                
                # 4. Similarities
                st.info("Computing multi-aspect similarity matrices...")
                title_scores = cosine_similarity_matrix(query_title_vector, dataset_result["title_embeddings"])
                project_scores = cosine_similarity_matrix(query_project_vector, dataset_result["project_embeddings"])
                skill_scores = cosine_similarity_matrix(query_skill_vector, dataset_result["skill_embeddings"])
                
                # 5. Modifiers
                st.info("Applying behavioral weights and traps detection...")
                final_scores_list = []
                honeypots_blocked = 0
                
                for idx, cid in enumerate(dataset_result["candidate_ids"]):
                    candidate_record = dataset_result["raw_records"][idx]
                    
                    # Honeypot check
                    if is_honeypot(candidate_record):
                        honeypots_blocked += 1
                        if exclude_honeypots:
                            final_scores_list.append(-1.0)
                            continue
                    
                    semantic_score = compute_scores(
                        title_scores[idx],
                        project_scores[idx],
                        skill_scores[idx],
                        w1=w1, w2=w2, w3=w3
                    )
                    
                    modifier = calculate_behavioral_multiplier(candidate_record)
                    
                    # Location flag override
                    if only_india:
                        profile = candidate_record.get("profile", {}) or {}
                        country = profile.get("country", "").strip().lower()
                        willing_relocate = candidate_record.get("redrob_signals", {}).get("willing_to_relocate", False)
                        if country != 'india' and not willing_relocate:
                            modifier *= 0.1
                            
                    final_scores_list.append(float(semantic_score * modifier))
                
                final_scores = np.array(final_scores_list)
                
                # 6. Tiebreaker Sort (score desc, candidate_id asc)
                candidate_items = []
                for idx in range(len(dataset_result["candidate_ids"])):
                    candidate_items.append((
                        idx,
                        final_scores[idx],
                        dataset_result["candidate_ids"][idx]
                    ))
                candidate_items.sort(key=lambda x: (-x[1], x[2]))
                
                sorted_indices = [x[0] for x in candidate_items]
                num_to_output = min(top_n, len(sorted_indices))
                top_indices = sorted_indices[:num_to_output]
                
                t_end = time.time()
                latency = t_end - t_start
                
                st.success(f"Candidate Search Completed in {latency:.2f} seconds!")
                
                # Summary metrics
                m_col1, m_col2, m_col3 = st.columns(3)
                m_col1.metric("Candidates Scanned", len(dataset_result["candidate_ids"]))
                m_col2.metric("Honeypots Blocked", honeypots_blocked)
                m_col3.metric("Search Latency", f"{latency:.2f}s")
                
                # Store results in session state
                st.session_state.results = {
                    "indices": top_indices,
                    "candidate_ids": [dataset_result["candidate_ids"][i] for i in top_indices],
                    "candidate_names": [dataset_result["candidate_names"][i] for i in top_indices],
                    "final_scores": [final_scores[i] for i in top_indices],
                    "title_scores": [title_scores[i] for i in top_indices],
                    "project_scores": [project_scores[i] for i in top_indices],
                    "skill_scores": [skill_scores[i] for i in top_indices],
                    "raw_records": [dataset_result["raw_records"][i] for i in top_indices]
                }

# Results Display Section
if "results" in st.session_state:
    st.markdown("---")
    st.subheader(f"🏆 Top {len(st.session_state.results['candidate_ids'])} Candidate Recommendations")
    
    results = st.session_state.results
    
    # Load reasonings cache
    from main import load_reasonings_cache, generate_fallback_reasoning
    reasonings_cache = load_reasonings_cache()
    
    for rank_idx, (cid, cname, final, t_sim, p_sim, s_sim, record) in enumerate(zip(
        results["candidate_ids"],
        results["candidate_names"],
        results["final_scores"],
        results["title_scores"],
        results["project_scores"],
        results["skill_scores"],
        results["raw_records"]
    )):
        profile = record.get("profile", {}) or {}
        signals = record.get("redrob_signals", {}) or {}
        skills = record.get("skills", []) or []
        career = record.get("career_history", []) or []
        
        # Inactivity
        last_active_str = signals.get("last_active_date", "2026-06-01")
        days_inactive = 0
        try:
            last_active = datetime.strptime(last_active_str, "%Y-%m-%d")
            ref_date = datetime(2026, 6, 1)
            days_inactive = (ref_date - last_active).days
        except:
            pass

        # Formatting values
        yoe = profile.get("years_of_experience", 0.0)
        curr_title = profile.get("current_title", "N/A")
        curr_company = profile.get("current_company", "N/A")
        location = profile.get("location", "N/A")
        notice = signals.get("notice_period_days", 30)
        resp_rate = signals.get("recruiter_response_rate", 1.0) * 100
        github = signals.get("github_activity_score", -1)
        
        # Display Candidate Card
        st.markdown(f"""
        <div class="candidate-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-size: 1.3rem; font-weight: bold; color: #f0f2f6;">
                    #{rank_idx + 1} — {cname} <span style="font-size: 0.9rem; color: #8f9cae; font-weight: normal;">(ID: {cid})</span>
                </span>
                <div>
                    <span class="badge badge-score">Match Score: {final:.4f}</span>
                    <span class="badge badge-signal">Notice: {notice}d</span>
                    <span class="badge badge-signal">Response Rate: {resp_rate:.0f}%</span>
                    {"<span class='badge badge-warning'>Honeypot/Trap Alert</span>" if is_honeypot(record) else ""}
                </div>
            </div>
            
            <div style="font-size: 0.95rem; color: #cbd5e0; margin-bottom: 12px;">
                💼 <b>{curr_title}</b> at <b>{curr_company}</b> | ⏳ <b>{yoe:.1f} YOE</b> | 📍 <b>{location}</b> | 💻 GitHub Activity: <b>{github if github >= 0 else 'N/A'}</b> | ⏱️ Active: <b>{days_inactive} days ago</b>
            </div>
            
            <div style="display: flex; gap: 24px; margin-bottom: 12px;">
                <div style="flex: 1;">
                    <div style="display: flex; justify-content: space-between;" class="sim-text">
                        <span>Title Match Score</span>
                        <span>{t_sim:.4f}</span>
                    </div>
                    <div class="sim-bar-container">
                        <div class="sim-bar-fill" style="width: {min(100, max(0, int(t_sim * 100)))}%; background-color: #4facfe;"></div>
                    </div>
                </div>
                <div style="flex: 1;">
                    <div style="display: flex; justify-content: space-between;" class="sim-text">
                        <span>Project Context Match</span>
                        <span>{p_sim:.4f}</span>
                    </div>
                    <div class="sim-bar-container">
                        <div class="sim-bar-fill" style="width: {min(100, max(0, int(p_sim * 100)))}%; background-color: #00f2fe;"></div>
                    </div>
                </div>
                <div style="flex: 1;">
                    <div style="display: flex; justify-content: space-between;" class="sim-text">
                        <span>Skill Overlap Score</span>
                        <span>{s_sim:.4f}</span>
                    </div>
                    <div class="sim-bar-container">
                        <div class="sim-bar-fill" style="width: {min(100, max(0, int(s_sim * 100)))}%; background-color: #38b2ac;"></div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Skills block
        st.markdown(f"**Core Skills:** " + ", ".join([f"`{s.get('name')}`" for s in skills[:8]]))
        
        # Reasoning lookup
        reasoning = reasonings_cache.get(cid)
        if not reasoning:
            reasoning = generate_fallback_reasoning(record, final, t_sim, p_sim, s_sim)
        
        # Accordion details
        with st.expander(f"🔍 Show Detailed Reasoning & Career History"):
            st.markdown(f"**Expert-Generated Reasoning:**")
            st.info(reasoning)
            
            st.markdown(f"**Career History Details:**")
            for job in career:
                job_title = job.get("title", "")
                company = job.get("company", "")
                s_date = job.get("start_date", "")
                e_date = job.get("end_date") or "Present"
                dur = job.get("duration_months", 0)
                desc = job.get("description", "")
                
                st.markdown(f"- **{job_title}** at *{company}* ({s_date} to {e_date}, {dur} months)")
                st.write(desc)
        
        st.markdown("</div>", unsafe_allow_html=True)
