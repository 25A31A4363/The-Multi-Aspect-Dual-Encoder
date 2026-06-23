# 🔍 Multi-Aspect Dual Encoder
### Intelligent Candidate Discovery & Ranking Engine
**India Runs Data & AI Hackathon — Track 01 | Redrob Challenge**

---

## 📌 Problem Statement

Traditional candidate ranking systems rely on **keyword matching** — a brittle approach that fails to capture what a resume truly communicates. A candidate who "led a team" is not matched against a JD asking for someone who "managed engineers," even though they mean the same thing. Worse, flat vector embeddings collapse the entire resume into a single noisy representation, masking the signal behind irrelevant content.

The result: strong candidates get buried, weak candidates get surfaced, and recruiters lose trust in the tool.

---

## 💡 Solution Overview

The **Multi-Aspect Dual Encoder** breaks every candidate profile into three structural aspects — **Job Titles**, **Projects**, and **Skills** — and embeds each independently using a state-of-the-art semantic model. These aspect embeddings are compared against the same aspects extracted from the Job Description (JD) using cosine similarity, then combined into a single weighted score.

A behavioral multiplier layer and a set of honeypot filters further refine the ranking — rewarding genuine fit and penalizing inflated or inconsistent profiles — before the final ranked list is written out.

---

## ✅ Key Features

- **Semantic understanding over keyword overlap** — captures meaning, not just word presence (e.g. "led a team" ≈ "managed engineers")
- **Multi-aspect precision** — a candidate with the right skills but wrong seniority, or the right title but no relevant project depth, is scored accurately instead of blended into one noisy vector
- **Explainable rankings** — every result surfaces per-aspect sub-scores (title / project / skill), not a black-box number
- **Tunable per role** — weights can be shifted at runtime (e.g. skills-heavy for niche tech roles, title-heavy for leadership positions)
- **Honeypot filtering** — catches inflated or internally inconsistent profiles before they pollute the ranking
- **Interactive UI** — Streamlit interface lets you paste a JD, adjust weight sliders, and inspect ranked results live

---

## ⚙️ How It Works

1. **Aspect Extraction** — Each resume is parsed into three structured fields: Job Titles, Projects, and Skills. The same extraction is applied to the incoming JD.

2. **Dual Encoding** — Each aspect pair (resume vs. JD) is independently encoded using `BAAI/bge-large-en-v1.5` and compared via cosine similarity, producing three sub-scores:
   - `score_title` — career trajectory and seniority alignment
   - `score_project` — technical depth and real-work context
   - `score_skill` — raw technical capability match

3. **Weighted Aggregation** — Sub-scores are combined using configurable weights:
   final_score = w1 × score_title + w2 × score_project + w3 × score_skill
   
5. **Behavioral Multiplier** — A scalar modifier adjusts the score based on availability, location fit, and experience range alignment.

6. **Honeypot Filtering** — Profiles flagged for inflation or inconsistency receive a penalty before the final ranking is produced.

7. **Output** — The top N candidates are written to `outputs/top_candidates.csv` with full sub-score breakdown.

---

## 🧠 Smart Decision-Making Logic

| Signal | How It's Used |
|---|---|
| Job Title match | Encodes seniority trajectory and role alignment |
| Project depth | Captures real-world technical context beyond claimed skills |
| Skills match | Direct technical capability comparison |
| Availability | Binary multiplier — unavailable candidates are down-ranked |
| Location fit | Soft multiplier — geographic mismatch reduces score |
| Experience range | Penalizes candidates significantly over or under the required band |
| Honeypot flags | Hard penalty for inflated YoE, skill count anomalies, or inconsistent signals |

Weights (`w1`, `w2`, `w3`) default to equal distribution but can be overridden at runtime to fit the hiring context.

---

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Semantic Embedding | `BAAI/bge-large-en-v1.5` (via HuggingFace) |
| Similarity Scoring | Cosine similarity |
| JD Parsing | Python-docx |
| Data Pipeline | Python, JSONL |
| Interactive UI | Streamlit |
| Output Format | CSV |

---

## 🏗️ Architecture Overview
Job Description (.docx)

│

▼

┌─────────────────────┐

│   Aspect Extractor  │ ──► [JD Titles, JD Projects, JD Skills]

└─────────────────────┘

│

candidates.jsonl                        │ cosine similarity (per aspect)

│                               │

▼                               │

┌─────────────────────┐                 │

│   Aspect Extractor  │ ──► [Resume Titles, Resume Projects, Resume Skills]

└─────────────────────┘

│

▼

┌──────────────────────────┐

│  BAAI/bge-large-en-v1.5  │  (dual encoder — JD side + Resume side)

└──────────────────────────┘

│

▼

┌──────────────────────────┐

│   Weighted Score Fusion  │  w1·title + w2·project + w3·skill

└──────────────────────────┘

│

▼

┌──────────────────────────┐

│  Behavioral Multiplier   │  availability · location · experience

└──────────────────────────┘

│

▼

┌──────────────────────────┐

│   Honeypot Filter Layer  │  penalize inflated / inconsistent profiles

└──────────────────────────┘

│

▼

outputs/top_candidates.csv

---

## 🎯 Alignment with Challenge Objectives

| Challenge Objective | How This System Addresses It |
|---|---|
| Semantic candidate matching | Per-aspect dual encoding captures meaning, not keywords |
| Ranked output with explainability | Sub-scores (title / project / skill) accompany every ranked result |
| Handling noisy/inflated profiles | Honeypot filter layer penalizes anomalous or inconsistent signals |
| Configurable ranking logic | Runtime weight flags allow role-specific tuning without code changes |
| Scalability | `--limit` flag enables fast local testing; batch encoding handles large datasets |

---

## 🚀 How to Run

**1. Clone and install**
```bash
git clone https://github.com/25A31A4363/The-Multi-Aspect-Dual-Encoder.git
cd The-Multi-Aspect-Dual-Encoder
pip install -r requirements.txt
```

**2. Add the dataset**

Place `candidates.jsonl` from the hackathon bundle into `src/dataset/`.

**3. Run the ranking**
```bash
python -m src.main --jd "src/dataset/job_description.docx" --top_n 100
```

With custom weights:
```bash
python -m src.main --jd "src/dataset/job_description.docx" --top_n 100 --w1 0.4 --w2 0.4 --w3 0.2
```

For quick local testing on a subset:
```bash
python -m src.main --jd "src/dataset/job_description.docx" --top_n 100 --limit 5000
```

**4. (Optional) Run the interactive UI**
```bash
streamlit run src/app.py
```
Opens at `http://localhost:8501` — paste a JD, adjust weight sliders, see ranked results live.

**5. Validate your output**
```bash
python src/dataset/validate_submission.py outputs/top_candidates.csv
```

---

## ♿ Accessibility

- The Streamlit UI uses standard HTML semantics and is screen-reader compatible out of the box
- All ranked output is written to a plain CSV — readable by any spreadsheet tool or downstream pipeline without proprietary software
- Weight controls are exposed as labeled sliders with visible numeric values, not opaque toggles
- Sub-scores are included in the output so results can be audited or re-ranked externally

---

## 🔮 Future Scope

- **Cross-lingual support** — extend to multilingual JDs and resumes using multilingual BGE variants
- **Dynamic weight learning** — learn per-role optimal weights from recruiter feedback signals rather than manual tuning
- **Graph-based re-ranking** — model candidate similarity as a graph and use PageRank-style propagation to surface diverse shortlists
- **Real-time streaming** — plug into ATS pipelines for live ranking as new applications arrive
- **Fine-tuned encoder** — fine-tune BGE on domain-specific recruiter-labeled pairs for higher precision in niche technical roles
- **Explainability dashboard** — highlight which specific phrases in the resume drove each sub-score, making the ranking auditable for hiring managers

---

## 🏁 Conclusion

The Multi-Aspect Dual Encoder moves candidate ranking from brittle keyword overlap to genuine semantic understanding — while keeping the ranking explainable, tunable, and resistant to profile inflation. By treating a resume as a structured set of signals rather than a flat document, it surfaces candidates who truly fit the role, not just candidates who wrote the right words.

Built for the **India Runs Data & AI Hackathon — Track 01**, Redrob Intelligent Candidate Discovery & Ranking challenge.
