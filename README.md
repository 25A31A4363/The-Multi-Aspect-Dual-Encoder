# 🔍 The Multi-Aspect Dual Encoder — Intelligent Candidate Discovery Engine

**India Runs Data & AI Hackathon — Track 01**
**Challenge:** Redrob Intelligent Candidate Discovery & Ranking

## 📖 About the Project

This is a multi-aspect dual-encoder system that ranks candidates against a Job Description (JD) using **semantic understanding instead of keyword matching**.

Instead of treating a resume as one block of text, every candidate profile is broken into three structural aspects:

- **Job Titles** — career trajectory and seniority signal
- **Projects** — technical depth and real work context
- **Skills** — raw technical capability

Each aspect is embedded separately with `BAAI/bge-large-en-v1.5`, compared against the same aspects extracted from the JD using cosine similarity, then combined into a single weighted score. A behavioral multiplier (availability, location fit, experience range, etc.) and a set of honeypot filters (catching inflated/inconsistent profiles) refine the ranking before the final list is written out.

## ⚙️ How to Run It

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

## ✅ Advantages

- **No flat keyword matching** — captures meaning, not just word overlap (e.g. "led a team" ≈ "managed engineers")
- **Multi-aspect precision** — a candidate with the right skills but wrong seniority, or right title but no relevant project depth, is scored accurately instead of blended into one noisy vector
- **Explainable rankings** — every result shows the title/project/skill sub-scores, not just a black-box number
- **Tunable per role** — weights can be shifted (e.g. skills-heavy for niche tech roles, title-heavy for leadership
