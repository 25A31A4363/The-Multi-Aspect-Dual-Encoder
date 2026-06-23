# 🔍 The Multi-Aspect Dual Encoder — Intelligent Candidate Discovery Engine

> **India Runs Data & AI Hackathon** — Track 01 Submission  
> Redrob Intelligent Candidate Discovery & Ranking Challenge

A production-grade, multi-aspect dual-encoder candidate ranking system that semantically matches candidates to a Job Description across three structural aspects: **job titles**, **project experience**, and **skills** — combined with behavioral signal modifiers and defensive honeypot filters.

---

## 🏗️ Architecture

```
candidates.jsonl ──▶ extractor.py ──▶ encoder.py ──▶ SQLite Cache
                                                         │
job_description ──▶ extractor.py ──▶ encoder.py ──▶ scoring.py
                                                         │
                                               weighted similarity
                                               + behavioral modifiers
                                               + honeypot filters
                                                         │
                                               outputs/top_candidates.csv
                                               + Streamlit UI (app.py)
```

### Scoring Formula

$$\text{Final Score} = \underbrace{(W_1 \cdot \text{Title Sim} + W_2 \cdot \text{Project Sim} + W_3 \cdot \text{Skill Sim})}_{\text{Semantic Score}} \times \underbrace{\text{Availability} \times \text{Location} \times \text{YOE} \times \text{Consulting} \times \text{Specialization}}_{\text{Behavioral Multiplier}}$$

Default weights: `W1=0.4, W2=0.4, W3=0.2`

---

## 📁 Project Structure

```
.
├── README.md
├── requirements.txt
├── config.yaml                       # Configurable model, cache, weights, paths
├── implementation_plan.md            # Architecture documentation
├── submission_metadata.yaml          # Hackathon submission metadata
├── src/
│   ├── __init__.py
│   ├── extractor.py                  # Structural aspect extraction (titles/projects/skills)
│   ├── encoder.py                    # BGE embedding model + SQLite cache
│   ├── pipeline.py                   # Batch JSONL processing + FAISS index
│   ├── scoring.py                    # Cosine similarity, behavioral modifiers, honeypot filters
│   ├── main.py                       # CLI entrypoint
│   ├── app.py                        # Interactive Streamlit UI
│   └── dataset/
│       ├── candidate_schema.json     # Official dataset schema
│       ├── sample_candidates.json    # 50 sample profiles for testing
│       ├── sample_submission.csv     # Format reference
│       ├── submission_spec.docx      # Submission specification
│       ├── validate_submission.py    # Official format validator
│       └── submission_metadata_template.yaml
└── outputs/
    └── top_candidates.csv            # Final ranked output (generated at runtime)
```

---

## ⚙️ Setup

### 1. Clone and install dependencies
```bash
git clone https://github.com/25A31A4363/The-Multi-Aspect-Dual-Encoder.git
cd The-Multi-Aspect-Dual-Encoder
pip install -r requirements.txt
```

### 2. Download the dataset
Place `candidates.jsonl` from the hackathon bundle into `src/dataset/`.

---

## 🚀 Run

### Command-line ranking
```bash
python -m src.main --jd "src/dataset/job_description.docx" --top_n 100
```

With custom weights:
```bash
python -m src.main --jd "src/dataset/job_description.docx" --top_n 100 --w1 0.4 --w2 0.4 --w3 0.2
```

With a dataset limit (for faster testing):
```bash
python -m src.main --jd "src/dataset/job_description.docx" --top_n 100 --limit 5000
```

### Interactive Web UI (Streamlit)
```bash
streamlit run src/app.py
```
Opens at **http://localhost:8501** — paste any JD, tune weights with sliders, and see ranked candidates with explainability cards.

### Validate submission output
```bash
python src/dataset/validate_submission.py outputs/top_candidates.csv
```

---

## 🎛️ Configuration (`config.yaml`)

```yaml
dataset_path: "src/dataset/candidates.jsonl"
output_path: "outputs/top_candidates.csv"
model_name: "BAAI/bge-large-en-v1.5"
cache_path: "outputs/embeddings_cache.sqlite"
default_weights:
  w1: 0.4   # Title relevance
  w2: 0.4   # Project/experience depth
  w3: 0.2   # Skill overlap
top_n: 100
```

Override any value at runtime via CLI flags.

---

## 🧠 How It Works

### 1. Structural Extraction (`extractor.py`)
Every candidate profile from `candidates.jsonl` is parsed into exactly **three structural fields**:
- `job_titles` — All role titles in career order (preserves seniority trajectory)
- `projects` — All job descriptions + profile summary (preserves technical context/depth)
- `skills` — Raw skill name tokens from the skills array

For raw text (like JDs), regex + heuristic section detection is used as a fallback.

### 2. Multi-Aspect Embeddings (`encoder.py`)
Three separate embeddings are generated per candidate using `BAAI/bge-large-en-v1.5` via `sentence-transformers`. Embeddings are **cached to SQLite** keyed by `(candidate_id, aspect)` — so re-runs skip unchanged candidates entirely.

### 3. Weighted Scoring (`scoring.py`)
Vectorized cosine similarity (NumPy) is computed per aspect. Weights are tunable and validated (warn if ≠ 1.0).

### 4. Behavioral Multiplier (`scoring.py`)
A multiplicative modifier (0.0–1.0) adjusts each candidate's raw semantic score based on:

| Signal | Effect |
|---|---|
| `open_to_work_flag = False` | ×0.9 |
| `recruiter_response_rate` | ×(0.3 + 0.7 × rate) |
| Days since last active (>30d) | Gradual decay up to ×0.6 |
| `notice_period_days > 30` | ×0.80–0.95 |
| Years of experience outside 5–9 range | ×0.5–0.9 |
| Location outside preferred India cities | ×0.3–0.8 |
| Only consulting-firm career history | ×0.5 |
| CV/Robotics skills, no NLP exposure | ×0.7 |

### 5. Honeypot Detection (`scoring.py`)
Three deterministic rules flag impossible profiles (score → -1.0, excluded from top-100):
1. **Expert proficiency with 0 months used** — "expert" skill but `duration_months == 0`
2. **Duration > total experience** — a single job's duration exceeds the candidate's stated total YOE by > 1 year
3. **Duration ≠ date span** — `duration_months` exceeds the `start_date` → `end_date` span by > 12 months

### 6. FAISS Indexing (`pipeline.py`)
Three FAISS `IndexFlatIP` indexes (one per aspect) are scaffolded over normalized embeddings. At the current 100k scale, exact search completes in milliseconds; the FAISS structure allows transparent migration to approximate ANN search (`IndexIVFFlat`) without changing the API.

---

## 📊 Output Format

The official submission CSV (`outputs/top_candidates.csv`) follows the exact hackathon spec:

```
candidate_id,rank,score,reasoning
CAND_0012345,1,0.823451,"Exceptional fit with 7.2 YOE as a Senior ML Engineer..."
```

An explainability CSV (`outputs/top_candidates_explainability.csv`) is also written for debugging:
```
rank,candidate_id,candidate_name,final_score,title_similarity,project_similarity,skill_similarity,years_of_experience,location,recruiter_response_rate,notice_period
```

---

## 📦 Dataset

- **Source:** `[PUB] India_runs_data_and_ai_challenge.zip` from the official hackathon bundle
- **Raw schema:** Each record in `candidates.jsonl` contains:
  - `candidate_id` — `CAND_XXXXXXX` format
  - `profile` — headline, summary, current title, location, years_of_experience
  - `career_history` — array of jobs with title, company, description, dates, duration
  - `skills` — array of `{name, proficiency, duration_months}`
  - `redrob_signals` — 23 behavioral platform signals

### Extraction assumptions
- `job_titles` → extracted from `career_history[*].title`, sorted chronologically, with `profile.current_title` as fallback
- `projects` → extracted from `career_history[*].description` + `profile.summary` (preserves full technical context)
- `skills` → extracted as raw `skill.name` strings (proficiency and duration used only for honeypot detection and behavioral modifiers, not for embedding text)

---

## 🔒 Compute Compliance

| Constraint | Limit | Our System |
|---|---|---|
| Runtime | ≤ 5 min | ~2–3 min (1st run with embedding) / <10s (cached) |
| Memory | ≤ 16 GB RAM | ~3–4 GB peak |
| Compute | CPU only | ✅ CPU-only (torch CPU build) |
| Network | Off during ranking | ✅ All embeddings pre-cached locally |

---

## 🧾 Assumptions

1. No hackathon-provided output spec beyond the validator script was found — defaulted to `candidate_id,rank,score,reasoning` CSV per `submission_spec.docx`.
2. Projects are distinguished from skills by source field (`career_history[].description` vs `skills[].name`) — no heuristics needed since the dataset is structured.
3. Honeypot detection uses deterministic profile inconsistency checks (not ML-based) to ensure reproducibility and auditability.
4. The reference date for "days since last active" calculations is `2026-06-01` based on the observed max `last_active_date` in the dataset.
