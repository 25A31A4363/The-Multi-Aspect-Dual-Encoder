# Implementation Plan: India Native AI Candidate Discovery Engine

## Goal
Build a high-accuracy, multi-aspect dual encoder candidate discovery engine optimized for semantic precision and speed. The system must avoid flat keyword matching and instead use structural candidate profile extraction with weighted scoring across titles, projects, and skills.

## Functional Phases

### 1. DATASET PROCESSING & STRUCTURAL EXTRACTION
- [ ] Confirm challenge dataset location and file format in the workspace.
- [ ] Create a Python pipeline to load candidate profiles from the dataset.
- [ ] Define structural schema for each profile with explicit fields:
  - `job_titles`
  - `projects`
  - `skills`
- [ ] Parse each profile into these three fields rather than treating the profile as a single text block.
- [ ] Validate that extracted fields preserve career trajectory, deep project context, and raw technical skill signals.

### 2. MULTI-ASPECT EMBEDDING PIPELINE
- [ ] Select an open-source embedding model suitable for retrieval and semantic similarity (e.g., HuggingFace `bge-large-en-v1.5`).
- [ ] Implement separate embedding generation for `job_titles`, `projects`, and `skills`.
- [ ] Cache or persist embeddings locally to minimize repeated model inference and speed retrieval.
- [ ] Design a local index for candidate vectors to support fast lookup.

### 3. DUAL-ENCODER QUERY PARSING
- [ ] Build a query ingestion module for Job Description (JD) text input.
- [ ] Use the same encoder model to generate JD embeddings.
- [ ] Ensure the JD embedding is compatible with candidate aspect embeddings.

### 4. STRUCTURAL WEIGHTED SCORING ENGINE
- [ ] Compute cosine similarity scores for each candidate aspect:
  - Title Similarity
  - Project Similarity
  - Skill Similarity
- [ ] Implement an adjustable scoring formula:
  - `Final_Score = (W1 * Title_Similarity) + (W2 * Project_Similarity) + (W3 * Skill_Similarity)`
- [ ] Set default weights:
  - `W1 = 0.4`
  - `W2 = 0.4`
  - `W3 = 0.2`
- [ ] Rank candidates by `Final_Score`.
- [ ] Add command-line or config support for tuning weights.

### 5. REQUIRED DELIVERABLES & OUTPUT FORMATTING
- [ ] Create clean, production-grade Python code with documentation.
- [ ] Provide a reproducible project structure.
- [ ] Generate output file containing top recommended candidates.
- [ ] Follow the hackathon's exact output formatting rules when available.
- [ ] Include a README with usage instructions and dependencies.

## Project Structure
- `implementation_plan.md`
- `README.md`
- `requirements.txt`
- `src/`
  - `data/` or `dataset/`
  - `pipeline.py`
  - `encoder.py`
  - `scoring.py`
  - `main.py`
- `outputs/`
  - `top_candidates.csv` or `top_candidates.json`

## Next Steps
1. Get your approval for the proposed architecture and pipeline.
2. Confirm where the dataset is located and whether any output formatting rules are already provided.
3. Scaffold the Python project and install required dependencies.

## Notes
- This plan is intentionally spec-driven and modular.
- The system will treat candidate profiles as multi-aspect by design and avoid flat keyword heuristics.
- Cached embeddings will ensure the system can scale across Hackathon dataset sizes.
