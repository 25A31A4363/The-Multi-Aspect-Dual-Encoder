import argparse
import logging
import os
import sys
import json
import zipfile
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional
import numpy as np
import yaml

from .extractor import extract_structural_aspects
from .encoder import EmbeddingCache, embed_aspect, load_model
from .pipeline import process_dataset
from .scoring import (
    compute_scores,
    cosine_similarity_matrix,
    validate_weights,
    calculate_behavioral_multiplier,
    is_honeypot
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def read_docx(docx_path: str) -> str:
    """Extract raw text from a DOCX file using built-in XML parsing."""
    try:
        with zipfile.ZipFile(docx_path) as z:
            xml_content = z.read('word/document.xml')
            root = ET.fromstring(xml_content)
            ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            paragraphs = []
            for p in root.findall('.//w:p', ns):
                text_runs = []
                for r in p.findall('.//w:r', ns):
                    t = r.find('w:t', ns)
                    if t is not None and t.text:
                        text_runs.append(t.text)
                if text_runs:
                    paragraphs.append(''.join(text_runs))
            return '\n'.join(paragraphs)
    except Exception as e:
        logger.warning("Failed to read docx file %s: %s", docx_path, e)
        return ""


def parse_jd_input(jd_input: str) -> str:
    """Read the JD from a path (txt/docx) or return it directly if it's text."""
    if os.path.exists(jd_input):
        _, ext = os.path.splitext(jd_input.lower())
        if ext == '.docx':
            text = read_docx(jd_input)
            if text:
                return text
        try:
            with open(jd_input, "r", encoding="utf-8") as handle:
                return handle.read().strip()
        except Exception as e:
            logger.warning("Error reading text file %s: %s", jd_input, e)
    return jd_input


def prepare_query_aspects(jd_text: str) -> Dict[str, str]:
    """Parse JD text into structural aspects or fall back to full JD text."""
    parsed = extract_structural_aspects(jd_text, candidate_id="job_description")
    return {
        "job_titles": " \n ".join(parsed["job_titles"]) if parsed["job_titles"] else jd_text,
        "projects": " \n ".join(parsed["projects"]) if parsed["projects"] else jd_text,
        "skills": " \n ".join(parsed["skills"]) if parsed["skills"] else jd_text,
    }


def load_reasonings_cache() -> Dict[str, str]:
    """Load pre-generated high-quality reasonings from cache JSON."""
    cache_path = os.path.join(os.path.dirname(__file__), "dataset", "reasonings_cache.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception as e:
            logger.warning("Failed to load reasonings cache: %s", e)
    return {}


def generate_fallback_reasoning(
    candidate_record: Dict[str, Any],
    score: float,
    title_sim: float,
    project_sim: float,
    skill_sim: float
) -> str:
    """Generate a structured, fact-based fallback reasoning for explainability."""
    profile = candidate_record.get("profile", {}) or {}
    redrob_signals = candidate_record.get("redrob_signals", {}) or {}
    skills = candidate_record.get("skills", []) or []

    yoe = profile.get("years_of_experience", 0.0)
    title = profile.get("current_title", "Software Engineer")
    skills_names = [s.get("name") for s in skills[:3] if s.get("name")]
    skills_str = ", ".join(skills_names) if skills_names else "matching technical skills"

    resp_rate = redrob_signals.get("recruiter_response_rate", 1.0) * 100
    notice = redrob_signals.get("notice_period_days", 30)

    # Contextual tone depending on score quality
    if score >= 0.7:
        reasoning = (
            f"Exceptional fit with {yoe:.1f} YOE as a {title}. Strong semantic match across "
            f"experience aspects (Titles: {title_sim:.2f}, Projects: {project_sim:.2f}) and skills ({skills_str}). "
            f"Excellent engagement signal of {resp_rate:.0f}% response rate and quick {notice}-day notice."
        )
    elif score >= 0.4:
        reasoning = (
            f"Solid candidate with {yoe:.1f} YOE as a {title}. Moderate similarity across titles ({title_sim:.2f}) "
            f"and projects ({project_sim:.2f}) with key skills like {skills_str}. Notice period is {notice} days."
        )
    else:
        reasoning = (
            f"Weak fit with {yoe:.1f} YOE. Low aspect matches (Title: {title_sim:.2f}, Projects: {project_sim:.2f}) "
            f"and potential availability gaps ({notice}-day notice period, {resp_rate:.0f}% recruiter response rate)."
        )
    return reasoning


def save_ranked_candidates(
    output_path: str,
    candidate_ids: List[str],
    candidate_names: List[str],
    final_scores: np.ndarray,
    title_scores: np.ndarray,
    project_scores: np.ndarray,
    skill_scores: np.ndarray,
    raw_records: List[Dict[str, Any]],
    top_n: int
) -> None:
    """Save ranked candidates to CSV. Writes official submission CSV and explainable metadata CSV."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    # Load high-quality reasoning cache if available
    reasonings_cache = load_reasonings_cache()

    # 1. Save official submission CSV (candidate_id,rank,score,reasoning)
    official_header = "candidate_id,rank,score,reasoning\n"
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(official_header)
        for i in range(top_n):
            cid = candidate_ids[i]
            score = final_scores[i]
            rank = i + 1
            
            # Lookup cached reasoning or fallback
            reasoning = reasonings_cache.get(cid)
            if not reasoning:
                reasoning = generate_fallback_reasoning(
                    raw_records[i],
                    score,
                    title_scores[i],
                    project_scores[i],
                    skill_scores[i]
                )
            
            # Escape reasoning field to comply with CSV quotes
            reasoning_escaped = reasoning.replace('"', '""')
            handle.write(f'{cid},{rank},{score:.6f},"{reasoning_escaped}"\n')

    # 2. Save explainable metadata CSV for internal review
    explain_path = output_path.replace(".csv", "_explainability.csv")
    explain_header = "rank,candidate_id,candidate_name,final_score,title_similarity,project_similarity,skill_similarity,years_of_experience,location,recruiter_response_rate,notice_period\n"
    with open(explain_path, "w", encoding="utf-8") as handle:
        handle.write(explain_header)
        for i in range(top_n):
            cid = candidate_ids[i]
            cname = candidate_names[i]
            score = final_scores[i]
            rank = i + 1
            rec = raw_records[i]
            prof = rec.get("profile", {}) or {}
            sig = rec.get("redrob_signals", {}) or {}
            yoe = prof.get("years_of_experience", 0.0)
            loc = prof.get("location", "")
            resp_rate = sig.get("recruiter_response_rate", 0.0)
            notice = sig.get("notice_period_days", 0)
            
            handle.write(f'{rank},{cid},"{cname}",{score:.6f},{title_scores[i]:.6f},{project_scores[i]:.6f},{skill_scores[i]:.6f},{yoe:.1f},"{loc}",{resp_rate:.2f},{notice}\n')


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Multi-Aspect Dual Encoder Candidate Discovery Engine")
    parser.add_argument("--jd", required=True, help="Job description text or path")
    parser.add_argument("--dataset", help="Path to the candidate dataset file (.jsonl)")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--top_n", type=int, default=None, help="Number of top candidates to output")
    parser.add_argument("--w1", type=float, default=None, help="Weight for title similarity")
    parser.add_argument("--w2", type=float, default=None, help="Weight for project similarity")
    parser.add_argument("--w3", type=float, default=None, help="Weight for skill similarity")
    parser.add_argument("--model_name", default=None, help="Embedding model name")
    parser.add_argument("--cache_path", default=None, help="Path to embedding cache database")
    parser.add_argument("--output_path", default=None, help="Output CSV path")
    parser.add_argument("--force_recompute", action="store_true", help="Recompute embeddings even if cached values exist")
    parser.add_argument("--limit", type=int, default=-1, help="Limit candidates processed for debug purposes")
    args = parser.parse_args(argv)

    # Load configuration
    config = load_config(args.config) if os.path.exists(args.config) else {}
    dataset_path = args.dataset or config.get("dataset_path") or "src/dataset/candidates.jsonl"
    output_path = args.output_path or config.get("output_path") or "outputs/top_candidates.csv"
    model_name = args.model_name or config.get("model_name") or "BAAI/bge-large-en-v1.5"
    cache_path = args.cache_path or config.get("cache_path") or "outputs/embeddings_cache.sqlite"
    top_n = args.top_n or config.get("top_n", 100)
    w1 = args.w1 if args.w1 is not None else config.get("default_weights", {}).get("w1", 0.4)
    w2 = args.w2 if args.w2 is not None else config.get("default_weights", {}).get("w2", 0.4)
    w3 = args.w3 if args.w3 is not None else config.get("default_weights", {}).get("w3", 0.2)

    logger.info("Initializing search engine with weights: Title=%.2f, Projects=%.2f, Skills=%.2f", w1, w2, w3)

    if not dataset_path or not os.path.exists(dataset_path):
        logger.error("Dataset path does not exist or was not specified: %s", dataset_path)
        return 1

    # Ingest query JD
    jd_text = parse_jd_input(args.jd)
    logger.info("Parsing Job Description...")
    query_aspects = prepare_query_aspects(jd_text)

    # Process all candidates (loads JSONL line-by-line, embeds, retrieves from cache, scaffolds FAISS)
    dataset_result = process_dataset(
        dataset_path,
        model_name,
        cache_path,
        force_recompute=args.force_recompute,
        limit=args.limit
    )
    if not dataset_result["candidate_ids"]:
        logger.error("No candidates were processed from the dataset.")
        return 1

    # Embed query aspects
    model = load_model(model_name)
    cache = EmbeddingCache(cache_path)
    query_title_vector = embed_aspect(model, cache, "query", "job_titles", query_aspects["job_titles"])
    query_project_vector = embed_aspect(model, cache, "query", "projects", query_aspects["projects"])
    query_skill_vector = embed_aspect(model, cache, "query", "skills", query_aspects["skills"])
    cache.close()

    # Compute similarity scores using exact cosine similarities
    title_scores = cosine_similarity_matrix(query_title_vector, dataset_result["title_embeddings"])
    project_scores = cosine_similarity_matrix(query_project_vector, dataset_result["project_embeddings"])
    skill_scores = cosine_similarity_matrix(query_skill_vector, dataset_result["skill_embeddings"])

    # Combine scores with aspect weights and apply behavioral modifiers & honeypot penalties
    validate_weights(w1, w2, w3)
    final_scores_list = []
    
    for idx, cid in enumerate(dataset_result["candidate_ids"]):
        candidate_record = dataset_result["raw_records"][idx]
        
        # Filter honeypots/trap profiles
        if is_honeypot(candidate_record):
            final_scores_list.append(-1.0)
        else:
            semantic_score = compute_scores(
                title_scores[idx],
                project_scores[idx],
                skill_scores[idx],
                w1=w1, w2=w2, w3=w3
            )
            # Apply behavioral modifier
            modifier = calculate_behavioral_multiplier(candidate_record)
            final_scores_list.append(float(semantic_score * modifier))

    final_scores = np.array(final_scores_list)

    # Deterministic tiebreaking: sort by score descending, then candidate_id ascending
    candidate_items = []
    for idx in range(len(dataset_result["candidate_ids"])):
        candidate_items.append((
            idx,
            final_scores[idx],
            dataset_result["candidate_ids"][idx]
        ))
    
    # Sort: score descending (-x[1]), candidate_id ascending (x[2])
    candidate_items.sort(key=lambda x: (-x[1], x[2]))
    
    # Get sorted indexes and slice to top_n
    sorted_indices = [x[0] for x in candidate_items]
    num_to_output = min(top_n, len(sorted_indices))
    top_indices = sorted_indices[:num_to_output]

    # Gather ranked outputs
    ranked_ids = [dataset_result["candidate_ids"][idx] for idx in top_indices]
    ranked_names = [dataset_result["candidate_names"][idx] for idx in top_indices]
    ranked_final = final_scores[top_indices]
    ranked_title = title_scores[top_indices]
    ranked_project = project_scores[top_indices]
    ranked_skill = skill_scores[top_indices]
    ranked_records = [dataset_result["raw_records"][idx] for idx in top_indices]

    # Save to file
    save_ranked_candidates(
        output_path,
        ranked_ids,
        ranked_names,
        ranked_final,
        ranked_title,
        ranked_project,
        ranked_skill,
        ranked_records,
        num_to_output
    )

    logger.info("Wrote top %d candidates to %s", num_to_output, output_path)
    
    # Log top 10 for terminal review
    for idx in range(min(10, num_to_output)):
        logger.info(
            "%d. %s (ID=%s) final=%.4f title_sim=%.4f project_sim=%.4f skill_sim=%.4f",
            idx + 1,
            ranked_names[idx] or "<unnamed>",
            ranked_ids[idx],
            ranked_final[idx],
            ranked_title[idx],
            ranked_project[idx],
            ranked_skill[idx],
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
