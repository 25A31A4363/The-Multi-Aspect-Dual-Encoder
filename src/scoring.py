import logging
from typing import Any, Dict, List, Tuple
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize rows of the embedding matrix."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return vectors / norms


def cosine_similarity_matrix(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between a query vector and candidate matrix."""
    if query.ndim == 1:
        query = query.reshape(1, -1)
    normalized_query = normalize_vectors(query)
    normalized_candidates = normalize_vectors(candidates)
    return np.squeeze(normalized_query.dot(normalized_candidates.T))


def validate_weights(w1: float, w2: float, w3: float) -> Tuple[float, float, float]:
    """Validate weight values and warn if they do not sum to 1."""
    total = w1 + w2 + w3
    if abs(total - 1.0) > 1e-6:
        logger.warning("Weights do not sum to 1.0 (sum is %s). Execution will continue.", total)
    return w1, w2, w3


def check_only_consulting(career_history: List[Dict[str, Any]]) -> bool:
    """Check if the candidate has worked ONLY at consulting/services firms."""
    if not career_history:
        return False
    consulting_firms = {'tcs', 'tata consultancy services', 'infosys', 'wipro', 'accenture', 'cognizant', 'capgemini', 'tech mahindra', 'hcl'}
    all_consulting = True
    for job in career_history:
        comp = job.get('company', '').lower().strip()
        is_consulting = False
        for firm in consulting_firms:
            if firm in comp:
                is_consulting = True
                break
        if not is_consulting:
            all_consulting = False
            break
    return all_consulting


def check_skills_specialization(skills: List[Dict[str, Any]]) -> float:
    """Apply cv/robotics vs nlp specialization factor.

    If primary skills are CV/Robotics without NLP/IR exposure, downweight slightly.
    """
    nlp_terms = {'nlp', 'natural language', 'search', 'information retrieval', 'retrieval', 'embeddings', 'rag', 'vector database', 'llm', 'transformers'}
    cv_robotics_terms = {'computer vision', 'image classification', 'object detection', 'speech recognition', 'robotics', 'vision'}

    skills_names = {s.get('name', '').lower() for s in skills}
    has_nlp = any(term in name for name in skills_names for term in nlp_terms)
    has_cv_robotics = any(term in name for name in skills_names for term in cv_robotics_terms)

    if has_cv_robotics and not has_nlp:
        return 0.7
    return 1.0


def calculate_behavioral_multiplier(candidate_record: Dict[str, Any]) -> float:
    """Compute the multiplicative behavioral modifier based on candidates' platform activities and criteria."""
    profile = candidate_record.get("profile", {}) or {}
    redrob_signals = candidate_record.get("redrob_signals", {}) or {}
    career_history = candidate_record.get("career_history", []) or []
    skills = candidate_record.get("skills", []) or []

    # 1. Availability / Open to Work Flag
    open_to_work = redrob_signals.get("open_to_work_flag", True)
    availability_factor = 1.0 if open_to_work else 0.9

    # 2. Recruiter Response Rate
    response_rate = redrob_signals.get("recruiter_response_rate", 1.0)
    response_factor = 0.3 + 0.7 * response_rate

    # 3. Last Active Date (Reference: 2026-06-01)
    last_active_str = redrob_signals.get("last_active_date", "2026-06-01")
    activity_factor = 1.0
    try:
        last_active = datetime.strptime(last_active_str, "%Y-%m-%d")
        ref_date = datetime(2026, 6, 1)
        days_inactive = (ref_date - last_active).days
        if days_inactive > 30:
            activity_factor = 1.0 - 0.4 * min(1.0, max(0.0, (days_inactive - 30) / 150.0))
    except Exception as exc:
        logger.debug("Error parsing last_active_date %s: %s", last_active_str, exc)

    # 4. Notice Period
    notice_days = redrob_signals.get("notice_period_days", 30)
    if notice_days <= 30:
        notice_factor = 1.0
    elif notice_days <= 60:
        notice_factor = 0.95
    elif notice_days <= 90:
        notice_factor = 0.90
    else:
        notice_factor = 0.80

    # 5. Experience Years (Preferred: 5-9)
    yoe = profile.get("years_of_experience", 0)
    if 5.0 <= yoe <= 9.0:
        yoe_factor = 1.0
    elif 4.0 <= yoe < 5.0 or 9.0 < yoe <= 11.0:
        yoe_factor = 0.9
    elif 3.0 <= yoe < 4.0 or 11.0 < yoe <= 13.0:
        yoe_factor = 0.7
    else:
        yoe_factor = 0.5

    # 6. Location Match
    country = profile.get("country", "").strip().lower()
    location = profile.get("location", "").strip().lower()
    willing_relocate = redrob_signals.get("willing_to_relocate", False)
    preferred_cities = {'pune', 'noida', 'delhi', 'ncr', 'gurgaon', 'hyderabad', 'mumbai', 'bangalore', 'chennai'}

    if country == 'india':
        has_pref_city = any(city in location for city in preferred_cities)
        if has_pref_city:
            location_factor = 1.0
        else:
            location_factor = 1.0 if willing_relocate else 0.8
    else:
        location_factor = 0.7 if willing_relocate else 0.3

    # 7. Consulting-Only Career Flag
    consulting_factor = 0.5 if check_only_consulting(career_history) else 1.0

    # 8. Specialization Match (CV/Robotics vs NLP)
    spec_factor = check_skills_specialization(skills)

    # Combined Multiplicative Factor
    multiplier = (
        availability_factor
        * response_factor
        * activity_factor
        * notice_factor
        * yoe_factor
        * location_factor
        * consulting_factor
        * spec_factor
    )
    return multiplier


def is_honeypot(candidate_record: Dict[str, Any]) -> bool:
    """Detect if a candidate profile is a honeypot (trap profile with impossible fields)."""
    profile = candidate_record.get("profile", {}) or {}
    skills = candidate_record.get("skills", []) or []
    career = candidate_record.get("career_history", []) or []

    # Rule 1: Expert proficiency in skills with 0 months used
    expert_zero_count = sum(1 for s in skills if s.get('proficiency') == 'expert' and s.get('duration_months', 0) == 0)
    if expert_zero_count > 0:
        return True

    # Rule 2: Single job duration is larger than total profile experience by > 1.0 year
    total_exp = profile.get('years_of_experience', 0)
    for job in career:
        dur_years = job.get('duration_months', 0) / 12.0
        if dur_years - total_exp > 1.0:
            return True

    # Rule 3: Single job duration exceeds the start/end date span by > 12 months
    for job in career:
        start = job.get('start_date')
        end = job.get('end_date') or "2026-06-01"
        duration = job.get('duration_months', 0)
        if start and end:
            try:
                sd = datetime.strptime(start, "%Y-%m-%d")
                ed = datetime.strptime(end, "%Y-%m-%d")
                span_months = (ed.year - sd.year) * 12 + ed.month - sd.month
                if duration - span_months > 12:
                    return True
            except:
                pass

    return False


def compute_scores(
    title_sim: np.ndarray,
    project_sim: np.ndarray,
    skill_sim: np.ndarray,
    w1: float = 0.4,
    w2: float = 0.4,
    w3: float = 0.2,
) -> np.ndarray:
    """Compute final weighted semantic scores from aspect similarities."""
    validate_weights(w1, w2, w3)
    return w1 * title_sim + w2 * project_sim + w3 * skill_sim


def test_weighted_scoring() -> None:
    """Unit test for the weighted scoring formula."""
    title_sim = np.array([0.5, 0.0, 1.0])
    project_sim = np.array([0.5, 1.0, 0.0])
    skill_sim = np.array([1.0, 0.5, 0.5])
    expected = np.array([0.6, 0.3, 0.7])
    scores = compute_scores(title_sim, project_sim, skill_sim, 0.4, 0.2, 0.4)
    assert np.allclose(scores, expected), f"Weighted scoring result mismatch: {scores} != {expected}"
    logger.info("Weighted scoring unit test passed.")


if __name__ == "__main__":
    test_weighted_scoring()
    print("Scoring test completed.")
