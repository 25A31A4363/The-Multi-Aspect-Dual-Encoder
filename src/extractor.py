import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

SECTION_PATTERNS = {
    "skills": [r"skills?", r"technical skills", r"tools", r"technology stack"],
    "projects": [r"projects?", r"selected projects", r"work experience", r"experience"],
    "job_titles": [r"experience", r"career", r"employment history", r"professional experience"],
}

SENTENCE_SPLIT_PATTERN = re.compile(r"[\.\n]+\s*")


def _normalize_text(text: str) -> str:
    return text.replace("\r", "\n").strip()


def _split_lines(text: str) -> List[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _find_section_boundaries(lines: List[str]) -> Dict[str, List[str]]:
    sections: Dict[str, List[str]] = {"job_titles": [], "projects": [], "skills": []}
    current_section = "projects"

    for line in lines:
        lowered = line.lower()
        if any(re.search(pattern, lowered) for pattern in SECTION_PATTERNS["skills"]):
            current_section = "skills"
            continue
        if any(re.search(pattern, lowered) for pattern in SECTION_PATTERNS["job_titles"]):
            current_section = "job_titles"
            continue
        if any(re.search(pattern, lowered) for pattern in SECTION_PATTERNS["projects"]):
            current_section = "projects"
            continue
        sections[current_section].append(line)

    return sections


def _extract_job_titles_from_text(text: str) -> List[str]:
    lines = _split_lines(text)
    titles: List[str] = []
    for line in lines:
        if re.search(r"\b(lead|manager|engineer|developer|architect|consultant|analyst|director|vp|principal)\b", line, re.I):
            titles.append(line)
        elif " at " in line.lower() and "," in line:
            titles.append(line)
    return titles[:10] if titles else lines[:3]


def _extract_skills_from_text(text: str) -> List[str]:
    if not text:
        return []
    skills = re.split(r"[,;\n]|\band\b|\bwith\b", text)
    skills = [skill.strip(" .-\t") for skill in skills if skill.strip()]
    return skills


def _extract_projects_from_text(text: str) -> List[str]:
    sentences = [s.strip() for s in SENTENCE_SPLIT_PATTERN.split(text) if s.strip()]
    projects: List[str] = []
    for sentence in sentences:
        if len(sentence.split()) >= 8:
            projects.append(sentence)
    return projects[:10]


def extract_structural_aspects(raw_record: Any, candidate_id: Optional[str] = None) -> Dict[str, Any]:
    """Extract job_titles, projects, and skills from a raw record.

    Args:
        raw_record: Raw text or a dictionary containing candidate data.
        candidate_id: Optional identifier for logging.

    Returns:
        A dictionary with keys `job_titles`, `projects`, and `skills`.
    """
    if isinstance(raw_record, dict):
        if all(key in raw_record for key in ("job_titles", "projects", "skills")):
            return {
                "job_titles": raw_record.get("job_titles") or [],
                "projects": raw_record.get("projects") or [],
                "skills": raw_record.get("skills") or [],
            }

        if "profile" in raw_record or "career_history" in raw_record or "skills" in raw_record:
            profile = raw_record.get("profile", {}) or {}
            career = raw_record.get("career_history", []) or []
            skills_list = raw_record.get("skills", []) or []

            # Sort career history chronologically (oldest to newest) to preserve career trajectory
            sorted_career = sorted(
                career,
                key=lambda x: x.get("start_date") or "1970-01-01"
            )

            job_titles = [job.get("title", "") for job in sorted_career if job.get("title")]
            # If no titles in career, fall back to current title / headline
            if not job_titles:
                if profile.get("current_title"):
                    job_titles.append(profile["current_title"])
                elif profile.get("headline"):
                    job_titles.append(profile["headline"])

            projects = [job.get("description", "") for job in sorted_career if job.get("description")]
            # Include summary as it contains project details
            if profile.get("summary"):
                projects.insert(0, profile["summary"])

            # If skills_list is a list of dicts:
            if skills_list and isinstance(skills_list[0], dict):
                skills = [s.get("name", "") for s in skills_list if s.get("name")]
            else:
                skills = [str(s) for s in skills_list if s]

            return {
                "job_titles": job_titles,
                "projects": projects,
                "skills": skills,
            }

        joined = "\n".join(str(raw_record.get(field, "")) for field in ["experience", "work_experience", "projects", "skills", "summary"] if raw_record.get(field))
    else:
        joined = str(raw_record)

    joined = _normalize_text(joined)
    lines = _split_lines(joined)
    sections = _find_section_boundaries(lines)

    job_titles = _extract_job_titles_from_text("\n".join(sections["job_titles"]))
    projects = _extract_projects_from_text("\n".join(sections["projects"]))
    skills = _extract_skills_from_text("\n".join(sections["skills"]))

    if not any([job_titles, projects, skills]):
        logger.warning("Malformed or empty record skipped: %s", candidate_id or "unknown")
        return {"job_titles": [], "projects": [], "skills": []}

    return {
        "job_titles": job_titles,
        "projects": projects,
        "skills": skills,
    }


def parse_candidate_record(record: Any, candidate_id: Optional[str] = None) -> Dict[str, Any]:
    """Convert a raw candidate record into a structured profile."""
    try:
        return extract_structural_aspects(record, candidate_id=candidate_id)
    except Exception as exc:
        logger.warning("Skipping malformed record %s: %s", candidate_id or "unknown", exc)
        return {"job_titles": [], "projects": [], "skills": []}


if __name__ == "__main__":
    sample_profiles = [
        {
            "name": "Asha Sharma",
            "experience": "Senior Software Engineer at Acme Corp\nLead developer for a payments platform, built microservices, APIs, and devops automation.\nSkills: Python, Django, AWS, Docker, Kubernetes",
        },
        "John Doe\nProject: Built an ML recommendation engine using Python and Spark. Skills: PyTorch, SQL, AWS, Kubernetes. Worked as Data Engineer at FinTech startup.",
        {
            "summary": "Engineering leader with experience in product delivery.",
            "work_experience": "Director of Engineering at XYZ\nManaged teams building SaaS analytics products.\nProjects: launched customer-facing metrics dashboard with React and Node.js.\nSkills: JavaScript, React, Node.js, PostgreSQL",
        },
    ]

    for idx, profile in enumerate(sample_profiles, start=1):
        parsed = parse_candidate_record(profile, candidate_id=str(idx))
        print(f"SAMPLE {idx}")
        print(parsed)
        print("---")
