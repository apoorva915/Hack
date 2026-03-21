from __future__ import annotations

from typing import Dict, List


class SkillGapAnalyzer:
    def compute_gap(self, resume_skills: list[str], jd_skills: list[str]) -> Dict[str, List[str]]:
        """
        Compute missing and matching skills.

        Gap = JD skills - Resume skills
        """
        resume_set = {skill.strip().lower() for skill in resume_skills if isinstance(skill, str) and skill.strip()}
        jd_set = {skill.strip().lower() for skill in jd_skills if isinstance(skill, str) and skill.strip()}

        missing = jd_set - resume_set
        matching = resume_set.intersection(jd_set)

        return {
            "resume_skills": sorted(resume_set),
            "jd_skills": sorted(jd_set),
            "missing_skills": sorted(missing),
            "matching_skills": sorted(matching),
        }
