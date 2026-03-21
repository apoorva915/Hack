from __future__ import annotations

import logging

import networkx as nx

from app.services.graph.priority_scorer import PriorityScorer
from app.services.graph.skill_graph_engine import SkillGraphEngine

logger = logging.getLogger(__name__)


class AdaptivePathGenerator:
    def __init__(self, graph_engine: SkillGraphEngine | None = None) -> None:
        self.graph_engine = graph_engine or SkillGraphEngine()
        self.graph = self.graph_engine.get_graph()
        self.scorer = PriorityScorer()
        self.max_path = 20

    def generate_path(self, user_skills: list[str], target_skills: list[str]) -> dict[str, list]:
        valid_nodes = set(self.graph.nodes)

        target_skills_valid = [s.lower().strip() for s in target_skills if s.lower().strip() in valid_nodes]
        user_skills_valid = [s.lower().strip() for s in user_skills if s.lower().strip() in valid_nodes]
        logger.info("Valid target skills=%d", len(target_skills_valid))

        learning_set: set[str] = set()
        user_skill_set = set(user_skills_valid)

        for skill in target_skills_valid:
            if skill not in self.graph:
                continue
            if skill not in user_skill_set:
                prereqs = nx.ancestors(self.graph, skill)
                for prereq in prereqs:
                    if prereq not in user_skill_set:
                        learning_set.add(prereq)
                learning_set.add(skill)

        if not learning_set:
            return {"ordered_path": [], "phases": []}

        subgraph = self.graph.subgraph(learning_set).copy()
        try:
            ordered = list(nx.topological_sort(subgraph))
        except nx.NetworkXUnfeasible:
            logger.warning("Cycle detected in subgraph. Falling back to deterministic sorted order.")
            ordered = sorted(learning_set)

        scored = [(skill, self.scorer.score(skill)) for skill in ordered]
        ordered = [item[0] for item in sorted(scored, key=lambda item: item[1], reverse=True)]
        ordered = ordered[: self.max_path]

        phases = self.build_phases(ordered)
        logger.info("Learning path size=%d", len(ordered))
        logger.info("Phase count=%d", len(phases))
        return {"ordered_path": ordered, "phases": phases}

    def build_phases(self, ordered: list[str]) -> list[list[str]]:
        phases: list[list[str]] = []
        remaining = set(ordered)

        while remaining:
            phase: list[str] = []
            for skill in list(remaining):
                prereqs = list(self.graph.predecessors(skill))
                if all(prereq not in remaining for prereq in prereqs):
                    phase.append(skill)

            phase.sort()
            if not phase:
                # Safety break for unexpected cycles.
                phase = sorted(remaining)

            phases.append(phase)
            remaining -= set(phase)

        return phases

    def build_phase_map(self, phases: list[list[str]]) -> dict[str, list[str]]:
        phase_map: dict[str, list[str]] = {"Phase 1": [], "Phase 2": [], "Phase 3": []}
        if not phases:
            return phase_map

        # Map dynamic phases into 3 display buckets for current frontend compatibility.
        if len(phases) <= 3:
            for index, phase in enumerate(phases):
                phase_map[f"Phase {index + 1}"] = phase
            return phase_map

        flat = [skill for phase in phases for skill in phase]
        total = len(flat)
        for index, skill in enumerate(flat):
            progress = (index + 1) / total
            if progress <= 1 / 3:
                phase_map["Phase 1"].append(skill)
            elif progress <= 2 / 3:
                phase_map["Phase 2"].append(skill)
            else:
                phase_map["Phase 3"].append(skill)
        return phase_map

    def build_graph_payload(self, learning_path: list[str]) -> dict[str, list[dict[str, str]]]:
        learning_set = set(learning_path)
        nodes = [{"id": skill} for skill in learning_path]
        edges: list[dict[str, str]] = []

        for source, target in self.graph.edges():
            if source in learning_set and target in learning_set:
                edges.append({"source": source, "target": target})
        return {"nodes": nodes, "edges": edges}

    def estimate_learning_time(self, learning_path: list[str]) -> int:
        return sum(self.scorer.get_difficulty(skill) for skill in learning_path)

    @staticmethod
    def _normalize_skills(skills: list[str]) -> set[str]:
        return {skill.strip().lower() for skill in skills if isinstance(skill, str) and skill.strip()}
