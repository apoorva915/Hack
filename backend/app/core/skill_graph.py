from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set


@dataclass(frozen=True)
class SkillNode:
    skill_id: str
    name: str
    prerequisites: List[str]
    default_level: str  # beginner/intermediate/advanced


class SkillGraph:
    def __init__(self, skills_db_path: Path):
        self._nodes: Dict[str, SkillNode] = {}
        self._load(skills_db_path)

    def _load(self, path: Path) -> None:
        import json

        raw = json.loads(path.read_text(encoding="utf-8"))
        for item in raw:
            node = SkillNode(
                skill_id=item["skill_id"],
                name=item.get("name", item["skill_id"]),
                prerequisites=item.get("prerequisites", []),
                default_level=item.get("default_level", "beginner"),
            )
            self._nodes[node.skill_id] = node

    def get(self, skill_id: str) -> Optional[SkillNode]:
        return self._nodes.get(skill_id)

    def prerequisites(self, skill_id: str) -> List[str]:
        node = self.get(skill_id)
        return list(node.prerequisites) if node else []

    def ensure_all(self, skill_ids: List[str]) -> List[str]:
        """
        Filters unknown skill IDs.
        """
        return [s for s in skill_ids if s in self._nodes]

    def topological_sort(self, skill_ids: List[str]) -> List[str]:
        """
        Topological sort for prerequisite ordering within the provided subset.
        """
        subset: Set[str] = set(self.ensure_all(skill_ids))

        # Kahn's algorithm
        in_degree: Dict[str, int] = {s: 0 for s in subset}
        graph: Dict[str, List[str]] = {s: [] for s in subset}

        for s in subset:
            for prereq in self.prerequisites(s):
                if prereq in subset:
                    graph[prereq].append(s)
                    in_degree[s] += 1

        queue = [s for s, deg in in_degree.items() if deg == 0]
        ordered: List[str] = []

        while queue:
            cur = queue.pop()
            ordered.append(cur)
            for nxt in graph.get(cur, []):
                in_degree[nxt] -= 1
                if in_degree[nxt] == 0:
                    queue.append(nxt)

        # If there is a cycle, fall back to input order for safety.
        if len(ordered) != len(subset):
            ordered = [s for s in skill_ids if s in subset]
        return ordered

    def expand_with_prerequisites(self, skill_ids: List[str]) -> List[str]:
        """
        Includes transitive prerequisites for each skill.
        Returns a unique list.
        """
        subset: Set[str] = set(self.ensure_all(skill_ids))
        stack = list(subset)
        while stack:
            cur = stack.pop()
            for prereq in self.prerequisites(cur):
                if prereq not in subset and prereq in self._nodes:
                    subset.add(prereq)
                    stack.append(prereq)
        return list(subset)

    def get_name(self, skill_id: str) -> str:
        node = self.get(skill_id)
        return node.name if node else skill_id

    def get_default_level(self, skill_id: str) -> str:
        node = self.get(skill_id)
        return node.default_level if node else "beginner"


def load_skill_graph() -> SkillGraph:
    data_dir = Path(__file__).resolve().parents[1] / "data"
    return SkillGraph(data_dir / "skills_db.json")

