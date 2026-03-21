import json
import logging
from functools import lru_cache
from pathlib import Path

import networkx as nx

logger = logging.getLogger(__name__)

GRAPH_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "skill_graph.json"


@lru_cache(maxsize=1)
def load_graph() -> nx.DiGraph:
    logger.info("Loading skill graph...")

    graph = nx.DiGraph()
    if not GRAPH_PATH.exists():
        logger.warning("Skill graph file not found at %s", GRAPH_PATH)
        return graph

    try:
        with GRAPH_PATH.open(encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        logger.exception("Invalid JSON in %s", GRAPH_PATH)
        return graph
    except OSError:
        logger.exception("Unable to read graph file %s", GRAPH_PATH)
        return graph

    if not isinstance(data, dict):
        logger.warning("Graph JSON root must be an object. Got %s", type(data).__name__)
        return graph

    edge_count = 0
    for raw_skill, raw_prereqs in data.items():
        if not isinstance(raw_skill, str):
            continue

        skill = raw_skill.lower().strip()
        if not skill:
            continue

        graph.add_node(skill)

        if raw_prereqs is None:
            prereqs = []
        elif isinstance(raw_prereqs, list):
            prereqs = raw_prereqs
        else:
            logger.warning("Invalid prereq format for '%s'. Expected list, got %s", skill, type(raw_prereqs).__name__)
            continue

        for prereq_raw in prereqs:
            if not isinstance(prereq_raw, str):
                continue
            prereq = prereq_raw.lower().strip()
            if not prereq:
                continue
            graph.add_node(prereq)
            graph.add_edge(prereq, skill)
            edge_count += 1

    logger.info("Graph loaded nodes=%d edges=%d", graph.number_of_nodes(), edge_count)
    logger.info("Sample edges: %s", list(graph.edges())[:10])
    return graph


class SkillGraphEngine:
    def __init__(self) -> None:
        self.graph = load_graph()

    def get_graph(self) -> nx.DiGraph:
        return self.graph

    def get_prerequisites(self, skill: str) -> set[str]:
        normalized_skill = skill.lower().strip()
        if normalized_skill not in self.graph:
            return set()
        return nx.ancestors(self.graph, normalized_skill)
