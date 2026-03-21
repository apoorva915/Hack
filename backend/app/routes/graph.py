from __future__ import annotations

from fastapi import APIRouter

from app.services.graph.skill_graph_engine import load_graph

router = APIRouter()


@router.get("/stats")
def graph_stats() -> dict[str, object]:
    graph = load_graph()
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "sample_edges": list(graph.edges())[:10],
    }
