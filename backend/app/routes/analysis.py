from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request

from ..models.schemas import AnalyzeRequest, AnalyzeResponse, RoadmapResponse
from ..services.learning_path import generate_learning_path
from ..services.reasoning import generate_reasoning_trace
from ..services.skill_gap import compute_skill_gaps
from ..services.skill_extractor import extract_skills_from_text


router = APIRouter(tags=["analysis"])


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: Request, payload: AnalyzeRequest):
    analysis_id = payload.analysis_id
    store = request.app.state.store
    graph = request.app.state.skill_graph

    try:
        item = store.get(analysis_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Invalid analysis_id. Call /upload first.")

    resume_skills = extract_skills_from_text(text=item.resume_text, source="resume")
    jd_skills = extract_skills_from_text(text=item.jd_text, source="job_description")

    gaps = compute_skill_gaps(resume_skills=resume_skills, jd_skills=jd_skills)
    recommended_skills, roadmap = generate_learning_path(gaps=gaps, graph=graph)
    reasoning_trace = generate_reasoning_trace(gaps=gaps, recommended_skills=recommended_skills, graph=graph)

    response = AnalyzeResponse(
        analysis_id=analysis_id,
        resume=resume_skills,
        job_description=jd_skills,
        skill_gaps=gaps,
        recommended_skills=recommended_skills,
        roadmap=roadmap,
        reasoning_trace=reasoning_trace,
    )

    store.set_result(analysis_id, response)
    return response


@router.get("/roadmap", response_model=RoadmapResponse)
async def roadmap(request: Request, analysis_id: str = Query(..., min_length=1)):
    store = request.app.state.store

    try:
        item = store.get(analysis_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Invalid analysis_id.")

    if item.analysis_result is None:
        raise HTTPException(
            status_code=400,
            detail="No analysis result found. Call POST /analyze first.",
        )

    result: AnalyzeResponse = item.analysis_result
    return RoadmapResponse(
        analysis_id=analysis_id, roadmap=result.roadmap, reasoning_trace=result.reasoning_trace
    )

