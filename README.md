# AI Adaptive Onboarding Engine

Hackathon-ready full-stack app that analyzes a user's resume and a job description to generate a personalized, prerequisite-aware learning roadmap.

## What it does

1. **Resume parsing**: Accepts PDF/DOCX resume uploads and extracts clean text.
2. **Job description parsing**: Accepts pasted text or uploaded JD files.
3. **Skill extraction**: Extracts relevant skills and infers a skill level (`beginner` / `intermediate` / `advanced`).
4. **Skill gap analysis**: Compares resume skills vs JD skills using embeddings + cosine similarity (FAISS when available).
5. **Adaptive learning path**: Builds a dependency graph and sequences prerequisites before advanced topics.
6. **Reasoning trace (MANDATORY)**: Explains why each recommended skill was included.
7. **Resource recommendation**: Suggests learning resources per skill from a small built-in dataset.

## API (FastAPI)

- `POST /upload`
  - Multipart form data: `resume_file` (required), `job_description_text` (optional), `job_description_file` (optional)
  - Returns: `{ analysis_id, resume_text_preview, jd_text_preview }`

- `POST /analyze`
  - Body: `{ "analysis_id": "<id>" }`
  - Returns a full structured report: extracted skills, gaps, roadmap, and reasoning trace.

- `GET /roadmap?analysis_id=<id>`
  - Returns: roadmap + reasoning trace

Swagger UI is available at `/docs`.

## Architecture (modular services)

- `backend/app/services/parser.py` : resume/JD text extraction
- `backend/app/services/skill_extractor.py` : skill + level extraction (heuristics, optional OpenAI)
- `backend/app/services/skill_gap.py` : fuzzy matching using embeddings/FAISS
- `backend/app/services/learning_path.py` : dependency-aware roadmap generation
- `backend/app/services/reasoning.py` : reasoning trace (“why recommended”)
- `backend/app/services/resource_recommender.py` : resource suggestions from `courses.json`
- `backend/app/core/skill_graph.py` : skill dependency graph from `skills_db.json`

## Setup (local)

### Backend

1. Install Python dependencies:
   - `pip install -r backend/requirements.txt`
2. Run the server:
   - `uvicorn app.main:app --host 0.0.0.0 --port 8000`
3. Open:
   - `http://localhost:8000/docs`

### Frontend

1. Install:
   - `cd frontend && npm install`
2. Run:
   - `npm run dev`
3. Open:
   - `http://localhost:5173`

## Environment variables

Copy `.env.example` to `.env` (optional).

- `VITE_BACKEND_URL` (frontend) defaults to `http://localhost:8000`
- Backend options:
  - `FRONTEND_ORIGIN`
  - `SIMILARITY_THRESHOLD`
  - `EMBEDDINGS_MODEL_NAME`
  - `OPENAI_API_KEY` (optional; improves extraction coverage)

## Docker (optional)

From the repo root:

- `docker-compose up --build`

Backend: `http://localhost:8000`
Frontend: `http://localhost:5173`

## Sample inputs

Example text files are included at:

- `backend/app/data/sample_resume.txt`
- `backend/app/data/sample_jd.txt`

You can upload these as `.txt` to quickly demo the pipeline.

