import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config.settings import get_settings
from .core.skill_graph import load_skill_graph
from .routes.analysis import router as analysis_router
from .routes.upload import router as upload_router
from .services.state import AnalysisStore


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


setup_logging()
settings = get_settings()

app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


app.include_router(upload_router)
app.include_router(analysis_router)


@app.on_event("startup")
async def startup_event():
    # Store extracted text + computed results across the 3-step API.
    app.state.store = AnalysisStore(ttl_seconds=settings.analysis_ttl_seconds)
    app.state.skill_graph = load_skill_graph()
    logging.getLogger("backend").info("Startup complete: store + skill graph ready")

