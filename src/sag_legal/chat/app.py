"""FastAPI chat UI over the finance SAG pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from sag_legal.chat.pipeline import (
    ChatResult,
    answer_query,
    compare_query,
    corpus_ready,
    get_corpus,
    keys_configured,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="SAG Legal Chat", version="0.1.0")

ChatMode = Literal["sag", "rag", "compare"]


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    mode: ChatMode = "sag"


class CitedOut(BaseModel):
    chunk_id: str
    document_id: str
    title: str
    citation_path: str
    text: str


class ChatStatsOut(BaseModel):
    seed_count: int
    context_count: int
    sag_added: int


class ArmOut(BaseModel):
    """One retrieval+draft arm (SAG or traditional RAG)."""

    label: str
    use_sag: bool
    answer: str
    abstained: bool
    cited: list[CitedOut]
    stats: ChatStatsOut


class ChatResponse(BaseModel):
    mode: ChatMode
    arms: list[ArmOut]


def _arm_from_result(result: ChatResult) -> ArmOut:
    label = "SAG" if result.use_sag else "RAG (no SAG)"
    return ArmOut(
        label=label,
        use_sag=result.use_sag,
        answer=result.answer,
        abstained=result.abstained,
        cited=[
            CitedOut(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                title=c.title,
                citation_path=c.citation_path,
                text=c.text,
            )
            for c in result.cited
        ],
        stats=ChatStatsOut(
            seed_count=result.stats.seed_count,
            context_count=result.stats.context_count,
            sag_added=result.stats.sag_added,
        ),
    )


@app.on_event("startup")
def _warm_corpus() -> None:
    """Load corpus in the background of startup when files/keys allow."""
    voyage_ok, _openai_ok = keys_configured()
    if corpus_ready() and voyage_ok:
        try:
            get_corpus()
        except Exception:
            # Health endpoint reports readiness; don't crash the server.
            pass


@app.get("/api/health")
def health() -> dict[str, object]:
    voyage_ok, openai_ok = keys_configured()
    ready = corpus_ready() and voyage_ok and openai_ok
    return {
        "status": "ok" if ready else "degraded",
        "corpus": corpus_ready(),
        "voyage": voyage_ok,
        "openai": openai_ok,
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat(body: ChatRequest) -> ChatResponse:
    voyage_ok, openai_ok = keys_configured()
    if not corpus_ready():
        raise HTTPException(
            status_code=503,
            detail="Corpus missing. Place data/raw/uts_vlc_processed.json.",
        )
    if not voyage_ok or not openai_ok:
        raise HTTPException(
            status_code=503,
            detail="Set VOYAGE_API_KEY and OPENAI_API_KEY in .env.",
        )
    try:
        if body.mode == "compare":
            # One Voyage shortlist, then RAG vs SAG draft (fair ablation).
            rag, sag = compare_query(body.query, bundle=get_corpus())
            arms = [_arm_from_result(rag), _arm_from_result(sag)]
        elif body.mode == "rag":
            # K-matched vs SAG (5 seeds + 10 expand): Voyage top-15.
            arms = [_arm_from_result(answer_query(body.query, use_sag=False))]
        else:
            arms = [_arm_from_result(answer_query(body.query, use_sag=True))]
    except Exception as exc:  # noqa: BLE001 — surface pipeline failures to UI
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ChatResponse(mode=body.mode, arms=arms)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
