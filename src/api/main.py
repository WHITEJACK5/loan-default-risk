"""HTTP surface for the loan-default-risk model.

Design constraints that shape this module:

* The model is **loaded, not trained, at startup.** A request handler must never
  enter a training loop; that made ``/predict`` take 130 seconds on the first
  call after deploy and made readiness meaningless as a signal.
* Every response carries the request ID so a client report can be traced to a
  single log line without shipping client IPs into storage.
* The body size is capped before the frame is parsed, because pydantic bounds
  a *known* field but cannot stop a 50MB JSON payload.
* Unhandled exceptions return 500 without a stack trace; the traceback goes to
  the log with the request ID attached.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.explain import explain
from src.api.model import load_model
from src.api.schemas import (
    AssignRequest,
    AssignResponse,
    ExplainResponse,
    LoanRequest,
    PredictResponse,
)
from src.experiments.assign import assign
from src.loan_default_risk.config import settings

# --- request guard rails (all overridable per deployment) ---
MAX_BODY_BYTES = int(os.environ.get("LDR_MAX_BODY_BYTES", 64 * 1024))
ALLOWED_ORIGINS = [o for o in os.environ.get("LDR_CORS_ORIGINS", "*").split(",") if o]

logging.basicConfig(
    level=os.environ.get("LDR_LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger("ldr.api")


class ModelNotReady(RuntimeError):
    """Raised when a prediction is attempted before the model finished loading."""


@asynccontextmanager
async def lifespan(app: FastAPI):
    started = time.perf_counter()
    app.state.model = load_model()
    app.state.model_version = "2.0.0"
    logger.info("model loaded version=%s", app.state.model_version)
    app.state.ready_at = time.perf_counter() - started
    yield
    app.state.model = None


app = FastAPI(
    title="loan-default-risk",
    version="2.0.0",
    description="Probability-of-default scoring with profit-thresholded decisions and adverse-action reasons.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Request-ID"],
)


@app.middleware("http")
async def guard_request_size(request: Request, call_next: Callable):
    """Reject oversized bodies with 413 before any parsing happens."""
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    request.state.request_id = request_id

    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > MAX_BODY_BYTES:
        logger.warning("rejected oversized body rid=%s bytes=%s limit=%s", request_id, declared, MAX_BODY_BYTES)
        return JSONResponse(status_code=413, content={"detail": "payload too large", "request_id": request_id})

    started = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        logger.info(
            "%s %s rid=%s status=%s %.1fms",
            request.method,
            request.url.path,
            request_id,
            status,
            (time.perf_counter() - started) * 1000,
        )


@app.exception_handler(Exception)
async def unhandled_error(request: Request, exc: Exception):
    """Never leak a traceback to the caller; log it with the request ID.

    ``exc_info`` is passed the traceback object rather than set to True because
    Starlette invokes this handler outside the original ``except`` block, where
    the implicit ``sys.exc_info()`` is no longer available.
    """
    request_id = getattr(request.state, "request_id", "-")
    logger.error(
        "unhandled error rid=%s path=%s",
        request_id,
        request.url.path,
        exc_info=exc.__traceback__,
    )
    return JSONResponse(status_code=500, content={"detail": "internal server error", "request_id": request_id})


def _require_model(request: Request):
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise ModelNotReady("model is not loaded; /health reports readiness")
    return model


@app.get("/health", tags=["ops"])
def health(request: Request) -> dict:
    """Liveness. Always 200 so a crash loops rather than a restart loop."""
    return {"status": "ok", "service": app.title, "version": app.version}


@app.get("/ready", tags=["ops"])
def ready(request: Request) -> dict:
    """Readiness. 503 while the model is being loaded."""
    if getattr(request.app.state, "model", None) is None:
        raise ModelNotReady("model unavailable")
    return {"model_loaded": True, "model_version": request.app.state.model_version}


def _score(request: Request, payload: LoanRequest) -> tuple[float, str]:
    model = _require_model(request)
    frame = payload.to_frame()
    probability = float(model.predict_proba(frame)[0, 1])
    return probability, "approve" if probability < settings.threshold else "reject"


@app.post("/predict", response_model=PredictResponse, tags=["scoring"])
def predict(payload: LoanRequest, request: Request) -> PredictResponse:
    """Score one application and apply the profit-maximising threshold."""
    probability, decision = _score(request, payload)
    return PredictResponse(
        pd=round(probability, 4),
        decision=decision,
        threshold=settings.threshold,
        model_version=getattr(request.app.state, "model_version", "unknown"),
        request_id=request.state.request_id,
    )


@app.post("/explain", response_model=ExplainResponse, tags=["scoring"])
def explain_application(payload: LoanRequest, request: Request) -> ExplainResponse:
    """Score one application and attach the features that raised its PD."""
    model = _require_model(request)
    probability = float(model.predict_proba(payload.to_frame())[0, 1])
    decision = "approve" if probability < settings.threshold else "reject"
    reasons = explain(model, payload.to_frame(), top_k=4)
    return ExplainResponse(
        pd=round(probability, 4),
        decision=decision,
        threshold=settings.threshold,
        model_version=getattr(request.app.state, "model_version", "unknown"),
        request_id=request.state.request_id,
        reasons=reasons,
    )


@app.post("/experiment/assign", response_model=AssignResponse, tags=["experiments"])
def experiment_assign(payload: AssignRequest, request: Request) -> AssignResponse:
    """Bucket a user deterministically for A/B assignment."""
    return AssignResponse(bucket=assign(payload.user_id, payload.exp_id))
