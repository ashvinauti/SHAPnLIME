"""
FastAPI REST API for XAI-IDS Pro.
Endpoints:
  POST /api/v1/detect         — Batch or single-sample detection
  POST /api/v1/explain/shap   — SHAP explanation for a sample
  POST /api/v1/explain/lime   — LIME explanation for a sample
  GET  /api/v1/model/info     — Model metadata
  GET  /api/v1/health         — Health check
  POST /api/v1/auth/token     — JWT login

Usage:
  uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
"""
import os
import time
import json
import io
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np

try:
    from fastapi import FastAPI, HTTPException, Depends, status, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
    from fastapi.responses import JSONResponse, HTMLResponse
    from pydantic import BaseModel as PydanticBase
    import jose.jwt as jwt
    from passlib.context import CryptContext
    FASTAPI_OK = True
except ImportError:
    FASTAPI_OK = False
    print("⚠️  FastAPI dependencies missing. Run: pip install fastapi uvicorn python-jose passlib[bcrypt]")

from .config import Config
from .logger import get_logger

logger = get_logger("xai_ids.api")

# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------
if FASTAPI_OK:
    class Sample(PydanticBase):
        features: List[float]
        feature_names: Optional[List[str]] = None

    class BatchRequest(PydanticBase):
        samples: List[List[float]]
        feature_names: Optional[List[str]] = None

    class DetectionResult(PydanticBase):
        sample_index: int
        prediction: int          # 0=normal, 1=anomaly
        anomaly_score: float
        label: str               # "NORMAL" | "ANOMALY"
        confidence: float

    class BatchResponse(PydanticBase):
        results: List[DetectionResult]
        total: int
        anomalies_detected: int
        processing_time_ms: float
        model_version: str

    class ExplanationResponse(PydanticBase):
        sample_index: int
        anomaly_score: float
        method: str
        feature_contributions: List[Dict[str, Any]]

    class HealthResponse(PydanticBase):
        status: str
        version: str
        model_loaded: bool
        uptime_seconds: float
        timestamp: str


# ---------------------------------------------------------------------------
# Factory — creates the FastAPI app with a loaded model
# ---------------------------------------------------------------------------
def create_app(config: Config, detector=None, preprocessor=None) -> "FastAPI":
    if not FASTAPI_OK:
        raise ImportError("Install fastapi, uvicorn, python-jose, passlib")

    app = FastAPI(
        title="XAI-IDS Pro API",
        description="Explainable AI Intrusion Detection System",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- State ---
    app.state.config = config
    app.state.detector = detector
    app.state.preprocessor = preprocessor
    app.state.start_time = time.time()
    app.state.model_version = "1.0.0"

    # --- Auth helpers ---
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token")

    # Demo user (replace with DB in production)
    DEMO_USERS = {
        "admin": {
            "username": "admin",
            "hashed_password": pwd_context.hash("admin123"),
            "role": "admin",
        },
        "analyst": {
            "username": "analyst",
            "hashed_password": pwd_context.hash("analyst123"),
            "role": "analyst",
        }
    }

    def _create_token(data: dict) -> str:
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=config.api.access_token_expire_minutes)
        to_encode["exp"] = expire
        return jwt.encode(to_encode, config.api.secret_key, algorithm=config.api.jwt_algorithm)

    async def _get_current_user(token: str = Depends(oauth2_scheme)):
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
        try:
            payload = jwt.decode(token, config.api.secret_key,
                                  algorithms=[config.api.jwt_algorithm])
            username = payload.get("sub")
            if username is None:
                raise credentials_exception
        except Exception:
            raise credentials_exception
        user = DEMO_USERS.get(username)
        if user is None:
            raise credentials_exception
        return user

    def _require_model():
        if app.state.detector is None:
            raise HTTPException(status_code=503, detail="Model not loaded. Train or load a model first.")
        return app.state.detector

    # -----------------------------------------------------------------------
    # Routes
    # -----------------------------------------------------------------------
    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def root():
        return """<html><body style="font-family:monospace;background:#0d1117;color:#e6edf3;padding:40px">
        <h1>🛡 XAI-IDS Pro API</h1>
        <p>Visit <a href="/docs" style="color:#58a6ff">/docs</a> for interactive API documentation.</p>
        </body></html>"""

    @app.get("/api/v1/health", response_model=HealthResponse)
    async def health():
        return HealthResponse(
            status="ok",
            version="1.0.0",
            model_loaded=app.state.detector is not None,
            uptime_seconds=round(time.time() - app.state.start_time, 1),
            timestamp=datetime.utcnow().isoformat(),
        )

    @app.post("/api/v1/auth/token")
    async def login(form_data: OAuth2PasswordRequestForm = Depends()):
        user = DEMO_USERS.get(form_data.username)
        if not user or not pwd_context.verify(form_data.password, user["hashed_password"]):
            raise HTTPException(status_code=400, detail="Incorrect username or password")
        token = _create_token({"sub": user["username"], "role": user["role"]})
        return {"access_token": token, "token_type": "bearer"}

    @app.post("/api/v1/detect", response_model=BatchResponse)
    async def detect(
        req: BatchRequest,
        current_user: dict = Depends(_get_current_user),
        detector=Depends(_require_model),
    ):
        t0 = time.time()
        try:
            X = np.array(req.samples, dtype=np.float32)
            predictions, scores, details = detector.predict(X)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Prediction error: {e}")

        results = [
            DetectionResult(
                sample_index=i,
                prediction=int(predictions[i]),
                anomaly_score=float(scores[i]),
                label="ANOMALY" if predictions[i] == 1 else "NORMAL",
                confidence=float(scores[i]) if predictions[i] == 1 else float(1 - scores[i]),
            )
            for i in range(len(X))
        ]

        elapsed_ms = (time.time() - t0) * 1000
        logger.info(f"🔍 Detected {int(predictions.sum())} anomalies in {len(X)} samples | {elapsed_ms:.1f}ms")
        return BatchResponse(
            results=results,
            total=len(X),
            anomalies_detected=int(predictions.sum()),
            processing_time_ms=round(elapsed_ms, 2),
            model_version=app.state.model_version,
        )

    @app.post("/api/v1/explain/lime")
    async def explain_lime(
        req: Sample,
        current_user: dict = Depends(_get_current_user),
        detector=Depends(_require_model),
    ):
        """LIME local explanation for one sample."""
        if not hasattr(app.state, "xai_explainer") or app.state.xai_explainer is None:
            raise HTTPException(status_code=503, detail="XAI explainer not initialized.")
        x = np.array(req.features, dtype=np.float32)
        result = app.state.xai_explainer.explain_lime(x, sample_label="api_sample")
        return ExplanationResponse(
            sample_index=0,
            anomaly_score=float(detector.anomaly_score_fn(x.reshape(1, -1))[0]),
            method="LIME",
            feature_contributions=[
                {"feature": f, "weight": w}
                for f, w in zip(result["features"], result["weights"])
            ],
        )

    @app.get("/api/v1/model/info")
    async def model_info(current_user: dict = Depends(_get_current_user)):
        det = app.state.detector
        if det is None:
            return {"loaded": False}
        return {
            "loaded": True,
            "threshold": det.threshold,
            "input_dim": det.input_dim,
            "ensemble": det.isolation_forest is not None,
            "model_version": app.state.model_version,
        }

    return app
