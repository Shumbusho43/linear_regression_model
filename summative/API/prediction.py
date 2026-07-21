"""
Student Performance Prediction API
==================================

Mission context
---------------
Many young people in Rwanda and across Africa miss out on education and
employment opportunities because of poor access to information, and students
from low-income backgrounds are hit hardest. This API exposes the regression
model built in Task 1, which predicts a student's final grade (G3, 0-20) from
socioeconomic and access-related factors. Schools and NGOs can use it to flag
at-risk students early and route scholarship, tutoring and training
information to the students who need it most.

Endpoints
---------
GET  /              service metadata
GET  /health        liveness probe
GET  /model-info    current model, features and input ranges
POST /predict       single prediction (the endpoint used by the Flutter app)
POST /stream-data   append streamed labelled records to the training store
POST /upload-data   append an uploaded labelled CSV to the training store
POST /retrain       retrain from the existing model on base + newly added data

Run locally:  uvicorn prediction:app --reload
Swagger UI:   http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import io
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------- #
# Paths and constants
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
DATA_DIR = BASE_DIR / "data"
MODEL_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "best_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
FEATURES_PATH = MODEL_DIR / "features.pkl"
BASE_DATA_PATH = DATA_DIR / "student-mat.csv"      # original UCI dataset
NEW_DATA_PATH = DATA_DIR / "new_data.csv"          # records added at runtime

FEATURES = ["G1", "failures", "Medu", "studytime",
            "absences", "internet", "higher", "age"]
TARGET = "G3"
RANDOM_STATE = 42

# Range constraints, kept in one place so /model-info, Pydantic and the
# retraining validator can never drift apart.
FEATURE_RANGES = {
    "G1":        (0, 20,  "First-period grade (0-20)"),
    "failures":  (0, 4,   "Number of past class failures (0-4)"),
    "Medu":      (0, 4,   "Mother's education: 0 none - 4 higher education"),
    "studytime": (1, 4,   "Weekly study time: 1 (<2h) - 4 (>10h)"),
    "absences":  (0, 93,  "Number of school absences (0-93)"),
    "internet":  (0, 1,   "Internet access at home: 0 no, 1 yes"),
    "higher":    (0, 1,   "Wants higher education: 0 no, 1 yes"),
    "age":       (15, 22, "Student age in years (15-22)"),
}

# --------------------------------------------------------------------------- #
# FastAPI application
# --------------------------------------------------------------------------- #
app = FastAPI(
    title="Student Performance Prediction API",
    description=(
        "Predicts a secondary-school student's final grade (0-20) from "
        "socioeconomic and information-access factors, so that at-risk "
        "students can be identified early and connected to scholarship, "
        "tutoring and training opportunities."
    ),
    version="1.0.0",
)

# --------------------------------------------------------------------------- #
# CORS middleware
# --------------------------------------------------------------------------- #
# WHAT IS ALLOWED, AND WHY
#
# allow_origins:
#     Defaults to "*" (any origin). The consumer of this API is a Flutter
#     *mobile* app. Native mobile clients are not browsers, so they send no
#     Origin header and are unaffected by CORS. A wildcard is used so that the
#     Swagger UI, a Flutter Web build and the graders' browsers can all call the
#     API from any host without a redeploy. This is acceptable here because the
#     API is public, read-only in effect, holds no user accounts and returns no
#     personal data. The value is read from the ALLOWED_ORIGINS environment
#     variable, so in production it can be narrowed to a known domain list
#     (e.g. ALLOWED_ORIGINS="https://school-dashboard.rw") with no code change.
#
# allow_methods:
#     Restricted to GET, POST and OPTIONS. Every endpoint here either reads
#     (GET) or submits data (POST); OPTIONS is required for the browser
#     preflight request. PUT, PATCH and DELETE are refused because the API
#     exposes no update or delete semantics, so permitting them would only
#     widen the attack surface.
#
# allow_headers:
#     Restricted to Content-Type (needed for application/json and multipart
#     uploads) and Accept. Arbitrary custom headers are refused.
#
# allow_credentials:
#     False. Cookies and Authorization headers are never sent to this service.
#     The CORS specification also forbids combining credentials with a wildcard
#     origin, so leaving this off is both safer and standards-compliant.
# --------------------------------------------------------------------------- #
ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Accept"],
)


# --------------------------------------------------------------------------- #
# Model loading (with self-healing fallback)
# --------------------------------------------------------------------------- #
def _load_base_dataframe() -> pd.DataFrame:
    """Load and clean the original UCI dataset."""
    df = pd.read_csv(BASE_DATA_PATH, sep=";", quotechar='"')
    df.columns = [c.strip() for c in df.columns]
    for grade in ["G1", "G2", "G3"]:
        if grade in df.columns:
            df[grade] = pd.to_numeric(df[grade])
    for col in ["internet", "higher"]:
        # pandas 2 types these as `object`, pandas 3 as `str`; checking for
        # "not numeric" keeps the encoding working on both.
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = (df[col].astype(str).str.strip().str.lower() == "yes").astype(int)
    return df[FEATURES + [TARGET]]


def _train_from_scratch() -> tuple[RandomForestRegressor, StandardScaler, dict]:
    """Train the Random Forest chosen in Task 1 and persist the artifacts."""
    df = _load_base_dataframe()
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df[TARGET], test_size=0.2, random_state=RANDOM_STATE
    )
    scaler = StandardScaler().fit(X_train)
    model = RandomForestRegressor(
        n_estimators=200, max_depth=8, random_state=RANDOM_STATE
    ).fit(scaler.transform(X_train), y_train)
    preds = model.predict(scaler.transform(X_test))
    metrics = {
        "mse": float(mean_squared_error(y_test, preds)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "mae": float(mean_absolute_error(y_test, preds)),
        "r2": float(r2_score(y_test, preds)),
    }
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(FEATURES, FEATURES_PATH)
    return model, scaler, metrics


def load_artifacts() -> tuple[RandomForestRegressor, StandardScaler]:
    """Load the saved model and scaler, retraining if they are missing.

    The fallback matters on a fresh host: if the pickled artifacts were written
    by a different scikit-learn build they may fail to load, and the service
    should recover on its own rather than return 500s for every request.
    """
    try:
        return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)
    except Exception:  # missing file or version mismatch
        model, scaler, _ = _train_from_scratch()
        return model, scaler


MODEL, SCALER = load_artifacts()
MODEL_METADATA: dict = {
    "algorithm": type(MODEL).__name__,
    "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "training_rows": int(len(_load_base_dataframe())),
}


# --------------------------------------------------------------------------- #
# Pydantic schemas: enforced data types and range constraints
# --------------------------------------------------------------------------- #
class StudentFeatures(BaseModel):
    """Input payload for a single prediction.

    Every field enforces both a data type and a realistic range. Anything
    outside these bounds is rejected by FastAPI with HTTP 422 before the model
    is ever called, so the model never extrapolates on impossible values such
    as a negative grade or a 40-year-old secondary-school pupil.
    """

    G1: float = Field(..., ge=0, le=20, description=FEATURE_RANGES["G1"][2], examples=[12])
    failures: int = Field(..., ge=0, le=4, description=FEATURE_RANGES["failures"][2], examples=[0])
    Medu: int = Field(..., ge=0, le=4, description=FEATURE_RANGES["Medu"][2], examples=[3])
    studytime: int = Field(..., ge=1, le=4, description=FEATURE_RANGES["studytime"][2], examples=[2])
    absences: int = Field(..., ge=0, le=93, description=FEATURE_RANGES["absences"][2], examples=[4])
    internet: int = Field(..., ge=0, le=1, description=FEATURE_RANGES["internet"][2], examples=[1])
    higher: int = Field(..., ge=0, le=1, description=FEATURE_RANGES["higher"][2], examples=[1])
    age: int = Field(..., ge=15, le=22, description=FEATURE_RANGES["age"][2], examples=[17])

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "G1": 12, "failures": 0, "Medu": 3, "studytime": 2,
                "absences": 4, "internet": 1, "higher": 1, "age": 17,
            }]
        }
    }


class LabelledStudent(StudentFeatures):
    """A training record: the eight features plus the observed final grade."""

    G3: float = Field(..., ge=0, le=20, description="Observed final grade (0-20)")


class PredictionResponse(BaseModel):
    predicted_grade: float = Field(..., description="Predicted final grade, 0-20")
    grade_percentage: float = Field(..., description="Predicted grade as a percentage")
    risk_level: Literal["High risk", "Moderate risk", "On track"]
    interpretation: str


class RetrainResponse(BaseModel):
    status: str
    promoted: bool
    previous_metrics: dict
    new_metrics: dict
    training_rows: int
    message: str


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _risk_band(grade: float) -> tuple[str, str]:
    """Translate a predicted grade into an actionable band for the mission."""
    if grade < 10:
        return ("High risk",
                "Below the pass mark. This student should be prioritised for "
                "tutoring and for direct outreach about scholarships and "
                "training opportunities.")
    if grade < 14:
        return ("Moderate risk",
                "Passing but not competitive for selective scholarships. "
                "Targeted study support and opportunity information would help.")
    return ("On track",
            "Performing well. Share competitive scholarship and internship "
            "openings with this student.")


def _combined_training_data() -> pd.DataFrame:
    """Original dataset plus every record added through the data endpoints."""
    df = _load_base_dataframe()
    if NEW_DATA_PATH.exists():
        extra = pd.read_csv(NEW_DATA_PATH)
        df = pd.concat([df, extra[FEATURES + [TARGET]]], ignore_index=True)
        # Uploading the same file twice is an easy mistake to make. Duplicated
        # rows would land in both the train and test split and make the
        # evaluation look better than it is, so they are dropped here.
        df = df.drop_duplicates(ignore_index=True)
    return df


def _append_records(records: pd.DataFrame) -> int:
    """Append validated records to the runtime training store."""
    records = records[FEATURES + [TARGET]]
    header = not NEW_DATA_PATH.exists()
    records.to_csv(NEW_DATA_PATH, mode="a", header=header, index=False)
    return len(records)


def _evaluate(model, scaler, X, y) -> dict:
    preds = model.predict(scaler.transform(X))
    return {
        "mse": round(float(mean_squared_error(y, preds)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y, preds))), 4),
        "mae": round(float(mean_absolute_error(y, preds)), 4),
        "r2": round(float(r2_score(y, preds)), 4),
    }


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/", tags=["Meta"])
def root() -> dict:
    return {
        "service": "Student Performance Prediction API",
        "mission": ("Identify students whose academic outcomes are held back by "
                    "poor access to information and support, early enough to act."),
        "docs": "/docs",
        "predict_endpoint": "POST /predict",
    }


@app.get("/health", tags=["Meta"])
def health() -> dict:
    return {"status": "healthy", "model_loaded": MODEL is not None}


@app.get("/model-info", tags=["Meta"])
def model_info() -> dict:
    return {
        **MODEL_METADATA,
        "target": "G3 (final grade, 0-20)",
        "features": [
            {"name": name, "min": lo, "max": hi, "description": desc}
            for name, (lo, hi, desc) in FEATURE_RANGES.items()
        ],
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(payload: StudentFeatures) -> PredictionResponse:
    """Predict a student's final grade from eight validated inputs.

    This is the endpoint consumed by the Flutter application.
    """
    try:
        row = pd.DataFrame([[getattr(payload, f) for f in FEATURES]], columns=FEATURES)
        raw = float(MODEL.predict(SCALER.transform(row))[0])
        grade = float(np.clip(raw, 0, 20))
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    level, message = _risk_band(grade)
    return PredictionResponse(
        predicted_grade=round(grade, 2),
        grade_percentage=round(grade / 20 * 100, 1),
        risk_level=level,
        interpretation=message,
    )


@app.post("/stream-data", tags=["Retraining"])
def stream_data(records: List[LabelledStudent]) -> dict:
    """Append streamed labelled records to the training store.

    Used when new observations arrive continuously (for example a school
    submitting end-of-term results). Records are validated against the same
    type and range rules as /predict before being stored.
    """
    if not records:
        raise HTTPException(status_code=400, detail="No records supplied.")
    frame = pd.DataFrame([r.model_dump() for r in records])
    added = _append_records(frame)
    return {
        "status": "accepted",
        "records_added": added,
        "total_training_rows": int(len(_combined_training_data())),
        "next_step": "Call POST /retrain to fold these records into the model.",
    }


@app.post("/upload-data", tags=["Retraining"])
async def upload_data(file: UploadFile = File(...)) -> dict:
    """Append an uploaded labelled CSV to the training store.

    The CSV must contain the eight feature columns plus G3. Comma-separated
    files and the semicolon-separated UCI format are both accepted.
    """
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")

    raw = await file.read()
    try:
        frame = pd.read_csv(io.BytesIO(raw))
        if len(frame.columns) == 1:  # semicolon-separated UCI export
            frame = pd.read_csv(io.BytesIO(raw), sep=";", quotechar='"')
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Unreadable CSV: {exc}") from exc

    frame.columns = [c.strip() for c in frame.columns]
    for col in ["internet", "higher"]:
        if col in frame.columns and not pd.api.types.is_numeric_dtype(frame[col]):
            frame[col] = (frame[col].astype(str).str.strip().str.lower() == "yes").astype(int)

    missing = [c for c in FEATURES + [TARGET] if c not in frame.columns]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing columns: {missing}")

    frame = frame[FEATURES + [TARGET]].apply(pd.to_numeric, errors="coerce").dropna()
    if frame.empty:
        raise HTTPException(status_code=422, detail="No valid numeric rows found.")

    # Enforce the same ranges as the prediction schema.
    for col, (lo, hi, _) in FEATURE_RANGES.items():
        frame = frame[(frame[col] >= lo) & (frame[col] <= hi)]
    frame = frame[(frame[TARGET] >= 0) & (frame[TARGET] <= 20)]
    if frame.empty:
        raise HTTPException(status_code=422, detail="All rows fell outside valid ranges.")

    added = _append_records(frame)
    return {
        "status": "accepted",
        "filename": file.filename,
        "records_added": added,
        "total_training_rows": int(len(_combined_training_data())),
        "next_step": "Call POST /retrain to fold these records into the model.",
    }


@app.post("/retrain", response_model=RetrainResponse, tags=["Retraining"])
def retrain(force: bool = False) -> RetrainResponse:
    """Retrain the model from the existing model on base + newly added data.

    The new model inherits the hyperparameters of the model currently in
    service, and is fitted on the original dataset combined with every record
    added through /stream-data and /upload-data. Old and new models are scored
    on the same held-out test split, and the new one is promoted only if its
    test MSE improves. Pass force=true to promote regardless.

    Note: a tree ensemble has to be refitted on the full history. A linear
    model trained with SGD could instead be updated incrementally with
    partial_fit on just the new batch.
    """
    global MODEL, SCALER, MODEL_METADATA

    df = _combined_training_data()
    if len(df) < 50:
        raise HTTPException(status_code=400, detail="Not enough data to retrain.")

    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df[TARGET], test_size=0.2, random_state=RANDOM_STATE
    )
    scaler = StandardScaler().fit(X_train)

    previous_metrics = _evaluate(MODEL, SCALER, X_test, y_test)

    params = MODEL.get_params()
    candidate = RandomForestRegressor(**params)
    candidate.fit(scaler.transform(X_train), y_train)
    new_metrics = _evaluate(candidate, scaler, X_test, y_test)

    promoted = force or new_metrics["mse"] <= previous_metrics["mse"]
    if promoted:
        joblib.dump(candidate, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        joblib.dump(FEATURES, FEATURES_PATH)
        MODEL, SCALER = candidate, scaler
        MODEL_METADATA = {
            "algorithm": type(candidate).__name__,
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "training_rows": int(len(df)),
        }
        message = "Retrained model improved on the test split and is now serving predictions."
    else:
        message = ("Retrained model did not improve on the test split, so the "
                   "previous model is still serving predictions. "
                   "Call /retrain?force=true to override.")

    return RetrainResponse(
        status="completed",
        promoted=promoted,
        previous_metrics=previous_metrics,
        new_metrics=new_metrics,
        training_rows=int(len(df)),
        message=message,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
