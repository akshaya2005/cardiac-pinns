from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from inference import run_prediction
from schemas import PredictRequest, PredictResponse

app = FastAPI(title="PINNs for Cardiac Modeling API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    return run_prediction(request)
