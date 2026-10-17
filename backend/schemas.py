from typing import Optional

from pydantic import BaseModel
import torch


class PredictRequest(BaseModel):
    model: str
    color_scheme: str
    initial_condition: str


class PredictResponse(BaseModel):
    network_image: str  # data URI, e.g. "data:image/png;base64,..."
    ground_truth_image: str  # data URI, e.g. "data:image/png;base64,..."
    training_points: Optional[int] = None
    accuracy: Optional[float] = None


