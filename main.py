"""

Trains a Random Forest on housing.csv when the server starts,
then serves predictions over HTTP.

Run:   uvicorn main:app --reload
Docs:  http://127.0.0.1:8000/docs 
Needs: pip install fastapi uvicorn pandas scikit-learn
"""

from contextlib import asynccontextmanager
from enum import Enum
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

CSV_PATH = Path(__file__).parent / "housing.csv"


state = {}



def train_model():
    data = pd.read_csv(CSV_PATH)
    data["total_bedrooms"] = data["total_bedrooms"].fillna(data["total_bedrooms"].median())
    data = pd.get_dummies(data, columns=["ocean_proximity"], dtype=int)

    X = data.drop(columns="median_house_value")
    y = data["median_house_value"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    state["model"] = model
    state["columns"] = list(X.columns)  # keeps input column order identical
    state["mae"] = mean_absolute_error(y_test, preds)
    state["r2"] = r2_score(y_test, preds)


@asynccontextmanager
async def lifespan(app: FastAPI):
    train_model()
    yield


app = FastAPI(title="House Price Predictor", lifespan=lifespan)


class OceanProximity(str, Enum):
    less_than_1h = "<1H OCEAN"
    inland = "INLAND"
    near_ocean = "NEAR OCEAN"
    near_bay = "NEAR BAY"
    island = "ISLAND"


class HouseInput(BaseModel):
    longitude: float = Field(..., examples=[-118.25])
    latitude: float = Field(..., examples=[34.05])
    housing_median_age: float = Field(..., ge=0, examples=[20])
    total_rooms: float = Field(..., gt=0, examples=[2500])
    total_bedrooms: float = Field(..., gt=0, examples=[500])
    population: float = Field(..., gt=0, examples=[1200])
    households: float = Field(..., gt=0, examples=[450])
    median_income: float = Field(
        ..., gt=0, description="In tens of thousands of dollars", examples=[5.0]
    )
    ocean_proximity: OceanProximity = Field(..., examples=["<1H OCEAN"])


class PredictionOutput(BaseModel):
    predicted_price: float



# Endpoints
@app.get("/")
def home():
    return {"message": "House Price Predictor API. Try /docs"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_avg_error_usd": round(state["mae"]),
        "model_r2": round(state["r2"], 3),
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(house: HouseInput):
    row = house.model_dump()
    category = row.pop("ocean_proximity").value

    
    df = pd.DataFrame([row])
    for col in state["columns"]:
        if col.startswith("ocean_proximity_"):
            df[col] = 1 if col == f"ocean_proximity_{category}" else 0
    df = df[state["columns"]]

    price = state["model"].predict(df)[0]
    return PredictionOutput(predicted_price=round(float(price), 2))
