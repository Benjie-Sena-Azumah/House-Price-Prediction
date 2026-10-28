# House-Price-Prediction
A FastAPI service that trains a Random Forest on housing data at startup and serves price predictions over HTTP.

House Price Predictor API
A FastAPI service that trains a Random Forest on California housing data at startup and serves price predictions over HTTP.
Overview
This project exposes a machine learning model as a REST API. When the server starts, it reads housing.csv, trains a Random Forest regressor, evaluates it on a held-out test set, and keeps the model in memory. Clients can then send details about a property to /predict and receive an estimated median house value in US dollars.
Interactive documentation is generated automatically by FastAPI, so you can try every endpoint from your browser.
Features
•	Model trains automatically on server start, with no separate training step
•	Input validation with Pydantic (types, ranges and allowed categories)
•	Dropdown-restricted ocean_proximity values using an enum
•	Health endpoint that reports the model's error and R² score
•	Auto-generated Swagger UI at /docs
Tech Stack
Component	Purpose
FastAPI	Web framework and API docs
Uvicorn	ASGI server
Pandas	Data loading and preprocessing
scikit-learn	Random Forest model and evaluation metrics
Pydantic	Request and response validation
Project Structure
.
├── main.py        # API and model training code
├── housing.csv    # Training data (must sit next to main.py)
└── README.md
Getting Started
1. Install dependencies
pip install fastapi uvicorn pandas scikit-learn
2. Add the dataset
Place housing.csv in the same folder as main.py. It must contain the California housing columns, including the target median_house_value.
3. Run the server
uvicorn main:app --reload
The server trains the model on startup, so allow a few seconds before it is ready.
4. Open the docs
Visit http://127.0.0.1:8000/docs to explore and test the endpoints.
How the Model Works
Step	Detail
Missing values	Missing total_bedrooms values are filled with the column median
Encoding	ocean_proximity is one-hot encoded into separate 0/1 columns
Target	median_house_value
Split	80% training, 20% testing (random_state=42)
Algorithm	RandomForestRegressor with 100 trees
Metrics	Mean Absolute Error (MAE) and R² on the test set
The column order from training is stored and reused at prediction time, so incoming requests always match what the model expects.
API Endpoints
Method	Path	Description
GET	/	Welcome message
GET	/health	Service status plus model MAE (USD) and R²
POST	/predict	Returns a predicted house price
GET /health
{
  "status": "ok",
  "model_avg_error_usd": 31500,
  "model_r2": 0.82
}
The values above are illustrative. Actual numbers depend on your dataset.
POST /predict
Request body
Field	Type	Rules	Example
longitude	float	none	-118.25
latitude	float	none	34.05
housing_median_age	float	0 or more	20
total_rooms	float	greater than 0	2500
total_bedrooms	float	greater than 0	500
population	float	greater than 0	1200
households	float	greater than 0	450
median_income	float	greater than 0, in tens of thousands of dollars	5.0
ocean_proximity	enum	<1H OCEAN, INLAND, NEAR OCEAN, NEAR BAY, ISLAND	<1H OCEAN
Example request
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "longitude": -118.25,
    "latitude": 34.05,
    "housing_median_age": 20,
    "total_rooms": 2500,
    "total_bedrooms": 500,
    "population": 1200,
    "households": 450,
    "median_income": 5.0,
    "ocean_proximity": "<1H OCEAN"
  }'
Example response
{
  "predicted_price": 285400.0
}
Invalid input, such as a negative room count or an unknown ocean_proximity value, returns a 422 error with details about the problem.
Notes and Limitations
•	The model is retrained every time the server starts, which adds startup time and means results can change if the data changes.
•	Predictions are held in memory only. The trained model is not saved to disk.
•	The dataset describes California census block groups, so predictions outside that region are not reliable.
•	Treat results as rough estimates, not appraisals.
Possible Improvements
•	Save the trained model with joblib and load it on startup
•	Add Docker support for easier deployment
•	Compare other models, such as Gradient Boosting
•	Add automated tests for the endpoints
•	Report feature importances through a new endpoint

