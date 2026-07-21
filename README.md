# Student Performance Prediction — Linear Regression Model

## Mission and problem

Many young people in Rwanda and across Africa miss education and employment opportunities because of poor access to information, and students from low-income backgrounds are affected most.
When they miss scholarships, training or jobs, they stay unemployed or under-skilled, feeding long-term poverty.
This project predicts a student's final grade (0–20) from socioeconomic and information-access factors such as parental education, internet access at home and study time.
Schools and NGOs can use it to spot at-risk students early and route scholarship, tutoring and training information to those who need it most.

## Live API

| Resource | URL |
| --- | --- |
| Swagger UI | `https://student-performance-api-3m5d.onrender.com/docs` |
| Prediction endpoint | `POST https://student-performance-api-3m5d.onrender.com/predict` |

> Replace `<your-service>` with your Render service name after deploying.

### Example request

```json
{
  "G1": 12, "failures": 0, "Medu": 3, "studytime": 2,
  "absences": 4, "internet": 1, "higher": 1, "age": 17
}
```

### Example response

```json
{
  "predicted_grade": 12.87,
  "grade_percentage": 64.4,
  "risk_level": "Moderate risk",
  "interpretation": "Passing but not competitive for selective scholarships. ..."
}
```

## Video demo

`https://youtu.be/<your-video-id>`

## Model

Dataset: [UCI Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance) (Cortez & Silva, 2008) — 395 students, 33 attributes.

| Model | Test MSE | Test RMSE | Test R² |
| --- | --- | --- | --- |
| **Random Forest — ensemble (saved as best)** | **4.19** | **2.05** | **0.796** |
| SGD Ridge Linear Regression — gradient descent, L2 | 6.23 | 2.50 | 0.696 |
| SGD Linear Regression — gradient descent | 6.25 | 2.50 | 0.695 |
| Decision Tree — tree | 6.66 | 2.58 | 0.675 |

The four algorithms cover the required categories: two linear regression models trained by gradient descent (plain and L2-regularized `SGDRegressor`), one tree, and one ensemble.

Features used (8): `G1`, `failures`, `Medu`, `studytime`, `absences`, `internet`, `higher`, `age`.
`G2` was deliberately dropped despite its 0.90 correlation with the target, because the model must work *early* in the school year, when only the first-period grade exists.

## API endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/` | Service metadata |
| GET | `/health` | Liveness check |
| GET | `/model-info` | Current model, features and valid input ranges |
| POST | `/predict` | Single prediction (used by the Flutter app) |
| POST | `/stream-data` | Append streamed labelled records |
| POST | `/upload-data` | Append a labelled CSV |
| POST | `/retrain` | Retrain from the existing model on base + new data |

### Input validation

Pydantic enforces a data type and a realistic range on every field. Anything outside the bounds is rejected with HTTP 422 before the model runs.

| Field | Type | Range |
| --- | --- | --- |
| `G1` | float | 0 – 20 |
| `failures` | int | 0 – 4 |
| `Medu` | int | 0 – 4 |
| `studytime` | int | 1 – 4 |
| `absences` | int | 0 – 93 |
| `internet` | int | 0 – 1 |
| `higher` | int | 0 – 1 |
| `age` | int | 15 – 22 |

### CORS configuration and reasoning

| Setting | Value | Reasoning |
| --- | --- | --- |
| `allow_origins` | `*` (override with `ALLOWED_ORIGINS`) | The consumer is a native Flutter mobile app, which sends no `Origin` header and is unaffected by CORS. A wildcard lets Swagger UI, a Flutter Web build and graders' browsers reach the API from any host. Acceptable because the service is public, holds no accounts and returns no personal data. |
| `allow_methods` | `GET, POST, OPTIONS` | Every endpoint only reads or submits; `OPTIONS` is needed for browser preflight. `PUT`, `PATCH` and `DELETE` are refused because no update or delete semantics exist, so allowing them would only widen the attack surface. |
| `allow_headers` | `Content-Type, Accept` | Enough for JSON and multipart uploads. Arbitrary custom headers are refused. |
| `allow_credentials` | `False` | No cookies or auth headers are ever sent, and the CORS spec forbids combining credentials with a wildcard origin. |

To restrict origins in production, set the environment variable without touching the code:

```
ALLOWED_ORIGINS=https://school-dashboard.rw,https://ngo-portal.org
```

### Retraining

1. Send new labelled records to `POST /stream-data` (JSON) or `POST /upload-data` (CSV). Both apply the same type and range validation as `/predict`, and duplicate rows are dropped.
2. Call `POST /retrain`. A new model inherits the current model's hyperparameters, is fitted on the original data plus everything added, and both models are scored on the same held-out test split.
3. The new model is promoted **only if its test MSE improves**. Use `POST /retrain?force=true` to promote regardless.

A tree ensemble must be refitted on the full history. A linear model trained with SGD could instead be updated incrementally with `partial_fit` on just the new batch.

> On Render's free tier the filesystem is ephemeral: uploaded data and retrained models are lost on restart or redeploy. Attach a persistent disk or use object storage to keep them.

## Repository structure

```
linear_regression_model/
├── README.md
├── render.yaml
├── summative/
│   ├── linear_regression/
│   │   ├── multivariate.ipynb      # EDA, feature engineering, 4-model comparison
│   │   ├── predict.py              # prediction function using the saved model
│   │   ├── student-mat.csv         # dataset
│   │   ├── best_model.pkl
│   │   ├── scaler.pkl
│   │   └── features.pkl
│   ├── API/
│   │   ├── prediction.py           # FastAPI service
│   │   ├── requirements.txt
│   │   ├── model/                  # artifacts served by the API
│   │   └── data/                   # base dataset + records added at runtime
│   ├── FlutterApp/
│   └── pyproject.toml
```

## Running locally

```bash
cd summative
uv sync                      # creates .venv and installs dependencies
uv run jupyter lab           # to open multivariate.ipynb

cd API
uv run uvicorn prediction:app --reload
# Swagger UI: http://127.0.0.1:8000/docs
```

## Deploying to Render

1. Push this repository to GitHub.
2. On [render.com](https://render.com): **New → Web Service**, connect the repo.
3. Settings:
   - **Root Directory:** `summative/API`
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn prediction:app --host 0.0.0.0 --port $PORT`
4. Environment variables: `PYTHON_VERSION` = `3.12.3`, `ALLOWED_ORIGINS` = `*`
5. Deploy, then open `https://<your-service>.onrender.com/docs`.

> The free tier sleeps after inactivity; the first request can take ~50 seconds. Wake the service before recording your demo.

## Running the mobile app

The app is a single page with eight text fields (one per model variable), a **Predict** button, and a display area that shows either the predicted grade or an error message.

### 1. Generate the platform scaffolding

`lib/main.dart` and `pubspec.yaml` are committed; the generated Android/iOS folders are not. From `summative/FlutterApp`:

```bash
flutter create .
flutter pub get
```

### 2. Point the app at your API

Open `lib/main.dart` and set your deployed URL near the top (no trailing slash):

```dart
const String kApiBaseUrl = 'https://your-service.onrender.com';
```

### 3. Allow internet access on Android

Add this line inside `android/app/src/main/AndroidManifest.xml`, above the `<application>` tag:

```xml
<uses-permission android:name="android.permission.INTERNET"/>
```

Flutter adds this automatically for debug builds only, so a release build fails without it.

### 4. Run it

```bash
flutter devices     # confirm a phone or emulator is connected
flutter run
```

### Input ranges

The app validates every field before sending a request, using the same bounds the API enforces: `G1` 0–20, `failures` 0–4, `Medu` 0–4, `studytime` 1–4, `absences` 0–93, `internet` 0/1, `higher` 0/1, `age` 15–22. Missing or out-of-range values produce an error message in the display area instead of a request.

> The first prediction after the API has been idle can take up to a minute while the free Render instance wakes up. Open `/docs` in a browser once before demoing.
