"""
Prediction script — Task 1 deliverable.

Loads the best-performing saved model (Random Forest) and the training-set
StandardScaler, and predicts a student's final grade (G3, 0-20) from
socioeconomic and access-related inputs. This function is imported by the
FastAPI service in Task 2 (summative/API/prediction.py).
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ARTIFACT_DIR = Path(__file__).parent
FEATURES = joblib.load(ARTIFACT_DIR / "features.pkl")
MODEL = joblib.load(ARTIFACT_DIR / "best_model.pkl")
SCALER = joblib.load(ARTIFACT_DIR / "scaler.pkl")


def predict_final_grade(
    G1: float,
    failures: int,
    Medu: int,
    studytime: int,
    absences: int,
    internet: int,
    higher: int,
    age: int,
) -> float:
    """Predict the final grade (0-20).

    Args:
        G1:        first-period grade, 0-20
        failures:  number of past class failures, 0-4
        Medu:      mother's education, 0 (none) - 4 (higher education)
        studytime: weekly study time, 1 (<2h) - 4 (>10h)
        absences:  number of school absences, 0-93
        internet:  internet access at home, 0 = no, 1 = yes
        higher:    wants to pursue higher education, 0 = no, 1 = yes
        age:       student age, 15-22
    """
    row = pd.DataFrame(
        [[G1, failures, Medu, studytime, absences, internet, higher, age]],
        columns=FEATURES,
    )
    prediction = MODEL.predict(SCALER.transform(row))[0]
    return float(np.clip(prediction, 0, 20))


if __name__ == "__main__":
    example = predict_final_grade(
        G1=8, failures=1, Medu=1, studytime=1,
        absences=6, internet=0, higher=1, age=17,
    )
    print(f"Predicted final grade: {example:.2f} / 20")
