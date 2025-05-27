from flask import Flask, request, render_template, jsonify
import numpy as np
import joblib
import os
from tensorflow.keras.models import load_model

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

models = {
    "Logistic Regression": joblib.load(os.path.join(BASE_DIR, 'model/Logistic_Regression_tuned_model.joblib')),
    "Decision Tree": joblib.load(os.path.join(BASE_DIR, 'model/Decision_Tree_tuned_model.joblib')),
    "KNN": joblib.load(os.path.join(BASE_DIR, 'model/KNN_tuned_model.joblib')),
    "SVM": joblib.load(os.path.join(BASE_DIR, 'model/SVM_tuned_model.joblib')),
    "XGBoost": joblib.load(os.path.join(BASE_DIR, 'model/XGBoost_tuned_model.joblib')),
    "Neural Network": load_model(os.path.join(BASE_DIR, 'model/Neural_Network_tuned_model.keras')),
    "Random Forest": joblib.load(os.path.join(BASE_DIR, 'model/Random_Forest_tuned_model.joblib'))
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        scaler = joblib.load(os.path.join(BASE_DIR, 'model/scaler.joblib'))
        input_data = np.array([
            float(request.form["age"]),
            float(request.form["oldpeak"]),
            float(request.form["Max_heart_rate"]),
            float(request.form["vessels_colored_by_flourosopy"]),
            float(request.form["sex"]),
            float(request.form["exercise_induced_angina"]),
            float(request.form["chest_pain_type"]),
            float(request.form["slope"]),
            float(request.form["thalassemia"])
        ]).reshape(1, -1)
        input_data = scaler.transform(input_data)
        model_predictions = {}
        total_prediction_proba = 0
        for model_name, model in models.items():
            if model_name == "Neural Network":
                prediction_proba = model.predict(input_data)[0][0] * 100
            elif model_name == "SVM":
                if hasattr(model, "predict_proba"):
                    prediction_proba = model.predict_proba(input_data)[0][1] * 100
                else:
                    decision = model.decision_function(input_data)[0]
                    prediction_proba = 100 / (1 + np.exp(-decision))
            else:
                prediction_proba = model.predict_proba(input_data)[0][1] * 100

            if prediction_proba >= 50:
                prediction_label = 'High Chance Of Heart Disease'
            else:
                prediction_label = 'Low Chance Of Heart Disease'

            model_predictions[model_name] = {
                'prediction_proba': prediction_proba,
                'prediction_label': prediction_label
            }

            total_prediction_proba += prediction_proba

        overall_prediction_proba = total_prediction_proba / len(models)

        return render_template(
            "result.html",
            overall_prediction_proba=round(overall_prediction_proba, 2),
            model_predictions=model_predictions 
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
