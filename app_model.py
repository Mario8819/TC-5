from flask import Flask,jsonify, request
import os
import pandas as pd
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, recall_score, confusion_matrix
from imblearn.over_sampling import SMOTE
# os.chdir(os.path.dirname(__file__))

app = Flask(__name__)


@app.route('/', methods=['GET'])
def hello(): # Ligado al endopoint "/" o sea el home, con el método GET
    return "Bienvenido a la API grupo 4 del modelo credits"

@app.route('/api/v1/predict', methods=['GET'])
def predict():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Obtener argumentos desde la URL
    revolving_util = request.args.get('revolving_util', None)
    age = request.args.get('age', None)
    past_due_30_59 = request.args.get('past_due_30_59', None)
    debt_ratio = request.args.get('debt_ratio', None)
    monthly_income = request.args.get('monthly_income', None)
    open_credit_lines = request.args.get('open_credit_lines', None)
    past_due_90 = request.args.get('past_due_90', None)
    real_estate_loans = request.args.get('real_estate_loans', None)
    past_due_60_89 = request.args.get('past_due_60_89', None)
    dependents = request.args.get('dependents', None)

    # Validación
    if None in [revolving_util, age, past_due_30_59, debt_ratio, monthly_income,
                open_credit_lines, past_due_90, real_estate_loans, past_due_60_89, dependents]:
        return "Missing arguments: please provide all required inputs", 400

    # Convertir a float
    try:
        features = [
            float(revolving_util), float(age), float(past_due_30_59), float(debt_ratio),
            float(monthly_income), float(open_credit_lines), float(past_due_90),
            float(real_estate_loans), float(past_due_60_89), float(dependents)
        ]
    except ValueError:
        return "Invalid input: all values must be numeric", 400

    # Hacer la predicción
    prediction = model.predict([features])

    return jsonify({'prediction': int(prediction[0])})

@app.route('/api/v1/retrain/', methods=['GET'])
def retrain():
    if os.path.exists("data/credit_data_new.csv"):
        # Cargar los datos
        data = pd.read_csv('data/credit_data_new.csv')

        # Separar variables
        X = data.drop(columns=['default_2yrs'])
        y = data['default_2yrs']

        # División train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        # Rebalancear con SMOTE
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        # Escalar
        scaler = StandardScaler()
        X_train_res_scaled = scaler.fit_transform(X_train_res)
        X_test_scaled = scaler.transform(X_test)

        # Entrenar modelo
        model = LogisticRegression(class_weight='balanced', max_iter=3000, random_state=42)
        model.fit(X_train_res_scaled, y_train_res)

        # Evaluación
        y_pred = model.predict(X_test_scaled)
        f1 = f1_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        # Retrain con todo el dataset completo
        X_full_res, y_full_res = smote.fit_resample(X, y)
        X_full_res_scaled = scaler.fit_transform(X_full_res)
        model.fit(X_full_res_scaled, y_full_res)

        # Guardar modelo entrenado (junto con el scaler si lo deseas)
        with open('ad_model.pkl', 'wb') as f:
            pickle.dump(model, f)

        with open('scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)

        return (
            f"<h2>Model retrained successfully.</h2>"
            f"<br>F1-score: {f1:.4f}"
            f"<br>Recall: {recall:.4f}"
            f"<br>Confusion Matrix:<br>{cm}"
        )
    else:
        return "<h2>New data for retrain NOT FOUND. Nothing done!</h2>"

if __name__ == '__main__':
    app.run(debug=True)