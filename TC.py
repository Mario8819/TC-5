import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


# Cambiar al directorio del script
os.chdir(os.path.dirname(__file__))

# Cargar y limpiar los datos
data = pd.read_csv("./data/credit_npo.csv")
data = data.dropna()  # Eliminar nulos
data.rename(columns={
    'SeriousDlqin2yrs': 'default_2yrs',
    'RevolvingUtilizationOfUnsecuredLines': 'revolving_util',
    'age': 'age',
    'NumberOfTime30-59DaysPastDueNotWorse': 'past_due_30_59',
    'DebtRatio': 'debt_ratio',
    'MonthlyIncome': 'monthly_income',
    'NumberOfOpenCreditLinesAndLoans': 'open_credit_lines',
    'NumberOfTimes90DaysLate': 'past_due_90',
    'NumberRealEstateLoansOrLines': 'real_estate_loans',
    'NumberOfTime60-89DaysPastDueNotWorse': 'past_due_60_89',
    'NumberOfDependents': 'dependents'
}, inplace=True)

# Separar variables predictoras y target

# Separar X e y
X = data.drop('default_2yrs', axis=1)
y = data['default_2yrs']

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Rebalancear con SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Escalar los datos
scaler = StandardScaler()
X_train_res_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

# Entrenar el modelo
model = LogisticRegression(class_weight='balanced', max_iter=3000, random_state=42)
model.fit(X_train_res_scaled, y_train_res)

# Hacer predicciones con el set correcto
y_pred = model.predict(X_test_scaled)

# Evaluación
print("F1-score Test:", f1_score(y_test, y_pred))
print("\nMatriz de Confusión:\n", confusion_matrix(y_test, y_pred))
print("\nReporte de Clasificación:\n", classification_report(y_test, y_pred))

# Guardar el modelo
model.fit(X, y)
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)