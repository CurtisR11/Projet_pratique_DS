import os
import glob
import pandas as pd
import numpy as np
import joblib
import ta
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# ----------------------------
# 1. DATA LOADING AND LABELING
# ----------------------------

def generate_labels(df):
    df['Close Horizon'] = df['Close'].shift(-20)
    df['Horizon Return'] = (df['Close Horizon'] - df['Close']) / df['Close']
    df['Label'] = df['Horizon Return'].apply(lambda x: 2 if x > 0.05 else (0 if x < -0.05 else 1))
    return df

def compute_features(df):
    df['SMA 20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['EMA 20'] = ta.trend.ema_indicator(df['Close'], window=20)
    df['RSI 14'] = ta.momentum.rsi(df['Close'], window=14)
    df['MACD'] = ta.trend.macd(df['Close'])
    df['MACD Signal'] = ta.trend.macd_signal(df['Close'])
    df['Bollinger High'] = ta.volatility.bollinger_hband(df['Close'])
    df['Bollinger Low'] = ta.volatility.bollinger_lband(df['Close'])
    df['Rolling Volatility 20'] = df['Close'].rolling(window=20).std()
    df['ROC 10'] = ta.momentum.roc(df['Close'], window=10)
    return df

def load_all_data(folder_path):
    files = glob.glob(f"{folder_path}/*.csv")
    dataframes = []

    for file in files:
        df = pd.read_csv(file)
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        if df['Close'].isnull().all():
            continue

        df = generate_labels(df)
        df = compute_features(df)
        df.dropna(inplace=True)
        dataframes.append(df)

    return pd.concat(dataframes, ignore_index=True)

# ----------------------------
# 2. PREPROCESSING
# ----------------------------

def preprocess_data(df):
    df = df.drop(columns=['Date'], errors='ignore')
    X = df.drop(columns=['Label', 'Close Horizon', 'Horizon Return'], errors='ignore')
    y = df['Label']
    X_scaled = StandardScaler().fit_transform(X)
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42), X.columns

# ----------------------------
# 3. MODELING + EVALUATION
# ----------------------------

def train_and_evaluate_model(name, model, param_grid, X_train, X_test, y_train, y_test, feature_names):
    print(f"\n🔍 Training {name}...")
    grid = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, scoring='accuracy')
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    print(f"\n📊 Classification report for {name}:\n")
    print(classification_report(y_test, y_pred))

    acc = accuracy_score(y_test, y_pred)

    if isinstance(best_model, (RandomForestClassifier, XGBClassifier)):
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_test)
        try:
            shap.summary_plot(shap_values[:, :, 2], X_test, feature_names=feature_names, show=False)
            plt.title(f"{name} - SHAP Summary (BUY)")
            plt.show()
            shap.summary_plot(shap_values[:, :, 0], X_test, feature_names=feature_names, show=False)
            plt.title(f"{name} - SHAP Summary (SELL)")
            plt.show()
        except:
            print("SHAP plotting skipped due to model output shape.")
    else:
        print(f"ℹ️ SHAP not supported for {name}")

    return name, acc

# ----------------------------
# 4. MAIN PIPELINE
# ----------------------------

def main():
    folder_path = os.path.expanduser("~/Desktop/Projet_python/data_tp1/companies")
    df = load_all_data(folder_path)
    (X_train, X_test, y_train, y_test), feature_names = preprocess_data(df)

    models = {
        "RandomForest": (RandomForestClassifier(), {'n_estimators': [50, 100]}),
        "XGBoost": (XGBClassifier(verbosity=0), {'n_estimators': [50], 'learning_rate': [0.05, 0.1]}),
        "KNN": (KNeighborsClassifier(), {'n_neighbors': [3, 5]}),
        "SVM": (SVC(), {'C': [1], 'kernel': ['linear']}),
        "LogisticRegression": (LogisticRegression(max_iter=500), {'C': [1.0]})
    }

    results = []
    for name, (model, params) in models.items():
        model_name, acc = train_and_evaluate_model(name, model, params, X_train, X_test, y_train, y_test, feature_names)
        results.append((model_name, acc))

    print("\n📈 Résumé des performances :")
    summary_df = pd.DataFrame(results, columns=['Model', 'Accuracy']).sort_values(by='Accuracy', ascending=False)
    print(summary_df)

if __name__ == "__main__":
    main()



print("\n📈 Résumé des performances :")
summary_df = pd.DataFrame(results, columns=['Model', 'Accuracy']).sort_values(by='Accuracy', ascending=False)
print(summary_df)

# 🔽 Créer le chemin de sauvegarde
desktop = os.path.join(os.path.expanduser("~"), "Desktop")
project_path = os.path.join(desktop, "Projet_python")
output_path = os.path.join(project_path, "output", "prediction_tp3")
os.makedirs(output_path, exist_ok=True)

# 💾 Sauvegarde du tableau des performances
results_csv_path = os.path.join(output_path, "model_performance.csv")
summary_df.to_csv(results_csv_path, index=False)
print(f"\n✅ Résultats sauvegardés dans : {results_csv_path}")

# 💾 Sauvegarde du meilleur modèle
best_model_name = summary_df.iloc[0]['Model']
best_model = dict(results)[best_model_name]
model_path = os.path.join(output_path, f"{best_model_name}_best_model.pkl")
joblib.dump(best_model, model_path)
print(f"✅ Meilleur modèle ({best_model_name}) sauvegardé dans : {model_path}")