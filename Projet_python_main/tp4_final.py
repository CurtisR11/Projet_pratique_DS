
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression

# Paramètres optimaux pour chaque entreprise
best_params = {
    "Tesla": {
        "XGBoost": {
            'colsample_bytree': 1.0,
            'learning_rate': 0.1,
            'max_depth': 3,
            'min_child_weight': 3,
            'n_estimators': 100,
            'subsample': 0.8,
            'objective': 'reg:squarederror',
            'seed': 42
        },
        "RandomForest": {
            'max_depth': 5,
            'min_samples_leaf': 2,
            'min_samples_split': 5,
            'n_estimators': 50,
            'random_state': 42
        },
        "KNN": {
            'n_neighbors': 9,
            'p': 1,
            'weights': 'uniform'
        }
    },
    "Samsung": {
        "XGBoost": {
            'colsample_bytree': 1.0,
            'learning_rate': 0.1,
            'max_depth': 3,
            'min_child_weight': 1,
            'n_estimators': 100,
            'subsample': 1.0,
            'objective': 'reg:squarederror',
            'seed': 42
        },
        "RandomForest": {
            'max_depth': 5,
            'min_samples_leaf': 1,
            'min_samples_split': 5,
            'n_estimators': 50,
            'random_state': 42
        },
        "KNN": {
            'n_neighbors': 3,
            'p': 2,
            'weights': 'distance'
        }
    },
    "Tencent": {
        "XGBoost": {
            'colsample_bytree': 0.8,
            'learning_rate': 0.1,
            'max_depth': 3,
            'min_child_weight': 1,
            'n_estimators': 100,
            'subsample': 1.0,
            'objective': 'reg:squarederror',
            'seed': 42
        },
        "RandomForest": {
            'max_depth': 10,
            'min_samples_leaf': 2,
            'min_samples_split': 2,
            'n_estimators': 50,
            'random_state': 42
        },
        "KNN": {
            'n_neighbors': 9,
            'p': 2,
            'weights': 'uniform'
        }
    },
    "Alibaba": {
        "XGBoost": {
            'colsample_bytree': 1.0,
            'learning_rate': 0.1,
            'max_depth': 7,
            'min_child_weight': 3,
            'n_estimators': 200,
            'subsample': 1.0,
            'objective': 'reg:squarederror',
            'seed': 42
        },
        "RandomForest": {
            'max_depth': None,
            'min_samples_leaf': 1,
            'min_samples_split': 2,
            'n_estimators': 200,
            'random_state': 42
        },
        "KNN": {
            'n_neighbors': 3,
            'p': 2,
            'weights': 'distance'
        }
    },
    "Sony": {
        "XGBoost": {
            'colsample_bytree': 1.0,
            'learning_rate': 0.1,
            'max_depth': 3,
            'min_child_weight': 3,
            'n_estimators': 50,
            'subsample': 0.8,
            'objective': 'reg:squarederror',
            'seed': 42
        },
        "RandomForest": {
            'max_depth': 3,
            'min_samples_leaf': 1,
            'min_samples_split': 2,
            'n_estimators': 50,
            'random_state': 42
        },
        "KNN": {
            'n_neighbors': 9,
            'p': 2,
            'weights': 'uniform'
        }
    },
    "Adobe": {
        "XGBoost": {
            'colsample_bytree': 1.0,
            'learning_rate': 0.1,
            'max_depth': 3,
            'min_child_weight': 3,
            'n_estimators': 50,
            'subsample': 1.0,
            'objective': 'reg:squarederror',
            'seed': 42
        },
        "RandomForest": {
            'max_depth': 5,
            'min_samples_leaf': 2,
            'min_samples_split': 5,
            'n_estimators': 50,
            'random_state': 42
        },
        "KNN": {
            'n_neighbors': 9,
            'p': 2,
            'weights': 'distance'
        }
    },
    "Johnson & Johnson": {
        "XGBoost": {
            'colsample_bytree': 0.8,
            'learning_rate': 0.1,
            'max_depth': 3,
            'min_child_weight': 3,
            'n_estimators': 100,
            'subsample': 1.0,
            'objective': 'reg:squarederror',
            'seed': 42
        },
        "RandomForest": {
            'max_depth': 5,
            'min_samples_leaf': 2,
            'min_samples_split': 2,
            'n_estimators': 100,
            'random_state': 42
        },
        "KNN": {
            'n_neighbors': 9,
            'p': 2,
            'weights': 'uniform'
        }
    },
    "Pfizer": {
        "XGBoost": {
            'colsample_bytree': 1.0,
            'learning_rate': 0.1,
            'max_depth': 3,
            'min_child_weight': 1,
            'n_estimators': 100,
            'subsample': 1.0,
            'objective': 'reg:squarederror',
            'seed': 42
        },
        "RandomForest": {
            'max_depth': None,
            'min_samples_leaf': 2,
            'min_samples_split': 5,
            'n_estimators': 100,
            'random_state': 42
        },
        "KNN": {
            'n_neighbors': 7,
            'p': 2,
            'weights': 'distance'
        }
    },
    "Louis Vuitton (LVMH)": {
        "XGBoost": {
            'colsample_bytree': 1.0,
            'learning_rate': 0.1,
            'max_depth': 3,
            'min_child_weight': 3,
            'n_estimators': 100,
            'subsample': 1.0,
            'objective': 'reg:squarederror',
            'seed': 42
        },
        "RandomForest": {
            'max_depth': 5,
            'min_samples_leaf': 2,
            'min_samples_split': 5,
            'n_estimators': 100,
            'random_state': 42
        },
        "KNN": {
            'n_neighbors': 9,
            'p': 2,
            'weights': 'uniform'
        }
    }
}

def load_close_data(file_path):
    df = pd.read_csv(file_path)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').set_index('Date')
    if 'Close' not in df.columns:
        raise ValueError(f"Le fichier {file_path} ne contient pas la colonne 'Close'.")
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df.dropna(subset=['Close'], inplace=True)
    return df[['Close']]

def create_target_features(df, n=30):
    x, y = [], []
    for i in range(n, df.shape[0]):
        x.append(df[i-n:i, 0])
        y.append(df[i, 0])
    return np.array(x), np.array(y)

def prepare_data_for_regression(file_path, n_days=30, test_ratio=0.2):
    df_close = load_close_data(file_path)
    close_vals = df_close.values
    scaler = MinMaxScaler()
    close_scaled = scaler.fit_transform(close_vals)
    X, Y = create_target_features(close_scaled, n=n_days)
    split_index = int((1 - test_ratio) * len(X))
    X_train, Y_train = X[:split_index], Y[:split_index]
    X_test, Y_test = X[split_index:], Y[split_index:]
    return X_train, Y_train, X_test, Y_test, scaler, close_vals

def build_xgb(params): return XGBRegressor(**params)
def build_rf(params): return RandomForestRegressor(**params)
def build_knn(params): return KNeighborsRegressor(**params)
def build_linear(): return LinearRegression()

def train_and_predict(model, X_train, Y_train, X_test, Y_test):
    model.fit(X_train, Y_train)
    pred = model.predict(X_test)
    mse = mean_squared_error(Y_test, pred)
    rmse = np.sqrt(mse)
    return pred, mse, rmse

def run_for_company(company_name, file_path, n_days=30):
    params_xgb = best_params[company_name]["XGBoost"]
    params_rf = best_params[company_name]["RandomForest"]
    params_knn = best_params[company_name]["KNN"]

    X_train, Y_train, X_test, Y_test, scaler, close_vals = prepare_data_for_regression(file_path, n_days=n_days, test_ratio=0.2)

    model_xgb = build_xgb(params_xgb)
    model_rf = build_rf(params_rf)
    model_knn = build_knn(params_knn)
    model_lr = build_linear()

    preds_scaled = {}
    preds_scaled["XGBoost"], _, _ = train_and_predict(model_xgb, X_train, Y_train, X_test, Y_test)
    preds_scaled["RandomForest"], _, _ = train_and_predict(model_rf, X_train, Y_train, X_test, Y_test)
    preds_scaled["KNN"], _, _ = train_and_predict(model_knn, X_train, Y_train, X_test, Y_test)
    preds_scaled["Linear"], _, _ = train_and_predict(model_lr, X_train, Y_train, X_test, Y_test)

    Y_test_true = scaler.inverse_transform(Y_test.reshape(-1,1)).ravel()
    results = []
    preds_true = {}

    for model_name, y_scaled in preds_scaled.items():
        y_true = scaler.inverse_transform(y_scaled.reshape(-1,1)).ravel()
        preds_true[model_name] = y_true
        mse_inv = mean_squared_error(Y_test_true, y_true)
        rmse_inv = np.sqrt(mse_inv)
        mse_norm = mean_squared_error(Y_test, preds_scaled[model_name])
        rmse_norm = np.sqrt(mse_norm)
        results.append({
            "Model": model_name,
            "MSE_norm": mse_norm,
            "RMSE_norm": rmse_norm,
            "MSE_inversed": mse_inv,
            "RMSE_inversed": rmse_inv
        })

    df_res = pd.DataFrame(results)

    real_values = close_vals.ravel()
    test_start_index = len(close_vals) - len(Y_test_true)
    x_axis_pred = range(test_start_index, test_start_index + len(Y_test_true))

    plt.figure(figsize=(12, 6))
    plt.plot(real_values, label="Vraies valeurs (Close)", color='black')
    for model_name, y_pred in preds_true.items():
        plt.plot(x_axis_pred, y_pred, label=f"{model_name} Pred")
    plt.title(f"Prédictions pour {company_name}")
    plt.xlabel("Jours indexés (jour 0 = début de l'historique)")
    plt.ylabel("Prix de clôture")
    plt.legend()
    plt.show()

    # Prédictions J+1 avec chaque modèle entraîné
    last_window = close_vals[-n_days:]
    last_window_scaled = scaler.transform(last_window.reshape(-1, 1)).reshape(1, -1)

    next_day_preds = {}
    for model_name, model in zip(
        ["XGBoost", "RandomForest", "KNN", "Linear"],
        [model_xgb, model_rf, model_knn, model_lr]
    ):
        pred_scaled = model.predict(last_window_scaled)
        pred_real = scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]
        next_day_preds[model_name] = round(pred_real, 2)

    return df_res, next_day_preds

def main():
    companies = {
        "Tesla": "TSLA",
        "Samsung": "005930.KS",
        "Tencent": "TCEHY",
        "Alibaba": "BABA",
        "Sony": "SONY",
        "Adobe": "ADBE",
        "Johnson & Johnson": "JNJ",
        "Pfizer": "PFE",
        "Louis Vuitton (LVMH)": "MC.PA",
    }

    user_path = os.path.expanduser("~")
    base_path = os.path.join(user_path, "Desktop", "Projet_python")
    data_folder = os.path.join(base_path, "data_tp1", "companies")
    output_folder = os.path.join(base_path, "output", "prediction_tp4")
    os.makedirs(output_folder, exist_ok=True)

    all_results = []
    next_day_all_models = []

    for company_name, symbol in companies.items():
        csv_file = f"{company_name}.csv"
        file_path = os.path.join(data_folder, csv_file)

        if not os.path.exists(file_path):
            print(f"Fichier introuvable : {file_path}")
            continue

        print(f"\n--- {company_name} ({symbol}) ---")
        df_res, next_day_preds = run_for_company(company_name, file_path, n_days=30)
        df_res["Company"] = company_name
        all_results.append(df_res)

        row = {"Company": company_name}
        row.update(next_day_preds)
        next_day_all_models.append(row)

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        print("\n=== Résumé final ===")
        print("\n=== Erreurs ===")
        print(final_df)
        output_csv_path = os.path.join(output_folder, "models_results_tp4.csv")
        final_df.to_csv(output_csv_path, index=False)
        print(f"\n✅ Résultats sauvegardés dans : {output_csv_path}")

        df_preds = pd.DataFrame(next_day_all_models)
        print("\n=== Prédictions J+1 ===")
        print(df_preds)
        pred_csv_path = os.path.join(output_folder, "j+1_predictions_models_tp4.csv")
        df_preds.to_csv(pred_csv_path, index=False)
        print(f"📤 Prédictions J+1 sauvegardées ici : {pred_csv_path}")
    else:
        print("Aucun fichier traité.")

if __name__ == "__main__":
    main()
