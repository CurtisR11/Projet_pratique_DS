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

tickers = list(companies.values())
len(tickers)

# 📊 Préparation des données
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

def prepare_data(df, feature_col="Close", window_size=30):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[[feature_col]])
    X, y = [], []
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i-window_size:i])
        y.append(scaled_data[i])
    return np.array(X), np.array(y), scaler

# 🔀 Séparation des données
from sklearn.model_selection import train_test_split
import tensorflow as tf
tf.random.set_seed(42)

# ✅ MLP optimisé
def build_mlp_model(input_shape, hidden_dims=[128, 64, 32], activation='relu', dropout_rate=0.4, optimizer='adam'):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=input_shape))
    for dim in hidden_dims:
        model.add(tf.keras.layers.Dense(dim, activation=activation, kernel_initializer='glorot_uniform'))
        model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

# ✅ RNN optimisé
def build_rnn_model(input_shape, units=100, dropout_rate=0.4, optimizer='adam'):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.SimpleRNN(units, activation='tanh', input_shape=input_shape, return_sequences=False,
                                        kernel_initializer='glorot_uniform'))
    model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

# ✅ LSTM optimisé
def build_lstm_model(input_shape, units=100, dropout_rate=0.4, optimizer='adam'):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(units, activation='tanh', input_shape=input_shape, return_sequences=False,
                                   kernel_initializer='glorot_uniform'))
    model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

# 🚀 Entraînement générique
def train_model(model_type, X_train, y_train, input_shape, **kwargs):

    if model_type == "MLP":
        model = build_mlp_model(input_shape, **kwargs)
    elif model_type == "RNN":
        model = build_rnn_model(input_shape, **kwargs)
    elif model_type == "LSTM":
        model = build_lstm_model(input_shape, **kwargs)
    else:
        raise ValueError("Type de modèle inconnu.")
    model.fit(X_train, y_train, epochs=kwargs.get("epochs", 100), batch_size=kwargs.get("batch_size", 32), verbose=kwargs.get("verbose", 0) )
    return model

def predict(model, X_test, y_test, scaler, model_type):
    y_pred = model.predict(X_test)
    y_pred_rescaled = scaler.inverse_transform(y_pred)
    y_test_rescaled = scaler.inverse_transform(y_test)

    mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
    rmse = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))

    print(f"{model_type} - MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    print("\n10 premières valeurs prédites vs vraies :")
    for p, t in zip(y_pred_rescaled[:10], y_test_rescaled[:10]):
        print(f"Prédit : {p[0]:.2f} | Réel : {t[0]:.2f}")

    plt.figure(figsize=(10, 4))
    plt.plot(y_test_rescaled, label="Réel")
    plt.plot(y_pred_rescaled, label="Prédit")
    plt.title(f"{model_type} - Prédictions vs Réel")
    plt.legend()
    plt.show()

    return mae, rmse

from tqdm import tqdm
import os

desktop = os.path.join(os.path.expanduser("~"), "Desktop")
project_path = os.path.join(desktop, "Projet_python/output/neurol_network_tp5")
os.makedirs(project_path, exist_ok=True)
print(project_path)

for company in companies.keys():
    print(f'Training For Company {company}')
    tf_model_path = os.path.join(project_path, f"{company}_models_results")
    os.makedirs(tf_model_path, exist_ok=True)
    
    # 1. Chargement des données
    user_path = os.path.expanduser("~")
    folder_path = os.path.join(user_path, "Desktop", "Projet_python", "data_tp1", "companies", f"{company}.csv")
    df = pd.read_csv(folder_path)

    # 2. Préparation
    X, y, scaler = prepare_data(df, window_size=30)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # 3. Tests des modèles
    results = []
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005)


    # MLP
    model_mlp = train_model("MLP", X_train, y_train, X_train.shape[1:] )
    mae_mlp, rmse_mlp = predict(model_mlp, X_test, y_test, scaler, "MLP")
    results.append(["MLP", mae_mlp, rmse_mlp])
    model_mlp.save(os.path.join(tf_model_path, "mlp_model.keras"))

    # RNN
    model_rnn = train_model("RNN", X_train, y_train, X_train.shape[1:] )
    mae_rnn, rmse_rnn = predict(model_rnn, X_test, y_test, scaler, "RNN")
    results.append(["RNN", mae_rnn, rmse_rnn])
    model_rnn.save(os.path.join(tf_model_path, "rnn_model.keras"))

    # LSTM
    model_lstm = train_model("LSTM", X_train, y_train, X_train.shape[1:] )
    mae_lstm, rmse_lstm = predict(model_lstm, X_test, y_test, scaler, "LSTM")
    results.append(["LSTM", mae_lstm, rmse_lstm])
    model_lstm.save(os.path.join(tf_model_path, "lstm_model.keras"))



    # Résumé
    results_df = pd.DataFrame(results, columns=["Modèle", "MAE", "RMSE"])
    results_df.to_csv(os.path.join(tf_model_path, f"{company}_results_tp5.csv"), index=False)
    print(f"\n📊 Résumé des performances pour {company}:")
    print(results_df)

