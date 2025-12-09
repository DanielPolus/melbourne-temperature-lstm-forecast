# Работа с данными
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Предобработка
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# LSTM-модель
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import mean_squared_error, root_mean_squared_error

# (если берём датасет с Kaggle)
import kagglehub

path = kagglehub.dataset_download("samfaraday/daily-minimum-temperatures-in-me")

print("Path to dataset files:", path)
print("Files in this folder:", os.listdir(path))

data_path = Path(path) / "daily-minimum-temperatures-in-me.csv"
df = pd.read_csv(
    data_path,
    on_bad_lines='skip',
    parse_dates=['Date']
)

temp_col = 'Daily minimum temperatures in Melbourne, Australia, 1981-1990'
df = df.rename(columns={temp_col: 'temp'})
df['temp'] = df['temp'].astype(str).str.strip()
df['temp'] = pd.to_numeric(df['temp'], errors='coerce')
print(df[df['temp'].isna()].head())
df = df.dropna(subset=['temp'])
df = df.sort_values('Date')

print(f"Dataframe's shape: {df.shape}")
print(f"Dataframe's dtypes: {df.dtypes}")
print(f"Dataframe's head: {df.head()}")
print(f"Dataframe's NaN: {df.isna().mean().sort_values(ascending=False)}")

values = df['temp'].values.reshape(-1, 1)

scaler = MinMaxScaler()
values_scaled = scaler.fit_transform(values)

def create_sequences(series, window_size):
    X = []
    y = []
    for i in range(window_size, len(series)):
        X_window = series[i-window_size:i]
        y_next = series[i]

        X.append(X_window)
        y.append(y_next)

    return np.array(X), np.array(y)

window_size = 30
X, y = create_sequences(values_scaled, window_size)

split_index = int(len(X) * 0.8)

X_train = X[:split_index]
y_train = y[:split_index]

X_test = X[split_index:]
y_test = y[split_index:]

model = Sequential()
model.add(LSTM(64, input_shape=(window_size, 1)))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.1,
    shuffle=False,
)

# предсказания в scaled виде
preds_scaled = model.predict(X_test)

# однажды делаем "обратное" масштабирование
y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
y_pred_lstm = scaler.inverse_transform(preds_scaled).flatten()

rmse_lstm = root_mean_squared_error(y_true, y_pred_lstm)
print(f"LSTM RMSE: {rmse_lstm:.4f}")

plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

plt.plot(y_test[:200], label='true')
plt.plot(y_pred_lstm[:200], label='pred')
plt.legend()
plt.show()

# ===== Naive: завтра = вчера =====
naive_scaled = X_test[:, -1, :]          # (n_samples, 1)
naive = scaler.inverse_transform(naive_scaled).flatten()

rmse_naive = root_mean_squared_error(y_true, naive)
print(f"Naive baseline RMSE: {rmse_naive:.4f}")

# ===== Среднее за 3 последних дня =====
last3_scaled = X_test[:, -3:, 0]                       # (n_samples, 3)
ma3_scaled = last3_scaled.mean(axis=1).reshape(-1, 1)  # (n_samples, 1)
ma3 = scaler.inverse_transform(ma3_scaled).flatten()

rmse_ma3 = root_mean_squared_error(y_true, ma3)
print(f"3-day mean baseline RMSE: {rmse_ma3:.4f}")
