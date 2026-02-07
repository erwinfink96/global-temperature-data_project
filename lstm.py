import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import tensorflow as tf


# Datensatz einlesen 
project_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(project_dir, 'GLB.Ts+dSSTnew.csv')

WINDOW = 120               # Eingabefenster in Monaten (120 Monate =10 Jahre)
TRAIN_END = "2025-12-31"   # Ende Trainingsperiode 
RANDOM_SEED = 0

# Liste mit Monaten erstellen 
months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

# Reproduzierbarkeit
#np.random.seed(RANDOM_SEED)
#tf.random.set_seed(RANDOM_SEED)


# 1. CSV laden ohne die Überschrift 
df = pd.read_csv(file_path, skiprows=1)

# erste geladene Zeile entfernen 
df = df.iloc[1:].reset_index(drop=True)



# 2. Monatswerte in Monatszeitreihe umwandeln

df_monthly = (
    df
    .set_index('Year')[months]
    .stack()
    .reset_index()
)

df_monthly.columns = ['Year', 'Month', 'value']

df_monthly['date'] = pd.to_datetime(
    df_monthly['Year'].astype(str) + '-' +
    (df_monthly['Month'].map(lambda m: months.index(m) + 1)).astype(str) +
    '-15'
)

df_monthly = df_monthly.set_index('date')[['value']].sort_index()



# 3. Zeitliche Aufteilung in Train- und Testdaten 
train_end = pd.to_datetime(TRAIN_END)
#train = df_monthly.loc[:train_end]  
train = df_monthly.loc['1980-01-01':train_end]
#test_start = train_end
#test = df_monthly.loc[test_start:]

# print(train)
# print(test)
# print(train_end)
# print(test_start)

# print(f"Monatliche Punkte: total={len(df_monthly)}, train={len(train)}, test={len(test)}")

# 4. Skalierung (fit nur auf Train)
scaler = MinMaxScaler()
train_vals = train['value'].values.reshape(-1,1)  # reshape für Skalierung 
#test_vals  = test['value'].values.reshape(-1,1)

scaler.fit(train_vals)
train_s = scaler.transform(train_vals)
#test_s  = scaler.transform(test_vals)

# 5. Sequenzen erzeugen 
def create_sequences(arr, window):
    x, y = [], []
    for i in range(window, len(arr)):
        x.append(arr[i-window:i, 0])     # erzeuge Liste mit arr[0:window,0], arr[1:window+1,0] usw... 
        y.append(arr[i, 0])              # erzeuge Liste mit darauffolgenden Monaten 
    x = np.array(x)

    return x.reshape((x.shape[0], x.shape[1], 1)), np.array(y)    #LSTM erwartet 3D-Input, daher reshape

x_train, y_train = create_sequences(train_s, WINDOW)
#x_test,  y_test  = create_sequences(test_s, WINDOW)

#print("Shapes:", x_train.shape, y_train.shape, x_test.shape, y_test.shape)


# 6. LSTM-Modell erstellen 
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(WINDOW, 1)),
    LSTM(32),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Early Stopping 
es = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)


# Modell auswerten 
history = model.fit(
    x_train, y_train,
    validation_split=0.1,
    epochs=300,
    batch_size=32,
    callbacks=[es],
    verbose=1
)

# # 7. Vorhersage auf Testset & Rückskalierung
# pred_s = model.predict(x_test).ravel()
# pred = scaler.inverse_transform(pred_s.reshape(-1,1)).ravel()
# y_true = scaler.inverse_transform(y_test.reshape(-1,1)).ravel()


# # Fehlerberechnung 
# rmse = np.sqrt(mean_squared_error(y_true, pred))
# mape = mean_absolute_percentage_error(y_true, pred) * 100
# print(f"Test RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")


# # 8. Plot: Test (echte Werte vs. Vorhersage)
# # für den Test werden die ersten WINDOW Monate weggelassen, 
# # da LSTM die ersten WINDOW Temperaturen für die Prognose braucht 

# test_index = test.index[WINDOW:]

# plt.figure(figsize=(12,5))
# plt.plot(test_index, y_true, label='True (test)')
# plt.plot(test_index, pred, label='Predicted (test)')
# plt.xlabel("Datum")
# plt.ylabel("Anomalie")
# plt.title("LSTM — Test: echte Werte vs Vorhersage")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()




# letztes WINDOW-Fenster (bis Dez 2025)
last_window = df_monthly['value'].iloc[-WINDOW:].values.reshape(-1,1)

# mit dem auf Train gefitteten Scaler skalieren
last_window_s = scaler.transform(last_window)

# LSTM erwartet Shape (1, WINDOW, 1)
x_2026 = last_window_s.reshape(1, WINDOW, 1)

# Vorhersage (skaliert → zurückskalieren)
pred_2026_s = model.predict(x_2026, verbose=0)
pred_2026 = scaler.inverse_transform(pred_2026_s)[0,0]

print(f"Vorhersage Temperatur-Anomalie für Jänner 2026: {pred_2026:.4f}")