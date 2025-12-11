import numpy as np
import joblib
import tensorflow as tf
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ruta_modelo = os.path.join(BASE_DIR, "modelo_lstm_ferreycorp.keras")
ruta_scaler = os.path.join(BASE_DIR, "scaler_lstm.pkl")
ruta_csv = os.path.join(BASE_DIR, "datos", "FERREYC1_2025-10-17_2025-11-17.csv")

modelo = tf.keras.models.load_model(ruta_modelo)
scaler = joblib.load(ruta_scaler)

N_COLUMNS = 5
IDX_CIERRE = 3


def cargar_ultimos_60_dias():
    df = pd.read_csv(ruta_csv, sep=';')
    df.columns = df.columns.str.strip()

    df['Fecha'] = pd.to_datetime(df['Fecha'], dayfirst=True, errors='coerce')
    df = df.sort_values("Fecha").reset_index(drop=True)

    # MISMO PROCESO QUE COLAB: dataset completo
    dataset = df[["Apertura", "Maximo", "Minimo", "Cierre", "Volumen"]].values

    # ESCALAR TODO EL DATASET (no solo el tail)
    dataset_scaled = scaler.transform(dataset)

    # ÚLTIMOS 60 IGUAL QUE COLAB
    bloque = dataset_scaled[-60:]

    return bloque.astype(np.float32)


def predecir_lstm(bloque):
    bloque = np.array(bloque, dtype=np.float32).reshape(1, 60, 5)

    pred_scaled = modelo.predict(bloque)

    dummy = np.zeros((1, 5), dtype=np.float32)
    dummy[0, IDX_CIERRE] = pred_scaled[0][0]

    pred_inv = scaler.inverse_transform(dummy)[0][IDX_CIERRE]
    return float(pred_inv)


def obtener_datos_y_prediccion():
    bloque_scaled = cargar_ultimos_60_dias()

    # PARA MOSTRAR EN TABLA → desescalar cada fila
    bloque_real = scaler.inverse_transform(bloque_scaled)

    prediccion = predecir_lstm(bloque_scaled)

    datos = []
    for fila in bloque_real:
        datos.append({
            "apertura": float(fila[0]),
            "maximo":   float(fila[1]),
            "minimo":   float(fila[2]),
            "cierre":   float(fila[3]),
            "volumen":  float(fila[4]),
        })

    return datos, prediccion
