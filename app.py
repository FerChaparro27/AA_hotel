# ============================================================
# app.py — Predicción de ocupación mensual por hotel
# ============================================================

import streamlit as st
import pandas as pd
import pickle

# ------------------------------------
# 1️⃣ Cargar datos y modelo
# ------------------------------------
# Cargar dataset base (para los promedios históricos)
df = pd.read_csv("data/hotel_bookings.csv")

# Codificar variables
month_order = ["January","February","March","April","May","June",
               "July","August","September","October","November","December"]
df["arrival_date_month"] = pd.Categorical(df["arrival_date_month"],
                                          categories=month_order, ordered=True)
df["month_num"] = df["arrival_date_month"].cat.codes + 1
df["hotel"] = df["hotel"].map({"City Hotel":0, "Resort Hotel":1})

# Calcular promedios históricos por hotel y mes
num_cols = ["lead_time","adr","stays_in_week_nights","stays_in_weekend_nights","total_of_special_requests"]
promedios = (
    df.groupby(['hotel','month_num'])[num_cols]
      .mean()
      .reset_index()
)

# Cargar modelo entrenado
with open('modelo_regresionsvm.pkl', 'rb') as f:
    modelo_svr = pickle.load(f)

svr = modelo_svr["model"]
scaler_X = modelo_svr["scaler_X"]
scaler_y = modelo_svr["scaler_y"]
features = modelo_svr["features"]

# ------------------------------------
# 2️⃣ Interfaz de usuario
# ------------------------------------
st.title("🏨 Predicción de ocupación mensual")

# Inputs usuario
inputs = {}
inputs['hotel'] = st.selectbox("Hotel", [0, 1], format_func=lambda x: "City Hotel" if x==0 else "Resort Hotel")
inputs['arrival_date_year'] = st.number_input("Año", min_value=2000, max_value=2030, value=2025)
inputs['month_num'] = st.selectbox("Mes", list(range(1,13)), format_func=lambda x: month_order[x-1])

# ------------------------------------
# 3️⃣ Completar automáticamente variables restantes
# ------------------------------------
fila_prom = promedios[(promedios['hotel']==inputs['hotel']) & (promedios['month_num']==inputs['month_num'])]

for col in num_cols:
    inputs[col] = fila_prom[col].values[0]

# ------------------------------------
# 4️⃣ Predicción
# ------------------------------------
if st.button("Predecir ocupación"):
    row = pd.DataFrame([inputs])[features]
    X_scaled = scaler_X.transform(row)
    y_scaled = svr.predict(X_scaled).reshape(-1,1)
    y_pred = scaler_y.inverse_transform(y_scaled).ravel()[0]
    y_pred = max(y_pred, 0.0)
    st.success(f"🎯 Predicción: {y_pred:.0f} reservas/mes")
