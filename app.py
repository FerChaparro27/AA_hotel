# ============================================================
# app.py — Predicción de ocupación mensual (Regresión)
# ============================================================

import streamlit as st
import pandas as pd
import pickle

# ------------------------------------
# 1️⃣ Cargar dataset base (para promedios históricos)
# ------------------------------------
df = pd.read_csv("Data/hotel_bookings.csv")

# Codificar variables
month_order = [
    "January","February","March","April","May","June",
    "July","August","September","October","November","December"
]
df["arrival_date_month"] = pd.Categorical(df["arrival_date_month"],
                                          categories=month_order, ordered=True)
df["month_num"] = df["arrival_date_month"].cat.codes + 1
df["hotel"] = df["hotel"].map({"City Hotel":0, "Resort Hotel":1})

# Calcular promedios históricos por hotel y mes
num_cols = ["lead_time","adr","stays_in_week_nights",
            "stays_in_weekend_nights","total_of_special_requests"]

promedios = (
    df.groupby(["hotel","month_num"])[num_cols]
      .mean()
      .reset_index()
)

# ------------------------------------
# 2️⃣ Cargar modelo de regresión
# ------------------------------------
with open("modelo_regresionsvm.pkl", "rb") as f:
    modelo_reg = pickle.load(f)

svr = modelo_reg["model"]
scaler_X_reg = modelo_reg["scaler_X"]
scaler_y_reg = modelo_reg["scaler_y"]
features_reg = modelo_reg["features"]

# ------------------------------------
# 3️⃣ Interfaz de usuario
# ------------------------------------
st.title("🏨 Predicción de Ocupación Mensual por Hotel")

inputs = {}
inputs["hotel"] = st.selectbox(
    "Hotel",
    [0, 1],
    format_func=lambda x: "City Hotel" if x==0 else "Resort Hotel"
)
inputs["arrival_date_year"] = st.number_input("Año", min_value=2015, max_value=2017, value=2015)
inputs["month_num"] = st.selectbox("Mes", list(range(1,13)),
                                   format_func=lambda x: month_order[x-1])

# ------------------------------------
# 4️⃣ Rellenar variables promedio automáticas
# ------------------------------------
fila_prom = promedios[
    (promedios["hotel"] == inputs["hotel"]) &
    (promedios["month_num"] == inputs["month_num"])
]

for col in num_cols:
    inputs[col] = fila_prom[col].values[0] if not fila_prom.empty else 0

# ------------------------------------
# 5️⃣ Predicción usando SVR
# ------------------------------------
if st.button("🔮 Predecir ocupación"):
    row = pd.DataFrame([inputs])[features_reg]
    X_scaled = scaler_X_reg.transform(row)
    y_scaled = svr.predict(X_scaled).reshape(-1, 1)
    y_pred = scaler_y_reg.inverse_transform(y_scaled).ravel()[0]
    y_pred = max(y_pred, 0.0)
    st.success(f"📊 Predicción estimada: **{y_pred:.0f} reservas/mes**")

# ------------------------------------
# 6️⃣ Créditos / Footer
# ------------------------------------
st.markdown("---")
st.caption("Desarrollado por Francisco Romero - Fernando Chaparro — Predicción de Ocupación Hotelera 🧠")
