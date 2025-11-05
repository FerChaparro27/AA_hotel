# ============================================================
# app.py — Predicción de ocupación mensual (SVR) + Cancelaciones (KNN)
# ============================================================

import streamlit as st
import pandas as pd
import pickle
import os

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
# 2️⃣ Cargar modelo de regresión (SVR)
# ------------------------------------
with open("modelo_regresionsvm.pkl", "rb") as f:
    modelo_reg = pickle.load(f)

svr = modelo_reg["model"]
scaler_X_reg = modelo_reg["scaler_X"]
scaler_y_reg = modelo_reg["scaler_y"]
features_reg = modelo_reg["features"]

# ------------------------------------
# 3️⃣ Interfaz de usuario - SVR
# ------------------------------------
st.title("🏨 Predicción de Ocupación y Cancelaciones de Reservas")

inputs = {}
inputs["hotel"] = st.selectbox(
    "Hotel",
    [0, 1],
    format_func=lambda x: "City Hotel" if x==0 else "Resort Hotel",
    key="hotel_svr"
)
inputs["arrival_date_year"] = st.number_input(
    "Año", min_value=2015, max_value=2017, value=2015, key="year_svr"
)
inputs["month_num"] = st.selectbox(
    "Mes", list(range(1,13)),
    format_func=lambda x: month_order[x-1],
    key="month_svr"
)

# Rellenar variables promedio automáticas
fila_prom = promedios[
    (promedios["hotel"] == inputs["hotel"]) &
    (promedios["month_num"] == inputs["month_num"])
]
for col in num_cols:
    inputs[col] = fila_prom[col].values[0] if not fila_prom.empty else 0

# Predicción SVR
if st.button("🔮 Predecir ocupación (SVR)", key="btn_svr"):
    row = pd.DataFrame([inputs])[features_reg]
    X_scaled = scaler_X_reg.transform(row)
    y_scaled = svr.predict(X_scaled).reshape(-1, 1)
    y_pred = scaler_y_reg.inverse_transform(y_scaled).ravel()[0]
    y_pred = max(y_pred, 0.0)
    st.success(f"📊 Predicción estimada (SVR): **{y_pred:.0f} reservas/mes**")

# ------------------------------------
# 4️⃣ KNN - Predicción de cancelación
# ------------------------------------
st.markdown("---")
st.subheader("🤖 Predicción de cancelación de reserva (KNN)")

if os.path.exists("modelo_knn.pkl"):
    with open("modelo_knn.pkl", "rb") as f:
        modelo_knn = pickle.load(f)

    knn = modelo_knn["model"]
    scaler_knn = modelo_knn["scaler"]
    features_knn = modelo_knn["features"]

    st.markdown("### 📋 Ingresá los datos de la reserva")
    # Inputs numéricos
    lead_time = st.number_input("Lead time (días)", 0, 1000, 50, key="lead_time")
    adr = st.number_input("ADR (tarifa diaria)", 0.0, 10000.0, 100.0, key="adr")
    stays_week = st.number_input("Noches entre semana", 0, 30, 3, key="stays_week")
    stays_weekend = st.number_input("Noches fin de semana", 0, 30, 1, key="stays_weekend")
    adults = st.number_input("Adultos", 1, 10, 2, key="adults")
    children = st.number_input("Niños", 0, 10, 0, key="children")
    babies = st.number_input("Bebés", 0, 5, 0, key="babies")
    special_requests = st.number_input("Solicitudes especiales", 0, 10, 1, key="special_requests")
    previous_cancellations = st.number_input("Cancelaciones previas", 0, 20, 0, key="prev_cancel")
    previous_bookings_not_canceled = st.number_input("Reservas previas no canceladas", 0, 50, 0, key="prev_not_cancel")
    booking_changes = st.number_input("Cambios de reserva", 0, 20, 0, key="booking_changes")
    days_in_waiting_list = st.number_input("Días en lista de espera", 0, 100, 0, key="days_wait")
    required_car_parking_spaces = st.number_input("Plazas de parking requeridas", 0, 5, 0, key="parking")
    hotel_input = st.selectbox(
        "Hotel",
        [0,1],
        format_func=lambda x: "City Hotel" if x==0 else "Resort Hotel",
        key="hotel_knn"
    )

    # Inputs categóricos
    customer_type = st.selectbox(
        "Tipo de cliente",
        ["Transient", "Contract", "Group", "Transient-Party"],
        key="customer_type"
    )
    deposit_type = st.selectbox(
        "Tipo de depósito",
        ["No Deposit", "Refundable", "Non Refund"],
        key="deposit_type"
    )
    market_segment = st.selectbox(
        "Segmento de mercado",
        ["Direct", "Corporate", "Online TA", "Groups", "Complementary", "Offline TA/TO"],
        key="market_segment"
    )

    # Crear DataFrame con todas las columnas
    data_knn = pd.DataFrame(columns=features_knn)

    # Asignar valores numéricos
    input_map = {
        "lead_time": lead_time,
        "adr": adr,
        "stays_in_week_nights": stays_week,
        "stays_in_weekend_nights": stays_weekend,
        "adults": adults,
        "children": children,
        "babies": babies,
        "total_of_special_requests": special_requests,
        "previous_cancellations": previous_cancellations,
        "previous_bookings_not_canceled": previous_bookings_not_canceled,
        "booking_changes": booking_changes,
        "days_in_waiting_list": days_in_waiting_list,
        "required_car_parking_spaces": required_car_parking_spaces,
        "hotel": hotel_input
    }
    for col in features_knn:
        data_knn.at[0, col] = input_map.get(col, 0)

    # Crear dummies automáticos
    for ct in ["customer_type_Group","customer_type_Transient","customer_type_Transient-Party"]:
        data_knn.at[0, ct] = 1 if ct.endswith(customer_type.replace(" ","-")) else 0
    for dt in ["deposit_type_Non Refund","deposit_type_Refundable"]:
        data_knn.at[0, dt] = 1 if dt.split("_")[-1] == deposit_type.replace(" ","") else 0
    for ms in ["market_segment_Complementary","market_segment_Corporate","market_segment_Direct",
               "market_segment_Groups","market_segment_Offline TA/TO","market_segment_Online TA"]:
        data_knn.at[0, ms] = 1 if ms.split("_")[-1].replace("/","") == market_segment.replace(" ","") else 0

    # Escalar y predecir
    X_scaled_knn = scaler_knn.transform(data_knn)
    y_pred = knn.predict(X_scaled_knn)[0]
    prob = knn.predict_proba(X_scaled_knn)[0][1]

    if st.button("🧮 Predecir cancelación", key="btn_knn"):
        if y_pred == 1:
            st.error(f"⚠️ La reserva probablemente se CANCELARÁ ({prob*100:.1f}% de probabilidad)")
        else:
            st.success(f"✅ La reserva probablemente NO se cancelará ({(1-prob)*100:.1f}% de probabilidad)")

else:
    st.warning("⚠️ No se encontró el archivo `modelo_knn.pkl`. Guardalo en la raíz del proyecto.")

# ------------------------------------
# 5️⃣ Créditos / Footer
# ------------------------------------
st.markdown("---")
st.caption("Desarrollado por Francisco Romero - Fernando Chaparro — Predicción de Ocupación Hotelera 🧠")

# ------------------------------------
# 4️⃣️⃣️⃣ Clustering de clientes (K-Means)
# ------------------------------------
st.markdown("---")
st.subheader("📊 Clasificación de reserva según comportamiento (Clustering K-Means)")

try:
    # Cargar modelo y scaler
    with open("modelo_clusters.pkl", "rb") as f:
        kmeans_final = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler_cluster = pickle.load(f)

    columnas_cluster = [
        'lead_time', 
        'adr', 
        'stays_in_week_nights',
        'stays_in_weekend_nights',
        'previous_cancellations',
        'previous_bookings_not_canceled',
        'is_repeated_guest',
        'total_of_special_requests',
        'adults',
        'children'
    ]

    st.markdown("### ✏️ Ingresá los valores para estimar el tipo de cliente o reserva")

    # Entradas del usuario
    lead_time_c = st.number_input("Lead time (días de anticipación)", 0, 1000, 100, key="lead_time_cluster")
    adr_c = st.number_input("ADR (tarifa diaria promedio, €)", 0.0, 1000.0, 120.0, key="adr_cluster")
    stays_week_c = st.number_input("Noches entre semana", 0, 30, 3, key="stays_week_cluster")
    stays_weekend_c = st.number_input("Noches de fin de semana", 0, 10, 1, key="stays_weekend_cluster")
    prev_cancel_c = st.number_input("Cancelaciones previas", 0, 20, 0, key="prev_cancel_cluster")
    prev_not_cancel_c = st.number_input("Reservas previas no canceladas", 0, 50, 0, key="prev_not_cancel_cluster")
    is_repeated_c = st.selectbox("Cliente repetido", [0, 1], format_func=lambda x: "Sí" if x==1 else "No", key="repeated_cluster")
    special_req_c = st.number_input("Solicitudes especiales", 0, 10, 1, key="special_req_cluster")
    adults_c = st.number_input("Adultos", 1, 10, 2, key="adults_cluster")
    children_c = st.number_input("Niños", 0, 10, 0, key="children_cluster")

    if st.button("🎯 Clasificar reserva (K-Means)", key="btn_cluster"):
        # Crear DataFrame con los valores del usuario
        df_cluster_input = pd.DataFrame([{
            'lead_time': lead_time_c,
            'adr': adr_c,
            'stays_in_week_nights': stays_week_c,
            'stays_in_weekend_nights': stays_weekend_c,
            'previous_cancellations': prev_cancel_c,
            'previous_bookings_not_canceled': prev_not_cancel_c,
            'is_repeated_guest': is_repeated_c,
            'total_of_special_requests': special_req_c,
            'adults': adults_c,
            'children': children_c
        }])

        # Escalar igual que durante el entrenamiento
        X_scaled_cluster = scaler_cluster.transform(df_cluster_input)

        # Predecir cluster
        cluster_pred = kmeans_final.predict(X_scaled_cluster)[0]

        st.success(f"📍 La reserva pertenece al **Cluster {cluster_pred}**")

        # Descripciones breves de los 4 clusters
        descripciones = {
            0: "🟢 **Cluster 0:** Clientes organizados, reservan con anticipación, pocas cancelaciones y estadías medias.",
            1: "🟣 **Cluster 1:** Clientes con alto gasto (ADR alto), muchas solicitudes especiales y estadías largas.",
            2: "🟠 **Cluster 2:** Clientes impulsivos o inconstantes, reservas cortas y cancelaciones frecuentes.",
            3: "🔴 **Cluster 3:** Casos menos comunes — grupos grandes o reservas con patrones atípicos."
        }

        st.info(descripciones.get(cluster_pred, "Sin descripción disponible."))

except FileNotFoundError:
    st.warning("⚠️ No se encontraron los archivos `modelo_clusters.pkl` o `scaler.pkl`. Guardalos en la raíz del proyecto.")
except Exception as e:
    st.error(f"❌ Error al ejecutar el modelo de clustering: {e}")

