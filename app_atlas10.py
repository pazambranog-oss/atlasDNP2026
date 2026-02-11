

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import plotly.express as px


st.set_page_config(
    page_title="Atlas Migrantes – DNP",
    layout="wide"
)

# =========================
# 📥 CARGA DE DATOS
# =========================

@st.cache_data
def cargar_bases():
    #ruta1 = "/Users/paozambrano/Desktop/ProyectoAtlasDNP/rsh_migrantes_limpio.parquet"
    #ruta2 = "/Users/paozambrano/Desktop/ProyectoAtlasDNP/Evolucion_match.parquet"

    ruta1 = "rsh_migrantes_limpio.parquet"
    ruta2 = "Evolucion_match.parquet"

    base1 = pd.read_parquet(ruta1)
    base2 = pd.read_parquet(ruta2)

    return base1, base2


base1, base2 = cargar_bases()

st.success("Bases cargadas correctamente")


# =========================
# 🧹 LIMPIEZA BÁSICA
# =========================

base1["género"] = base1["Sexo"].map({"M": "Masculino", "F": "Femenino"})
base1["edad"] = pd.to_numeric(base1["edad"], errors="coerce")

programas = [
    "familias_en_accion",
    "jovenes_en_accion",
    "ingreso_solidario",
    "devolucion_iva",
    "colombia_mayor",
    "transferencias_condicionadas",
    "renta_ciudadana",
    "renta_joven",
    "fondo_emprender",
    "iraca",
    "mi_negocio",
    "resa"
]

# =========================
# 🎛️ SIDEBAR – FILTROS
# =========================

st.sidebar.header("Filtros")

edad_min = int(base1["edad"].min())
edad_max = int(base1["edad"].max())

edad_f = st.sidebar.slider(
    "Rango de edad",
    edad_min,
    edad_max,
    (edad_min, edad_max)
)

sexo_f = st.sidebar.multiselect(
    "Género",
    options=sorted(base1["género"].dropna().unique()),
    default=sorted(base1["género"].dropna().unique())
)

doc_f = st.sidebar.multiselect(
    "Tipo de documento",
    options=sorted(base1["tipo_documento"].dropna().unique()),
    default=sorted(base1["tipo_documento"].dropna().unique())
)

# =========================
# 🔎 APLICAR FILTROS
# =========================

df1 = base1[
    (base1["edad"].between(edad_f[0], edad_f[1])) &
    (base1["género"].isin(sexo_f)) &
    (base1["tipo_documento"].isin(doc_f))
].copy()

# =========================
# 🗂️ TABS PRINCIPALES
# =========================

tab1, tab2, tab3 = st.tabs([
    "👥 Caracterización y Programas",
    "🔄 Evolución Documental",
    "📊 Evolución documental y acceso a programas"
])

# =========================================================
# 🟦 TAB 1 – CARACTERIZACIÓN Y PROGRAMAS
# =========================================================

with tab1:
    st.title("Caracterización de población migrante")

    # ---------- MÉTRICAS ----------
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Personas", df1["Bdua_Afl_id"].nunique())
    c2.metric("Edad promedio", round(df1["edad"].mean(), 1))
    c3.metric("Mujeres (%)", round((df1["género"] == "Femenino").mean() * 100, 1))
    c4.metric("Hombres (%)", round((df1["género"] == "Masculino").mean() * 100, 1))

    st.divider()

    # ---------- PROGRAMAS ----------
    st.subheader("Acceso a programas")

    prog_counts = (
        df1[programas]
        .apply(lambda x: (x == "Sí").sum())
        .reset_index()
    )
    prog_counts.columns = ["Programa", "Personas"]

    st.bar_chart(prog_counts.set_index("Programa"))

    st.divider()

    # ---------- PROGRAMAS POR EDAD ----------
    st.subheader("Acceso a programas por grupo etario")

    df1["grupo_edad"] = pd.cut(
        df1["edad"],
        bins=[0, 17, 29, 44, 59, 120],
        labels=["0–17", "18–29", "30–44", "45–59", "60+"]
    )

    prog_edad = (
        df1.groupby("grupo_edad")[programas]
        .apply(lambda d: (d == "Sí").sum())
        .reset_index()
    )

    st.dataframe(prog_edad, use_container_width=True)

    st.divider()

    # ---------- DOCUMENTO ----------
    st.subheader("Distribución por tipo de documento")

    doc_dist = (
        df1["tipo_documento"]
        .value_counts()
        .reset_index()
    )
    doc_dist.columns = ["Documento", "Personas"]

    st.bar_chart(doc_dist.set_index("Documento"))

# =========================================================
# 🟨 TAB 2 – EVOLUCIÓN DOCUMENTAL
# =========================================================

with tab2:
    st.title("Evolución del tipo de documento")

    # Personas con más de un tipo de documento
    cambios = (
        base2.groupby("afl_id")["TPS_IDN_ID"]
        .nunique()
        .reset_index(name="num_documentos")
    )

    total_cambios = (cambios["num_documentos"] > 1).sum()

    st.metric(
        "Personas con más de un tipo de documento",
        int(total_cambios)
    )

    st.divider()

    st.subheader("Distribución del número de documentos por persona")

    dist_docs = (
        cambios["num_documentos"]
        .value_counts()
        .sort_index()
        .reset_index()
    )
    dist_docs.columns = ["Número de documentos", "Personas"]

    st.bar_chart(dist_docs.set_index("Número de documentos"))

    st.divider()

    st.subheader("Vista de ejemplo (personas con cambios)")

    ejemplo = cambios[cambios["num_documentos"] > 1].head(20)
    st.dataframe(ejemplo, use_container_width=True)



with tab3:
    st.title("Evolución documental y acceso a programas")

    st.markdown(
        """
        Esta sección presenta patrones agregados de cambio documental 
        y su relación con el acceso a programas sociales.
        """
    )

    # ======================================================
    # 1️⃣ CAMBIO DE DOCUMENTO (BARRAS)
    # ======================================================

    st.subheader("1️⃣ Personas que cambiaron tipo de documento")

    doc_hist = (
        base2.sort_values(["afl_id", "HST_IDN_FECHA_INICIO"])
        .groupby("afl_id")["TPS_IDN_ID"]
        .agg(doc_inicial="first", doc_final="last")
        .reset_index()
    )

    doc_hist["cambio_doc"] = np.where(
        doc_hist["doc_inicial"] != doc_hist["doc_final"],
        "Cambió documento",
        "No cambió documento"
    )

    resumen_cambio = (
        doc_hist["cambio_doc"]
        .value_counts()
        .reset_index()
    )
    resumen_cambio.columns = ["Condición", "Personas"]

    fig1 = px.bar(
        resumen_cambio,
        x="Personas",
        y="Condición",
        orientation="h",
        text="Personas"
    )

    fig1.update_layout(height=400)
    st.plotly_chart(fig1, width="stretch")

    st.divider()

    # ======================================================
    # 2️⃣ DOCUMENTO INICIAL Y ACCESO A PROGRAMA
    # ======================================================

    st.subheader("2️⃣ Documento inicial y acceso a programa")

    programa_sel = st.selectbox("Selecciona un programa", programas)

    base1["Bdua_Afl_id"] = base1["Bdua_Afl_id"].astype(str)
    doc_hist["afl_id"] = doc_hist["afl_id"].astype(str)

    df_merge = base1.merge(
        doc_hist,
        left_on="Bdua_Afl_id",
        right_on="afl_id",
        how="inner"
    )

    df_merge["Acceso"] = df_merge[programa_sel].map(
        {"Sí": "Accedió", "No": "No accedió"}
    )

    resumen_doc_prog = (
        df_merge
        .groupby(["doc_inicial", "Acceso"])
        .size()
        .reset_index(name="Personas")
    )

    fig2 = px.bar(
        resumen_doc_prog,
        x="doc_inicial",
        y="Personas",
        color="Acceso",
        barmode="stack"
    )

    fig2.update_layout(
        xaxis_title="Documento inicial",
        yaxis_title="Número de personas",
        height=500
    )

    st.plotly_chart(fig2, width="stretch")

    st.divider()

    # ======================================================
    # 3️⃣ CAMBIO DOCUMENTAL Y PROBABILIDAD DE ACCESO
    # ======================================================

    st.subheader("3️⃣ Cambio documental y acceso al programa")

    resumen_cambio_prog = (
        df_merge
        .groupby(["cambio_doc", "Acceso"])
        .size()
        .reset_index(name="Personas")
    )

    # Calcular porcentajes
    resumen_cambio_prog["Total_grupo"] = (
        resumen_cambio_prog
        .groupby("cambio_doc")["Personas"]
        .transform("sum")
    )

    resumen_cambio_prog["Porcentaje"] = (
        resumen_cambio_prog["Personas"] /
        resumen_cambio_prog["Total_grupo"] * 100
    )

    fig3 = px.bar(
        resumen_cambio_prog,
        x="cambio_doc",
        y="Porcentaje",
        color="Acceso",
        barmode="stack",
        text=round(resumen_cambio_prog["Porcentaje"], 1)
    )

    fig3.update_layout(
        yaxis_title="Porcentaje dentro del grupo",
        xaxis_title="Condición documental",
        height=500
    )

    st.plotly_chart(fig3, width="stretch")

    st.divider()

    # ======================================================
    # 4️⃣ TABLA RESUMEN PARA ANÁLISIS
    # ======================================================

    st.subheader("4️⃣ Tabla resumen")

    tabla_final = (
        resumen_cambio_prog[[
            "cambio_doc",
            "Acceso",
            "Personas",
            "Porcentaje"
        ]]
        .sort_values(["cambio_doc", "Acceso"])
    )

    st.dataframe(tabla_final, use_container_width=True)



# =========================
# 🧠 FOOTER
# =========================

st.caption(
    "Atlas Migrantes – DNP | Prototipo analítico con datos administrativos"
)
