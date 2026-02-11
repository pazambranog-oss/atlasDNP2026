import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import geopandas as gpd

st.set_page_config(
    page_title="Atlas Migrantes – DNP",
    layout="wide"
)

# =========================
# 📥 CARGA DE DATOS
# =========================

@st.cache_data
def cargar_bases():
    ruta1 = "rsh_migrantes_limpio.parquet"
    ruta2 = "Evolucion_match.parquet"

    base1 = pd.read_parquet(ruta1)
    base2 = pd.read_parquet(ruta2)
    return base1, base2

base1, base2 = cargar_bases()

base1["Bdua_Afl_id"] = base1["Bdua_Afl_id"].astype(str)
base2["afl_id"] = base2["afl_id"].astype(str)

# =========================
# 🧹 LIMPIEZA
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
# 🎛️ FILTROS GENERALES
# =========================

st.sidebar.header("Filtros generales")

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
    sorted(base1["género"].dropna().unique()),
    default=sorted(base1["género"].dropna().unique())
)

doc_f = st.sidebar.multiselect(
    "Tipo de documento",
    sorted(base1["tipo_documento"].dropna().unique()),
    default=sorted(base1["tipo_documento"].dropna().unique())
)

# 🔥 NUEVO: opción Todos
opciones_programa = ["Todos los programas"] + programas

programa_f = st.sidebar.selectbox(
    "Programa",
    opciones_programa
)

# =========================
# 🔎 APLICAR FILTROS
# =========================

df_filtrado = base1[
    (base1["edad"].between(edad_f[0], edad_f[1])) &
    (base1["género"].isin(sexo_f)) &
    (base1["tipo_documento"].isin(doc_f))
].copy()

# =========================
# 🔁 LÓGICA PROGRAMA
# =========================

if programa_f == "Todos los programas":
    df_filtrado["acceso_programa"] = (
        df_filtrado[programas]
        .apply(lambda row: (row == "Sí").any(), axis=1)
    )
    columna_programa = "acceso_programa"
else:
    columna_programa = programa_f

# =========================
# 🗂️ TABS
# =========================

tab1, tab2, tab3, tab4 = st.tabs([
    "👥 Caracterización",
    "🔄 Evolución documental",
    "📊 Documento y programa",
    "📈 Cambio y acceso"
])

# =========================================================
# TAB 1
# =========================================================

with tab1:

    st.title("Caracterización general")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Personas", df_filtrado["Bdua_Afl_id"].nunique())
    c2.metric("Edad promedio", round(df_filtrado["edad"].mean(), 1))
    c3.metric("Mujeres (%)", round((df_filtrado["género"] == "Femenino").mean()*100,1))
    c4.metric("Hombres (%)", round((df_filtrado["género"] == "Masculino").mean()*100,1))

    st.divider()

    if programa_f == "Todos los programas":
        prog_count = df_filtrado["acceso_programa"].sum()
        st.metric("Personas con al menos un programa", prog_count)
    else:
        prog_count = (df_filtrado[columna_programa] == "Sí").sum()
        st.metric(f"Personas en {programa_f.replace('_',' ').title()}", prog_count)

    fig = px.histogram(df_filtrado, x="edad", nbins=30)
    st.plotly_chart(fig, width="stretch")

# =========================================================
# TAB 2
# =========================================================

with tab2:

    st.title("Evolución documental")

    cambios = (
        base2.groupby("afl_id")["TPS_IDN_ID"]
        .nunique()
        .reset_index(name="num_documentos")
    )

    st.metric(
        "Personas con más de un documento",
        int((cambios["num_documentos"] > 1).sum())
    )

    conteo_docs = (
        cambios["num_documentos"]
        .value_counts()
        .sort_index()
        .rename_axis("num_documentos")
        .reset_index(name="count")
    )

    fig = px.bar(
        conteo_docs,
        x="num_documentos",
        y="count"
    )

    st.plotly_chart(fig, width="stretch")

# =========================================================
# TAB 3
# =========================================================

with tab3:

    st.title("Documento y acceso")

    if programa_f == "Todos los programas":
        resumen = (
            df_filtrado
            .groupby("tipo_documento")["acceso_programa"]
            .sum()
            .reset_index(name="Personas")
        )
    else:
        resumen = (
            df_filtrado
            .groupby("tipo_documento")[columna_programa]
            .apply(lambda x: (x == "Sí").sum())
            .reset_index(name="Personas")
        )

    fig = px.bar(
        resumen,
        x="tipo_documento",
        y="Personas"
    )

    st.plotly_chart(fig, width="stretch")

# =========================================================
# TAB 4
# =========================================================

with tab4:

    st.title("Cambio documental y acceso")

    doc_hist = (
        base2.sort_values(["afl_id","HST_IDN_FECHA_INICIO"])
        .groupby("afl_id")["TPS_IDN_ID"]
        .agg(doc_inicial="first", doc_final="last")
        .reset_index()
    )

    doc_hist["cambio_doc"] = np.where(
        doc_hist["doc_inicial"] != doc_hist["doc_final"],
        "Cambió documento",
        "No cambió documento"
    )

    df_merge = df_filtrado.merge(
        doc_hist,
        left_on="Bdua_Afl_id",
        right_on="afl_id",
        how="inner"
    )

    if programa_f == "Todos los programas":
        df_merge["Acceso"] = np.where(
            df_merge["acceso_programa"],
            "Accedió",
            "No accedió"
        )
    else:
        df_merge["Acceso"] = df_merge[columna_programa].map(
            {"Sí": "Accedió", "No": "No accedió"}
        )

    resumen = (
        df_merge.groupby(["cambio_doc","Acceso"])
        .size()
        .reset_index(name="Personas")
    )

    fig = px.bar(
        resumen,
        x="cambio_doc",
        y="Personas",
        color="Acceso",
        barmode="stack"
    )

    st.plotly_chart(fig, width="stretch")

st.caption("Atlas Migrantes – DNP | Prototipo analítico")
