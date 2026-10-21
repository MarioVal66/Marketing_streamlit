import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import streamlit as st

# Configuración de la página
st.set_page_config(
    page_title="Analytics de Retención Educativa",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 Portal de Analytics: Retención de Alumnos y Riesgo de Abandono")

# Definición de las 6 pestañas
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1. Entender los datos",
    "2. Limpiar los datos",
    "3. Estadística descriptiva",
    "4. Estadística predictiva",
    "5. Estadística prescriptiva",
    "6. Toma de decisiones",
])

# Carga de datos
df_raw = pd.read_csv("data/dataset_clientes_sinteticos_segmentos_marketing.csv")


# -------------------------------
# PESTAÑA 1: ENTENDER LOS DATOS
# -------------------------------
with tab1:
    st.header("1. Entender los datos")
    st.markdown(
        "**Pregunta guía:** ¿Quiénes son tus alumnos y qué información tienes de"
        " ellos?"
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Número de casos (Alumnos)", len(df_raw))
    col2.metric("Número de variables", len(df_raw.columns))
    col3.metric("Unidad de análisis", "Alumno registrado")

    st.subheader("Vista previa del dataset")
    st.dataframe(df_raw.head(10), use_container_width=True)

    st.subheader("Variables disponibles y tipos de datos")
    df_tipos = pd.DataFrame({
        "Variable": df_raw.columns,
        "Tipo de dato": [str(t) for t in df_raw.dtypes],
        "Ejemplo": [str(df_raw[col].dropna().iloc[0]) for col in df_raw.columns],
    })
    st.table(df_tipos)


# -------------------------------
# PESTAÑA 2: LIMPIAR LOS DATOS
# -------------------------------
with tab2:
    st.header("2. Limpiar los datos")
    st.markdown("**Pregunta guía:** ¿Qué tan confiables son los datos?")

    col_dup, col_null = st.columns(2)

    with col_dup:
        duplicados = df_raw.duplicated().sum()
        st.metric("Registros duplicados", duplicados)

    with col_null:
        nulos_totales = df_raw.isnull().sum().sum()
        st.metric("Valores nulos totales", nulos_totales)

    st.subheader("Detalle de nulos por columna")
    st.dataframe(df_raw.isnull().sum(), use_container_width=True)

    st.subheader("Decisión y estrategia de limpieza")
    st.markdown("""
    - **Valores nulos numéricos:** Imputados con la **mediana** para evitar distorsiones por valores extremos.
    - **Valores nulos categóricos:** Imputados con la **moda** (categoría más frecuente).
    - **Duplicados:** Eliminación automática de registros repetidos.
    """)

    df = df_raw.copy().drop_duplicates()

    num_cols = df.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())

    cat_cols = df.select_dtypes(include=["object"]).columns
    for c in cat_cols:
        df[c] = df[c].fillna(df[c].mode()[0])

    st.success("✅ Datos limpiados correctamente.")


# -------------------------------
# PESTAÑA 3: ESTADÍSTICA DESCRIPTIVA
# -------------------------------
with tab3:
    st.header("3. Estadística descriptiva")
    st.markdown(
        "**Pregunta guía:** ¿Cómo se comportan en el journey, qué segmentos"
        " existen y qué KPIs los describen mejor?"
    )

    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Promedio General", f"{df['promedio_acumulado'].mean():.2f}")
    kpi2.metric(
        "Asistencia Promedio", f"{df['porcentaje_asistencia'].mean():.1f}%"
    )
    kpi3.metric(
        "Tasa de Riesgo de Abandono",
        f"{(df['riesgo_abandono'].mean()*100):.1f}%",
    )

    st.subheader("Comportamiento por Segmento")
    var_sel = st.selectbox(
        "Selecciona una variable categórica para analizar la distribución:",
        ["modalidad", "tipo_beca", "nivel_socioeconomico"],
    )

    st.bar_chart(df[var_sel].value_counts())


# -------------------------------
# PESTAÑA 4: ESTADÍSTICA PREDICTIVA
# -------------------------------
with tab4:
    st.header("4. Estadística predictiva")
    st.markdown(
        "**Pregunta guía:** ¿Qué características tienen quienes están en riesgo"
        " de abandonar y podemos predecirlo?"
    )

    features = [
        "promedio_acumulado",
        "porcentaje_asistencia",
        "materias_reprobadas",
        "entregas_a_tiempo",
    ]
    X = df[features]
    y = df["riesgo_abandono"]

    modelo = LogisticRegression()
    modelo.fit(X, y)

    df["prob_abandono"] = modelo.predict_proba(X)[:, 1]

    st.subheader("Pesos del Modelo (Importancia de Variables)")
    pesos_df = pd.DataFrame({
        "Variable Predictora": features,
        "Peso (Coeficiente)": modelo.coef_[0],
    }).sort_values(by="Peso (Coeficiente)", ascending=False)

    st.dataframe(pesos_df, use_container_width=True)
    st.info(
        "💡 **Interpretación:** Coeficientes positivos altos (como"
        " *materias_reprobadas*) aumentan el riesgo de deserción; coeficientes"
        " negativos altos (como *promedio* o *asistencia*) reducen el riesgo."
    )


# -------------------------------
# PESTAÑA 5: ESTADÍSTICA PRESCRIPTIVA
# -------------------------------
with tab5:
    st.header("5. Estadística prescriptiva")
    st.markdown(
        "**Pregunta guía:** ¿Qué acción deberíamos tomar con cada caso?"
    )

    def asignar_segmento(prob):
        if prob >= 0.60:
            return "Alto Riesgo (Atención Urgente)"
        elif prob >= 0.30:
            return "Riesgo Medio (Seguimiento)"
        else:
            return "Bajo Riesgo (Monitoreo Regular)"

    df["segmento_prescriptivo"] = df["prob_abandono"].apply(asignar_segmento)

    st.subheader("Matriz de Acciones Recomendadas")
    matriz_acciones = pd.DataFrame({
        "Nivel de Riesgo Estimado": [
            "Alto Riesgo (>= 60%)",
            "Riesgo Medio (30% - 59%)",
            "Bajo Riesgo (< 30%)",
        ],
        "Acción Recomendada": [
            "Asignación prioritaria de tutoría personalizada y beca de rescate",
            "Envío de alertas preventivas e invitación a talleres de regularización",
            "Mantener comunicación estándar y reconocimientos de avance",
        ],
    })
    st.table(matriz_acciones)

    st.subheader("Alumnos categorizados por nivel de probabilidad")
    st.dataframe(
        df[[
            "id_alumno",
            "promedio_acumulado",
            "prob_abandono",
            "segmento_prescriptivo",
        ]].head(10),
        use_container_width=True,
    )


# -------------------------------
# PESTAÑA 6: TOMA DE DECISIONES
# -------------------------------
with tab6:
    st.header("6. Toma de decisiones")
    st.markdown(
        "**Pregunta guía:** ¿Qué mensaje y estrategia deberíamos mostrarle a"
        " cada tipo de caso?"
    )

    opcion = st.selectbox(
        "Selecciona el segmento de alumnos:",
        [
            "Alto Riesgo (Atención Urgente)",
            "Riesgo Medio (Seguimiento)",
            "Bajo Riesgo (Monitoreo Regular)",
        ],
    )

    if opcion == "Alto Riesgo (Atención Urgente)":
        st.error("🚨 **Estrategia: Intervención Directa e Inmediata**")
        st.markdown(
            "**Mensaje clave para el alumno:** *'Hola, detectamos que estás"
            " teniendo dificultades en algunas materias. Queremos ofrecerte"
            " una tutoría 1 a 1 y asesoría para un apoyo económico este"
            " semestre. Agenda tu cita aquí.'*"
        )
        st.markdown(
            "**Justificación basada en datos:** Tienen una probabilidad"
            " estimada de deserción mayor al 60%. Una intervención rápida evita"
            " la baja definitiva de la institución."
        )

    elif opcion == "Riesgo Medio (Seguimiento)":
        st.warning("⚠️ **Estrategia: Nutrición y Prevención**")
        st.markdown(
            "**Mensaje clave para el alumno:** *'¡Aún estás a tiempo de"
            " mejorar tu promedio! Únete a nuestros grupos de estudio e"
            " inscríbete a los talleres de regularización antes de exámenes'*"
        )
        st.markdown(
            "**Justificación basada en datos:** Tienen entre 30% y 59% de riesgo"
            " estimado. Con apoyos ligeros de asesoría preventiva evitamos que"
            " caigan en el segmento de alto riesgo."
        )

    else:
        st.success("✅ **Estrategia: Reconocimiento y Fidelización**")
        st.markdown(
            "**Mensaje clave para el alumno:** *'¡Excelente trabajo este"
            " ciclo! Consulta los talleres extracurriculares y conferencias"
            " disponibles para seguir fortaleciendo tu perfil.'*"
        )
        st.markdown(
            "**Justificación basada en datos:** Alumnos con riesgo menor a"
            " 30%. No requieren intervención de asesores humanos, lo que optimiza"
            " el presupuesto del área educativa."
        )