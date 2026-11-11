"""Aplicación Interactiva en Streamlit – Proyecto Final de Modelación (Laboratorio 3).

Especialización en Analítica Estratégica de Datos · Estadística para Analítica de Datos · UPTC
Profesor: Duván Cataño
Integrantes: Juan Beltrán · Daniel Castellanos · Daniel Rico

Cumple los 4 requisitos del Laboratorio 3:
1. Descripción del proyecto, diccionario de variables, limpieza e ingeniería de características.
2. Visualizaciones interactivas (EDA univariado, bivariado y multivariado con Plotly y Seaborn).
3. Entradas y filtros dinámicos del usuario (segmentación poblacional en tiempo real y controles).
4. Predicciones en vivo usando los pipelines serializados (`.joblib`) de Regresión Lineal y Regresión Logística.

Ejecución desde la terminal:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import datos, estadistica as E, graficos as G, modelos as M


def _asegurar_modulos_actualizados() -> None:
    """Recarga los submódulos de `src` si cambiaron en disco (evita caché obsoleto en Streamlit Cloud)."""
    for mod in (datos, E, M, G):
        ruta = getattr(mod, "__file__", None)
        if not ruta:
            continue
        mtime = Path(ruta).stat().st_mtime
        if getattr(mod, "_loaded_mtime", None) != mtime:
            importlib.reload(mod)
            mod._loaded_mtime = Path(ruta).stat().st_mtime


_asegurar_modulos_actualizados()

# ===========================================================================
# CONFIGURACIÓN DE LA PÁGINA Y ESTILOS
# ===========================================================================
st.set_page_config(
    page_title="Saber 11 · Proyecto Final Modelación (Lab 3)",
    page_icon="🎓",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main-banner {
        background: linear-gradient(90deg, #222222 0%, #3a3a3a 100%);
        color: #FFFFFF;
        padding: 1.2rem 1.8rem;
        border-radius: 10px;
        border-left: 8px solid #FFC000;
        margin-bottom: 1.2rem;
    }
    .main-banner h1 {
        color: #FFC000;
        margin: 0;
        font-size: 2.0rem;
        font-weight: 800;
    }
    .main-banner p {
        color: #F0F0F0;
        margin: 0.35rem 0 0 0;
        font-size: 0.98rem;
    }
    .pred-card {
        background-color: #FFFDF2;
        border: 2px solid #FFC000;
        border-radius: 10px;
        padding: 1.1rem;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ===========================================================================
# FUNCIONES CACHEADAS DE CARGA
# ===========================================================================
@st.cache_data(show_spinner="Cargando y depurando la base Saber 11 (periodo 20224)…")
def cargar_base():
    crudo = datos.cargar_muestra_cruda()
    df, rep = datos.depurar(crudo)
    mecanismo_na = E.evaluar_mecanismo_faltantes(crudo)
    return df, rep, mecanismo_na


@st.cache_resource(show_spinner="Cargando pipelines predictivos (.joblib)…")
def obtener_modelos(sklearn_version: str = M.sklearn.__version__):
    _ = sklearn_version
    return M.cargar_modelos_guardados()


def mostrar_mpl(fig):
    st.pyplot(fig)
    plt.close(fig)


# ===========================================================================
# ENCABEZADO PRINCIPAL
# ===========================================================================
st.markdown(
    """
    <div class="main-banner">
        <h1>🎓 Analítica y Modelación Predictiva del Desempeño en Saber 11 (ICFES)</h1>
        <p><b>Especialización en Analítica Estratégica de Datos · Estadística para Analítica de Datos · UPTC (2026-2)</b><br>
        Profesor: Duván Cataño · Integrantes: Juan Beltrán · Daniel Castellanos · Daniel Rico · Laboratorio # 3</p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    df_full, rep, tabla_mnar = cargar_base()
    artefactos = obtener_modelos()
    resumen = artefactos["resumen"]
except Exception as exc:  # noqa: BLE001
    st.error(f"Error al cargar datos o modelos: {exc}")
    st.stop()


# ===========================================================================
# BARRA LATERAL: NAVEGACIÓN Y FILTROS GLOBALES DEL USUARIO
# ===========================================================================
st.sidebar.markdown("## 🧭 Módulo del Proyecto")
modulo = st.sidebar.radio(
    "Seleccione una vista:",
    [
        "1 · Descripción y Preprocesamiento",
        "2 · EDA Interactivo con Filtros",
        "3 · Pruebas de Hipótesis e Inferencia",
        "4 · Pipelines y Regresión Lineal",
        "5 · Clasificación: Regresión Logística",
        "6 · 🔮 Simulador y Predicción en Vivo",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown("## 🔎 Filtros Globales de Población")
st.sidebar.caption("Afectan en tiempo real las métricas y visualizaciones de los módulos 1, 2 y 3.")

f_nat = st.sidebar.multiselect(
    "Naturaleza del colegio",
    options=sorted(df_full["cole_naturaleza"].unique()),
    default=sorted(df_full["cole_naturaleza"].unique()),
)
f_zona = st.sidebar.multiselect(
    "Zona de ubicación",
    options=sorted(df_full["cole_area_ubicacion"].unique()),
    default=sorted(df_full["cole_area_ubicacion"].unique()),
)
f_int = st.sidebar.multiselect(
    "Internet en el hogar",
    options=["Si", "No", "Desconocido"],
    default=["Si", "No", "Desconocido"],
)
f_est = st.sidebar.multiselect(
    "Estrato de la vivienda",
    options=datos.orden_categorias("fami_estratovivienda", df_full["fami_estratovivienda"].unique()),
    default=datos.orden_categorias("fami_estratovivienda", df_full["fami_estratovivienda"].unique()),
)
rango_pg = st.sidebar.slider(
    "Rango de Puntaje Global",
    min_value=int(df_full["punt_global"].min()),
    max_value=int(df_full["punt_global"].max()),
    value=(int(df_full["punt_global"].min()), int(df_full["punt_global"].max())),
)

df = df_full[
    df_full["cole_naturaleza"].isin(f_nat)
    & df_full["cole_area_ubicacion"].isin(f_zona)
    & df_full["fami_tieneinternet"].isin(f_int)
    & df_full["fami_estratovivienda"].isin(f_est)
    & df_full["punt_global"].between(rango_pg[0], rango_pg[1])
].copy()

st.sidebar.markdown("---")
st.sidebar.metric("Muestra Filtrada Activa", f"{len(df):,} est.", f"{len(df)/len(df_full)*100:.2f} % del total")

if len(df) < 30:
    st.warning("El filtro seleccionado contiene menos de 30 registros. Amplíe los criterios en la barra lateral.")
    st.stop()


# ===========================================================================
# MÓDULO 1: DESCRIPCIÓN DEL PROYECTO, VARIABLES Y PREPROCESAMIENTO
# ===========================================================================
if modulo.startswith("1"):
    st.subheader("1. Descripción General del Proyecto y Continuidad (Laboratorios 1 → 2 → 3)")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Estudiantes Únicos", f"{rep['n_final']:,}", f"-{rep['n_duplicados']:,} duplicados")
    c2.metric("Media Puntaje Global", f"{df['punt_global'].mean():.2f} pts", f"Mediana: {df['punt_global'].median():.1f}")
    c3.metric("Desviación Estándar", f"{df['punt_global'].std():.2f} pts", f"CV: {df['punt_global'].std()/df['punt_global'].mean()*100:.2f} %")
    c4.metric("Superan Promedio (≥250)", f"{df['supera_promedio'].mean()*100:.2f} %", "Objetivo Logística")
    c5.metric("Sin Cuestionario (MNAR)", f"{rep['n_sin_cuestionario']:,}", f"{rep['pct_sin_cuestionario']:.2f} %")

    st.markdown(
        """
        ### 🎯 Objetivo del Proyecto Aplicado
        Modelar y cuantificar el efecto de las **condiciones socioeconómicas, familiares e institucionales** sobre el desempeño académico en el examen de Estado **Saber 11 (ICFES, periodo 20224)** mediante un flujo reproducible de extremo a extremo que articula:
        - **Laboratorio 1 (Definición y Calidad):** Extracción vía API Socrata (`kgxf-xxbe`, $n=80.000$, semilla 42), detección de $2.173$ duplicados operacionales por `estu_consecutivo`, identificación del mecanismo **MNAR (*Missing Not At Random*)** en el cuestionario socioeconómico y prevención de **fuga de información (*data leakage*)** excluyendo los puntajes de las 5 pruebas como predictores directos en el modelo socioeconómico.
        - **Laboratorio 2 (EDA e Inferencia No Paramétrica/Paramétrica):** Caracterización de forma (asimetría y curtosis), robustez de Spearman/Kendall frente a atípicos, dependencia categórica ($\\chi^2$, $V$ de Cramér, Información Mutua) y pruebas de contraste de grupos (ANOVA de Welch y Kruskal-Wallis), demostrando que `fami_educacionmadre` ($\\varepsilon^2 = 0.1275$) y `fami_tieneinternet` ($\\varepsilon^2 = 0.0795$) lideran la explicación de las brechas.
        - **Laboratorio 3 (Pipelines, Escalamiento y Modelación Predictiva):** Construcción de arquitecturas `Pipeline` + `ColumnTransformer` (`SimpleImputer`, `OneHotEncoder(drop='first')` y `StandardScaler`), entrenamiento y validación cruzada ($k=10$) de modelos de **Regresión Lineal** (predicción continua de `punt_global` + diagnóstico MCO en `statsmodels`) y **Regresión Logística** (clasificación binaria de `supera_promedio` $\\ge 250$ con `RandomizedSearchCV`).
        """
    )

    tab_dic, tab_na, tab_imp = st.tabs([
        "📋 Diccionario e Ingeniería de Variables",
        "🔍 Diagnóstico MNAR de Valores Faltantes",
        "🧪 Experimento Comparativo de Imputación Numérica",
    ])

    with tab_dic:
        t_tipos = pd.DataFrame(datos.TIPOS, index=["Tipología Estadística", "Significado y Escala", "Rol en el Proyecto"]).T
        t_tipos["Valores Distintos"] = [df_full[c].nunique() if c in df_full.columns else "-" for c in t_tipos.index]
        st.dataframe(t_tipos, use_container_width=True)

    with tab_na:
        st.markdown(
            """
            **Evidencia empírica del mecanismo MNAR (*Missing Not At Random*):**
            Quienes omiten las preguntas de estrato, educación de la madre o internet obtienen en promedio **~22,4 puntos menos** en `punt_global` ($p < 0.0001$ tanto en $t$ de Welch como en Mann-Whitney). Eliminar esas filas (*listwise deletion*) o imputarlas con la moda sesgaría el modelo hacia arriba; por ello se preserva la señal con la categoría explícita `'Desconocido'` y las indicadoras de categorías especiales (`es_sin_estrato`, `educ_no_sabe`).
            """
        )
        st.dataframe(E.formatear_pvalores(tabla_mnar).style.format(precision=4), use_container_width=True)

    with tab_imp:
        st.markdown(
            """
            Para variables cuantitativas continuas (como `punt_ingles`, que presenta 282 valores faltantes en los estudiantes únicos), se comparan cuatro técnicas de imputación del curso (**Media**, **Mediana**, **KNN ($k=5$)** e **Iterativa tipo MICE**) simulando un 20 % de datos faltantes sobre una submuestra de $n=2.000$ registros:
            """
        )
        res_imp = E.comparar_imputacion_numerica(df_full, n_muestra=2000)
        c_imp1, c_imp2 = st.columns([1, 2])
        with c_imp1:
            st.dataframe(res_imp["resultados"].style.format(precision=4), use_container_width=True)
        with c_imp2:
            mostrar_mpl(G.fig_imputacion_comparativa(res_imp, "punt_ingles"))


# ===========================================================================
# MÓDULO 2: EDA INTERACTIVO CON FILTROS
# ===========================================================================
elif modulo.startswith("2"):
    st.subheader("2. Análisis Exploratorio de Datos (EDA) Interactivo")
    t_uni, t_cual, t_corr = st.tabs([
        "📈 Variables Cuantitativas y Atípicos (IQR)",
        "📊 Variables Cualitativas y Brechas por Grupo",
        "🔗 Correlación Lineal y No Paramétrica",
    ])

    with t_uni:
        c_v1, c_v2 = st.columns([2, 1])
        var_num = c_v1.selectbox("Seleccione variable cuantitativa:", datos.COLS_CUANT, index=0)
        color_by = c_v2.selectbox(
            "Segmentar histograma por:",
            [None, "cole_naturaleza", "cole_area_ubicacion", "fami_tieneinternet", "estu_genero"],
            index=1,
        )
        st.plotly_chart(G.plotly_distribucion_interactiva(df, var_num, color_by), use_container_width=True)

        desc_df = E.descriptivas(df, datos.COLS_CUANT)
        st.markdown("#### Tabla de Medidas Estructurales, Posición, Forma y Atípicos (4 cifras decimales)")
        st.dataframe(
            desc_df.drop(columns=["forma_asimetria", "forma_curtosis"]).T.style.format(precision=4),
            use_container_width=True,
        )
        st.markdown(E.interpretar_univariada(desc_df))

    with t_cual:
        c_q1, c_q2 = st.columns(2)
        var_cat = c_q1.selectbox("Variable cualitativa:", datos.COLS_CUAL, index=2)
        var_y = c_q2.selectbox("Variable de desempeño (eje X):", datos.COLS_CUANT, index=0)
        st.plotly_chart(G.plotly_boxplot_interactivo(df, var_cat, var_y), use_container_width=True)

        c_f1, c_f2 = st.columns([1.1, 1.3])
        with c_f1:
            st.markdown(f"**Tabla de frecuencias de `{var_cat}`**")
            st.dataframe(E.tabla_frecuencias_cualitativa(df, var_cat).style.format(precision=4), use_container_width=True)
        with c_f2:
            mostrar_mpl(G.barras(df, var_cat))

    with t_corr:
        c_x, c_y = st.columns(2)
        vx = c_x.selectbox("Variable X:", datos.COLS_CUANT, index=5)
        vy = c_y.selectbox("Variable Y:", [c for c in datos.COLS_CUANT if c != vx], index=0)
        r_par = E.correlacion_par(df, vx, vy)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Covarianza", f"{r_par['covarianza']:.4f}")
        m2.metric("Pearson (r)", f"{r_par['pearson']:.4f}")
        m3.metric("Spearman (ρ)", f"{r_par['spearman']:.4f}")
        m4.metric("Kendall (τ)", f"{r_par['kendall']:.4f}")
        mostrar_mpl(G.scatter_par(df, vx, vy))
        st.markdown(E.interpretar_correlacion(r_par))


# ===========================================================================
# MÓDULO 3: PRUEBAS DE HIPÓTESIS E INTERPRETACIÓN
# ===========================================================================
elif modulo.startswith("3"):
    st.subheader("3. Pruebas de Hipótesis Estadísticas e Interpretación")
    t_mix, t_chi, t_rank = st.tabs([
        "⚖️ ANOVA / Welch / Kruskal-Wallis (Cualitativa vs Cuantitativa)",
        "🎲 Independencia Chi-cuadrado (χ²) e Información Mutua",
        "🏆 Síntesis y Ranking de Predictores",
    ])

    with t_mix:
        cm1, cm2 = st.columns(2)
        cat_h = cm1.selectbox("Factor Categórico (Grupos):", datos.COLS_CUAL, index=2)
        y_h = cm2.selectbox("Variable Respuesta Continua:", datos.COLS_CUANT, index=0)
        grupos_activos = E._grupos(df, cat_h, y_h)
        if len(grupos_activos) < 2:
            st.warning(f"Con los filtros actuales de la barra lateral solo queda {len(grupos_activos)} grupo válido en `{cat_h}`. Amplíe los filtros para comparar grupos.")
        else:
            res_m = E.analisis_mixto(df, cat_h, y_h)
            st.markdown("#### Resultados de las Pruebas de Hipótesis de Homocedasticidad y Comparación de Grupos")
            st.dataframe(E.formatear_pvalores(E.tabla_pruebas_mixtas(res_m)).style.format(precision=4), use_container_width=True)

            st.markdown("#### Tamaños del Efecto: $\\eta^2$ (Eta cuadrado), $\\omega^2$ (Omega cuadrado) y $\\varepsilon^2$ (Épsilon cuadrado)")
            st.dataframe(
                E.tabla_tamanos_efecto(res_m).style.format({
                    "Valor (0–1)": "{:.4f}",
                    "Porcentaje Explicado (%)": "{:.2f} %",
                }),
                use_container_width=True,
            )
            ef_min = min(res_m["eta2"], res_m["omega2"], res_m["epsilon2"]) * 100.0
            ef_max = max(res_m["eta2"], res_m["omega2"], res_m["epsilon2"]) * 100.0
            st.info(
                f"**¿Qué significan estos porcentajes?** Con $N = {res_m['N']:,}$ estudiantes, el p-valor (`< 0.0001`) "
                f"solo confirma que existe diferencia estadística, pero el **tamaño del efecto** mide **qué tan grande e importante** "
                f"es esa diferencia (de $0$ a $1$, o de $0\\%$ a $100\\%$):\n"
                f"- Para **`{cat_h}`**, entre un **{ef_min:.2f} %** y un **{ef_max:.2f} %** de toda la variabilidad en **`{y_h}`** "
                f"se explica únicamente por pertenecer a distintos grupos de esta variable; el porcentaje restante proviene de otros "
                f"factores (colegio, internet, preparación individual, etc.).\n"
                f"- **Escala de referencia (Cohen):** $< 1\\%$ despreciable · $1\\%–6\\%$ pequeño · $6\\%–14\\%$ mediano · $\\ge 14\\%$ grande."
            )

            st.markdown(E.interpretar_mixto(res_m))
            with st.expander("Ver estadísticas y pruebas de normalidad por grupo + Gráficos Q-Q"):
                st.dataframe(E.formatear_pvalores(res_m["tabla_grupos"]).style.format(precision=4), use_container_width=True)
                mostrar_mpl(G.qq_grupos(res_m))

    with t_chi:
        cc1, cc2 = st.columns(2)
        ca = cc1.selectbox("Variable Categórica A:", datos.COLS_CUAL, index=1)
        cb = cc2.selectbox("Variable Categórica B:", [c for c in datos.COLS_CUAL if c != ca], index=1)
        res_c = E.independencia_categoricas(df, ca, cb, n_perm=100)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Estadístico χ²", f"{res_c['chi2']:.4f}")
        k2.metric("p-valor χ²", E.fmt_p(res_c["p_chi2"]))
        k3.metric("V de Cramér (corregida)", f"{res_c['cramer_v_corregido']:.4f}")
        k4.metric("Información Mutua Norm.", f"{res_c['MI_normalizada']:.4f}")
        st.markdown(E.interpretar_independencia(res_c))
        mostrar_mpl(G.heatmap_residuos(res_c))

    with t_rank:
        rank_c = E.ranking_asociacion(df, "punt_global")
        st.dataframe(E.formatear_pvalores(rank_c).style.format(precision=4), use_container_width=True)
        mostrar_mpl(G.barras_ranking(rank_c, "epsilon2", "Tamaño de Efecto No Paramétrico (ε² Kruskal-Wallis) sobre punt_global"))


# ===========================================================================
# MÓDULO 4: PIPELINES, ESCALAMIENTO Y REGRESIÓN LINEAL
# ===========================================================================
elif modulo.startswith("4"):
    st.subheader("4. Construcción de Pipelines, Escalamiento y Modelo de Regresión Lineal")

    enfoque_reg = st.radio(
        "Seleccione la especificación del Pipeline de Regresión Lineal:",
        [
            "Modelo A · Contexto Socioeconómico Puro (Principal — Cero Data Leakage)",
            "Modelo B · Contexto Socioeconómico + Competencia Transversal Inglés (`punt_ingles`)",
        ],
        horizontal=True,
    )
    clave_reg = "regresion_socio" if enfoque_reg.startswith("Modelo A") else "regresion_acad"
    info_reg = resumen[clave_reg]
    mt = info_reg["metricas_test"]

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("MAE (Test)", f"{mt['MAE']:.4f} pts")
    m2.metric("RMSE (Test)", f"{mt['RMSE']:.4f} pts")
    m3.metric("MAPE (Test)", f"{mt['MAPE']:.4f}", f"{mt['MAPE (%)']:.2f} %")
    m4.metric("R² (Test)", f"{mt['R²']:.4f}")
    m5.metric("R² Ajustado", f"{mt['R² Ajustado']:.4f}")

    t_pipe, t_eval, t_ols = st.tabs([
        "⚙️ Arquitectura del Pipeline y Escalamiento",
        "📊 Evaluación en Test, Validación Cruzada (k=10) e Importancia",
        "📐 Inferencia Estadística MCO (Statsmodels, VIF, Jarque-Bera, Breusch-Pagan)",
    ])

    with t_pipe:
        st.markdown(
            """
            ### Estructura Reproducible del `Pipeline` y `ColumnTransformer`
            Siguiendo las mejores prácticas, el preprocesamiento y el estimador se encapsulan en un único objeto `Pipeline` para impedir cualquier fuga de información (*data leakage*) entre entrenamiento y prueba o durante los 10 pliegues de validación cruzada:
            1. **Rama Categórica (`cat_pipeline`):** `SimpleImputer(strategy='constant', fill_value='Desconocido')` → `OneHotEncoder(drop='first', handle_unknown='ignore')` (codificación *Dummy* $k-1$ que evita la trampa de multicolinealidad perfecta).
            2. **Rama Numérica (`num_pipeline`):** `SimpleImputer(strategy='median')` → `StandardScaler()` ($z = \\frac{x - \\mu}{\\sigma}$, centrando cada variable en media 0 y desviación estándar 1 para hacer directamente comparables sus coeficientes parciales estandarizados).
            """
        )
        dem_esc = E.demostracion_escalamiento_distancias(df_full)
        c_e1, c_e2 = st.columns(2)
        with c_e1:
            st.markdown("**Perfiles de prueba en escalas distintas (`educ_madre_nivel` 0–9 vs `punt_ingles` 0–100):**")
            st.dataframe(dem_esc["perfiles"].style.format(precision=2), use_container_width=True)
        with c_e2:
            st.markdown("**Efecto del escalamiento sobre la Distancia Euclidiana $d(P_i, P_j)$:**")
            st.dataframe(dem_esc["distancias"].style.format(precision=4), use_container_width=True)

    with t_eval:
        c_ev1, c_ev2 = st.columns(2)
        with c_ev1:
            st.markdown("#### Comparación Entrenamiento (80 %) vs Prueba (20 %)")
            df_m = pd.DataFrame({
                "Entrenamiento (80 %)": info_reg["metricas_train"],
                "Prueba / Test (20 %)": info_reg["metricas_test"],
            })
            st.dataframe(df_m.style.format(precision=4), use_container_width=True)
        with c_ev2:
            st.markdown("#### Validación Cruzada ($k = 10$ folds)")
            df_cv = pd.DataFrame(info_reg["tabla_cv"]).set_index("Métrica")
            st.dataframe(df_cv.style.format(precision=4), use_container_width=True)

        imp_df = pd.DataFrame(info_reg["importance"])
        st.plotly_chart(
            G.plotly_importancia_coeficientes(
                imp_df,
                "Coeficiente",
                f"Importancia de Variables mediante Coeficientes — {enfoque_reg.split('·')[1].strip()}",
            ),
            use_container_width=True,
        )

    with t_ols:
        st.markdown("#### Tabla de Inferencia Econométrica MCO Clásico vs Errores Estándar Robustos (HC3)")
        df_ols = pd.DataFrame(info_reg["ols_params"]).set_index("Variable")
        st.dataframe(E.formatear_pvalores(df_ols).style.format(precision=4), use_container_width=True)

        c_d1, c_d2, c_d3 = st.columns(3)
        with c_d1:
            st.markdown("**Factor de Inflación de la Varianza (VIF)**")
            df_vif = pd.DataFrame(info_reg["vif"]).set_index("Variable")
            st.dataframe(df_vif.style.format({"VIF": "{:.4f}"}), use_container_width=True)
            st.success(f"VIF máximo = {df_vif['VIF'].max():.4f} (< 5): ausencia total de multicolinealidad severa.")
        with c_d2:
            jb = info_reg["jarque_bera"]
            st.markdown("**Prueba de Normalidad de Jarque-Bera**")
            st.write(f"- Estadístico JB: `{jb['JB_stat']:.4f}`")
            st.write(f"- p-valor: `{E.fmt_p(jb['p_valor'])}`")
            st.write(f"- Asimetría residual: `{jb['asimetria']:.4f}`")
            st.write(f"- Exceso de curtosis: `{jb['exceso_curtosis']:.4f}`")
            st.caption("Con n > 60.000, JB rechaza por desviaciones mínimas, pero la asimetría cercana a 0 y el TCL garantizan validez asintótica.")
        with c_d3:
            bp = info_reg["breusch_pagan"]
            st.markdown("**Prueba de Homocedasticidad de Breusch-Pagan**")
            st.write(f"- Estadístico LM: `{bp['LM_stat']:.4f}`")
            st.write(f"- p-valor LM: `{E.fmt_p(bp['p_valor_LM'])}`")
            st.write(f"- Estadístico F: `{bp['F_stat']:.4f}`")
            st.write(f"- p-valor F: `{E.fmt_p(bp['p_valor_F'])}`")
            st.caption("Al rechazarse H₀ (heterocedasticidad), se reportan errores estándar robustos de White-MacKinnon (HC3).")


# ===========================================================================
# MÓDULO 5: CLASIFICACIÓN CON REGRESIÓN LOGÍSTICA
# ===========================================================================
elif modulo.startswith("5"):
    st.subheader("5. Modelo de Clasificación Binaria: Regresión Logística")
    st.markdown(
        """
        **Definición de la variable objetivo binaria (`supera_promedio`):**
        $$Y = \\begin{cases} 1, & \\text{si } \\text{punt\\_global} \\ge 250 \\text{ (Supera el umbral nacional de referencia)} \\\\ 0, & \\text{si } \\text{punt\\_global} < 250 \\text{ (Bajo el umbral nacional)} \\end{cases}$$
        """
    )

    enfoque_log = st.radio(
        "Seleccione la especificación del Pipeline de Regresión Logística:",
        [
            "Modelo A · Contexto Socioeconómico Puro (Principal — Cero Data Leakage)",
            "Modelo B · Contexto Socioeconómico + Competencia Transversal Inglés (`punt_ingles`)",
        ],
        horizontal=True,
    )
    clave_log = "logistica_socio" if enfoque_log.startswith("Modelo A") else "logistica_acad"
    info_log = resumen[clave_log]
    mb = info_log["metricas_best"]

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Accuracy (Test)", f"{mb['Accuracy']:.4f}", f"{mb['Accuracy']*100:.2f} %")
    k2.metric("Precision (Test)", f"{mb['Precision']:.4f}")
    k3.metric("Recall (Test)", f"{mb['Recall']:.4f}")
    k4.metric("F1-Score (Test)", f"{mb['F1-Score']:.4f}")
    k5.metric("ROC-AUC (Test)", f"{mb['ROC-AUC']:.4f}")

    st.plotly_chart(
        G.plotly_curva_roc_matriz(info_log["cm_best"], info_log["fpr_best"], info_log["tpr_best"], mb["ROC-AUC"]),
        use_container_width=True,
    )

    c_l1, c_l2 = st.columns(2)
    with c_l1:
        st.markdown("#### Modelo Base vs Optimizado (`RandomizedSearchCV`)")
        df_comp = pd.DataFrame({
            "Modelo Base": info_log["metricas_base"],
            "Optimizado (RandomSearch)": info_log["metricas_best"],
        })
        df_comp["Diferencia"] = df_comp["Optimizado (RandomSearch)"] - df_comp["Modelo Base"]
        st.dataframe(df_comp.style.format(precision=4), use_container_width=True)
        st.caption(f"Mejores hiperparámetros: `{info_log['best_params']}`")
    with c_l2:
        st.markdown("#### Validación Cruzada Estratificada ($k = 10$ folds)")
        df_cv_log = pd.DataFrame(info_log["tabla_cv"]).set_index("Métrica")
        st.dataframe(df_cv_log.style.format(precision=4), use_container_width=True)

    st.markdown("#### Interpretación de Coeficientes (Log-Odds) y Razones de Probabilidad (*Odds Ratios* $e^{\\beta_j}$)")
    imp_log = pd.DataFrame(info_log["importance"])
    st.dataframe(
        imp_log[["Etiqueta", "Coeficiente (Log-Odds)", "Odds Ratio (e^β)", "Cambio % en Odds", "Signo"]].style.format({
            "Coeficiente (Log-Odds)": "{:+.4f}",
            "Odds Ratio (e^β)": "{:.4f}",
            "Cambio % en Odds": "{:+.2f} %",
        }),
        use_container_width=True,
    )
    st.plotly_chart(
        G.plotly_importancia_coeficientes(
            imp_log,
            "Coeficiente (Log-Odds)",
            "Efecto sobre el Log-Odds de Superar 250 Puntos en Saber 11",
        ),
        use_container_width=True,
    )


# ===========================================================================
# MÓDULO 6: SIMULADOR INTERACTIVO DE PREDICCIÓN EN VIVO (.JOBLIB)
# ===========================================================================
else:
    st.subheader("6. 🔮 Simulador Predictivo en Vivo (Regresión Lineal y Regresión Logística)")
    st.markdown(
        """
        Configure el perfil socioeconómico, familiar e institucional de un estudiante para obtener en tiempo real:
        1. **Predicción de Regresión Lineal (`model_linear_regression.joblib`):** Estimación puntual del `punt_global` (0–500), intervalo de predicción e impacto individual de cada factor.
        2. **Predicción de Regresión Logística (`model_logistic_regression.joblib`):** Probabilidad $\\hat{p}(Y=1|X)$ de superar el promedio nacional ($\\ge 250$ puntos), *Odds* asociadas y clasificación según el umbral de decisión $\\tau$.
        """
    )

    modo_pred = st.radio(
        "Seleccione el motor de predicción (.joblib):",
        [
            "Modo 1 · Socioeconómico Puro (Solo variables de contexto familiar y escolar)",
            "Modo 2 · Académico Integrado (Contexto + Puntaje diagnóstico en Inglés)",
        ],
        horizontal=True,
    )
    usar_acad = modo_pred.startswith("Modo 2")

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        in_educ = st.selectbox(
            "🎓 Nivel educativo de la madre (`fami_educacionmadre`)",
            options=datos.ORDEN_EDUCACION + ["Desconocido"],
            index=4,  # Secundaria completa
        )
        in_estrato = st.selectbox(
            "🏠 Estrato de la vivienda (`fami_estratovivienda`)",
            options=datos.ORDEN_ESTRATO + ["Desconocido"],
            index=1,  # Estrato 2
        )
    with col_f2:
        in_internet = st.selectbox(
            "🌐 Servicio de internet en el hogar (`fami_tieneinternet`)",
            options=["Si", "No", "Desconocido"],
            index=0,
        )
        in_cole = st.selectbox(
            "🏫 Naturaleza del colegio (`cole_naturaleza`)",
            options=["OFICIAL", "NO OFICIAL"],
            index=0,
        )
    with col_f3:
        in_zona = st.selectbox(
            "📍 Zona de ubicación de la sede (`cole_area_ubicacion`)",
            options=["URBANO", "RURAL"],
            index=0,
        )
        in_genero = st.selectbox(
            "👤 Género del estudiante (`estu_genero`)",
            options=["F", "M"],
            index=0,
        )

    c_extra1, c_extra2 = st.columns(2)
    with c_extra1:
        in_ingles = st.slider(
            "🇬🇧 Puntaje en prueba de Inglés (`punt_ingles`, escala 0–100)",
            min_value=0.0,
            max_value=100.0,
            value=50.0,
            step=1.0,
            disabled=not usar_acad,
            help="Habilitado en el Modo 2 (Académico Integrado).",
        )
    with c_extra2:
        umbral_tau = st.slider(
            "⚖️ Umbral de clasificación logística (τ)",
            min_value=0.10,
            max_value=0.90,
            value=0.50,
            step=0.05,
            help="Si P(Y=1|X) >= τ, el estudiante se clasifica en 'Supera Promedio (>= 250 pts)'.",
        )

    # Construir registro de entrada y aplicar ingeniería de variables derivadas
    df_usuario_raw = pd.DataFrame([{
        "fami_educacionmadre": in_educ,
        "fami_estratovivienda": in_estrato,
        "fami_tieneinternet": in_internet,
        "cole_naturaleza": in_cole,
        "cole_area_ubicacion": in_zona,
        "estu_genero": in_genero,
        "punt_ingles": in_ingles,
    }])
    df_usuario = datos.preparar_entrada_usuario(df_usuario_raw)

    pipe_reg = artefactos["reg_acad"] if usar_acad else artefactos["reg_socio"]
    feats_reg = artefactos["features_reg_acad"] if usar_acad else artefactos["features_reg_socio"]
    pipe_log = artefactos["log_acad"] if usar_acad else artefactos["log_socio"]
    feats_log = artefactos["features_log_acad"] if usar_acad else artefactos["features_log_socio"]
    info_r = resumen["regresion_acad" if usar_acad else "regresion_socio"]

    pred_puntaje = float(pipe_reg.predict(df_usuario[feats_reg])[0])
    rse = float(info_r["metricas_test"]["RSE"])
    ic_inf = max(0.0, pred_puntaje - 1.96 * rse)
    ic_sup = min(500.0, pred_puntaje + 1.96 * rse)
    percentil_est = float((df_full["punt_global"] <= pred_puntaje).mean() * 100.0)

    prob_supera = float(pipe_log.predict_proba(df_usuario[feats_log])[0, 1])
    odds_val = prob_supera / max(1e-9, 1.0 - prob_supera)
    logit_val = float(np.log(max(1e-9, odds_val)))
    clase_pred = 1 if prob_supera >= umbral_tau else 0

    st.markdown("---")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric(
        "🎯 Puntaje Global Estimado (Regresión Lineal)",
        f"{pred_puntaje:.2f} / 500",
        f"{pred_puntaje - df_full['punt_global'].mean():+.2f} pts vs media nacional",
    )
    r2.metric(
        "📏 Intervalo de Predicción (95 %)",
        f"[{ic_inf:.1f} , {ic_sup:.1f}]",
        f"Percentil estimado: P{percentil_est:.1f}",
        delta_color="off",
    )
    r3.metric(
        "📈 Probabilidad P(Puntaje ≥ 250) (Logística)",
        f"{prob_supera * 100:.2f} %",
        f"Odds = {odds_val:.3f} (Logit = {logit_val:+.3f})",
        delta_color="off",
    )
    r4.metric(
        f"🏷️ Clasificación (Umbral τ = {umbral_tau:.2f})",
        "SUPERA (≥ 250)" if clase_pred == 1 else "BAJO PROMEDIO (< 250)",
        "Clase Positiva (1)" if clase_pred == 1 else "Clase Negativa (0)",
        delta_color="normal" if clase_pred == 1 else "inverse",
    )

    # Desglose de contribuciones del modelo lineal para el perfil ingresado
    prep_step = pipe_reg.named_steps["preprocessing"]
    reg_step = pipe_reg.named_steps["regressor"]
    x_trans = prep_step.transform(df_usuario[feats_reg])[0]
    nombres_f = list(prep_step.get_feature_names_out())
    contribuciones = x_trans * reg_step.coef_

    df_contrib = pd.DataFrame({
        "Factor": [M.etiquetar_feature(f) for f in nombres_f],
        "Valor Transformado (X_j)": x_trans,
        "Coeficiente (β_j)": reg_step.coef_,
        "Aporte en Puntos (β_j · X_j)": contribuciones,
    }).sort_values("Aporte en Puntos (β_j · X_j)", ascending=True)

    c_g1, c_g2 = st.columns([1.35, 1.0])
    with c_g1:
        fig_c = px.bar(
            df_contrib,
            x="Aporte en Puntos (β_j · X_j)",
            y="Factor",
            orientation="h",
            text="Aporte en Puntos (β_j · X_j)",
            color=np.where(df_contrib["Aporte en Puntos (β_j · X_j)"] >= 0, "Impulsa (+)", "Penaliza (-)"),
            color_discrete_map={"Impulsa (+)": G.AMARILLO, "Penaliza (-)": "#566573"},
            title=f"Descomposición del Puntaje Estimado (Intercepto β₀ = {reg_step.intercept_:.2f} pts)",
        )
        fig_c.add_vline(x=0, line_dash="dash", line_color=G.NEGRO)
        fig_c.update_traces(texttemplate="%{text:+.2f} pts", textposition="outside")
        fig_c.update_layout(template="plotly_white", height=420, yaxis_title="", legend_title="Efecto")
        st.plotly_chart(fig_c, use_container_width=True)

    with c_g2:
        st.markdown("#### 📄 Resumen del Vector de Entrada y Contribuciones")
        st.dataframe(
            df_contrib.sort_values("Aporte en Puntos (β_j · X_j)", ascending=False).style.format({
                "Valor Transformado (X_j)": "{:.4f}",
                "Coeficiente (β_j)": "{:+.4f}",
                "Aporte en Puntos (β_j · X_j)": "{:+.4f}",
            }),
            use_container_width=True,
        )
        st.info(
            f"**Ecuación evaluada:** Intercepto ({reg_step.intercept_:.2f}) + "
            f"Suma de aportes ({contribuciones.sum():+.2f}) = **{pred_puntaje:.2f} puntos**."
        )

