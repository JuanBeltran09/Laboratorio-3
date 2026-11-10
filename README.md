# Laboratorio 3 · Proyecto Final de Modelación Predictiva — Saber 11 (ICFES, Periodo 20224)

**Especialización en Analítica Estratégica de Datos**  
**Asignatura:** Estadística para Analítica de Datos – UPTC (2026-2)  
**Profesor:** Duván Cataño  
**Integrantes:** Juan Beltrán · Daniel Castellanos · Daniel Rico  

---

## 📌 Descripción General

Este repositorio contiene el desarrollo integral del **Laboratorio # 3 (Proyecto Final – Modelación)** sobre los resultados oficiales del examen de Estado de la educación media **Saber 11 (ICFES)** publicados en Datos Abiertos de Colombia (`kgxf-xxbe`, periodo `20224`).

El proyecto articula de principio a fin los hallazgos y decisiones metodológicas de los **Laboratorios 1 y 2** con las técnicas de preprocesamiento, imputación, escalamiento, construcción de `Pipelines`, modelación predictiva (**Regresión Lineal** y **Regresión Logística**), inferencia econométrica y despliegue interactivo en **Streamlit**.

---

## 🗂️ Estructura del Proyecto

```text
Laboratorio 3/
├── README.md                                      # Guía principal, ejecución y síntesis de resultados
├── 01_Guia_General_y_Continuidad_Labs_1_2_3.md    # Trazabilidad metodológica entre los Laboratorios 1, 2 y 3
├── 02_Diccionario_Variables_e_Ingenieria.md       # Diccionario estadístico, variables derivadas y control de data leakage
├── 03_Preprocesamiento_Imputacion_y_Pipelines.md  # Duplicados, MNAR, Simple/KNN/MICE, StandardScaler y ColumnTransformer
├── 04_EDA_y_Pruebas_de_Hipotesis.md               # Medidas estructurales, forma, IQR y pruebas paramétricas/no paramétricas
├── 05_Modelacion_Regresion_y_Clasificacion.md     # Regresión Lineal (MCO/HC3) y Regresión Logística (Log-Odds/RandomSearch)
├── 06_Manual_Aplicacion_Streamlit.md              # Arquitectura y manual de uso de la aplicación interactiva en Streamlit
├── requirements.txt                               # Dependencias reproducibles de Python
├── data/
│   └── saber11_muestra_20224.csv                  # Muestra oficial reproducible (80.000 filas crudas -> 77.827 únicos)
├── models/
│   ├── model_linear_regression.joblib             # Pipeline entrenado Regresión Lineal (Modelo A: Socioeconómico Puro)
│   ├── features_linear_regression.joblib          # Lista de predictores de Regresión Lineal (Modelo A)
│   ├── model_linear_academico.joblib              # Pipeline entrenado Regresión Lineal (Modelo B: Socioeconómico + Inglés)
│   ├── features_linear_academico.joblib           # Lista de predictores de Regresión Lineal (Modelo B)
│   ├── model_logistic_regression.joblib           # Pipeline entrenado Regresión Logística (Modelo A: Clasificación >= 250)
│   ├── features_logistic_regression.joblib        # Lista de predictores de Regresión Logística (Modelo A)
│   ├── model_logistic_academico.joblib            # Pipeline entrenado Regresión Logística (Modelo B: Socioeconómico + Inglés)
│   ├── features_logistic_academico.joblib         # Lista de predictores de Regresión Logística (Modelo B)
│   └── metricas_resumen.json                      # Resumen serializado de métricas, CV y pruebas diagnósticas
├── figuras/                                       # Gráficos de alta resolución generados por el Notebook
├── src/
│   ├── __init__.py
│   ├── datos.py                                   # Extracción Socrata, depuración e ingeniería de variables derivadas
│   ├── estadistica.py                             # Motor estadístico: descriptivas, MNAR, imputación, correlación y pruebas
│   ├── modelos.py                                 # Pipelines, Regresión Lineal, OLS/HC3, Regresión Logística y joblib
│   └── graficos.py                                # Visualizaciones en Matplotlib/Seaborn (paleta UPTC) y Plotly
├── notebooks/
│   └── Laboratorio3_Modelacion_Saber11.ipynb      # Entregable 1: Notebook ejecutado paso a paso (secciones 1 a 9)
└── app/
    └── streamlit_app.py                           # Entregable 2: Aplicación interactiva en Streamlit
```

---

## 🚀 Cómo Ejecutar el Proyecto

### 1. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 2. Abrir el Notebook de Jupyter (Entregable 1)
El notebook [`notebooks/Laboratorio3_Modelacion_Saber11.ipynb`](notebooks/Laboratorio3_Modelacion_Saber11.ipynb) ya se encuentra **completamente ejecutado y con todas las salidas guardadas**. Para abrirlo o volverlo a correr:
```bash
jupyter lab notebooks/Laboratorio3_Modelacion_Saber11.ipynb
```

### 3. Ejecutar la Aplicación Interactiva en Streamlit (Entregable 2)
```bash
streamlit run app/streamlit_app.py
```

### 4. (Opcional) Reentrenar y regenerar los modelos `.joblib` por consola
```bash
python -m src.modelos
```

---

## 📊 Resumen Ejecutivo de Resultados de Modelación (4 cifras decimales)

### A. Modelos de Regresión Lineal (Variable Objetivo: `punt_global`, escala 0–500)

| Métrica (Conjunto de Prueba 20 %, $n=15.566$) | Modelo A: Contexto Socioeconómico Puro (Cero Leakage) | Modelo B: Contexto + Competencia Base Inglés (`punt_ingles`) |
|---|---:|---:|
| **MAE** (Error Absoluto Medio en puntos) | `38.3033` | `24.4325` |
| **RMSE** (Raíz del Error Cuadrático Medio) | `46.8585` | `30.5989` |
| **MAPE** (Error Porcentual Absoluto Medio) | `0.1625` (`16.2482 %`) | `0.1041` (`10.4061 %`) |
| **$R^2$** (Coeficiente de Determinación) | `0.1888` | `0.6541` |
| **$R^2_{\text{ajustado}}$** | `0.1883` | `0.6539` |
| **CV $R^2$ Promedio ($k=10$ folds)** | `0.1875` | `0.6494` |
| **VIF Máximo (Multicolinealidad)** | `1.4018` ($< 5$) | `1.4660` ($< 5$) |

### B. Modelos de Clasificación con Regresión Logística (Variable Objetivo: `supera_promedio`, $\text{punt\_global} \ge 250$)

| Métrica (Conjunto de Prueba 30 %, $n=23.349$) | Modelo A Base | Modelo A Optimizado (`RandomSearch`) | Modelo B Optimizado (Contexto + Inglés) |
|---|---:|---:|---:|
| **Accuracy** (Exactitud global) | `0.6468` | `0.6456` | `0.8152` |
| **Precision** (Precisión clase positiva) | `0.6287` | `0.6191` | `0.8041` |
| **Recall** (Sensibilidad / Tasa de verdaderos positivos) | `0.6497` | **`0.6850`** | **`0.8144`** |
| **F1-Score** (Media armónica Precision–Recall) | `0.6391` | **`0.6504`** | **`0.8092`** |
| **ROC-AUC** (Área bajo la curva ROC) | `0.7014` | **`0.7014`** | **`0.8957`** |

---

## 📚 Mapa de Documentación Detallada (`.md`)

Para una revisión exhaustiva de **qué se hace, cómo se hace, para qué sirve y cómo aplica lo desarrollado en los Laboratorios 1 y 2**, consulte los seis documentos técnicos incluidos en esta carpeta:

1. [`01_Guia_General_y_Continuidad_Labs_1_2_3.md`](01_Guia_General_y_Continuidad_Labs_1_2_3.md): Hilo conductor completo desde la formulación del problema en el Laboratorio 1 y las pruebas de asociación del Laboratorio 2 hasta la arquitectura predictiva del Laboratorio 3.
2. [`02_Diccionario_Variables_e_Ingenieria.md`](02_Diccionario_Variables_e_Ingenieria.md): Especificación detallada de las variables originales y derivadas, justificación de las escalas ordinales e indicadoras de categorías especiales y control de fuga de información.
3. [`03_Preprocesamiento_Imputacion_y_Pipelines.md`](03_Preprocesamiento_Imputacion_y_Pipelines.md): Demostración estadística del mecanismo MNAR, comparación experimental de imputación numérica (Media, Mediana, KNN, MICE), estandarización vs normalización y diseño del `ColumnTransformer`.
4. [`04_EDA_y_Pruebas_de_Hipotesis.md`](04_EDA_y_Pruebas_de_Hipotesis.md): Análisis descriptivo univariado/bivariado, impacto de atípicos IQR y contrastes de hipótesis paramétricos y no paramétricos.
5. [`05_Modelacion_Regresion_y_Clasificacion.md`](05_Modelacion_Regresion_y_Clasificacion.md): Desarrollo matemático y computacional de la Regresión Lineal (MCO, inferencia `statsmodels`, Jarque-Bera, Breusch-Pagan, VIF, errores robustos HC3) y la Regresión Logística (función sigmoide, *Log-Odds*, *Odds Ratios*, regularización $L_1/L_2$ y `RandomizedSearchCV`).
6. [`06_Manual_Aplicacion_Streamlit.md`](06_Manual_Aplicacion_Streamlit.md): Guía funcional y técnica del dashboard interactivo y del simulador de predicción en tiempo real.

