"""Carga, muestreo, depuración e ingeniería de variables para Resultados únicos Saber 11 (ICFES).

Mantiene continuidad estricta con los Laboratorios 1 y 2:
- Mismo dataset oficial de Datos Abiertos de Colombia (kgxf-xxbe), periodo 20224.
- Mismo límite de consulta en la API Socrata (700.000 registros) y misma semilla (42) para la muestra de 80.000 filas.
- Misma regla de depuración: eliminación de duplicados por `estu_consecutivo` (77.827 estudiantes únicos),
  preservación de categorías especiales (`Sin Estrato`, `No sabe`, `No Aplica`) y asignación de `Desconocido`
  a valores faltantes del cuestionario socioeconómico junto con el indicador `sin_cuestionario`.
- Agrega las variables derivadas propuestas en el Diccionario de Variables del Laboratorio 1 para la etapa
  de modelación (Laboratorio 3): `estrato_num` (1-6), `educ_madre_nivel` (0-9), `es_sin_estrato`, `educ_no_sabe`,
  `supera_promedio` (>= 250 puntos) y `alto_desempeno` (>= 300 puntos).
"""
from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
RUTA_CACHE = RAIZ / "data" / "saber11_muestra_20224.csv"

DATASET_ID = "kgxf-xxbe"      # Resultados únicos Saber 11 (datos.gov.co, 2010-2022)
PERIODO = "20224"
LIMITE_API = 700_000          # Igual que en los Laboratorios 1 y 2
N_MUESTRA = 80_000
SEMILLA = 42
UMBRAL_PROMEDIO = 250         # Umbral nacional de referencia en punt_global (escala 0-500)
UMBRAL_ALTO = 300             # Umbral de alto desempeño (elegibilidad de incentivos/becas)

ID_COL = "estu_consecutivo"
COLS_CUANT = [
    "punt_global",
    "punt_matematicas",
    "punt_lectura_critica",
    "punt_c_naturales",
    "punt_sociales_ciudadanas",
    "punt_ingles",
]
COLS_CUAL = [
    "estu_genero",
    "fami_estratovivienda",
    "fami_educacionmadre",
    "fami_tieneinternet",
    "cole_naturaleza",
    "cole_area_ubicacion",
]
COLUMNAS_API = [ID_COL, "periodo"] + COLS_CUANT + COLS_CUAL
COLS_FALTANTES = ["fami_estratovivienda", "fami_educacionmadre", "fami_tieneinternet"]

ORDEN_ESTRATO = [
    "Estrato 1", "Estrato 2", "Estrato 3",
    "Estrato 4", "Estrato 5", "Estrato 6", "Sin Estrato",
]
ORDEN_EDUCACION = [
    "Ninguno",
    "Primaria incompleta",
    "Primaria completa",
    "Secundaria (Bachillerato) incompleta",
    "Secundaria (Bachillerato) completa",
    "Técnica o tecnológica incompleta",
    "Técnica o tecnológica completa",
    "Educación profesional incompleta",
    "Educación profesional completa",
    "Postgrado",
    "No sabe",
    "No Aplica",
]

MAPA_ESTRATO_NUM = {
    "Estrato 1": 1.0,
    "Estrato 2": 2.0,
    "Estrato 3": 3.0,
    "Estrato 4": 4.0,
    "Estrato 5": 5.0,
    "Estrato 6": 6.0,
}

MAPA_EDUCACION_NUM = {
    "Ninguno": 0.0,
    "Primaria incompleta": 1.0,
    "Primaria completa": 2.0,
    "Secundaria (Bachillerato) incompleta": 3.0,
    "Secundaria (Bachillerato) completa": 4.0,
    "Técnica o tecnológica incompleta": 5.0,
    "Técnica o tecnológica completa": 6.0,
    "Educación profesional incompleta": 7.0,
    "Educación profesional completa": 8.0,
    "Postgrado": 9.0,
}

# Especificación de variables predictoras según el diseño del Laboratorio 1 (§10 y Diccionario de Variables):
# - Variables ordinales codificadas en escala numérica (`educ_madre_nivel` 0-9, `estrato_num` 1-6) -> Imputación mediana + StandardScaler
# - Variables nominales e indicadoras de categorías especiales fuera de escala -> OneHotEncoder(drop='first') sin colinealidad (VIF < 1.5)
CAT_COLS_SOCIO = [
    "fami_tieneinternet",
    "cole_naturaleza",
    "cole_area_ubicacion",
    "estu_genero",
    "es_sin_estrato",
    "educ_no_sabe",
]
CATEGORIAS_ORDEN_MODELO = [
    ["No", "Si", "Desconocido"],
    ["NO OFICIAL", "OFICIAL"],
    ["RURAL", "URBANO"],
    ["F", "M"],
    ["No", "Si"],
    ["No", "Si"],
]
NUM_COLS_SOCIO = [
    "educ_madre_nivel",
    "estrato_num",
]
FEATURES_SOCIO = CAT_COLS_SOCIO + NUM_COLS_SOCIO

# Especificación del Modelo Académico Ampliado (Contexto Socioeconómico + Competencia Transversal Bilingüe `punt_ingles`)
CAT_COLS_ACAD = CAT_COLS_SOCIO.copy()
NUM_COLS_ACAD = [
    "punt_ingles",
    "educ_madre_nivel",
    "estrato_num",
]
FEATURES_ACAD = CAT_COLS_ACAD + NUM_COLS_ACAD

# Especificación alternativa Categórica Completa (22 dummies One-Hot sobre las 6 variables originales)
CAT_COLS_FULL_DUMMIES = [
    "fami_educacionmadre",
    "fami_estratovivienda",
    "fami_tieneinternet",
    "cole_naturaleza",
    "cole_area_ubicacion",
    "estu_genero",
]

TIPOS = {
    ID_COL: ("Cualitativa nominal", "Identificador único del examinando (control de duplicados)", "Control"),
    "periodo": ("Variable temporal", "Periodo de aplicación del examen (constante: 20224)", "Filtro"),
    "punt_global": ("Cuantitativa continua", "Puntaje global Saber 11, escala 0-500", "Objetivo Regresión (Y)"),
    "punt_matematicas": ("Cuantitativa continua", "Puntaje en Matemáticas, escala 0-100", "EDA (excluida del modelo por leakage)"),
    "punt_lectura_critica": ("Cuantitativa continua", "Puntaje en Lectura Crítica, escala 0-100", "EDA (excluida del modelo por leakage)"),
    "punt_c_naturales": ("Cuantitativa continua", "Puntaje en Ciencias Naturales, escala 0-100", "EDA (excluida del modelo por leakage)"),
    "punt_sociales_ciudadanas": ("Cuantitativa continua", "Puntaje en Sociales y Ciudadanas, escala 0-100", "EDA (excluida del modelo por leakage)"),
    "punt_ingles": ("Cuantitativa continua", "Puntaje en Inglés, escala 0-100 (282 faltantes en únicos)", "EDA / Predictor en Modelo Académico"),
    "estu_genero": ("Cualitativa nominal (dicotómica)", "Género del examinando (F / M)", "Predictor (control demográfico)"),
    "fami_estratovivienda": ("Cualitativa ordinal", "Estrato socioeconómico de la vivienda (E1-E6, Sin Estrato)", "Predictor socioeconómico"),
    "fami_educacionmadre": ("Cualitativa ordinal", "Máximo nivel educativo alcanzado por la madre", "Predictor familiar principal"),
    "fami_tieneinternet": ("Cualitativa nominal (dicotómica)", "Acceso a servicio de internet en el hogar (Si / No / Desconocido)", "Predictor tecnológico"),
    "cole_naturaleza": ("Cualitativa nominal (dicotómica)", "Naturaleza jurídica del colegio (OFICIAL / NO OFICIAL)", "Predictor institucional"),
    "cole_area_ubicacion": ("Cualitativa nominal (dicotómica)", "Zona geográfica de la sede (URBANO / RURAL)", "Predictor territorial"),
    "estrato_num": ("Cuantitativa discreta (ordinal)", "Codificación numérica del estrato (1 a 6; NaN si Sin Estrato/Desconocido)", "Predictor numérico (escalado)"),
    "educ_madre_nivel": ("Cuantitativa discreta (ordinal)", "Escala educativa materna (0=Ninguno a 9=Postgrado; NaN si especial/vacío)", "Predictor numérico (escalado)"),
    "es_sin_estrato": ("Cualitativa nominal (dicotómica)", "Indicador ('Si'/'No') para hogares en categoría especial 'Sin Estrato'", "Predictor categoría especial"),
    "educ_no_sabe": ("Cualitativa nominal (dicotómica)", "Indicador ('Si'/'No') cuando el estudiante reporta 'No sabe' educación materna", "Predictor categoría especial"),
    "sin_cuestionario": ("Cuantitativa discreta (binaria)", "Indicador 1 si faltan simultáneamente las 3 variables fami_*", "Diagnóstico de no respuesta (MNAR)"),
    "supera_promedio": ("Cualitativa nominal (binaria)", "1 si punt_global >= 250 (supera media nacional), 0 si < 250", "Objetivo Clasificación (Y_bin)"),
    "alto_desempeno": ("Cualitativa nominal (binaria)", "1 si punt_global >= 300 (alto desempeño / perfil beca), 0 si < 300", "Objetivo Clasificación Desbalanceada"),
}


def descargar_muestra() -> pd.DataFrame:
    """Descarga desde la API Socrata de datos.gov.co y toma muestra aleatoria reproducible (semilla 42)."""
    filtro = f"periodo = '{PERIODO}'"
    url = (
        f"https://www.datos.gov.co/resource/{DATASET_ID}.csv"
        f"?$select={quote(','.join(COLUMNAS_API))}&$where={quote(filtro)}&$limit={LIMITE_API}"
    )
    try:
        total = pd.read_csv(url, low_memory=False)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "No se pudo descargar la base desde datos.gov.co. Verifique la conexión a internet "
            f"o coloque el archivo saber11_muestra_20224.csv en data/.\nDetalle: {e}"
        ) from e
    total.columns = total.columns.str.lower().str.strip()
    return total.sample(n=min(N_MUESTRA, len(total)), random_state=SEMILLA).reset_index(drop=True)


def cargar_muestra_cruda(forzar_descarga: bool = False) -> pd.DataFrame:
    """Lee la muestra cruda (80.000 filas) desde data/saber11_muestra_20224.csv o la descarga si no existe."""
    if RUTA_CACHE.exists() and not forzar_descarga:
        return pd.read_csv(RUTA_CACHE, low_memory=False)
    df = descargar_muestra()
    RUTA_CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RUTA_CACHE, index=False)
    return df


def agregar_variables_derivadas(df: pd.DataFrame) -> pd.DataFrame:
    """Construye las variables derivadas propuestas en los Laboratorios 1 y 2 para el modelado."""
    d = df.copy()
    if "fami_estratovivienda" in d.columns:
        d["estrato_num"] = d["fami_estratovivienda"].map(MAPA_ESTRATO_NUM).astype(float)
        d["es_sin_estrato"] = np.where(d["fami_estratovivienda"] == "Sin Estrato", "Si", "No")
    if "fami_educacionmadre" in d.columns:
        d["educ_madre_nivel"] = d["fami_educacionmadre"].map(MAPA_EDUCACION_NUM).astype(float)
        d["educ_no_sabe"] = np.where(d["fami_educacionmadre"] == "No sabe", "Si", "No")
    if "sin_cuestionario" not in d.columns and all(c in d.columns for c in COLS_FALTANTES):
        d["sin_cuestionario"] = (
            (d["fami_estratovivienda"] == "Desconocido")
            & (d["fami_educacionmadre"] == "Desconocido")
            & (d["fami_tieneinternet"] == "Desconocido")
        ).astype(int)
    if "punt_global" in d.columns:
        d["supera_promedio"] = (d["punt_global"] >= UMBRAL_PROMEDIO).astype(int)
        d["alto_desempeno"] = (d["punt_global"] >= UMBRAL_ALTO).astype(int)
    return d


def preparar_entrada_usuario(df_input: pd.DataFrame) -> pd.DataFrame:
    """Transforma un DataFrame con las variables seleccionadas por el usuario en las columnas del Pipeline."""
    return agregar_variables_derivadas(df_input)


def depurar(crudo: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Aplica el protocolo de limpieza y preprocesamiento definido en los Laboratorios 1 y 2."""
    df = crudo.copy()
    df.columns = df.columns.str.lower().str.strip()
    for c in COLS_CUANT + ["periodo"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    faltantes_crudo = df[COLS_FALTANTES + ["punt_ingles"]].isna().sum()
    n0 = len(df)
    n_dup_exactos = int(df.duplicated().sum())
    n_dup_id = int(df[ID_COL].duplicated().sum())

    # Eliminar duplicados por identificador único del estudiante
    df = df.drop_duplicates(subset=ID_COL).reset_index(drop=True)
    faltantes_dedup = df[COLS_FALTANTES + ["punt_ingles"]].isna().sum()

    # Indicador de no diligenciamiento del cuestionario socioeconómico (3 variables fami_* ausentes a la vez)
    df["sin_cuestionario"] = df[COLS_FALTANTES].isna().all(axis=1).astype(int)
    n_sin_cuestionario = int(df["sin_cuestionario"].sum())
    n_al_menos_un_faltante = int(df[COLS_FALTANTES].isna().any(axis=1).sum())

    # Tratamiento de vacíos en cualitativas como categoría explícita 'Desconocido'
    for c in COLS_CUAL:
        df[c] = df[c].astype(object).where(df[c].notna(), "Desconocido").astype(str).str.strip()

    # Ingeniería de variables derivadas
    df = agregar_variables_derivadas(df)

    reporte = {
        "n_inicial": n0,
        "n_duplicados": n_dup_id,
        "n_duplicados_exactos": n_dup_exactos,
        "pct_duplicados": 100.0 * n_dup_id / n0,
        "n_final": len(df),
        "faltantes_antes": faltantes_crudo,
        "faltantes_dedup": faltantes_dedup,
        "n_sin_cuestionario": n_sin_cuestionario,
        "pct_sin_cuestionario": 100.0 * n_sin_cuestionario / len(df),
        "n_al_menos_un_faltante": n_al_menos_un_faltante,
        "pct_al_menos_un_faltante": 100.0 * n_al_menos_un_faltante / len(df),
        "n_faltantes_ingles": int(df["punt_ingles"].isna().sum()),
        "pct_supera_promedio": float(df["supera_promedio"].mean() * 100.0),
        "pct_alto_desempeno": float(df["alto_desempeno"].mean() * 100.0),
    }
    return df, reporte


def cargar_datos(forzar_descarga: bool = False) -> tuple[pd.DataFrame, dict]:
    """Atajo reproducible: carga la muestra cruda y ejecuta la depuración e ingeniería de variables."""
    return depurar(cargar_muestra_cruda(forzar_descarga))


def orden_categorias(col: str, valores) -> list[str]:
    """Orden lógico de las categorías (ordinales en su jerarquía natural; 'Desconocido' siempre al final)."""
    valores = list(pd.unique(pd.Series(valores).astype(str)))
    base = {"fami_estratovivienda": ORDEN_ESTRATO, "fami_educacionmadre": ORDEN_EDUCACION}.get(col)
    desconocido = [v for v in valores if v == "Desconocido"]
    resto = [v for v in valores if v != "Desconocido"]
    if base is None:
        return sorted(resto) + desconocido
    ordenadas = [v for v in base if v in resto]
    return ordenadas + sorted(v for v in resto if v not in ordenadas) + desconocido


def ponderacion_oficial(df: pd.DataFrame) -> pd.Series:
    """Puntaje global reconstruido según la fórmula oficial del ICFES: 5 * (3*LC + 3*MAT + 3*SC + 3*CN + 1*ING) / 13."""
    num = (
        3 * df["punt_lectura_critica"]
        + 3 * df["punt_matematicas"]
        + 3 * df["punt_sociales_ciudadanas"]
        + 3 * df["punt_c_naturales"]
        + df["punt_ingles"]
    )
    return 5 * num / 13
