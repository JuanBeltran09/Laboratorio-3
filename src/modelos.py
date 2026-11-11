"""Construcción de Pipelines, Entrenamiento, Validación Cruzada, Diagnóstico OLS y Serialización (Laboratorio 3).

Implementa la metodología completa de los Notebooks 7, 8 y 9 del curso:
- Preprocesamiento con `Pipeline` y `ColumnTransformer` (`SimpleImputer`, `OneHotEncoder(drop='first')`, `StandardScaler`).
- Modelo de Regresión Lineal (`LinearRegression` en Scikit-Learn + `OLS` en Statsmodels):
  Métricas MAE, RMSE, MAPE y R² (más R² ajustado y RSE), Validación Cruzada k=10,
  inferencia de parámetros (t, F, p-valores, IC 95 %), VIF, Prueba de Jarque-Bera y Prueba de Breusch-Pagan.
- Modelo de Clasificación con Regresión Logística (`LogisticRegression`):
  Métricas Accuracy, Precision, Recall y F1-score (más ROC-AUC, Matriz de Confusión y Curva ROC),
  Validación Cruzada Estratificada (`StratifiedKFold`), optimización de hiperparámetros (`RandomizedSearchCV`)
  e interpretación mediante Log-Odds y Odds Ratios (OR = exp(beta)).
- Serialización reproducible en `models/*.joblib` para consumo directo en la aplicación de Streamlit.
"""
from __future__ import annotations

import inspect
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy import stats
from scipy.stats import uniform
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import jarque_bera

from . import datos

RUTA_MODELOS = datos.RAIZ / "models"

ETIQUETAS_FEATURES = {
    "const": "Intercepto (β₀)",
    "cat__fami_tieneinternet_Si": "Internet en casa: Sí (vs. No)",
    "cat__fami_tieneinternet_Desconocido": "Cuestionario ausente / Internet Desconocido (vs. No)",
    "cat__cole_naturaleza_OFICIAL": "Colegio Oficial / Público (vs. No Oficial)",
    "cat__cole_area_ubicacion_URBANO": "Zona del colegio: Urbana (vs. Rural)",
    "cat__estu_genero_M": "Género: Masculino (vs. Femenino)",
    "cat__es_sin_estrato_Si": "Vivienda: Sin Estrato (vs. En escala E1–E6)",
    "cat__educ_no_sabe_Si": "Educación madre: Reporta 'No sabe' (vs. Nivel conocido)",
    "num__educ_madre_nivel": "Nivel educativo de la madre (escalado Z, 0–9)",
    "num__estrato_num": "Estrato socioeconómico numérico (escalado Z, 1–6)",
    "num__punt_ingles": "Puntaje en Inglés (escalado Z, 0–100)",
}


def etiquetar_feature(nombre: str) -> str:
    return ETIQUETAS_FEATURES.get(nombre, nombre)


# --------------------------------------------------------------------------- 1. Construcción de Pipelines
def construir_preprocesador(
    cat_cols: list[str] | None = None,
    num_cols: list[str] | None = None,
    categorias_orden: list[list[str]] | str | None = None,
    scaler: str = "standard",
) -> ColumnTransformer:
    """Construye el ColumnTransformer integrando imputación, codificación Dummy (k-1) y escalamiento."""
    cat_cols = cat_cols if cat_cols is not None else datos.CAT_COLS_SOCIO
    num_cols = num_cols if num_cols is not None else datos.NUM_COLS_SOCIO
    if categorias_orden is None and cat_cols == datos.CAT_COLS_SOCIO:
        categorias_orden = datos.CATEGORIAS_ORDEN_MODELO
    elif categorias_orden is None:
        categorias_orden = "auto"

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Desconocido")),
        (
            "onehot",
            OneHotEncoder(
                categories=categorias_orden,
                drop="first",
                handle_unknown="ignore",
                sparse_output=False,
            ),
        ),
    ])

    escalador = StandardScaler() if scaler == "standard" else MinMaxScaler()
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", escalador),
    ])

    return ColumnTransformer([
        ("cat", cat_pipeline, cat_cols),
        ("num", num_pipeline, num_cols),
    ])


def construir_pipeline_regresion(
    cat_cols: list[str] | None = None,
    num_cols: list[str] | None = None,
    categorias_orden: list[list[str]] | str | None = None,
    scaler: str = "standard",
) -> Pipeline:
    """Pipeline reproducible para Regresión Lineal Múltiple (MCO)."""
    prep = construir_preprocesador(cat_cols, num_cols, categorias_orden, scaler)
    return Pipeline([
        ("preprocessing", prep),
        ("regressor", LinearRegression()),
    ])


def _usa_l1_ratio_logistica() -> bool:
    """Detecta si scikit-learn (>= 1.8) deprecó `penalty` en favor de `l1_ratio`."""
    param = inspect.signature(LogisticRegression.__init__).parameters.get("penalty")
    return param is None or param.default == "deprecated"


def _kwargs_regularizacion_logistica(penalty: str | None, C: float) -> dict:
    """Devuelve argumentos de regularización compatibles con scikit-learn 1.3–1.8+."""
    if not _usa_l1_ratio_logistica():
        return {"penalty": penalty, "C": C}
    if penalty == "l1":
        return {"l1_ratio": 1.0, "C": C}
    if penalty == "l2":
        return {"l1_ratio": 0.0, "C": C}
    if penalty == "elasticnet":
        return {"l1_ratio": 0.5, "C": C}
    return {"C": np.inf if penalty is None else C}


def construir_pipeline_logistica(
    cat_cols: list[str] | None = None,
    num_cols: list[str] | None = None,
    categorias_orden: list[list[str]] | str | None = None,
    scaler: str = "standard",
    C: float = 1.0,
    penalty: str = "l2",
    solver: str = "liblinear",
    class_weight: str | None = None,
    semilla: int = datos.SEMILLA,
) -> Pipeline:
    """Pipeline reproducible para Regresión Logística (Clasificación Binaria)."""
    prep = construir_preprocesador(cat_cols, num_cols, categorias_orden, scaler)
    reg_kwargs = _kwargs_regularizacion_logistica(penalty=penalty, C=C)
    return Pipeline([
        ("preprocessing", prep),
        (
            "model",
            LogisticRegression(
                solver=solver,
                class_weight=class_weight,
                max_iter=1000,
                random_state=semilla,
                **reg_kwargs,
            ),
        ),
    ])


# --------------------------------------------------------------------------- 2. Regresión Lineal y Diagnóstico MCO
def calcular_metricas_regresion(y_true: np.ndarray, y_pred: np.ndarray, p_features: int) -> dict[str, float]:
    """Calcula MAE, RMSE, MAPE, MAPE (%), R², R² ajustado y RSE (Error Estándar de los Residuos)."""
    n = len(y_true)
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mape = float(mean_absolute_percentage_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    r2_adj = float(1.0 - (1.0 - r2) * (n - 1) / max(1, n - p_features - 1))
    rss = float(np.sum((y_true - y_pred) ** 2))
    rse = float(np.sqrt(rss / max(1, n - p_features - 1)))
    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "MAPE (%)": mape * 100.0,
        "R²": r2,
        "R² Ajustado": r2_adj,
        "RSE": rse,
    }


def ajustar_ols_statsmodels(X_encoded_df: pd.DataFrame, y: pd.Series) -> dict:
    """Ajusta el modelo OLS en Statsmodels y ejecuta pruebas de Jarque-Bera, Breusch-Pagan y VIF."""
    X_reset = X_encoded_df.reset_index(drop=True)
    y_reset = y.reset_index(drop=True).astype(float)
    X_sm = sm.add_constant(X_reset)

    model_ols = sm.OLS(y_reset, X_sm).fit()
    model_hc3 = sm.OLS(y_reset, X_sm).fit(cov_type="HC3")

    residuals = model_ols.resid
    fitted = model_ols.fittedvalues

    # Prueba de normalidad de Jarque-Bera
    jb_stat, jb_pval, jb_skew, jb_kurt = jarque_bera(residuals)

    # Prueba de homocedasticidad de Breusch-Pagan
    bp_lm, bp_lm_pval, bp_f, bp_f_pval = het_breuschpagan(residuals, model_ols.model.exog)

    # Factor de Inflación de la Varianza (VIF)
    exog_arr = X_sm.to_numpy()
    vifs = []
    for i, col in enumerate(X_sm.columns):
        if col == "const":
            continue
        vifs.append({
            "Variable": col,
            "Etiqueta": etiquetar_feature(col),
            "VIF": float(variance_inflation_factor(exog_arr, i)),
        })
    vif_df = pd.DataFrame(vifs).set_index("Variable")

    # Tabla de inferencia de parámetros (MCO clásico y errores robustos HC3)
    ci = model_ols.conf_int(alpha=0.05)
    tabla_params = pd.DataFrame({
        "Etiqueta": [etiquetar_feature(c) for c in X_sm.columns],
        "Coeficiente (β)": model_ols.params,
        "Error Estándar (MCO)": model_ols.bse,
        "t (MCO)": model_ols.tvalues,
        "p_valor (MCO)": model_ols.pvalues,
        "Error Estándar (HC3)": model_hc3.bse,
        "z (HC3)": model_hc3.tvalues,
        "p_valor (HC3)": model_hc3.pvalues,
        "IC 95% Inf": ci[0],
        "IC 95% Sup": ci[1],
    })
    tabla_params.index.name = "Variable"

    return {
        "model_ols": model_ols,
        "model_hc3": model_hc3,
        "tabla_params": tabla_params,
        "vif_df": vif_df,
        "residuals": residuals,
        "fitted": fitted,
        "jarque_bera": {
            "JB_stat": float(jb_stat),
            "p_valor": float(jb_pval),
            "asimetria": float(jb_skew),
            "curtosis": float(jb_kurt),
            "exceso_curtosis": float(jb_kurt - 3.0),
        },
        "breusch_pagan": {
            "LM_stat": float(bp_lm),
            "p_valor_LM": float(bp_lm_pval),
            "F_stat": float(bp_f),
            "p_valor_F": float(bp_f_pval),
        },
        "F_global": float(model_ols.fvalue),
        "p_F_global": float(model_ols.f_pvalue),
        "R2": float(model_ols.rsquared),
        "R2_adj": float(model_ols.rsquared_adj),
        "AIC": float(model_ols.aic),
        "BIC": float(model_ols.bic),
        "cond_number": float(model_ols.condition_number),
    }


def entrenar_evaluar_regresion(
    df: pd.DataFrame,
    cat_cols: list[str] | None = None,
    num_cols: list[str] | None = None,
    target: str = "punt_global",
    categorias_orden: list[list[str]] | str | None = None,
    test_size: float = 0.20,
    semilla: int = datos.SEMILLA,
    cv_folds: int = 10,
) -> dict:
    """Ejecuta el flujo completo de entrenamiento, evaluación en test, validación cruzada e inferencia OLS."""
    cat_cols = cat_cols if cat_cols is not None else datos.CAT_COLS_SOCIO
    num_cols = num_cols if num_cols is not None else datos.NUM_COLS_SOCIO
    features = cat_cols + num_cols

    X = df[features].copy()
    y = df[target].astype(float).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=semilla
    )

    model = construir_pipeline_regresion(cat_cols, num_cols, categorias_orden)
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    feature_names = list(model.named_steps["preprocessing"].get_feature_names_out())
    p_features = len(feature_names)

    metricas_train = calcular_metricas_regresion(y_train.to_numpy(), y_pred_train, p_features)
    metricas_test = calcular_metricas_regresion(y_test.to_numpy(), y_pred_test, p_features)

    tabla_metricas = pd.DataFrame({
        "Entrenamiento (80 %)": metricas_train,
        "Prueba / Test (20 %)": metricas_test,
    })
    tabla_metricas.index.name = "Métrica"

    # DataFrame de resultados y errores sobre el conjunto de prueba (Notebook 8)
    df_resultados = X_test.reset_index(drop=True).copy()
    df_resultados["y_real"] = y_test.reset_index(drop=True)
    df_resultados["y_pred"] = pd.Series(y_pred_test)
    df_resultados["error"] = df_resultados["y_real"] - df_resultados["y_pred"]
    df_resultados["abs_error"] = df_resultados["error"].abs()
    df_resultados["pct_error"] = (df_resultados["abs_error"] / df_resultados["y_real"]) * 100.0

    # Validación Cruzada k-fold (Notebook 8)
    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "mape": "neg_mean_absolute_percentage_error",
        "r2": "r2",
    }
    cv_raw = cross_validate(model, X, y, cv=cv_folds, scoring=scoring, n_jobs=1)
    cv_df = pd.DataFrame({
        "MAE": -cv_raw["test_mae"],
        "RMSE": -cv_raw["test_rmse"],
        "MAPE": -cv_raw["test_mape"],
        "R2": cv_raw["test_r2"],
    })
    tabla_cv = pd.DataFrame({
        "Métrica": ["CV MAE", "CV RMSE", "CV MAPE", "CV MAPE (%)", "CV R²", "FIT TIME (s)", "SCORE TIME (s)"],
        "Promedio": [
            float(cv_df["MAE"].mean()),
            float(cv_df["RMSE"].mean()),
            float(cv_df["MAPE"].mean()),
            float(cv_df["MAPE"].mean() * 100.0),
            float(cv_df["R2"].mean()),
            float(cv_raw["fit_time"].mean()),
            float(cv_raw["score_time"].mean()),
        ],
        "Desv. Estándar": [
            float(cv_df["MAE"].std(ddof=1)),
            float(cv_df["RMSE"].std(ddof=1)),
            float(cv_df["MAPE"].std(ddof=1)),
            float(cv_df["MAPE"].std(ddof=1) * 100.0),
            float(cv_df["R2"].std(ddof=1)),
            float(cv_raw["fit_time"].std(ddof=1)),
            float(cv_raw["score_time"].std(ddof=1)),
        ],
    }).set_index("Métrica")

    # Importancia de variables mediante coeficientes
    coefs = model.named_steps["regressor"].coef_
    importance_df = pd.DataFrame({
        "Variable": feature_names,
        "Etiqueta": [etiquetar_feature(f) for f in feature_names],
        "Coeficiente": coefs,
        "Importancia": np.abs(coefs),
        "Signo": np.where(coefs >= 0, "Positivo", "Negativo"),
    }).sort_values("Importancia", ascending=False).reset_index(drop=True)

    # Inferencia OLS sobre la matriz de diseño de entrenamiento
    X_train_encoded = pd.DataFrame(
        model.named_steps["preprocessing"].transform(X_train),
        columns=feature_names,
    )
    diagnostico_ols = ajustar_ols_statsmodels(X_train_encoded, y_train)

    return {
        "model": model,
        "features": features,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "feature_names": feature_names,
        "intercepto": float(model.named_steps["regressor"].intercept_),
        "metricas_train": metricas_train,
        "metricas_test": metricas_test,
        "tabla_metricas": tabla_metricas,
        "df_resultados": df_resultados,
        "cv_df": cv_df,
        "tabla_cv": tabla_cv,
        "importance_df": importance_df,
        "ols": diagnostico_ols,
        "X_train_shape": X_train.shape,
        "X_test_shape": X_test.shape,
    }


# --------------------------------------------------------------------------- 3. Regresión Logística (Clasificación)
def calcular_metricas_clasificacion(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
) -> dict[str, float]:
    """Calcula las métricas exigidas para Clasificación: Accuracy, Precision, Recall, F1-score y ROC-AUC."""
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1-Score": float(f1_score(y_true, y_pred, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y_true, y_prob)),
    }


def entrenar_evaluar_logistica(
    df: pd.DataFrame,
    cat_cols: list[str] | None = None,
    num_cols: list[str] | None = None,
    target: str = "supera_promedio",
    categorias_orden: list[list[str]] | str | None = None,
    test_size: float = 0.30,
    semilla: int = datos.SEMILLA,
    cv_folds: int = 10,
    n_iter_search: int = 12,
) -> dict:
    """Ejecuta el flujo completo de Regresión Logística (Notebook 9 del curso):
    Partición estratificada 70/30, modelo base, validación cruzada estratificada k=10,
    optimización de hiperparámetros con RandomizedSearchCV, métricas en test, curva ROC y Odds Ratios.
    """
    cat_cols = cat_cols if cat_cols is not None else datos.CAT_COLS_SOCIO
    num_cols = num_cols if num_cols is not None else datos.NUM_COLS_SOCIO
    features = cat_cols + num_cols

    X = df[features].copy()
    y = df[target].astype(int).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=semilla, stratify=y
    )

    # 1. Modelo Base
    pipeline_base = construir_pipeline_logistica(cat_cols, num_cols, categorias_orden, semilla=semilla)
    pipeline_base.fit(X_train, y_train)

    y_pred_base = pipeline_base.predict(X_test)
    y_prob_base = pipeline_base.predict_proba(X_test)[:, 1]
    metricas_base = calcular_metricas_clasificacion(y_test.to_numpy(), y_pred_base, y_prob_base)

    # 2. Validación Cruzada Estratificada k=10
    cv_strat = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=semilla)
    scoring_cls = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }
    cv_raw = cross_validate(
        estimator=pipeline_base,
        X=X,
        y=y,
        cv=cv_strat,
        scoring=scoring_cls,
        return_train_score=False,
        n_jobs=1,
    )
    tabla_cv = pd.DataFrame({
        "Métrica": ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
        "Promedio": [
            float(cv_raw["test_accuracy"].mean()),
            float(cv_raw["test_precision"].mean()),
            float(cv_raw["test_recall"].mean()),
            float(cv_raw["test_f1"].mean()),
            float(cv_raw["test_roc_auc"].mean()),
        ],
        "Desv. Std": [
            float(cv_raw["test_accuracy"].std(ddof=1)),
            float(cv_raw["test_precision"].std(ddof=1)),
            float(cv_raw["test_recall"].std(ddof=1)),
            float(cv_raw["test_f1"].std(ddof=1)),
            float(cv_raw["test_roc_auc"].std(ddof=1)),
        ],
    }).set_index("Métrica")

    # 3. Optimización de Hiperparámetros con RandomizedSearchCV (Notebook 9)
    cv_search = StratifiedKFold(n_splits=5, shuffle=True, random_state=semilla)
    clave_reg_param = "model__l1_ratio" if _usa_l1_ratio_logistica() else "model__penalty"
    vals_reg_param = [0.0, 1.0] if _usa_l1_ratio_logistica() else ["l1", "l2"]
    param_dist = {
        "model__C": uniform(0.01, 15.0),
        clave_reg_param: vals_reg_param,
        "model__solver": ["liblinear"],
        "model__class_weight": [None, "balanced"],
    }
    random_search = RandomizedSearchCV(
        estimator=pipeline_base,
        param_distributions=param_dist,
        n_iter=n_iter_search,
        scoring="f1",
        cv=cv_search,
        random_state=semilla,
        n_jobs=1,
    )
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_

    y_pred_best = best_model.predict(X_test)
    y_prob_best = best_model.predict_proba(X_test)[:, 1]
    metricas_best = calcular_metricas_clasificacion(y_test.to_numpy(), y_pred_best, y_prob_best)

    tabla_comparativa = pd.DataFrame({
        "Modelo Base": metricas_base,
        "Modelo Optimizado (RandomSearch)": metricas_best,
    })
    tabla_comparativa["Diferencia"] = (
        tabla_comparativa["Modelo Optimizado (RandomSearch)"] - tabla_comparativa["Modelo Base"]
    )
    tabla_comparativa.index.name = "Métrica"

    # Matriz de confusión y curva ROC (sobre modelo optimizado y base)
    cm_base = confusion_matrix(y_test, y_pred_base)
    cm_best = confusion_matrix(y_test, y_pred_best)
    fpr_base, tpr_base, _ = roc_curve(y_test, y_prob_base)
    fpr_best, tpr_best, umbrales_roc = roc_curve(y_test, y_prob_best)

    reporte_cls = classification_report(
        y_test,
        y_pred_best,
        target_names=["Bajo Promedio (<250)", "Supera Promedio (>=250)"],
        output_dict=True,
    )
    reporte_texto = classification_report(
        y_test,
        y_pred_best,
        target_names=["Bajo Promedio (<250)", "Supera Promedio (>=250)"],
        digits=4,
    )

    # Importancia de variables, Log-Odds y Odds Ratios (OR = e^beta)
    feature_names = list(best_model.named_steps["preprocessing"].get_feature_names_out())
    coefs = best_model.named_steps["model"].coef_[0]
    odds_ratios = np.exp(coefs)
    importance_df = pd.DataFrame({
        "Variable": feature_names,
        "Etiqueta": [etiquetar_feature(f) for f in feature_names],
        "Coeficiente (Log-Odds)": coefs,
        "Odds Ratio (e^β)": odds_ratios,
        "Cambio % en Odds": (odds_ratios - 1.0) * 100.0,
        "Importancia": np.abs(coefs),
        "Signo": np.where(coefs >= 0, "Positivo (Aumenta P(Y=1))", "Negativo (Reduce P(Y=1))"),
    }).sort_values("Importancia", ascending=False).reset_index(drop=True)

    return {
        "pipeline_base": pipeline_base,
        "best_model": best_model,
        "best_params": {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in random_search.best_params_.items()},
        "best_cv_score": float(random_search.best_score_),
        "features": features,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "feature_names": feature_names,
        "intercepto": float(best_model.named_steps["model"].intercept_[0]),
        "metricas_base": metricas_base,
        "metricas_best": metricas_best,
        "tabla_comparativa": tabla_comparativa,
        "tabla_cv": tabla_cv,
        "cm_base": cm_base,
        "cm_best": cm_best,
        "fpr_base": fpr_base,
        "tpr_base": tpr_base,
        "fpr_best": fpr_best,
        "tpr_best": tpr_best,
        "umbrales_roc": umbrales_roc,
        "reporte_dict": reporte_cls,
        "reporte_texto": reporte_texto,
        "importance_df": importance_df,
        "X_train_shape": X_train.shape,
        "X_test_shape": X_test.shape,
        "balance_clases": y.value_counts(normalize=True).to_dict(),
    }


# --------------------------------------------------------------------------- 4. Entrenamiento Final y Guardado (.joblib + JSON)
def _extraer_hiperparametros_logistica(best_params: dict) -> dict:
    """Normaliza los hiperparámetros guardados de Logística (compatibilidad penalty / l1_ratio)."""
    if "model__penalty" in best_params:
        penalty = best_params["model__penalty"]
    elif "model__l1_ratio" in best_params:
        penalty = "l1" if float(best_params["model__l1_ratio"]) >= 0.5 else "l2"
    else:
        penalty = "l2"
    return {
        "C": float(best_params.get("model__C", 1.0)),
        "penalty": penalty,
        "solver": best_params.get("model__solver", "liblinear"),
        "class_weight": best_params.get("model__class_weight", None),
    }


def _ajustar_modelos_produccion(
    df: pd.DataFrame,
    bp_socio: dict,
    bp_acad: dict,
    guardar_en_disco: bool = True,
    ignorar_error_disco: bool = False,
) -> dict:
    """Ajusta los 4 pipelines finales con el 100 % de los datos usando la versión activa de scikit-learn."""
    df_fit = df.copy()
    for col_cat in set(datos.CAT_COLS_SOCIO) | set(datos.CAT_COLS_ACAD):
        if col_cat in df_fit.columns:
            df_fit[col_cat] = df_fit[col_cat].astype(object)

    y_reg = df_fit["punt_global"].astype(float)
    y_log = df_fit["supera_promedio"].astype(int)

    final_reg_socio = construir_pipeline_regresion(datos.CAT_COLS_SOCIO, datos.NUM_COLS_SOCIO)
    final_reg_socio.fit(df_fit[datos.FEATURES_SOCIO], y_reg)

    final_reg_acad = construir_pipeline_regresion(datos.CAT_COLS_ACAD, datos.NUM_COLS_ACAD)
    final_reg_acad.fit(df_fit[datos.FEATURES_ACAD], y_reg)

    params_s = _extraer_hiperparametros_logistica(bp_socio)
    final_log_socio = construir_pipeline_logistica(
        datos.CAT_COLS_SOCIO, datos.NUM_COLS_SOCIO, **params_s
    )
    final_log_socio.fit(df_fit[datos.FEATURES_SOCIO], y_log)

    params_a = _extraer_hiperparametros_logistica(bp_acad)
    final_log_acad = construir_pipeline_logistica(
        datos.CAT_COLS_ACAD, datos.NUM_COLS_ACAD, **params_a
    )
    final_log_acad.fit(df_fit[datos.FEATURES_ACAD], y_log)

    artefactos = {
        "reg_socio": final_reg_socio,
        "features_reg_socio": list(datos.FEATURES_SOCIO),
        "reg_acad": final_reg_acad,
        "features_reg_acad": list(datos.FEATURES_ACAD),
        "log_socio": final_log_socio,
        "features_log_socio": list(datos.FEATURES_SOCIO),
        "log_acad": final_log_acad,
        "features_log_acad": list(datos.FEATURES_ACAD),
    }

    if guardar_en_disco:
        try:
            RUTA_MODELOS.mkdir(parents=True, exist_ok=True)
            joblib.dump(final_reg_socio, RUTA_MODELOS / "model_linear_regression.joblib")
            joblib.dump(list(datos.FEATURES_SOCIO), RUTA_MODELOS / "features_linear_regression.joblib")
            joblib.dump(final_reg_acad, RUTA_MODELOS / "model_linear_academico.joblib")
            joblib.dump(list(datos.FEATURES_ACAD), RUTA_MODELOS / "features_linear_academico.joblib")
            joblib.dump(final_log_socio, RUTA_MODELOS / "model_logistic_regression.joblib")
            joblib.dump(list(datos.FEATURES_SOCIO), RUTA_MODELOS / "features_logistic_regression.joblib")
            joblib.dump(final_log_acad, RUTA_MODELOS / "model_logistic_academico.joblib")
            joblib.dump(list(datos.FEATURES_ACAD), RUTA_MODELOS / "features_logistic_academico.joblib")
        except OSError:
            if not ignorar_error_disco:
                raise

    return artefactos


def _validar_artefactos_predictivos(artefactos: dict) -> None:
    """Verifica que los 4 pipelines cargados ejecuten transform y predict sin errores de versión."""
    df_smoke_raw = pd.DataFrame([
        {
            "fami_educacionmadre": "Secundaria (Bachillerato) completa",
            "fami_estratovivienda": "Estrato 2",
            "fami_tieneinternet": "Si",
            "cole_naturaleza": "OFICIAL",
            "cole_area_ubicacion": "URBANO",
            "estu_genero": "F",
            "punt_ingles": 50.0,
        },
        {
            "fami_educacionmadre": "Desconocido",
            "fami_estratovivienda": "Sin Estrato",
            "fami_tieneinternet": "Desconocido",
            "cole_naturaleza": "NO OFICIAL",
            "cole_area_ubicacion": "RURAL",
            "estu_genero": "M",
            "punt_ingles": np.nan,
        },
    ])
    df_smoke = datos.preparar_entrada_usuario(df_smoke_raw)

    for clave_reg, clave_feats in (("reg_socio", "features_reg_socio"), ("reg_acad", "features_reg_acad")):
        pipe = artefactos[clave_reg]
        feats = artefactos[clave_feats]
        prep = pipe.named_steps["preprocessing"]
        _ = pipe.predict(df_smoke[feats])
        _ = prep.transform(df_smoke[feats])
        _ = prep.get_feature_names_out()

    for clave_log, clave_feats in (("log_socio", "features_log_socio"), ("log_acad", "features_log_acad")):
        pipe = artefactos[clave_log]
        feats = artefactos[clave_feats]
        _ = pipe.predict_proba(df_smoke[feats])


def entrenar_y_guardar_modelos_finales(df: pd.DataFrame | None = None) -> dict:
    """Entrena los pipelines finales sobre toda la base depurada y los guarda en `models/`."""
    if df is None:
        df, _ = datos.cargar_datos()

    RUTA_MODELOS.mkdir(parents=True, exist_ok=True)

    # Evaluación completa de los dos enfoques para guardar métricas y diagnósticos
    reg_socio = entrenar_evaluar_regresion(
        df, datos.CAT_COLS_SOCIO, datos.NUM_COLS_SOCIO, target="punt_global"
    )
    reg_acad = entrenar_evaluar_regresion(
        df, datos.CAT_COLS_ACAD, datos.NUM_COLS_ACAD, target="punt_global"
    )
    log_socio = entrenar_evaluar_logistica(
        df, datos.CAT_COLS_SOCIO, datos.NUM_COLS_SOCIO, target="supera_promedio"
    )
    log_acad = entrenar_evaluar_logistica(
        df, datos.CAT_COLS_ACAD, datos.NUM_COLS_ACAD, target="supera_promedio"
    )

    # Ajustar modelos finales con el 100 % de los datos para producción en Streamlit
    _ajustar_modelos_produccion(
        df,
        log_socio["best_params"],
        log_acad["best_params"],
        guardar_en_disco=True,
        ignorar_error_disco=False,
    )

    # Guardar resumen JSON para carga instantánea en el dashboard de Streamlit
    resumen_json = {
        "regresion_socio": {
            "metricas_train": reg_socio["metricas_train"],
            "metricas_test": reg_socio["metricas_test"],
            "tabla_cv": reg_socio["tabla_cv"].reset_index().to_dict(orient="records"),
            "importance": reg_socio["importance_df"].to_dict(orient="records"),
            "ols_params": reg_socio["ols"]["tabla_params"].reset_index().to_dict(orient="records"),
            "vif": reg_socio["ols"]["vif_df"].reset_index().to_dict(orient="records"),
            "jarque_bera": reg_socio["ols"]["jarque_bera"],
            "breusch_pagan": reg_socio["ols"]["breusch_pagan"],
            "F_global": reg_socio["ols"]["F_global"],
            "p_F_global": reg_socio["ols"]["p_F_global"],
            "cond_number": reg_socio["ols"]["cond_number"],
            "intercepto": reg_socio["intercepto"],
        },
        "regresion_acad": {
            "metricas_train": reg_acad["metricas_train"],
            "metricas_test": reg_acad["metricas_test"],
            "tabla_cv": reg_acad["tabla_cv"].reset_index().to_dict(orient="records"),
            "importance": reg_acad["importance_df"].to_dict(orient="records"),
            "ols_params": reg_acad["ols"]["tabla_params"].reset_index().to_dict(orient="records"),
            "vif": reg_acad["ols"]["vif_df"].reset_index().to_dict(orient="records"),
            "jarque_bera": reg_acad["ols"]["jarque_bera"],
            "breusch_pagan": reg_acad["ols"]["breusch_pagan"],
            "F_global": reg_acad["ols"]["F_global"],
            "p_F_global": reg_acad["ols"]["p_F_global"],
            "cond_number": reg_acad["ols"]["cond_number"],
            "intercepto": reg_acad["intercepto"],
        },
        "logistica_socio": {
            "metricas_base": log_socio["metricas_base"],
            "metricas_best": log_socio["metricas_best"],
            "best_params": log_socio["best_params"],
            "tabla_cv": log_socio["tabla_cv"].reset_index().to_dict(orient="records"),
            "importance": log_socio["importance_df"].to_dict(orient="records"),
            "cm_base": log_socio["cm_base"].tolist(),
            "cm_best": log_socio["cm_best"].tolist(),
            "fpr_best": [float(x) for x in log_socio["fpr_best"][::max(1, len(log_socio["fpr_best"]) // 250)]],
            "tpr_best": [float(x) for x in log_socio["tpr_best"][::max(1, len(log_socio["tpr_best"]) // 250)]],
            "intercepto": log_socio["intercepto"],
        },
        "logistica_acad": {
            "metricas_base": log_acad["metricas_base"],
            "metricas_best": log_acad["metricas_best"],
            "best_params": log_acad["best_params"],
            "tabla_cv": log_acad["tabla_cv"].reset_index().to_dict(orient="records"),
            "importance": log_acad["importance_df"].to_dict(orient="records"),
            "cm_base": log_acad["cm_base"].tolist(),
            "cm_best": log_acad["cm_best"].tolist(),
            "fpr_best": [float(x) for x in log_acad["fpr_best"][::max(1, len(log_acad["fpr_best"]) // 250)]],
            "tpr_best": [float(x) for x in log_acad["tpr_best"][::max(1, len(log_acad["tpr_best"]) // 250)]],
            "intercepto": log_acad["intercepto"],
        },
    }
    with open(RUTA_MODELOS / "metricas_resumen.json", "w", encoding="utf-8") as f:
        json.dump(resumen_json, f, ensure_ascii=False, indent=2)

    return {
        "reg_socio": reg_socio,
        "reg_acad": reg_acad,
        "log_socio": log_socio,
        "log_acad": log_acad,
        "resumen_json": resumen_json,
    }


def _leer_artefactos_joblib() -> dict:
    """Carga los 4 pipelines `.joblib` y sus listas de variables desde `models/`."""
    return {
        "reg_socio": joblib.load(RUTA_MODELOS / "model_linear_regression.joblib"),
        "features_reg_socio": joblib.load(RUTA_MODELOS / "features_linear_regression.joblib"),
        "reg_acad": joblib.load(RUTA_MODELOS / "model_linear_academico.joblib"),
        "features_reg_acad": joblib.load(RUTA_MODELOS / "features_linear_academico.joblib"),
        "log_socio": joblib.load(RUTA_MODELOS / "model_logistic_regression.joblib"),
        "features_log_socio": joblib.load(RUTA_MODELOS / "features_logistic_regression.joblib"),
        "log_acad": joblib.load(RUTA_MODELOS / "model_logistic_academico.joblib"),
        "features_log_acad": joblib.load(RUTA_MODELOS / "features_logistic_academico.joblib"),
    }


def cargar_modelos_guardados() -> dict:
    """Carga los pipelines `.joblib` y el resumen de métricas desde `models/`.

    Si los `.joblib` faltan o fueron serializados con otra versión de scikit-learn
    (p. ej. `SimpleImputer._fill_dtype` en Streamlit Cloud con Python 3.14 / sklearn 1.8+),
    reajusta automáticamente los 4 pipelines finales en la versión activa (~1 s)
    preservando los hiperparámetros óptimos de `metricas_resumen.json`.
    """
    ruta_resumen = RUTA_MODELOS / "metricas_resumen.json"
    if not ruta_resumen.exists():
        entrenar_y_guardar_modelos_finales()

    with open(ruta_resumen, encoding="utf-8") as f:
        resumen = json.load(f)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            artefactos = _leer_artefactos_joblib()
        _validar_artefactos_predictivos(artefactos)
    except Exception:
        df_dep, _ = datos.cargar_datos()
        artefactos = _ajustar_modelos_produccion(
            df_dep,
            resumen["logistica_socio"]["best_params"],
            resumen["logistica_acad"]["best_params"],
            guardar_en_disco=True,
            ignorar_error_disco=True,
        )
        _validar_artefactos_predictivos(artefactos)

    artefactos["resumen"] = resumen
    artefactos["sklearn_version"] = sklearn.__version__
    return artefactos


if __name__ == "__main__":
    res = entrenar_y_guardar_modelos_finales()
    print("Modelos entrenados y guardados en:", RUTA_MODELOS)

