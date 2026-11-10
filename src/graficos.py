"""Visualizaciones estáticas (Matplotlib/Seaborn) e interactivas (Plotly) para el Laboratorio 3.

Combina la identidad visual UPTC (amarillo #FFC000 y negro #222222) de los Laboratorios 1 y 2
con los gráficos analíticos y de diagnóstico de modelos de los Notebooks 6, 8 y 9 del curso.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from scipy import stats
from scipy.stats import gaussian_kde

from .datos import orden_categorias
from .estadistica import limites_iqr

AMARILLO, NEGRO, GRIS, ROJO, AZUL = "#FFC000", "#222222", "#8C8C8C", "#C0392B", "#1F618D"
sns.set_theme(style="whitegrid")


# ===========================================================================
# 1. VISUALIZACIONES EDA Y PRUEBAS DE HIPÓTESIS (MATPLOTLIB / SEABORN)
# ===========================================================================
def hist_box(df: pd.DataFrame, col: str, bins: int = 50):
    """Histograma (con KDE y normal teórica) + boxplot con límites 1,5·IQR."""
    s = df[col].dropna()
    q1, q3, iqr, li, ls = limites_iqr(s)
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [2, 1.2]})
    sns.histplot(s, bins=bins, stat="density", color=AMARILLO, edgecolor="white", ax=ax[0])
    sns.kdeplot(s, color=NEGRO, ax=ax[0], label="KDE")
    xs = np.linspace(s.min(), s.max(), 300)
    ax[0].plot(xs, stats.norm.pdf(xs, s.mean(), s.std()), "--", color=ROJO, lw=1.2, label="Normal teórica")
    ax[0].axvline(s.mean(), color=ROJO, lw=1.1, label=f"Media {s.mean():.2f}")
    ax[0].axvline(s.median(), color=AZUL, lw=1.1, ls=":", label=f"Mediana {s.median():.2f}")
    ax[0].set_title(f"Distribución e Histograma de {col}")
    ax[0].legend(fontsize=8)
    sns.boxplot(x=s, color=AMARILLO, ax=ax[1], flierprops=dict(marker="o", markersize=3, markerfacecolor=GRIS))
    ax[1].axvline(li, color=ROJO, ls="--", lw=0.9)
    ax[1].axvline(ls, color=ROJO, ls="--", lw=0.9)
    ax[1].set_title(f"Boxplot (límites IQR: {li:.1f} y {ls:.1f})")
    fig.tight_layout()
    return fig


def grilla_hist(df: pd.DataFrame, cols: list[str], ncols: int = 3, bins: int = 40):
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.2 * nrows))
    for ax, c in zip(np.ravel(axs), cols):
        sns.histplot(df[c].dropna(), bins=bins, color=AMARILLO, edgecolor="white", kde=True, ax=ax)
        ax.set_title(c, fontsize=10, fontweight="bold")
    for ax in np.ravel(axs)[len(cols):]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def grilla_box(df: pd.DataFrame, cols: list[str], ncols: int = 3):
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 2.4 * nrows))
    for ax, c in zip(np.ravel(axs), cols):
        sns.boxplot(
            x=df[c].dropna(),
            color=AMARILLO,
            ax=ax,
            flierprops=dict(marker="o", markersize=2, markerfacecolor=GRIS),
        )
        ax.set_title(c, fontsize=10, fontweight="bold")
    for ax in np.ravel(axs)[len(cols):]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def barras(df: pd.DataFrame, col: str, ax=None):
    """Gráfico de barras (frecuencia relativa %) de una variable cualitativa."""
    f = df[col].astype(str).value_counts(normalize=True) * 100.0
    f = f.reindex(orden_categorias(col, f.index))
    if ax is None:
        fig, ax = plt.subplots(figsize=(8.5, 0.45 * len(f) + 1.6))
    else:
        fig = ax.figure
    sns.barplot(x=f.values, y=f.index, color=AMARILLO, edgecolor=NEGRO, ax=ax)
    for i, v in enumerate(f.values):
        ax.text(v + 0.3, i, f"{v:.2f} %", va="center", fontsize=8)
    ax.set_xlim(0, f.max() * 1.15)
    ax.set_xlabel("% de estudiantes")
    ax.set_ylabel("")
    ax.set_title(f"Distribución porcentual de {col}")
    fig.tight_layout()
    return fig


def scatter_par(df: pd.DataFrame, x: str, y: str, n_max: int = 8000, semilla: int = 42):
    """Dispersión con recta MCO, media por intervalos y mapa hexbin de densidad."""
    d = df[[x, y]].dropna()
    m = d.sample(min(n_max, len(d)), random_state=semilla)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    ax[0].scatter(m[x], m[y], s=6, alpha=0.25, color=NEGRO, rasterized=True)
    coef = np.polyfit(d[x], d[y], 1)
    xs = np.linspace(d[x].min(), d[x].max(), 100)
    ax[0].plot(xs, np.polyval(coef, xs), color=ROJO, lw=2, label="Recta MCO")
    bins = pd.qcut(d[x], 20, duplicates="drop")
    med = d.groupby(bins, observed=True).agg(xm=(x, "mean"), ym=(y, "mean"))
    ax[0].plot(med.xm, med.ym, color=AMARILLO, lw=2.5, marker="o", ms=4, label="Media por vintiles")
    ax[0].set_xlabel(x)
    ax[0].set_ylabel(y)
    ax[0].set_title(f"Dispersión: {x} vs {y} (muestra n={len(m):,})")
    ax[0].legend(fontsize=8)
    hb = ax[1].hexbin(d[x], d[y], gridsize=45, cmap="YlOrBr", mincnt=1, bins="log")
    fig.colorbar(hb, ax=ax[1], label="log10(conteo)")
    ax[1].set_xlabel(x)
    ax[1].set_ylabel(y)
    ax[1].set_title(f"Densidad Conjunta Hexbin (n = {len(d):,})")
    fig.tight_layout()
    return fig


def heatmap_corr(df: pd.DataFrame, cols: list[str], metodo: str = "pearson", ax=None):
    c = df[cols].corr(method=metodo)
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.2, 5.5))
    else:
        fig = ax.figure
    sns.heatmap(c, annot=True, fmt=".4f", cmap="YlOrBr", vmin=0, vmax=1, ax=ax, cbar=True, annot_kws={"size": 8})
    ax.set_title(f"Matriz de Correlación ({metodo.capitalize()})")
    fig.tight_layout()
    return fig


def heatmap_residuos(res: dict):
    fig, ax = plt.subplots(figsize=(1.1 * res["residuos"].shape[1] + 4, 0.55 * res["residuos"].shape[0] + 2.5))
    sns.heatmap(res["residuos"], annot=True, fmt=".1f", cmap="RdBu_r", center=0, ax=ax, annot_kws={"size": 8})
    ax.set_title(
        f"Residuos estandarizados ajustados: {res['a']} × {res['b']}\n"
        "(|z| > 1.96 indica desviación estadísticamente significativa frente a independencia)"
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    return fig


def boxplot_grupos(df: pd.DataFrame, cat: str, y: str):
    d = df[[cat, y]].dropna()
    orden = orden_categorias(cat, d[cat].unique())
    fig, ax = plt.subplots(figsize=(9.5, 0.5 * len(orden) + 2))
    sns.boxplot(
        data=d,
        y=cat,
        x=y,
        order=orden,
        color=AMARILLO,
        ax=ax,
        flierprops=dict(marker="o", markersize=2, markerfacecolor=GRIS),
    )
    ax.axvline(d[y].mean(), color=ROJO, ls="--", lw=1.1, label=f"Media global ({d[y].mean():.1f})")
    ax.set_title(f"Distribución de {y} según {cat}")
    ax.set_ylabel("")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def qq_grupos(res: dict, max_grupos: int = 8):
    g = list(res["grupos"].items())[:max_grupos]
    n = len(g)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3 * nrows))
    for ax, (nombre, a) in zip(np.ravel(axs), g):
        stats.probplot(a, dist="norm", plot=ax)
        ax.get_lines()[0].set_markerfacecolor(AMARILLO)
        ax.get_lines()[0].set_markeredgecolor(NEGRO)
        ax.get_lines()[0].set_markersize(2)
        ax.set_title(nombre[:28], fontsize=8)
        ax.set_xlabel("")
        ax.set_ylabel("")
    for ax in np.ravel(axs)[n:]:
        ax.axis("off")
    fig.suptitle("Gráficos Q-Q por grupo (alineación con la recta roja ⇒ normalidad)", y=1.01)
    fig.tight_layout()
    return fig


def barras_ranking(rank: pd.DataFrame, col: str = "epsilon2", titulo: str = ""):
    r = rank.sort_values(col)
    fig, ax = plt.subplots(figsize=(8.5, 0.5 * len(r) + 1.6))
    sns.barplot(x=r[col], y=r.index, color=AMARILLO, edgecolor=NEGRO, ax=ax)
    for i, v in enumerate(r[col].values):
        ax.text(v + 0.002, i, f"{v:.4f}", va="center", fontsize=8)
    ax.set_xlim(0, r[col].max() * 1.18)
    ax.set_xlabel(col)
    ax.set_ylabel("")
    ax.set_title(titulo or f"Fuerza de asociación ({col})")
    fig.tight_layout()
    return fig


# ===========================================================================
# 2. GRÁFICOS DE IMPUTACIÓN, REGRESIÓN LINEAL Y LOGÍSTICA (MATPLOTLIB)
# ===========================================================================
def fig_imputacion_comparativa(res_imp: dict, col: str = "punt_ingles"):
    """Compara el error (RMSE/MAE) de los métodos de imputación y sus densidades KDE (Notebook 6)."""
    df_res = res_imp["resultados"]
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.2))

    # Panel izquierdo: Barras agrupadas RMSE y MAE
    x = np.arange(len(df_res))
    w = 0.36
    b1 = ax[0].bar(x - w / 2, df_res["RMSE"], width=w, color=NEGRO, label="RMSE")
    b2 = ax[0].bar(x + w / 2, df_res["MAE"], width=w, color=AMARILLO, edgecolor=NEGRO, label="MAE")
    for b in b1:
        ax[0].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.15, f"{b.get_height():.2f}", ha="center", fontsize=8)
    for b in b2:
        ax[0].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.15, f"{b.get_height():.2f}", ha="center", fontsize=8)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(df_res.index, rotation=15, ha="right", fontsize=8.5)
    ax[0].set_ylabel("Error en puntos (escala 0-100)")
    ax[0].set_title("Error de Reconstrucción por Método de Imputación")
    ax[0].legend()

    # Panel derecho: Curvas KDE sobre la variable seleccionada
    orig = res_imp["original"][col].to_numpy()
    xs = np.linspace(orig.min(), orig.max(), 300)
    ax[1].plot(xs, gaussian_kde(orig)(xs), color=NEGRO, lw=2.5, label="Original (sin vacíos)")
    estilos = [
        ("Media (SimpleImputer)", ROJO, ":"),
        ("KNN (k=5)", AZUL, "--"),
        ("Iterativa (MICE)", "#27AE60", "-."),
    ]
    for nombre, color, ls in estilos:
        vals = res_imp["imputados"][nombre][col].to_numpy()
        ax[1].plot(xs, gaussian_kde(vals)(xs), color=color, ls=ls, lw=1.8, label=nombre)
    ax[1].set_xlabel(col)
    ax[1].set_ylabel("Densidad KDE")
    ax[1].set_title(f"Preservación de la Distribución KDE ({col})")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    return fig


def fig_evaluacion_regresion(df_resultados: pd.DataFrame, titulo_modelo: str = "Regresión Lineal"):
    """Valores Reales vs Predichos + Distribución de Errores con estadísticos de forma (Notebook 8)."""
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.4))

    muestra = df_resultados.sample(min(4000, len(df_resultados)), random_state=42)
    r, p = stats.pearsonr(df_resultados["y_real"], df_resultados["y_pred"])
    ax[0].scatter(muestra["y_real"], muestra["y_pred"], s=8, alpha=0.25, color=NEGRO, label="Observaciones (test)")
    lims = [
        min(df_resultados["y_real"].min(), df_resultados["y_pred"].min()),
        max(df_resultados["y_real"].max(), df_resultados["y_pred"].max()),
    ]
    ax[0].plot(lims, lims, "--", color=ROJO, lw=2, label="Predicción Perfecta (y = ŷ)")
    coef = np.polyfit(df_resultados["y_real"], df_resultados["y_pred"], 1)
    xs = np.linspace(lims[0], lims[1], 100)
    ax[0].plot(xs, np.polyval(coef, xs), color=AMARILLO, lw=2.2, label="Tendencia MCO")
    ax[0].set_xlabel("Puntaje Global Real (y)")
    ax[0].set_ylabel("Puntaje Global Predicho (ŷ)")
    ax[0].set_title(f"{titulo_modelo}: Reales vs Predichos (r = {r:.4f})")
    ax[0].legend(fontsize=8)

    err = df_resultados["error"]
    sns.histplot(err, bins=40, stat="density", color=AMARILLO, edgecolor="white", ax=ax[1])
    sns.kdeplot(err, color=NEGRO, lw=1.8, ax=ax[1], label="Densidad real")
    xs_e = np.linspace(err.min(), err.max(), 300)
    ax[1].plot(xs_e, stats.norm.pdf(xs_e, err.mean(), err.std()), "--", color=ROJO, lw=1.4, label="Normal teórica")
    ax[1].axvline(0, color=NEGRO, ls=":", lw=1.2)
    ax[1].set_xlabel("Error Residual (e = y - ŷ)")
    ax[1].set_ylabel("Densidad")
    ax[1].set_title(
        f"Distribución de Errores\nMedia={err.mean():.2f} | SD={err.std():.2f} | "
        f"Asim={stats.skew(err):.2f} | Ex.Curt={stats.kurtosis(err):.2f}"
    )
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    return fig


def fig_cv_regresion(cv_df: pd.DataFrame):
    """Subplots 2x2 con las densidades KDE de MAE, RMSE, MAPE y R² en Validación Cruzada (Notebook 8)."""
    fig, axs = plt.subplots(2, 2, figsize=(11, 6.2))
    metricas = [("MAE", "MAE (puntos)"), ("RMSE", "RMSE (puntos)"), ("MAPE", "MAPE (proporción)"), ("R2", "R² (varianza explicada)")]
    for ax, (col, etiq) in zip(np.ravel(axs), metricas):
        vals = cv_df[col].to_numpy()
        xs = np.linspace(vals.min() - 0.2 * vals.std(), vals.max() + 0.2 * vals.std(), 200)
        kde = gaussian_kde(vals)(xs)
        ax.plot(xs, kde, color=NEGRO, lw=2)
        ax.fill_between(xs, kde, color=AMARILLO, alpha=0.45)
        ax.axvline(vals.mean(), color=ROJO, ls="--", lw=1.5, label=f"Media = {vals.mean():.4f}")
        ax.scatter(vals, np.zeros_like(vals), color=NEGRO, zorder=5, s=20, label="10 Folds")
        ax.set_title(f"Distribución CV: {etiq}", fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def fig_importancia_coeficientes(importance_df: pd.DataFrame, col_coef: str = "Coeficiente", titulo: str = ""):
    """Gráfico horizontal de importancia de variables según magnitud y signo del coeficiente."""
    d = importance_df.sort_values("Importancia", ascending=True).copy()
    colores = [AMARILLO if c >= 0 else "#566573" for c in d[col_coef]]
    fig, ax = plt.subplots(figsize=(9.5, 0.48 * len(d) + 1.8))
    bars = ax.barh(d["Etiqueta"], d[col_coef], color=colores, edgecolor=NEGRO)
    ax.axvline(0, color=NEGRO, ls="--", lw=1.2)
    rango = d[col_coef].max() - d[col_coef].min()
    offset = 0.02 * (rango if rango > 0 else 1.0)
    for bar, val in zip(bars, d[col_coef]):
        pos_x = val + offset if val >= 0 else val - offset
        align = "left" if val >= 0 else "right"
        ax.text(pos_x, bar.get_y() + bar.get_height() / 2, f"{val:+.4f}", va="center", ha=align, fontsize=8)
    ax.set_xlim(d[col_coef].min() - 0.22 * abs(rango), d[col_coef].max() + 0.22 * abs(rango))
    ax.set_xlabel("Valor del Coeficiente Estimado")
    ax.set_title(titulo or "Importancia y Dirección de Efecto de las Variables (Coeficientes)")
    fig.tight_layout()
    return fig


def fig_diagnostico_residuos_ols(residuals: pd.Series, fitted: pd.Series):
    """Diagnóstico econométrico MCO: Residuos vs Ajustados y Gráfico Q-Q Normal (Notebook 8)."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    idx = np.random.default_rng(42).choice(len(residuals), size=min(4000, len(residuals)), replace=False)
    f_sub = fitted.iloc[idx].to_numpy()
    r_sub = residuals.iloc[idx].to_numpy()

    ax[0].scatter(f_sub, r_sub, s=7, alpha=0.25, color=NEGRO)
    ax[0].axhline(0, color=ROJO, ls="--", lw=1.5, label="Residuo = 0")
    orden = np.argsort(f_sub)
    tendencia = np.polyval(np.polyfit(f_sub, r_sub, 2), f_sub[orden])
    ax[0].plot(f_sub[orden], tendencia, color=AMARILLO, lw=2.2, label="Tendencia polinomial")
    ax[0].set_xlabel("Valores Ajustados (ŷ)")
    ax[0].set_ylabel("Residuos MCO (e)")
    ax[0].set_title("Homocedasticidad y Linealidad: Residuos vs Ajustados")
    ax[0].legend(fontsize=8)

    stats.probplot(r_sub, dist="norm", plot=ax[1])
    ax[1].get_lines()[0].set_markerfacecolor(AMARILLO)
    ax[1].get_lines()[0].set_markeredgecolor(NEGRO)
    ax[1].get_lines()[0].set_markersize(3)
    ax[1].set_title("Gráfico Q-Q Normal de los Residuos")
    fig.tight_layout()
    return fig


def fig_evaluacion_logistica(cm: np.ndarray, fpr: np.ndarray, tpr: np.ndarray, auc: float, titulo: str = ""):
    """Matriz de Confusión y Curva ROC para Regresión Logística (Notebook 9)."""
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3))
    etiquetas = ["Bajo (<250)", "Supera (>=250)"]
    sns.heatmap(
        cm,
        annot=True,
        fmt=",d",
        cmap="YlOrBr",
        xticklabels=[f"Pred {e}" for e in etiquetas],
        yticklabels=[f"Real {e}" for e in etiquetas],
        ax=ax[0],
        annot_kws={"size": 11, "weight": "bold"},
    )
    ax[0].set_title(f"Matriz de Confusión {titulo}".strip())

    ax[1].plot(fpr, tpr, color=NEGRO, lw=2.5, label=f"Regresión Logística (AUC = {auc:.4f})")
    ax[1].fill_between(fpr, tpr, color=AMARILLO, alpha=0.30)
    ax[1].plot([0, 1], [0, 1], "--", color=ROJO, lw=1.5, label="Clasificador Aleatorio (AUC = 0.5000)")
    ax[1].set_xlabel("Tasa de Falsos Positivos (1 - Especificidad)")
    ax[1].set_ylabel("Tasa de Verdaderos Positivos (Sensibilidad / Recall)")
    ax[1].set_title(f"Curva ROC {titulo}".strip())
    ax[1].legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()
    return fig


# ===========================================================================
# 3. VISUALIZACIONES INTERACTIVAS CON PLOTLY (PARA STREAMLIT Y NOTEBOOK)
# ===========================================================================
def plotly_distribucion_interactiva(df: pd.DataFrame, col: str, color_col: str | None = None):
    """Histograma interactivo con boxplot marginal (estilo Notebooks 8 y estrato_eda.py)."""
    fig = px.histogram(
        df,
        x=col,
        color=color_col,
        nbins=45,
        marginal="box",
        opacity=0.82,
        color_discrete_sequence=[AMARILLO, NEGRO, ROJO, AZUL, "#27AE60", "#8E44AD"],
        title=f"Distribución Interactiva de {col}" + (f" por {color_col}" if color_col else ""),
    )
    fig.update_layout(template="plotly_white", height=480)
    return fig


def plotly_boxplot_interactivo(df: pd.DataFrame, cat: str, y: str):
    """Boxplot interactivo ordenado por jerarquía de categorías."""
    orden = orden_categorias(cat, df[cat].unique())
    fig = px.box(
        df,
        x=y,
        y=cat,
        category_orders={cat: orden},
        color_discrete_sequence=[AMARILLO],
        title=f"Distribución de {y} según {cat}",
    )
    fig.add_vline(
        x=float(df[y].mean()),
        line_dash="dash",
        line_color=ROJO,
        annotation_text=f"Media = {df[y].mean():.2f}",
    )
    fig.update_layout(template="plotly_white", height=max(420, 42 * len(orden)))
    return fig


def plotly_importancia_coeficientes(importance_df: pd.DataFrame, col_coef: str = "Coeficiente", titulo: str = ""):
    """Gráfico de barras horizontal interactivo de importancia de variables (Notebooks 8 y 9)."""
    d = importance_df.sort_values("Importancia", ascending=True).copy()
    fig = px.bar(
        d,
        x=col_coef,
        y="Etiqueta",
        color="Signo",
        orientation="h",
        text=col_coef,
        color_discrete_map={
            "Positivo": AMARILLO,
            "Negativo": "#566573",
            "Positivo (Aumenta P(Y=1))": AMARILLO,
            "Negativo (Reduce P(Y=1))": "#566573",
        },
        title=titulo or "Importancia de Variables mediante Coeficientes",
    )
    fig.add_vline(x=0, line_width=2, line_dash="dash", line_color=NEGRO)
    fig.update_traces(texttemplate="%{text:+.4f}", textposition="outside")
    fig.update_layout(template="plotly_white", height=max(420, 48 * len(d)), yaxis_title="", legend_title="Efecto")
    return fig


def plotly_curva_roc_matriz(cm: list[list[int]], fpr: list[float], tpr: list[float], auc: float):
    """Subplots interactivos en Plotly: Matriz de Confusión y Curva ROC (Notebook 9)."""
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Matriz de Confusión (Conjunto de Prueba)", f"Curva ROC (AUC = {auc:.4f})"),
        horizontal_spacing=0.14,
    )
    etiquetas = ["Bajo (<250)", "Supera (>=250)"]
    z_text = [[f"{val:,}" for val in fila] for fila in cm]
    fig.add_trace(
        go.Heatmap(
            z=cm,
            x=[f"Pred {e}" for e in etiquetas],
            y=[f"Real {e}" for e in etiquetas],
            text=z_text,
            texttemplate="%{text}",
            colorscale="YlOrBr",
            showscale=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"ROC (AUC={auc:.4f})",
            line=dict(color=NEGRO, width=3),
            fill="tozeroy",
            fillcolor="rgba(255, 192, 0, 0.25)",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Aleatorio (0.50)",
            line=dict(color=ROJO, dash="dash", width=2),
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text="Predicción", row=1, col=1)
    fig.update_yaxes(title_text="Valor Real", autorange="reversed", row=1, col=1)
    fig.update_xaxes(title_text="Tasa de Falsos Positivos (FPR)", row=1, col=2)
    fig.update_yaxes(title_text="Tasa de Verdaderos Positivos (TPR)", row=1, col=2)
    fig.update_layout(template="plotly_white", height=440, showlegend=True)
    return fig
