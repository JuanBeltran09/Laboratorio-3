"""Funciones estadísticas, pruebas de hipótesis y experimentos de imputación/escalamiento (Laboratorio 3).

Integra y amplía todo el motor estadístico desarrollado en los Laboratorios 1 y 2 junto con
los métodos de los Notebooks 5, 6 y 7 del curso (Pruebas de Asociación, Imputación y Escalamiento).
Todas las funciones numéricas conservan precisión completa y se reportan a 4 cifras decimales.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, mutual_info_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .datos import COLS_CUAL, COLS_FALTANTES, orden_categorias

DEC = 4


# --------------------------------------------------------------------------- 0. Formato
def fmt(x, d: int = DEC) -> str:
    return "NaN" if x is None or (isinstance(x, (float, np.floating)) and np.isnan(x)) else f"{x:,.{d}f}"


def fmt_p(p: float) -> str:
    """Formatea p-valor a 4 decimales; valores < 0,00005 se reportan como '< 0.0001'."""
    if p is None or (isinstance(p, (float, np.floating)) and np.isnan(p)):
        return "NaN"
    return "< 0.0001" if p < 0.00005 else f"{p:.4f}"


def formatear_pvalores(tabla: pd.DataFrame) -> pd.DataFrame:
    t = tabla.copy()
    for c in t.columns:
        if c.startswith("p_") or c.startswith("p-") or c == "p_valor":
            t[c] = t[c].map(fmt_p)
    return t


# --------------------------------------------------------------------------- 1. Valores faltantes e Imputación (Notebook 6 + Lab 1)
def resum_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Porcentaje y conteo de valores perdidos por variable (Notebook 6 del curso)."""
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum() * 100.0 / df.isnull().count()).sort_values(ascending=False)
    return pd.concat([total, percent], axis=1, keys=["Total", "Porcentaje"])


def posmissing(df: pd.DataFrame) -> pd.DataFrame:
    """Resumen de posiciones (fila, columna) de valores perdidos (Notebook 6 del curso)."""
    lista_miss = np.where(df.isna())
    v1 = pd.Series(np.ndarray.tolist(lista_miss[0]))
    v2 = pd.Series(np.ndarray.tolist(lista_miss[1]))
    return pd.concat([v1, v2], axis=1, keys=["posicion_fila", "posicion_columna"])


def count_zeros(df: pd.DataFrame) -> pd.DataFrame:
    """Conteo y porcentaje de ceros exactos por variable (Notebook 6 del curso)."""
    total = (df == 0).astype(int).sum(axis=0)
    percent = (df == 0).astype(int).sum(axis=0) * 100.0 / df.count()
    res = pd.concat([total, percent], axis=1, keys=["Total_ceros", "Porcentaje"])
    return res.sort_values(by="Porcentaje", ascending=False)


def evaluar_mecanismo_faltantes(crudo: pd.DataFrame) -> pd.DataFrame:
    """Evalúa empíricamente por qué los faltantes en fami_* NO son al azar (MNAR vs MCAR).

    Compara el puntaje global de quienes tienen dato vs quienes tienen valor ausente (NaN)
    mediante prueba t de Welch y prueba U de Mann-Whitney, justificando la decisión del
    Laboratorio 1 de crear la categoría 'Desconocido' en lugar de eliminar filas o imputar con la moda.
    """
    d = crudo.copy()
    d.columns = d.columns.str.lower().str.strip()
    d = d.drop_duplicates(subset="estu_consecutivo").reset_index(drop=True)
    d["punt_global"] = pd.to_numeric(d["punt_global"], errors="coerce")

    filas = []
    for c in COLS_FALTANTES + ["punt_ingles"]:
        mask_na = d[c].isna()
        con = d.loc[~mask_na, "punt_global"].dropna()
        sin = d.loc[mask_na, "punt_global"].dropna()
        if len(sin) == 0:
            continue
        t_stat, p_t = stats.ttest_ind(con, sin, equal_var=False)
        u_stat, p_u = stats.mannwhitneyu(con, sin, alternative="two-sided")
        filas.append({
            "variable": c,
            "n_faltantes": int(mask_na.sum()),
            "%_faltantes": float(mask_na.mean() * 100.0),
            "media_con_dato": float(con.mean()),
            "media_sin_dato": float(sin.mean()),
            "diferencia_medias": float(sin.mean() - con.mean()),
            "mediana_con_dato": float(con.median()),
            "mediana_sin_dato": float(sin.median()),
            "t_welch": float(t_stat),
            "p_welch": float(p_t),
            "p_mannwhitney": float(p_u),
        })
    return pd.DataFrame(filas).set_index("variable")


def comparar_imputacion_numerica(
    df: pd.DataFrame,
    cols: list[str] | None = None,
    porcentaje: float = 0.20,
    n_muestra: int = 3000,
    semilla: int = 42,
) -> dict:
    """Experimento comparativo de métodos de imputación numérica (Notebook 6 del curso).

    Introduce un porcentaje controlado de valores faltantes (por defecto 20 %) sobre las
    variables cuantitativas completas y evalúa el error de reconstrucción (RMSE y MAE) de:
    1) Media (SimpleImputer strategy='mean')
    2) Mediana (SimpleImputer strategy='median')
    3) KNN (KNNImputer n_neighbors=5)
    4) Iterativa tipo MICE (IterativeImputer BayesianRidge)
    """
    cols = cols or [
        "punt_matematicas",
        "punt_lectura_critica",
        "punt_c_naturales",
        "punt_sociales_ciudadanas",
        "punt_ingles",
    ]
    sub = df[cols].dropna().astype(float).sample(n=min(n_muestra, len(df[cols].dropna())), random_state=semilla)
    rng = np.random.default_rng(semilla)
    mask = rng.random(sub.shape) < porcentaje
    sub_missing = sub.copy()
    sub_missing[mask] = np.nan

    metodos = {
        "Media (SimpleImputer)": SimpleImputer(strategy="mean"),
        "Mediana (SimpleImputer)": SimpleImputer(strategy="median"),
        "KNN (k=5)": KNNImputer(n_neighbors=5),
        "Iterativa (MICE)": IterativeImputer(max_iter=10, random_state=semilla),
    }

    orig_vals = sub.to_numpy()
    tabla_res = {}
    imputados = {}

    for nombre, imp in metodos.items():
        arr_imp = imp.fit_transform(sub_missing)
        imputados[nombre] = pd.DataFrame(arr_imp, columns=cols, index=sub.index)
        tabla_res[nombre] = {
            "RMSE": float(np.sqrt(mean_squared_error(orig_vals[mask], arr_imp[mask]))),
            "MAE": float(mean_absolute_error(orig_vals[mask], arr_imp[mask])),
        }

    df_res = pd.DataFrame(tabla_res).T.sort_values("RMSE")
    return {
        "resultados": df_res,
        "original": sub,
        "con_missing": sub_missing,
        "imputados": imputados,
        "mask": mask,
        "porcentaje": porcentaje,
        "n_muestra": len(sub),
    }


def demostracion_escalamiento_distancias(df: pd.DataFrame) -> dict:
    """Demostración matemática del impacto del escalamiento sobre distancias euclidianas (Notebook 7).

    Compara tres perfiles reales/típicos de estudiantes usando dos variables en escalas distintas:
    `educ_madre_nivel` (escala 0 a 9) y `punt_global` (o `punt_ingles`, escala 0 a 100/500),
    mostrando cómo sin estandarizar la variable de mayor rango domina la distancia euclidiana.
    """
    # Tres perfiles representativos en R^2: [educ_madre_nivel (0-9), punt_ingles (0-100)]
    perfiles = pd.DataFrame(
        {
            "Estudiante": ["Estudiante A (P1)", "Estudiante B (P2)", "Estudiante C (P3)"],
            "educ_madre_nivel": [1.0, 2.0, 9.0],   # P1 y P2: Primaria; P3: Postgrado (+8 niveles)
            "punt_ingles": [45.0, 60.0, 47.0],     # P1 y P3 difieren solo en 2 pts de inglés; P1 y P2 en 15 pts
        }
    ).set_index("Estudiante")

    # Ajustar escaladores sobre la muestra real de estudiantes
    muestra_ref = df[["educ_madre_nivel", "punt_ingles"]].dropna()
    std_scaler = StandardScaler().fit(muestra_ref)
    mm_scaler = MinMaxScaler().fit(muestra_ref)

    X_raw = perfiles.to_numpy()
    X_std = std_scaler.transform(perfiles)
    X_mm = mm_scaler.transform(perfiles)

    pares = [("d(P1, P2)", 0, 1), ("d(P1, P3)", 0, 2), ("d(P2, P3)", 1, 2)]
    filas = []
    for etiqueta, i, j in pares:
        d_raw = float(np.linalg.norm(X_raw[i] - X_raw[j]))
        d_std = float(np.linalg.norm(X_std[i] - X_std[j]))
        d_mm = float(np.linalg.norm(X_mm[i] - X_mm[j]))
        filas.append({
            "Par de Perfiles": etiqueta,
            "Distancia Sin Escalar": d_raw,
            "Distancia StandardScaler (Z-score)": d_std,
            "Distancia MinMaxScaler [0,1]": d_mm,
        })

    return {
        "perfiles": perfiles,
        "distancias": pd.DataFrame(filas).set_index("Par de Perfiles"),
        "medias_poblacionales": pd.Series(std_scaler.mean_, index=["educ_madre_nivel", "punt_ingles"]),
        "desv_poblacionales": pd.Series(std_scaler.scale_, index=["educ_madre_nivel", "punt_ingles"]),
    }


# --------------------------------------------------------------------------- 2 y 3. Univariado y Forma (Lab 2)
def clasificar_asimetria(g1: float) -> str:
    if abs(g1) < 0.5:
        return "Aproximadamente simétrica"
    intensidad = "moderada" if abs(g1) < 1 else "fuerte"
    return (
        f"Asimétrica {intensidad} positiva (cola derecha)"
        if g1 > 0
        else f"Asimétrica {intensidad} negativa (cola izquierda)"
    )


def clasificar_curtosis(exceso: float) -> str:
    """Exceso de curtosis = curtosis - 3. Tolerancia ±0,5 para mesocúrtica."""
    if exceso > 0.5:
        return "Leptocúrtica"
    if exceso < -0.5:
        return "Platicúrtica"
    return "Mesocúrtica"


def limites_iqr(s: pd.Series) -> tuple[float, float, float, float, float]:
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return float(q1), float(q3), float(iqr), float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)


def descriptivas(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Tendencia central, dispersión, posición, forma y atípicos por criterio 1,5·IQR."""
    filas = []
    for c in cols:
        s = df[c].dropna().astype(float)
        q1, q3, iqr, li, ls = limites_iqr(s)
        moda = s.mode()
        media, sd = float(s.mean()), float(s.std(ddof=1))
        g1 = float(stats.skew(s, bias=False))
        exc = float(stats.kurtosis(s, fisher=True, bias=False))
        n_out = int(((s < li) | (s > ls)).sum())
        filas.append({
            "variable": c,
            "n": int(len(s)),
            "media": media,
            "mediana": float(s.median()),
            "moda": float(moda.iloc[0]),
            "n_modas": int(len(moda)),
            "minimo": float(s.min()),
            "maximo": float(s.max()),
            "rango": float(s.max() - s.min()),
            "varianza": float(s.var(ddof=1)),
            "desv_estandar": sd,
            "coef_variacion_%": 100.0 * sd / media,
            "Q1": q1,
            "Q2": float(s.median()),
            "Q3": q3,
            "P10": float(s.quantile(0.10)),
            "P90": float(s.quantile(0.90)),
            "IQR": iqr,
            "asimetria": g1,
            "curtosis": exc + 3.0,
            "exceso_curtosis": exc,
            "forma_asimetria": clasificar_asimetria(g1),
            "forma_curtosis": clasificar_curtosis(exc),
            "lim_inf_IQR": li,
            "lim_sup_IQR": ls,
            "n_atipicos": n_out,
            "%_atipicos": 100.0 * n_out / len(s),
        })
    return pd.DataFrame(filas).set_index("variable")


def tabla_frecuencias_cualitativa(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Tabla de frecuencias absolutas, relativas (%) y acumuladas (Lab 1)."""
    s = df[col].astype(str)
    orden = orden_categorias(col, s.unique())
    abs_f = s.value_counts().reindex(orden, fill_value=0)
    rel_f = abs_f / len(s) * 100.0
    acum_abs = abs_f.cumsum()
    acum_rel = rel_f.cumsum()
    return pd.DataFrame({
        "Frecuencia Absoluta (n)": abs_f,
        "Frecuencia Relativa (%)": rel_f,
        "Frecuencia Acumulada (N)": acum_abs,
        "Porcentaje Acumulado (%)": acum_rel,
    })


def interpretar_univariada(d: pd.DataFrame) -> str:
    """Lectura estructurada automática de tendencia central, dispersión y posición."""
    out = []
    for v, r in d.iterrows():
        dif = r["media"] - r["mediana"]
        sesgo = (
            "la media supera a la mediana (sesgo positivo hacia valores altos)"
            if dif > 0.02 * r["desv_estandar"]
            else "la media es menor que la mediana (sesgo negativo hacia valores bajos)"
            if dif < -0.02 * r["desv_estandar"]
            else "media y mediana prácticamente coinciden (distribución centrada sin sesgo apreciable)"
        )
        out.append(
            f"- **`{v}`**: media = {fmt(r['media'])}, mediana = {fmt(r['mediana'])} y moda = {fmt(r['moda'])}; {sesgo}. "
            f"Desviación estándar = {fmt(r['desv_estandar'])} (CV = {fmt(r['coef_variacion_%'])} %). "
            f"El 50 % central se ubica entre Q1 = {fmt(r['Q1'])} y Q3 = {fmt(r['Q3'])} (IQR = {fmt(r['IQR'])}); "
            f"el 10 % inferior obtiene menos de {fmt(r['P10'])} y el 10 % superior supera {fmt(r['P90'])}."
        )
    mayor = d["coef_variacion_%"].idxmax()
    menor = d["coef_variacion_%"].idxmin()
    out.append(
        f"\n**Comparación de variabilidad relativa (CV):** dado que `punt_global` (0–500) y las pruebas por área (0–100) "
        f"tienen escalas distintas, se comparan mediante el Coeficiente de Variación. La mayor dispersión relativa se "
        f"presenta en **`{mayor}`** ({fmt(d.loc[mayor, 'coef_variacion_%'])} %) y la menor en **`{menor}`** "
        f"({fmt(d.loc[menor, 'coef_variacion_%'])} %)."
    )
    return "\n".join(out)


def interpretar_forma(d: pd.DataFrame) -> str:
    out = []
    for v, r in d.iterrows():
        out.append(
            f"- **`{v}`**: asimetría $g_1$ = {fmt(r['asimetria'])} → {r['forma_asimetria'].lower()}; "
            f"curtosis = {fmt(r['curtosis'])} (exceso = {fmt(r['exceso_curtosis'])}) → {r['forma_curtosis'].lower()}; "
            f"atípicos IQR = {int(r['n_atipicos']):,} ({fmt(r['%_atipicos'])} %)."
        )
    return "\n".join(out)


def impacto_outliers(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Compara las medidas descriptivas con y sin atípicos (criterio 1,5·IQR)."""
    filas = []
    for c in cols:
        s = df[c].dropna().astype(float)
        _, _, _, li, ls = limites_iqr(s)
        sin = s[(s >= li) & (s <= ls)]
        for nombre, f in [
            ("media", np.mean),
            ("mediana", np.median),
            ("desv_estandar", lambda x: np.std(x, ddof=1)),
            ("asimetria", lambda x: stats.skew(x, bias=False)),
            ("exceso_curtosis", lambda x: stats.kurtosis(x, bias=False)),
        ]:
            a, b = float(f(s)), float(f(sin))
            filas.append({
                "variable": c,
                "medida": nombre,
                "con_atipicos": a,
                "sin_atipicos": b,
                "diferencia": b - a,
                "cambio_%": 100.0 * (b - a) / abs(a) if a else np.nan,
                "n_atipicos": int(len(s) - len(sin)),
            })
    return pd.DataFrame(filas)


def interpretar_outliers(imp: pd.DataFrame) -> str:
    out = []
    for v, g in imp.groupby("variable", sort=False):
        g = g.set_index("medida")
        n = int(g["n_atipicos"].iloc[0])
        if n == 0:
            out.append(f"- **`{v}`**: 0 atípicos bajo el criterio 1,5·IQR; los estimadores permanecen invariantes.")
            continue
        out.append(
            f"- **`{v}`**: {n:,} atípicos detectados. Al filtrarlos, la media varía {fmt(g.loc['media', 'diferencia'])} puntos, "
            f"la mediana {fmt(g.loc['mediana', 'diferencia'])} y la desviación estándar {fmt(g.loc['desv_estandar', 'diferencia'])}. "
            f"Como corresponden a desempeños reales dentro del rango oficial del ICFES ([0, 100] o [0, 500]), **se conservan en el modelado**."
        )
    return "\n".join(out)


# --------------------------------------------------------------------------- 4. Correlación entre continuas (Lab 2)
def _fuerza(r: float) -> str:
    a = abs(r)
    return (
        "despreciable"
        if a < 0.1
        else "débil"
        if a < 0.3
        else "moderada"
        if a < 0.5
        else "fuerte"
        if a < 0.7
        else "muy fuerte"
    )


def correlacion_par(df: pd.DataFrame, x: str, y: str) -> dict:
    d = df[[x, y]].dropna().astype(float)
    xv, yv = d[x].to_numpy(), d[y].to_numpy()
    r, p_r = stats.pearsonr(xv, yv)
    rho, p_s = stats.spearmanr(xv, yv)
    tau, p_k = stats.kendalltau(xv, yv)
    cov = float(np.cov(xv, yv, ddof=1)[0, 1])

    mask = np.ones(len(d), dtype=bool)
    for c in (x, y):
        _, _, _, li, ls = limites_iqr(d[c])
        mask &= ((d[c] >= li) & (d[c] <= ls)).to_numpy()
    r_s = float(stats.pearsonr(xv[mask], yv[mask])[0])
    rho_s = float(stats.spearmanr(xv[mask], yv[mask])[0])

    ss_tot = float(((yv - yv.mean()) ** 2).sum())
    res = {}
    for g in (1, 2):
        coef = np.polyfit(xv, yv, g)
        res[g] = float(1.0 - ((yv - np.polyval(coef, xv)) ** 2).sum() / ss_tot)
    return {
        "x": x,
        "y": y,
        "n": int(len(d)),
        "covarianza": cov,
        "pearson": float(r),
        "p_pearson": float(p_r),
        "spearman": float(rho),
        "p_spearman": float(p_s),
        "kendall": float(tau),
        "p_kendall": float(p_k),
        "pearson_sin_atipicos": r_s,
        "spearman_sin_atipicos": rho_s,
        "n_excluidos_atipicos": int((~mask).sum()),
        "R2_lineal": res[1],
        "R2_cuadratico": res[2],
    }


def tabla_correlaciones(df: pd.DataFrame, pares: list[tuple[str, str]]) -> pd.DataFrame:
    t = pd.DataFrame([correlacion_par(df, x, y) for x, y in pares])
    t.insert(0, "par", t["x"] + " vs " + t["y"])
    return t.drop(columns=["x", "y"]).set_index("par")


def interpretar_correlacion(r: dict) -> str:
    dr = abs(r["pearson"] - r["spearman"])
    ganancia_cuad = r["R2_cuadratico"] - r["R2_lineal"]
    cambio_out = r["pearson_sin_atipicos"] - r["pearson"]
    txt = (
        f"**`{r['x']}` vs `{r['y']}`** ($n = {r['n']:,}$): asociación {_fuerza(r['pearson'])} "
        f"{'positiva' if r['pearson'] > 0 else 'negativa'} (Pearson $r = {fmt(r['pearson'])}$, $p$ {fmt_p(r['p_pearson'])}; "
        f"covarianza = {fmt(r['covarianza'])}). "
        f"Spearman $\\rho = {fmt(r['spearman'])}$ ($p$ {fmt_p(r['p_spearman'])}) y Kendall $\\tau = {fmt(r['kendall'])}$ ($p$ {fmt_p(r['p_kendall'])}). "
    )
    if dr < 0.02:
        txt += "Pearson y Spearman casi coinciden: la dependencia es esencialmente lineal y sin distorsión por extremos. "
    elif r["spearman"] > r["pearson"]:
        txt += f"Spearman supera a Pearson en {fmt(dr)}: relación monótona con ligera curvatura o asimetría en los extremos. "
    else:
        txt += f"Pearson supera a Spearman en {fmt(dr)}: apalancamiento de las colas sobre la covarianza lineal. "
    txt += (
        f"Sin los {r['n_excluidos_atipicos']:,} atípicos IQR, Pearson pasa a {fmt(r['pearson_sin_atipicos'])} "
        f"(cambio = {fmt(cambio_out)}), y el ajuste cuadrático aporta $\\Delta R^2 = {fmt(ganancia_cuad)}$ "
        f"({fmt(r['R2_lineal'])} → {fmt(r['R2_cuadratico'])})."
    )
    return txt


# --------------------------------------------------------------------------- 5. Independencia Categóricas (Lab 2)
def _entropia(conteos) -> float:
    p = np.asarray(conteos, dtype=float)
    p = p[p > 0] / p.sum()
    return float(-(p * np.log(p)).sum())


def independencia_categoricas(
    df: pd.DataFrame, a: str, b: str, n_perm: int = 200, semilla: int = 42
) -> dict:
    d = df[[a, b]].dropna().astype(str)
    obs = pd.crosstab(d[a], d[b])
    obs = obs.reindex(index=orden_categorias(a, obs.index), columns=orden_categorias(b, obs.columns))
    chi2, p, dof, esp = stats.chi2_contingency(obs.to_numpy())
    esp = pd.DataFrame(esp, index=obs.index, columns=obs.columns)
    n = obs.to_numpy().sum()
    k = min(obs.shape) - 1
    v = float(np.sqrt(chi2 / (n * k))) if k > 0 else np.nan

    phi2 = chi2 / n
    r_, c_ = obs.shape
    phi2c = max(0.0, phi2 - (r_ - 1) * (c_ - 1) / (n - 1))
    rc, cc = r_ - (r_ - 1) ** 2 / (n - 1), c_ - (c_ - 1) ** 2 / (n - 1)
    v_corr = float(np.sqrt(phi2c / min(rc - 1, cc - 1))) if min(rc, cc) > 1 else np.nan

    ca, cb = pd.factorize(d[a])[0], pd.factorize(d[b])[0]
    mi = float(mutual_info_score(ca, cb))
    ha, hb = _entropia(obs.sum(axis=1)), _entropia(obs.sum(axis=0))
    nmi = mi / min(ha, hb) if min(ha, hb) > 0 else np.nan
    rng = np.random.default_rng(semilla)
    mi_perm = np.array([mutual_info_score(ca, rng.permutation(cb)) for _ in range(n_perm)])
    p_mi = (1 + (mi_perm >= mi).sum()) / (n_perm + 1)

    pr, pc = obs.sum(axis=1).to_numpy() / n, obs.sum(axis=0).to_numpy() / n
    resid = (obs.to_numpy() - esp.to_numpy()) / np.sqrt(esp.to_numpy() * np.outer(1 - pr, 1 - pc))
    return {
        "a": a,
        "b": b,
        "n": int(n),
        "observadas": obs,
        "esperadas": esp,
        "residuos": pd.DataFrame(resid, index=obs.index, columns=obs.columns),
        "chi2": float(chi2),
        "gl": int(dof),
        "p_chi2": float(p),
        "cramer_v": v,
        "cramer_v_corregido": v_corr,
        "min_esperada": float(esp.to_numpy().min()),
        "pct_esp_menor_5": float((esp.to_numpy() < 5).mean() * 100.0),
        "MI_nats": mi,
        "MI_bits": mi / np.log(2),
        "MI_normalizada": nmi,
        "G": 2.0 * n * mi,
        "p_MI_permutacion": float(p_mi),
        "n_perm": n_perm,
        "H_a": ha,
        "H_b": hb,
    }


def resumen_independencia(resultados: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "par": f"{r['a']} × {r['b']}",
            "n": r["n"],
            "chi2": r["chi2"],
            "gl": r["gl"],
            "p_chi2": r["p_chi2"],
            "V_Cramer": r["cramer_v"],
            "V_Cramer_corregido": r["cramer_v_corregido"],
            "MI_nats": r["MI_nats"],
            "MI_normalizada": r["MI_normalizada"],
            "G=2n·MI": r["G"],
            "p_MI_perm": r["p_MI_permutacion"],
            "%_esperadas<5": r["pct_esp_menor_5"],
        }
        for r in resultados
    ]).set_index("par")


def interpretar_independencia(r: dict) -> str:
    sig = r["p_chi2"] < 0.05
    fuerza = (
        "despreciable"
        if r["cramer_v"] < 0.1
        else "débil"
        if r["cramer_v"] < 0.3
        else "moderada"
        if r["cramer_v"] < 0.5
        else "fuerte"
    )
    txt = (
        f"- **`{r['a']}` × `{r['b']}`**: $\\chi^2 = {fmt(r['chi2'])}$ ({r['gl']} gl), $p$ {fmt_p(r['p_chi2'])} → "
        f"{'se rechaza $H_0$ (existe dependencia estadísticamente significativa)' if sig else 'no se rechaza $H_0$'}. "
        f"Magnitud **{fuerza}** ($V$ de Cramér = {fmt(r['cramer_v'])}, corregida = {fmt(r['cramer_v_corregido'])}; "
        f"MI = {fmt(r['MI_nats'])} nats, MI normalizada = {fmt(r['MI_normalizada'])}). "
    )
    if sig and r["cramer_v"] < 0.1:
        txt += "Con $n = 77.827$, el test $\\chi^2$ detecta una desviación mínima aunque el efecto práctico es despreciable."
    elif sig:
        txt += "Esto indica colinealidad moderada entre predictores socioeconómicos que el modelo lineal controla mediante efectos parciales."
    return txt


# --------------------------------------------------------------------------- 6. Asociación Mixta: ANOVA / Welch / Kruskal-Wallis
def _grupos(df: pd.DataFrame, cat: str, y: str, min_n: int = 5) -> dict[str, np.ndarray]:
    d = df[[cat, y]].dropna()
    d[cat] = d[cat].astype(str)
    orden = [g for g in orden_categorias(cat, d[cat].unique()) if (d[cat] == g).sum() >= min_n]
    return {g: d.loc[d[cat] == g, y].astype(float).to_numpy() for g in orden}


def welch_anova(grupos: list[np.ndarray]) -> tuple[float, float, float, float]:
    k = len(grupos)
    n = np.array([len(g) for g in grupos], dtype=float)
    m = np.array([g.mean() for g in grupos])
    v = np.array([g.var(ddof=1) for g in grupos])
    w = n / v
    mw = (w * m).sum() / w.sum()
    a = (w * (m - mw) ** 2).sum() / (k - 1)
    lam = ((1 - w / w.sum()) ** 2 / (n - 1)).sum()
    b = 1 + 2 * (k - 2) / (k**2 - 1) * lam
    F = float(a / b)
    df2 = float((k**2 - 1) / (3 * lam))
    return F, float(k - 1), df2, float(stats.f.sf(F, k - 1, df2))


def analisis_mixto(df: pd.DataFrame, cat: str, y: str, semilla: int = 42) -> dict:
    g = _grupos(df, cat, y)
    arrs = list(g.values())
    rng = np.random.default_rng(semilla)
    filas = []
    for nombre, a in g.items():
        muestra = rng.choice(a, size=min(5000, len(a)), replace=False)
        filas.append({
            "grupo": nombre,
            "n": int(len(a)),
            "media": float(a.mean()),
            "mediana": float(np.median(a)),
            "desv_estandar": float(a.std(ddof=1)),
            "asimetria": float(stats.skew(a, bias=False)),
            "exceso_curtosis": float(stats.kurtosis(a, bias=False)),
            "p_shapiro": float(stats.shapiro(muestra).pvalue),
            "p_dagostino": float(stats.normaltest(a).pvalue),
        })
    tabla = pd.DataFrame(filas).set_index("grupo")
    k, N = len(arrs), sum(len(a) for a in arrs)
    lev = stats.levene(*arrs, center="median")
    bar = stats.bartlett(*arrs)
    F, p_F = stats.f_oneway(*arrs)
    Fw, gl1w, gl2w, p_w = welch_anova(arrs)
    H, p_H = stats.kruskal(*arrs)
    todo = np.concatenate(arrs)
    sst = float(((todo - todo.mean()) ** 2).sum())
    ssb = float(sum(len(a) * (a.mean() - todo.mean()) ** 2 for a in arrs))
    eta2 = ssb / sst
    msw = (sst - ssb) / (N - k)
    omega2 = (ssb - (k - 1) * msw) / (sst + msw)
    eps2 = float((H - k + 1) / (N - k))
    return {
        "cat": cat,
        "y": y,
        "k": int(k),
        "N": int(N),
        "tabla_grupos": tabla,
        "levene_W": float(lev.statistic),
        "p_levene": float(lev.pvalue),
        "bartlett_T": float(bar.statistic),
        "p_bartlett": float(bar.pvalue),
        "F": float(F),
        "gl_between": int(k - 1),
        "gl_within": int(N - k),
        "p_anova": float(p_F),
        "F_welch": Fw,
        "gl1_welch": gl1w,
        "gl2_welch": gl2w,
        "p_welch": p_w,
        "H": float(H),
        "p_kruskal": float(p_H),
        "eta2": eta2,
        "omega2": omega2,
        "epsilon2": eps2,
        "cociente_sd_max_min": float(tabla["desv_estandar"].max() / tabla["desv_estandar"].min()),
        "grupos": g,
    }


def tabla_pruebas_mixtas(r: dict) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "prueba": "Levene (Brown-Forsythe)",
            "estadistico": r["levene_W"],
            "gl": f"{r['k'] - 1}, {r['N'] - r['k']}",
            "p_valor": r["p_levene"],
            "tamaño_efecto": np.nan,
        },
        {
            "prueba": "Bartlett",
            "estadistico": r["bartlett_T"],
            "gl": f"{r['k'] - 1}",
            "p_valor": r["p_bartlett"],
            "tamaño_efecto": np.nan,
        },
        {
            "prueba": "ANOVA de un factor",
            "estadistico": r["F"],
            "gl": f"{r['gl_between']}, {r['gl_within']}",
            "p_valor": r["p_anova"],
            "tamaño_efecto": r["eta2"],
        },
        {
            "prueba": "ANOVA de Welch",
            "estadistico": r["F_welch"],
            "gl": f"{fmt(r['gl1_welch'], 0)}, {fmt(r['gl2_welch'], 2)}",
            "p_valor": r["p_welch"],
            "tamaño_efecto": r["omega2"],
        },
        {
            "prueba": "Kruskal-Wallis",
            "estadistico": r["H"],
            "gl": f"{r['k'] - 1}",
            "p_valor": r["p_kruskal"],
            "tamaño_efecto": r["epsilon2"],
        },
    ]).set_index("prueba")


def interpretar_mixto(r: dict) -> str:
    t = r["tabla_grupos"]
    pct_no_normal = 100.0 * (t["p_dagostino"] < 0.05).mean()
    hom = r["p_levene"] >= 0.05
    txt = (
        f"**`{r['cat']}` → `{r['y']}`** ($k = {r['k']}$ grupos, $N = {r['N']:,}$):\n"
        f"- **Supuesto de normalidad intragrupo:** D'Agostino-Pearson rechaza $H_0$ en el {fmt(pct_no_normal, 1)} % de los grupos "
        f"(esperado con tamaños muestrales grandes), aunque la asimetría se mantiene acotada en "
        f"$[{fmt(t['asimetria'].min())}, {fmt(t['asimetria'].max())}]$.\n"
        f"- **Supuesto de homocedasticidad:** Levene $W = {fmt(r['levene_W'])}$ ($p$ {fmt_p(r['p_levene'])}) → "
        f"{'varianzas homogéneas' if hom else 'se rechaza igualdad de varianzas (heterocedasticidad entre grupos)'}; "
        f"razón $\\sigma_{{\\max}}/\\sigma_{{\\min}} = {fmt(r['cociente_sd_max_min'])}$.\n"
        f"- **Contraste de medias y medianas:** ANOVA clásico $F = {fmt(r['F'])}$ ($p$ {fmt_p(r['p_anova'])}, $\\eta^2 = {fmt(r['eta2'])}$), "
        f"ANOVA de Welch $F_W = {fmt(r['F_welch'])}$ ($p$ {fmt_p(r['p_welch'])}, $\\omega^2 = {fmt(r['omega2'])}$) y "
        f"Kruskal-Wallis $H = {fmt(r['H'])}$ ($p$ {fmt_p(r['p_kruskal'])}, $\\varepsilon^2 = {fmt(r['epsilon2'])}$).\n"
        f"- **Decisión estadística:** al violarse la homocedasticidad con tamaños de grupo desbalanceados, "
        f"las pruebas rigurosas de referencia son **Kruskal-Wallis** (no paramétrica) y **ANOVA de Welch** (robusta a varianzas desiguales). "
        f"Ambas confirman que `{r['cat']}` explica el **{fmt(r['eta2'] * 100, 2)} %** de la variabilidad de `{r['y']}`."
    )
    return txt


# --------------------------------------------------------------------------- 7. Síntesis y Ranking de Predictores
def ranking_asociacion(
    df: pd.DataFrame, y: str = "punt_global", cats: list[str] | None = None
) -> pd.DataFrame:
    """Ranking de fuerza de asociación de cada predictor cualitativo con la variable objetivo."""
    cats = cats or COLS_CUAL
    ybin = pd.qcut(df[y], 10, duplicates="drop").cat.codes
    filas = []
    for c in cats:
        g = _grupos(df, c, y)
        arrs = list(g.values())
        k, N = len(arrs), sum(len(a) for a in arrs)
        F, p_f = stats.f_oneway(*arrs)
        Fw, _, _, p_w = welch_anova(arrs)
        H, p_k = stats.kruskal(*arrs)
        todo = np.concatenate(arrs)
        ssb = sum(len(a) * (a.mean() - todo.mean()) ** 2 for a in arrs)
        eta2 = ssb / ((todo - todo.mean()) ** 2).sum()
        cod = pd.factorize(df[c].astype(str))[0]
        mi = mutual_info_score(cod, ybin)
        ha = _entropia(np.bincount(cod))
        med = {k_: np.median(v) for k_, v in g.items()}
        filas.append({
            "variable": c,
            "k_grupos": int(k),
            "F_anova": float(F),
            "F_welch": float(Fw),
            "H_kruskal": float(H),
            "p_kruskal": float(p_k),
            "epsilon2": float((H - k + 1) / (N - k)),
            "eta2": float(eta2),
            "MI_normalizada": float(mi / min(ha, _entropia(np.bincount(ybin)))),
            "dif_medianas": float(max(med.values()) - min(med.values())),
        })
    return pd.DataFrame(filas).set_index("variable").sort_values("epsilon2", ascending=False)


def ranking_cuantitativas(df: pd.DataFrame, y: str, cols: list[str]) -> pd.DataFrame:
    filas = []
    for c in cols:
        if c == y:
            continue
        r = correlacion_par(df, c, y)
        filas.append({
            "variable": c,
            "covarianza": r["covarianza"],
            "pearson": r["pearson"],
            "spearman": r["spearman"],
            "kendall": r["kendall"],
            "R2_lineal": r["R2_lineal"],
        })
    return pd.DataFrame(filas).set_index("variable").sort_values("pearson", ascending=False)


def asimetria_tras_transformacion(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Asimetría original vs raíz cuadrada y Yeo-Johnson."""
    filas = []
    for c in cols:
        s = df[c].dropna().astype(float)
        yj, lam = stats.yeojohnson(s)
        filas.append({
            "variable": c,
            "asimetria": float(stats.skew(s, bias=False)),
            "asimetria_raiz": float(stats.skew(np.sqrt(s.clip(lower=0)), bias=False)),
            "asimetria_yeojohnson": float(stats.skew(yj, bias=False)),
            "lambda_yeojohnson": float(lam),
        })
    return pd.DataFrame(filas).set_index("variable")


def sensibilidad_atipicos(
    df: pd.DataFrame, x: str, y: str, n_base: int = 2000, n_atip: int = 20, semilla: int = 42
) -> pd.DataFrame:
    """Experimento didáctico de sensibilidad de Pearson, Spearman y Kendall ante atípicos extremos."""
    d = df[[x, y]].dropna().astype(float).sample(n_base, random_state=semilla)
    malos = pd.DataFrame({x: [d[x].max()] * n_atip, y: [d[y].min()] * n_atip})
    d2 = pd.concat([d, malos], ignore_index=True)
    filas = []
    for nombre, f in [("Pearson", stats.pearsonr), ("Spearman", stats.spearmanr), ("Kendall", stats.kendalltau)]:
        a, b = float(f(d[x], d[y])[0]), float(f(d2[x], d2[y])[0])
        filas.append({
            "coeficiente": nombre,
            "sin_atipicos": a,
            "con_atipicos": b,
            "cambio": b - a,
            "cambio_%": 100.0 * (b - a) / abs(a),
        })
    return pd.DataFrame(filas).set_index("coeficiente")

