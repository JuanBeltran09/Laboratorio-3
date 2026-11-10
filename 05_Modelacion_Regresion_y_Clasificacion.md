# 05 — Modelación Predictiva: Regresión Lineal y Regresión Logística

Este documento presenta la fundamentación matemática, la implementación computacional en [`src/modelos.py`](file:///c:/Users/beltr/Documents/Universidad/Estadistica/Laboratorio%203/src/modelos.py) y la interpretación estadística detallada de las **Secciones 7, 8 y 9** del **Laboratorio 3**: el entrenamiento, validación cruzada, optimización de hiperparámetros, evaluación de métricas y diagnóstico de supuestos para los modelos de **Regresión Lineal Múltiple** y **Regresión Logística Binaria**.

---

## 1. Estrategia Dual de Modelado: ¿Por qué Modelo A y Modelo B?

Siguiendo la progresión analítica de los **Laboratorios 1 y 2**, donde se estudiaron por un lado los determinantes socioeconómicos e institucionales del hogar y por otro lado el desempeño específico en inglés (`punt_ingles`), en el **Laboratorio 3** entrenamos y serializamos **dos especificaciones complementarias** para cada familia de modelos:

1. **Modelo A — Determinantes Socioeconómicos e Institucionales Puros (`8 predictores`):**
   - **Variables:** `fami_tieneinternet`, `cole_naturaleza`, `cole_area_ubicacion`, `estu_genero`, `es_sin_estrato`, `educ_no_sabe`, `educ_madre_nivel` y `estrato_num`.
   - **Propósito:** Responder qué proporción del desempeño académico (`punt_global`) y de la probabilidad de superar los $250$ puntos (`supera_promedio`) está determinada **exclusivamente por las condiciones de origen del estudiante y su entorno escolar**, *antes* de conocer cualquier resultado parcial del examen. Es el modelo de **alerta temprana y política pública**.
2. **Modelo B — Socioeconómico + Competencia en Inglés (`9 predictores`, añade `punt_ingles`):**
   - **Variables:** Las $8$ variables del Modelo A más `punt_ingles` (escalado con `StandardScaler`).
   - **Propósito:** Evaluar el salto en poder predictivo cuando se incorpora una competencia académica transversal evaluada en la misma prueba, y cuantificar el **efecto de mediación** (cuánto del efecto socioeconómico opera a través de las brechas previas en bilingüismo y capital escolar).

---

## 2. Parte I: Regresión Lineal Múltiple (`punt_global`)

### 2.1 Formulación Matemática y Estimación MCO (OLS)

El modelo de Regresión Lineal Múltiple postula que la variable respuesta continua $Y_i = \text{punt\_global}_i \in [0, 500]$ se relaciona linealmente con el vector de características transformadas $\mathbf{x}_i = [1, x_{i1}, x_{i2}, \dots, x_{ip}]^\top$ tras aplicar el `ColumnTransformer`:

$$Y_i = \beta_0 + \sum_{j=1}^{p} \beta_j x_{ij} + \varepsilon_i, \qquad \varepsilon_i \sim \text{i.i.d. } (0, \sigma^2)$$

El estimador de **Mínimos Cuadrados Ordinarios (MCO / OLS)** minimiza la Suma de Cuadrados de los Residuos ($\text{SSR}$):

$$\hat{\boldsymbol{\beta}}_{\text{OLS}} = \arg\min_{\boldsymbol{\beta}} \sum_{i=1}^{n} \left(y_i - \beta_0 - \sum_{j=1}^{p}\beta_j x_{ij}\right)^2 = (\mathbf{X}^\top \mathbf{X})^{-1}\mathbf{X}^\top \mathbf{y}$$

En [`src/modelos.py::entrenar_regresion_lineal`](file:///c:/Users/beltr/Documents/Universidad/Estadistica/Laboratorio%203/src/modelos.py#L72-L195) se realiza una doble estimación sincronizada:
1. **Pipeline predictivo de `scikit-learn` (`LinearRegression`):** Entrenado sobre el $80\%$ de los datos (`X_train`, $n = 62,261$) y evaluado sobre el $20\%$ de prueba (`X_test`, $n = 15,566$, `random_state=42`) y mediante validación cruzada de $10$ pliegues (`KFold(n_splits=10, shuffle=True, random_state=42)`), garantizando cero fuga de datos (*data leakage*).
2. **Ajuste inferencial econométrico en `statsmodels` (`sm.OLS`):** Ajustado sobre la matriz de diseño preprocesada de entrenamiento para obtener estadísticos $t$, $p$-valores exactos, intervalos de confianza al $95\%$, estadístico $F$ global, pruebas de diagnóstico residual y **errores estándar robustos a la heterocedasticidad (`HC3`)**:
   $$\widehat{\text{Var}}_{\text{HC3}}(\hat{\boldsymbol{\beta}}) = (\mathbf{X}^\top \mathbf{X})^{-1} \left( \sum_{i=1}^{n} \frac{\hat{\varepsilon}_i^2}{(1 - h_{ii})^2} \mathbf{x}_i \mathbf{x}_i^\top \right) (\mathbf{X}^\top \mathbf{X})^{-1}$$
   donde $h_{ii} = \mathbf{x}_i^\top(\mathbf{X}^\top \mathbf{X})^{-1}\mathbf{x}_i$ es el *leverage* de la observación $i$.

---

### 2.2 Definición Matemática de las Métricas de Regresión

Cumpliendo con los requerimientos de `Laboratorio_3.pdf` y la metodología del notebook `8.Model_Regression_Housing.ipynb` de la clase, en [`src/modelos.py::calcular_metricas_regresion`](file:///c:/Users/beltr/Documents/Universidad/Estadistica/Laboratorio%203/src/modelos.py#L46-L69) se calculan:

1. **MAE (*Mean Absolute Error* — Error Absoluto Medio):**
   $$\text{MAE} = \frac{1}{n}\sum_{i=1}^{n} |y_i - \hat{y}_i|$$
   Mide el error promedio en las mismas unidades de la variable respuesta (**puntos de Saber 11**), penalizando linealmente las desviaciones (robusto a valores extremos).
2. **RMSE (*Root Mean Squared Error* — Raíz del Error Cuadrático Medio):**
   $$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$
   Penaliza cuadráticamente los errores grandes. Siempre cumple $\text{RMSE} \ge \text{MAE}$; la diferencia entre ambos indica la presencia de residuos grandes en las colas.
3. **MAPE (*Mean Absolute Percentage Error* — Error Porcentual Absoluto Medio):**
   $$\text{MAPE} = \frac{1}{n_{\text{val}}}\sum_{y_i > 1} \left|\frac{y_i - \hat{y}_i}{y_i}\right|$$
   Expresa el error promedio como proporción (o porcentaje) del puntaje real del estudiante (excluyendo únicamente el caso patológico $y_i = 0$ para evitar división por cero).
4. **Coeficiente de Determinación ($R^2$) y $R^2$ Ajustado ($R^2_{\text{adj}}$):**
   $$R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}, \qquad R^2_{\text{adj}} = 1 - (1 - R^2)\frac{n - 1}{n - p - 1}$$
   Cuantifica la fracción de la varianza total de `punt_global` explicada por los predictores del modelo, penalizando en $R^2_{\text{adj}}$ la inclusión de parámetros irrelevantes.
5. **RSE (*Residual Standard Error* — Error Estándar Residual):**
   $$\text{RSE} = \sqrt{\frac{1}{n - p - 1}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$
   Estimador insesgado de la desviación estándar de las perturbaciones $\sigma$.

---

### 2.3 Resultados de Evaluación y Validación Cruzada (Regresión Lineal)

| Métrica | Modelo A (Socioeconómico Puro) — Train ($80\%$) | Modelo A (Socioeconómico Puro) — Test ($20\%$) | Modelo A — CV 10-Fold (Media $\pm$ DE) | Modelo B (Socioeconómico + Inglés) — Train ($80\%$) | Modelo B (Socioeconómico + Inglés) — Test ($20\%$) | Modelo B — CV 10-Fold (Media $\pm$ DE) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **MAE (puntos)** | $38.0207$ | **$38.3033$** | $38.0775 \pm 0.2138$ | $24.5113$ | **$24.4325$** | $24.4946 \pm 0.1271$ |
| **RMSE (puntos)** | $46.5567$ | **$46.8585$** | $46.6218 \pm 0.2325$ | $30.6285$ | **$30.5989$** | $30.6259 \pm 0.1304$ |
| **MAPE (proporción)** | $0.1607$ | **$0.1625$** | $0.1611 \pm 0.0011$ | $0.1041$ | **$0.1041$** | $0.1041 \pm 0.0008$ |
| **MAPE (%)** | $16.0739\%$ | **$16.2482\%$** | $16.1090\% \pm 0.1118\%$ | $10.4107\%$ | **$10.4061\%$** | $10.4065\% \pm 0.0802\%$ |
| **$R^2$** | $0.1875$ | **$0.1888$** | $0.1874 \pm 0.0058$ | $0.6483$ | **$0.6541$** | $0.6493 \pm 0.0053$ |
| **$R^2$ Ajustado** | $0.1873$ | **$0.1883$** | — | $0.6483$ | **$0.6539$** | — |
| **RSE (puntos)** | $46.5604$ | **$46.8736$** | — | $30.6313$ | **$30.6097$** | — |
| **Estadístico $F$ Global** | $1595.6977$ ($p < 0.0001$) | — | — | $11476.2160$ ($p < 0.0001$) | — | — |

#### Análisis de Generalización y Capacidad Explicativa:
1. **Ausencia total de sobreajuste (*Overfitting*):** En el **Modelo A**, el $R^2$ de entrenamiento ($0.1875$), el $R^2$ de prueba ($0.1888$) y el promedio de la validación cruzada de $10$ pliegues ($0.1874 \pm 0.0058$) difieren en menos de $0.0015$. En el **Modelo B**, el $R^2$ de prueba es $0.6541$ frente a $0.6483$ en entrenamiento y $0.6493 \pm 0.0053$ en CV.
2. **Significado del $18.88\%$ de varianza explicada en el Modelo A:** En ciencias sociales y economía de la educación, que casi una quinta parte ($18.88\%$) de la varianza total de un examen cognitivo estandarizado esté predeterminada por $8$ indicadores demográficos y socioeconómicos básicos demuestra una **fuerte desigualdad de oportunidades de origen**. El error medio absoluto es de $38.30$ puntos ($\text{MAPE} = 16.25\%$).
3. **Salto al $65.41\%$ en el Modelo B:** Al incorporar `punt_ingles`, la varianza explicada se multiplica por $3.46\times$ ($R^2 = 0.6541$), reduciendo el $\text{MAE}$ a **$24.43$ puntos** y el $\text{MAPE}$ al **$10.41\%$**.

---

### 2.4 Interpretación Econométrica de los Coeficientes ($\hat{\beta}_j$)

#### Tabla de Inferencia OLS — Modelo A (Socioeconómico Puro)

| Variable en el Pipeline | Etiqueta Descriptiva | $\hat{\beta}_j$ (puntos) | Error Est. (MCO) | Error Est. (`HC3`) | Estadístico $z$ (`HC3`) | $p$-valor (`HC3`) | IC $95\%$ Inferior | IC $95\%$ Superior |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `const` | **Intercepto ($\beta_0$: Perfil Base de Referencia)** | **$241.4038$** | $0.7218$ | $0.7222$ | $334.2805$ | $< 0.0001$ | $239.9891$ | $242.8186$ |
| `cat__es_sin_estrato_Si` | Vivienda: Sin Estrato (vs. En escala E1–E6) | **$-25.0213$** | $1.0026$ | $0.9280$ | $-26.9631$ | $< 0.0001$ | $-26.9865$ | $-23.0561$ |
| `cat__fami_tieneinternet_Si` | Internet en casa: Sí (vs. No) | **$+15.6274$** | $0.4741$ | $0.4601$ | $+33.9623$ | $< 0.0001$ | $+14.6981$ | $+16.5567$ |
| `cat__cole_naturaleza_OFICIAL` | Colegio Oficial / Público (vs. No Oficial) | **$-14.3629$** | $0.4811$ | $0.5174$ | $-27.7589$ | $< 0.0001$ | $-15.3059$ | $-13.4200$ |
| `num__educ_madre_nivel` | Nivel educativo madre (escalado $Z$, 0–9) | **$+12.7983$** | $0.2064$ | $0.2138$ | $+59.8713$ | $< 0.0001$ | $+12.3939$ | $+13.2028$ |
| `cat__fami_tieneinternet_Desconocido` | Cuestionario ausente / MNAR (vs. Sin Internet) | **$-10.4637$** | $0.8549$ | $0.8616$ | $-12.1442$ | $< 0.0001$ | $-12.1394$ | $-8.7881$ |
| `cat__cole_area_ubicacion_URBANO` | Zona del colegio: Urbana (vs. Rural) | **$+9.3467$** | $0.5245$ | $0.5068$ | $+18.4417$ | $< 0.0001$ | $+8.3188$ | $+10.3747$ |
| `cat__educ_no_sabe_Si` | Educación madre: Reporta 'No sabe' (vs. Conocido) | **$+8.3381$** | $1.2152$ | $1.3237$ | $+6.2990$ | $< 0.0001$ | $+5.9563$ | $+10.7200$ |
| `cat__estu_genero_M` | Género: Masculino (vs. Femenino) | **$+5.5161$** | $0.3761$ | $0.3785$ | $+14.5741$ | $< 0.0001$ | $+4.7790$ | $+6.2532$ |
| `num__estrato_num` | Estrato numérico (escalado $Z$, 1–6) | **$-0.5692$** | $0.2023$ | $0.2149$ | $-2.6488$ | $0.0081$ | $-0.9656$ | $-0.1728$ |

#### Interpretación Ceteris Paribus de los Efectos Clave:
1. **Intercepto ($\beta_0 = 241.4038$ puntos):** Representa el puntaje global esperado para una **mujer (`F`)**, matriculada en un **colegio privado (`NO OFICIAL`)** de **zona rural (`RURAL`)**, **sin internet en el hogar (`No`)**, cuya vivienda está estratificada ($1\text{–}6$) con un **estrato promedio ($Z = 0 \implies \approx 2.09$)** y cuya madre tiene una **escolaridad promedio ($Z = 0 \implies \approx 4.58$, bachillerato)**.
2. **Penalización severa por vulnerabilidad extrema (`es_sin_estrato_Si`: $-25.0213$ puntos):** Manteniendo constantes todas las demás variables, habitar en una vivienda clasificada como `'Sin Estrato'` reduce el puntaje esperado en **$25.02$ puntos** respecto a una vivienda dentro de la escala DANE E1–E6. Es el predictor binario de mayor impacto negativo.
3. **Prima de conectividad digital (`fami_tieneinternet_Si`: $+15.6274$ puntos):** Contar con internet en casa aporta **$+15.63$ puntos** adicionales *ceteris paribus* frente a no tener internet. Por el contrario, pertenecer al bloque **MNAR de cuestionario ausente (`fami_tieneinternet_Desconocido`)** resta **$-10.46$ puntos** adicionales incluso frente a quienes reportan explícitamente no tener internet, lo que supone una brecha neta de **$26.09$ puntos** ($15.63 - (-10.46)$) entre un hogar conectado y un estudiante con cuestionario ausente.
4. **Brecha institucional público-privada (`cole_naturaleza_OFICIAL`: $-14.3629$ puntos):** A igual nivel educativo materno, igual conectividad e igual zona geográfica, asistir a un colegio oficial reduce el puntaje esperado en **$14.36$ puntos** frente a un colegio no oficial.
5. **Efecto del capital cultural materno (`educ_madre_nivel`: $+12.7983$ puntos por desviación estándar):** Es el predictor con el mayor estadístico $t/z$ de todo el modelo ($z_{\text{HC3}} = 59.87$, $p < 10^{-300}$). Un incremento de $1$ desviación estándar ($\approx 2.31$ escalones educativos, por ejemplo pasar de secundaria incompleta a educación técnica) eleva el puntaje global en **$+12.80$ puntos** ($\approx +5.53$ puntos por cada escalón educativo de la escala $0\text{–}9$).
6. **¿Por qué `estrato_num` tiene un coeficiente cercano a cero ($-0.5692$) en el modelo multivariable si en el análisis bivariado tenía $r = +0.2651$? (Efecto de Mediación y Supresión Clásico):**
   - En Colombia, el estrato socioeconómico es una clasificación física/catastral del inmueble y del barrio. Su relación positiva bruta con el puntaje Saber 11 ($+64.3$ puntos entre E6 y E1) opera casi íntegramente **a través de cuatro canales mediadores**: (a) la educación de los padres (`educ_madre_nivel`), (b) la conectividad digital (`fami_tieneinternet`), (c) el pago de educación privada (`cole_naturaleza`) y (d) la ubicación urbana (`cole_area_ubicacion`).
   - Cuando controlamos simultáneamente por esos cuatro canales y además separamos la condición de `'Sin Estrato'`, el estrato físico de la fachada por sí solo no aporta capital cognitivo adicional; de hecho, a igual escolaridad materna y tipo de colegio, se observa un ligero efecto de supresión ($\hat{\beta} = -0.57$ puntos por DE en Modelo A, y $-3.50$ en Modelo B).
7. **¿Qué ocurre en el Modelo B al añadir `punt_ingles` ($\hat{\beta} = +40.0386$ puntos por DE)?**
   - Un aumento de $1$ desviación estándar en inglés ($\approx 13.09$ puntos en la escala $0\text{–}100$) se traduce en **$+40.04$ puntos** en el puntaje global ($\approx +3.06$ puntos globales por cada punto adicional en inglés).
   - Al controlar por `punt_ingles`, el coeficiente de `cole_naturaleza_OFICIAL` pasa de **$-14.3629$** a **$+1.1596$** puntos ($p = 0.0002$). Esto revela un hallazgo sustantivo extraordinario: **toda la ventaja promedio de los colegios privados se explica por su superioridad en bilingüismo y por la selección socioeconómica de sus estudiantes**; cuando se compara a un estudiante de colegio oficial y uno de colegio privado que obtuvieron *el mismo puntaje en inglés* y tienen el mismo entorno familiar, el estudiante de colegio oficial obtiene ligeramente más puntaje global ($+1.16$ puntos).

---

### 2.5 Diagnóstico de Supuestos de la Regresión Lineal

| Diagnóstico / Supuesto | Prueba Aplicada | Modelo A (Socioeconómico Puro) | Modelo B (Socioeconómico + Inglés) | Interpretación y Solución |
| :--- | :--- | :---: | :---: | :--- |
| **No Multicolinealidad** | Factor de Inflación de la Varianza ($\text{VIF}_j = \frac{1}{1 - R_j^2}$) | $\text{VIF}_{\text{máx}} = \mathbf{1.4045}$<br>(`cond_number` $= 11.43$) | $\text{VIF}_{\text{máx}} = \mathbf{1.4316}$<br>(`cond_number` $= 11.44$) | **Cumplimiento perfecto ($\text{VIF} \ll 5.0$).** La ingeniería de variables eliminó por completo la colinealidad estructural. |
| **Media Cero de Residuos** | $\frac{1}{n}\sum \hat{\varepsilon}_i$ | $0.0000$ | $0.0000$ | **Cumplimiento exacto** garantizado por la inclusión de intercepto en MCO. |
| **Normalidad de Residuos** | Prueba de Jarque-Bera ($JB$), Asimetría y Curtosis | $JB = 549.33$ ($p < 0.0001$)<br>Asimetría $= 0.1368$<br>Curtosis $= 2.6300$ ($\text{Exceso} = -0.3700$) | $JB = 67.01$ ($p < 0.0001$)<br>Asimetría $= 0.0402$<br>Curtosis $= 3.1392$ ($\text{Exceso} = +0.1392$) | **Normalidad práctica excelente** (en el Modelo B la asimetría residual es casi nula, $0.0402$, y la curtosis es $3.14 \approx 3.0$). Por el Teorema Central del Límite ($n = 62,261$), los estimadores $\hat{\boldsymbol{\beta}}$ son asintóticamente normales. |
| **Homocedasticidad** | Prueba de Breusch-Pagan ($LM$ y $F$) | $LM = 979.68$ ($p < 0.0001$)<br>$F = 110.58$ ($p < 0.0001$) | $LM = 275.77$ ($p < 0.0001$)<br>$F = 27.69$ ($p < 0.0001$) | Existe heterocedasticidad moderada (confirmando lo hallado con Levene en la Sección 5). **Solución:** se emplean los errores estándar robustos `HC3` de White-MacKinnon, bajo los cuales todos los regresores mantienen su significancia intacta. |

---

## 3. Parte II: Regresión Logística Binaria (`supera_promedio`)

### 3.1 Formulación Matemática: Sigmoide, Logit y Regularización

Sea $Y_i = \text{supera\_promedio}_i \in \{0, 1\}$ la variable objetivo binaria que toma el valor $1$ si $\text{punt\_global}_i \ge 250$ ($48.13\%$ de la muestra) y $0$ en caso contrario ($51.87\%$). La Regresión Logística modela la probabilidad condicional $p_i = P(Y_i = 1 \mid \mathbf{x}_i)$ mediante la **función logística sigmoide**:

$$p_i = \sigma(z_i) = \frac{1}{1 + e^{-z_i}}, \qquad z_i = \beta_0 + \sum_{j=1}^{p} \beta_j x_{ij}$$

La transformación inversa define el **Logit (Log-Odds)**:

$$\ln\left(\frac{p_i}{1 - p_i}\right) = \beta_0 + \sum_{j=1}^{p} \beta_j x_{ij} \implies \text{Odds}(Y_i = 1 \mid \mathbf{x}_i) = \frac{p_i}{1 - p_i} = e^{\beta_0} \prod_{j=1}^{p} \left(e^{\beta_j}\right)^{x_{ij}}$$

Por lo tanto, el **Odds Ratio ($\text{OR}_j = e^{\beta_j}$)** representa el factor multiplicativo por el cual cambian las probabilidades relativas (*odds*) de superar los $250$ puntos ante un incremento unitario en $x_{ij}$:
- Si $\beta_j > 0 \implies \text{OR}_j = e^{\beta_j} > 1$: la característica **aumenta** la probabilidad de éxito en un $(e^{\beta_j} - 1)\times 100\%$.
- Si $\beta_j < 0 \implies \text{OR}_j = e^{\beta_j} < 1$: la característica **reduce** las *odds* de éxito en un $(1 - e^{\beta_j})\times 100\%$.

En [`src/modelos.py::entrenar_regresion_logistica`](file:///c:/Users/beltr/Documents/Universidad/Estadistica/Laboratorio%203/src/modelos.py#L220-L353) se realiza:
1. **Partición estratificada (`70% Train / 30% Test`):** `train_test_split(..., test_size=0.30, random_state=42, stratify=y)`, produciendo $n_{\text{train}} = 54,478$ y $n_{\text{test}} = 23,349$ con idéntica proporción de clases ($48.13\%$ positivos).
2. **Comparación Modelo Base vs. Modelo Optimizado (`RandomizedSearchCV`):**
   - **Modelo Base:** `LogisticRegression(max_iter=1000, random_state=42)` con umbral estándar $\tau = 0.50$.
   - **Modelo Optimizado:** Búsqueda aleatoria (`n_iter=12`, `StratifiedKFold(n_splits=5)`, métrica objetivo `scoring='f1'`) explorando la fuerza de regularización inversa $C \sim \text{LogUniform}(10^{-3}, 10^2)$, el tipo de penalización (`l1` Lasso vs. `l2` Ridge con solver `liblinear`) y el esquema de pesos (`class_weight=None` vs. `class_weight='balanced'`).

---

### 3.2 Definición Matemática de las Métricas de Clasificación

A partir de la Matriz de Confusión sobre el conjunto de prueba ($n_{\text{test}} = 23,349$):
- $\text{VN}$ (*Verdaderos Negativos*): Estudiantes con $\text{punt\_global} < 250$ correctamente clasificados como `0`.
- $\text{FP}$ (*Falsos Positivos* — Error Tipo I): Estudiantes con $\text{punt\_global} < 250$ clasificados erróneamente como `1`.
- $\text{FN}$ (*Falsos Negativos* — Error Tipo II): Estudiantes con $\text{punt\_global} \ge 250$ clasificados erróneamente como `0`.
- $\text{VP}$ (*Verdaderos Positivos*): Estudiantes con $\text{punt\_global} \ge 250$ correctamente clasificados como `1`.

Se calculan las 5 métricas estándar:
1. **Accuracy (Exactitud Global):**
   $$\text{Accuracy} = \frac{\text{VP} + \text{VN}}{\text{VP} + \text{VN} + \text{FP} + \text{FN}}$$
2. **Precision (Valor Predictivo Positivo):**
   $$\text{Precision} = \frac{\text{VP}}{\text{VP} + \text{FP}}$$
3. **Recall (Sensibilidad / Tasa de Verdaderos Positivos — TPR):**
   $$\text{Recall} = \frac{\text{VP}}{\text{VP} + \text{FN}}$$
4. **F1-Score (Media Armónica entre Precision y Recall):**
   $$\text{F1-Score} = 2 \cdot \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2\,\text{VP}}{2\,\text{VP} + \text{FP} + \text{FN}}$$
5. **ROC-AUC (*Area Under the Receiver Operating Characteristic Curve*):**
   Área bajo la curva que traza $\text{TPR}(\tau)$ frente a $\text{FPR}(\tau) = \frac{\text{FP}}{\text{FP} + \text{VN}}$ para todos los umbrales de probabilidad $\tau \in [0, 1]$. Equivale a la probabilidad de que el clasificador asigne un score mayor a un estudiante positivo elegido al azar que a uno negativo elegido al azar.

---

### 3.3 Resultados de Clasificación y Matrices de Confusión (`Test 30%`, $n = 23,349$)

| Métrica / Hiperparámetro | Modelo A (Socioeconómico) — Base | Modelo A (Socioeconómico) — Optimizado (`RandomizedSearchCV`) | Modelo A — CV 5-Fold (Media $\pm$ DE) | Modelo B (Socioec. + Inglés) — Base | Modelo B (Socioec. + Inglés) — Optimizado (`RandomizedSearchCV`) | Modelo B — CV 5-Fold (Media $\pm$ DE) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mejor Penalización / $C$** | `l2`, $C = 1.0$ | `l2`, $C = 0.3188$ | — | `l2`, $C = 1.0$ | `l1`, $C = 4.5736$ | — |
| **`class_weight`** | `None` | `'balanced'` | — | `None` | `'balanced'` | — |
| **Accuracy** | $0.6468$ ($64.68\%$) | **$0.6456$ ($64.56\%$)** | $0.6481 \pm 0.0041$ | $0.8167$ ($81.67\%$) | **$0.8152$ ($81.52\%$)** | $0.8175 \pm 0.0039$ |
| **Precision** | $0.6287$ ($62.87\%$) | **$0.6191$ ($61.91\%$)** | $0.6317 \pm 0.0053$ | $0.8126$ ($81.26\%$) | **$0.8041$ ($80.41\%$)** | $0.8145 \pm 0.0051$ |
| **Recall (Sensibilidad)** | $0.6497$ ($64.97\%$) | **$0.6850$ ($68.50\%$)** | $0.6446 \pm 0.0043$ | $0.8047$ ($80.47\%$) | **$0.8144$ ($81.44\%$)** | $0.8041 \pm 0.0059$ |
| **F1-Score** | $0.6391$ ($63.91\%$) | **$0.6504$ ($65.04\%$)** | $0.6381 \pm 0.0032$ | $0.8086$ ($80.86\%$) | **$0.8092$ ($80.92\%$)** | $0.8092 \pm 0.0041$ |
| **ROC-AUC** | $0.7014$ | **$0.7014$** | $0.7035 \pm 0.0044$ | $0.8957$ | **$0.8957$** | $0.8961 \pm 0.0032$ |
| **Verdaderos Negativos ($\text{VN}$)** | $7,801$ | **$7,376$** | — | $10,023$ | **$9,886$** | — |
| **Falsos Positivos ($\text{FP}$)** | $4,311$ | **$4,736$** | — | $2,089$ | **$2,226$** | — |
| **Falsos Negativos ($\text{FN}$)** | $3,936$ | **$3,540$** | — | $2,195$ | **$2,086$** | — |
| **Verdaderos Positivos ($\text{VP}$)** | $7,301$ | **$7,697$** | — | $9,042$ | **$9,151$** | — |

#### Interpretación del Efecto de Optimización (`RandomizedSearchCV`):
- Como la clase positiva (`supera_promedio = 1`) representa el $48.13\%$ de la muestra (ligeramente minoritaria frente al $51.87\%$ de la clase `0`), el optimizador `RandomizedSearchCV` selecciona `class_weight='balanced'` al maximizar `F1-Score`.
- En el **Modelo A**, esto incrementa la sensibilidad (**Recall**) del **$64.97\%$ al $68.50\%$** ($+396$ estudiantes exitosos adicionales correctamente detectados, pasando de $7,301$ a $7,697$ $\text{VP}$), elevando el **F1-Score** de **$0.6391$ a $0.6504$** y manteniendo un **ROC-AUC de $0.7014$**.
- En el **Modelo B**, al añadir `punt_ingles`, el **ROC-AUC** alcanza un sobresaliente **$0.8957$**, con un **Accuracy de $81.52\%$** y un **F1-Score de $0.8092$**.

---

### 3.4 Interpretación de Coeficientes Log-Odds y Odds Ratios ($e^{\beta_j}$)

#### Tabla de Odds Ratios — Modelo A (Socioeconómico Puro, Optimizado)

| Variable | Etiqueta Descriptiva | Coeficiente Log-Odds ($\hat{\beta}_j$) | Odds Ratio ($\text{OR} = e^{\hat{\beta}_j}$) | Cambio Porcentual en las Odds | Efecto sobre $P(Y=1)$ |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `const` | **Intercepto ($\beta_0$)** | $-0.4987$ | $0.6073$ | — | Probabilidad base $= 37.78\%$ |
| `cat__es_sin_estrato_Si` | Vivienda: Sin Estrato (vs. En escala E1–E6) | **$-1.0838$** | **$0.3383$** | **$-66.17\%$** | Reduce en dos tercios las odds de superar 250 pts |
| `cat__fami_tieneinternet_Si` | Internet en casa: Sí (vs. No) | **$+0.6376$** | **$1.8920$** | **$+89.20\%$** | Casi duplica las odds de superar 250 pts |
| `num__educ_madre_nivel` | Nivel educativo madre (escalado $Z$, 0–9) | **$+0.4581$** | **$1.5811$** | **$+58.11\%$** | $+58.1\%$ en odds por cada $+1\text{ DE}$ de educación materna |
| `cat__cole_naturaleza_OFICIAL` | Colegio Oficial / Público (vs. No Oficial) | **$-0.4089$** | **$0.6644$** | **$-33.56\%$** | Reduce en un tercio las odds frente a colegio privado |
| `cat__cole_area_ubicacion_URBANO` | Zona del colegio: Urbana (vs. Rural) | **$+0.3868$** | **$1.4723$** | **$+47.23\%$** | Incrementa en $47.2\%$ las odds frente a zona rural |
| `cat__educ_no_sabe_Si` | Educación madre: Reporta 'No sabe' (vs. Conocido) | **$+0.2964$** | **$1.3450$** | **$+34.50\%$** | Incrementa en $34.5\%$ las odds respecto a la mediana imputada |
| `cat__fami_tieneinternet_Desconocido` | Cuestionario ausente / MNAR (vs. Sin Internet) | **$-0.2037$** | **$0.8157$** | **$-18.43\%$** | Reduce en $18.4\%$ las odds incluso frente a no tener internet |
| `cat__estu_genero_M` | Género: Masculino (vs. Femenino) | **$+0.1945$** | **$1.2147$** | **$+21.47\%$** | Incrementa en $21.5\%$ las odds a favor de los hombres |
| `num__estrato_num` | Estrato numérico (escalado $Z$, 1–6) | **$-0.0347$** | **$0.9659$** | **$-3.41\%$** | Efecto directo marginal una vez controlados los mediadores |

#### Consistencia Total entre Regresión Lineal y Regresión Logística:
El orden de importancia y el signo de los $9$ términos del **Modelo A** y de los $10$ términos del **Modelo B** coinciden exactamente entre el modelo de Regresión Lineal (`LinearRegression`) y el modelo de Regresión Logística (`LogisticRegression`):
- En el **Modelo A**, tener internet en el hogar multiplica las *odds* de superar el promedio por **$1.8920$ ($+89.20\%$)**, mientras que vivir en una vivienda `'Sin Estrato'` las reduce a **$0.3383$ ($-66.17\%$)**.
- En el **Modelo B**, un incremento de $1$ desviación estándar en `punt_ingles` tiene un Log-Odds de **$+2.3449$**, lo que equivale a un **Odds Ratio de $10.4318$ ($+943.18\%$ en las odds de superar los $250$ puntos globales)**.

