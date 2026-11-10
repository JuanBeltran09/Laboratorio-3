# 02 · Diccionario de Variables e Ingeniería de Características

**Especialización en Analítica Estratégica de Datos · Estadística para Analítica de Datos · UPTC**  
**Profesor:** Duván Cataño · **Integrantes:** Juan Beltrán · Daniel Castellanos · Daniel Rico  

---

## 1. Origen y Especificación de la Fuente de Datos

| Aspecto | Especificación Técnica |
|---|---|
| **Nombre oficial del dataset** | Resultados únicos Saber 11 |
| **Entidad publicadora** | Instituto Colombiano para la Evaluación de la Educación (ICFES) |
| **Portal de Datos Abiertos** | https://www.datos.gov.co/d/kgxf-xxbe |
| **Endpoint API Socrata (SODA)** | `https://www.datos.gov.co/resource/kgxf-xxbe.csv` |
| **Periodo de estudio** | `20224` (aplicación calendario A, segundo semestre de 2022) |
| **Tamaño muestral bruto** | $80.000$ registros (`random_state = 42` sobre los primeros $700.000$ de la API) |
| **Tamaño muestral depurado** | **$77.827$ estudiantes únicos** (tras eliminar $2.173$ duplicados de `estu_consecutivo`) |

---

## 2. Diccionario Completo de Variables Originales (14 Columnas)

| # | Variable | Tipología Estadística | Escala / Categorías | Faltantes (en 77.827 únicos) | Rol en el Laboratorio 3 |
|---|---|---|---|---|---|
| 1 | `estu_consecutivo` | Cualitativa nominal | Código alfanumérico único (`SB1120224...`) | `0` (`0,0000 %`) | Control de duplicados (no entra al modelo) |
| 2 | `periodo` | Variable temporal | Constante `20224` | `0` (`0,0000 %`) | Filtro de cohorte (no entra al modelo) |
| 3 | `punt_global` | **Cuantitativa continua** | $0$ a $500$ puntos (observado: $12$ a $477$; media $249{,}9086$) | `0` (`0,0000 %`) | **Variable Objetivo de Regresión ($Y$)** |
| 4 | `punt_matematicas` | Cuantitativa continua | $0$ a $100$ puntos (media $50{,}6365$) | `0` (`0,0000 %`) | EDA y pruebas de hipótesis (excluida como predictor por *data leakage*) |
| 5 | `punt_lectura_critica` | Cuantitativa continua | $0$ a $100$ puntos (media $53{,}1644$) | `0` (`0,0000 %`) | EDA y pruebas de hipótesis (excluida como predictor por *data leakage*) |
| 6 | `punt_c_naturales` | Cuantitativa continua | $0$ a $100$ puntos (media $48{,}8115$) | `0` (`0,0000 %`) | EDA y pruebas de hipótesis (excluida como predictor por *data leakage*) |
| 7 | `punt_sociales_ciudadanas` | Cuantitativa continua | $0$ a $100$ puntos (media $47{,}5353$) | `0` (`0,0000 %`) | EDA y pruebas de hipótesis (excluida como predictor por *data leakage*) |
| 8 | `punt_ingles` | Cuantitativa continua | $0$ a $100$ puntos (media $50{,}1569$) | `282` (`0,3623 %`) | EDA + **Predictor en el Modelo B (Académico Ampliado)** |
| 9 | `estu_genero` | Cualitativa nominal (dicotómica) | `F` ($54{,}12\\%$), `M` ($45{,}88\\%$) | `0` (`0,0000 %`) | Predictor demográfico de control |
| 10 | `fami_estratovivienda` | Cualitativa ordinal | `Estrato 1` a `Estrato 6`, `Sin Estrato`, `Desconocido` | `5.288` (`6,7946 %`) | Fuente de `estrato_num` y `es_sin_estrato` |
| 11 | `fami_educacionmadre` | Cualitativa ordinal | $10$ niveles (`Ninguno` a `Postgrado`), `No sabe`, `No Aplica`, `Desconocido` | `4.568` (`5,8694 %`) | Fuente de `educ_madre_nivel` y `educ_no_sabe` |
| 12 | `fami_tieneinternet` | Cualitativa nominal | `Si` ($68{,}09\\%$), `No` ($25{,}94\\%$), `Desconocido` ($5{,}97\\%$) | `4.649` (`5,9735 %`) | Predictor tecnológico del hogar |
| 13 | `cole_naturaleza` | Cualitativa nominal (dicotómica) | `OFICIAL` ($77{,}13\\%$), `NO OFICIAL` ($22{,}87\\%$) | `0` (`0,0000 %`) | Predictor institucional |
| 14 | `cole_area_ubicacion` | Cualitativa nominal (dicotómica) | `URBANO` ($82{,}91\\%$), `RURAL` ($17{,}09\\%$) | `0` (`0,0000 %`) | Predictor territorial |

---

## 3. Variables Derivadas Construidas para la Modelación (7 Columnas Nuevas)

En el **Laboratorio 1** ([`diccionario_variables.md`](../Laboratorio%201/diccionario_variables.md) y [`documento-final.md`](../Laboratorio%201/documento-final.md) §10) se diseñó una estrategia de ingeniería de características que fue validada estadísticamente en el **Laboratorio 2** y programada en [`src/datos.py`](src/datos.py) (`agregar_variables_derivadas`):

### 3.1 `educ_madre_nivel` — Escala Cuantitativa Ordinal de Educación Materna ($0$ a $9$)
- **¿Qué es?** La codificación numérica entera de los $10$ peldaños jerárquicos de escolaridad de la madre:
  - `0`: Ninguno
  - `1`: Primaria incompleta
  - `2`: Primaria completa
  - `3`: Secundaria (Bachillerato) incompleta
  - `4`: Secundaria (Bachillerato) completa
  - `5`: Técnica o tecnológica incompleta
  - `6`: Técnica o tecnológica completa
  - `7`: Educación profesional incompleta
  - `8`: Educación profesional completa
  - `9`: Postgrado
  - `NaN`: Para las respuestas fuera de escala (`No sabe`, `No Aplica`, `Desconocido`), que el `num_pipeline` imputa con la mediana (`4.0`, Secundaria completa) antes de aplicar `StandardScaler()`.
- **¿Para qué sirve y cómo aplica los Laboratorios 1 y 2?** En los Laboratorios 1 y 2 se demostró que la mediana de `punt_global` crece de forma casi lineal y monótona a lo largo de estos 10 niveles (de $204$ puntos en `Ninguno` a $314$ puntos en `Postgrado`, una brecha de $110$ puntos y $\varepsilon^2 = 0.1275$). Codificarla como variable numérica continua/discreta y estandarizarla con `StandardScaler()` resume en un solo grado de libertad toda la pendiente educativa materna ($\hat{\beta} = +13{,}0018$ puntos por cada desviación estándar).

### 3.2 `estrato_num` — Escala Cuantitativa Ordinal del Estrato ($1$ a $6$)
- **¿Qué es?** La transformación de `Estrato 1` $\to 1.0$, …, `Estrato 6` $\to 6.0$, dejando como `NaN` las categorías fuera de la escala (`Sin Estrato` y `Desconocido`), las cuales son imputadas con la mediana (`2.0`, Estrato 2) y estandarizadas en el `num_pipeline`.
- **¿Para qué sirve?** Permite introducir el estrato en la rama numérica del `ColumnTransformer` y medir su efecto parcial una vez descontados la educación de la madre, la tenencia de internet y la naturaleza del colegio.

### 3.3 `es_sin_estrato` y `educ_no_sabe` — Indicadoras de Categorías Especiales Fuera de Orden
- **¿Qué son?** Variables binarias categóricas (`'Si'` / `'No'`) que identifican respectivamente:
  - `es_sin_estrato`: hogares clasificados como `'Sin Estrato'` ($2.888$ estudiantes únicos, $3{,}71\\%$).
  - `educ_no_sabe`: estudiantes que reportaron `'No sabe'` en el nivel educativo de la madre ($1.912$ estudiantes únicos, $2{,}46\\%$).
- **¿Por qué son indispensables? (Aplicación directa del Laboratorio 1):**
  En el Laboratorio 1 se descubrió que:
  1. Los estudiantes en `Sin Estrato` obtienen una mediana de **$202{,}0000$ puntos** (media $211{,}1070$), el desempeño más bajo de toda la base de datos. Si simplemente se imputaran con la mediana del estrato (`Estrato 2`, mediana $249$), se perdería una señal crítica de vulnerabilidad extrema. Gracias a `es_sin_estrato`, el modelo captura un efecto parcial exacto de **$-24{,}8730$ puntos** ($p < 0.0001$).
  2. Los estudiantes que responden `No sabe` en educación materna tienen una mediana de **$263{,}0000$ puntos** (media $261{,}8264$), superando por $+13$ puntos el promedio nacional. Gracias a `educ_no_sabe`, el modelo estima una prima de **$+8{,}6299$ puntos** ($p < 0.0001$) sobre la mediana imputada.
  3. Al separar las escalas ordinales (`educ_madre_nivel`, `estrato_num`) de estas dos indicadoras puntuales, se evita la colinealidad que ocurriría si se incluyeran simultáneamente todas las dummies de `fami_estratovivienda` y `fami_educacionmadre` junto con las variables numéricas (logrando un **$\text{VIF máximo} = 1{,}4018$**).

### 3.4 `sin_cuestionario` — Indicador de Ausencia Simultánea (Diagnóstico MNAR)
- **¿Qué es?** Vale $1$ cuando el estudiante tiene `NaN` simultáneamente en `fami_estratovivienda`, `fami_educacionmadre` y `fami_tieneinternet` ($4.370$ estudiantes únicos, $5{,}6150\\%$), y $0$ en otro caso.
- **¿Cómo se representa en el modelo?** En la rama categórica del `ColumnTransformer`, la categoría `'Desconocido'` de `fami_tieneinternet` (`cat__fami_tieneinternet_Desconocido`) actúa como el regresor que absorbe este efecto de no diligenciamiento del cuestionario (estimado en **$-10{,}1291$ puntos** adicionales respecto a no tener internet), sin repetir dummies redundantes de `'Desconocido'` en estrato y educación materna.

### 3.5 `supera_promedio` y `alto_desempeno` — Variables Objetivo para Clasificación (Regresión Logística)
- **¿Qué son?**
  $$\text{supera\_promedio} = \mathbb{I}(\text{punt\_global} \ge 250) \in \{0, 1\} \qquad\qquad \text{alto\_desempeno} = \mathbb{I}(\text{punt\_global} \ge 300) \in \{0, 1\}$$
- **¿Para qué sirven?** Permiten implementar el componente de **Clasificación Binaria con Regresión Logística** exigido en la guía del Laboratorio 3 sobre la misma base de datos Saber 11:
  - `supera_promedio` ($\ge 250$ puntos, mitad de la escala $0\text{–}500$ y media nacional $\approx 249{,}9$): Presenta un balance de clases casi perfecto (**$48{,}1273\\%$ clase 1** vs **$51{,}8727\\%$ clase 0**), ideal para clasificar si un estudiante alcanzará un desempeño satisfactorio por encima de la media nacional.
  - `alto_desempeno` ($\ge 300$ puntos, **$18{,}7724\\%$ clase 1**): Representa el segmento de excelencia académica.

---

## 4. Prevención de Fuga de Información (*Data Leakage*): Modelo A vs Modelo B

Uno de los criterios técnicos más importantes establecidos desde el Laboratorio 1 es la distinción entre variables explicativas legítimas y componentes algebraicos de la respuesta:

1. **¿Por qué no se usan los 5 puntajes por área juntos como predictores de `punt_global`?**
   El ICFES define oficialmente:
   $$\text{punt\_global} = \text{round}\left(\frac{5 \times (3\cdot\text{LC} + 3\cdot\text{MAT} + 3\cdot\text{SC} + 3\cdot\text{CN} + 1\cdot\text{ING})}{13}\right)$$
   En la sección 3.1 del Notebook se comprueba que la diferencia media entre esta fórmula y `punt_global` es apenas $0{,}2494$ puntos (producto exclusivo del redondeo al entero más cercano). Entrenar una regresión con las 5 áreas produce un $R^2 = 0.9999$ que carece de utilidad predictiva previa al examen.

2. **Diseño Dual Implementado en el Laboratorio 3:**
   - **Modelo A (Contexto Socioeconómico Puro — Modelo Principal):**
     - **Variables de entrada (8):** `fami_tieneinternet`, `cole_naturaleza`, `cole_area_ubicacion`, `estu_genero`, `es_sin_estrato`, `educ_no_sabe`, `educ_madre_nivel` y `estrato_num`.
     - **Propósito:** Medir con rigor econométrico cuánto del resultado del examen de Estado está condicionado exclusivamente por el origen socioeconómico, familiar y territorial antes de presentar cualquier prueba ($R^2 = 0.1888$, $\text{ROC-AUC} = 0.7014$).
   - **Modelo B (Contexto Socioeconómico + Competencia Transversal en Inglés `punt_ingles`):**
     - **Variables de entrada (9):** Las 8 variables del Modelo A más `punt_ingles` (que tiene $282$ datos faltantes reales y solo pondera $1/13 = 7{,}69\\%$ en la fórmula del puntaje global).
     - **Propósito:** Demostrar en la práctica la imputación numérica (`SimpleImputer('median')`) y el escalamiento multivariado (`StandardScaler`) sobre variables con rangos muy dispares ($[0, 100]$, $[0, 9]$, $[1, 6]$), así como cuantificar cuánto mejora la precisión predictiva ($R^2 = 0.6541$, $\text{ROC-AUC} = 0.8957$) cuando se dispone de una prueba diagnóstica de competencias base.

