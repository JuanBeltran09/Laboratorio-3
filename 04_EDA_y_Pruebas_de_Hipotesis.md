# 04 — Análisis Exploratorio de Datos (EDA) y Pruebas de Hipótesis

Este documento detalla exhaustivamente las **Secciones 4 y 5** del **Laboratorio 3**: el Análisis Exploratorio de Datos (EDA) univariado/bivariado y la batería de pruebas de hipótesis estadísticas implementadas en [`src/estadistica.py`](file:///c:/Users/beltr/Documents/Universidad/Estadistica/Laboratorio%203/src/estadistica.py) y visualizadas mediante [`src/graficos.py`](file:///c:/Users/beltr/Documents/Universidad/Estadistica/Laboratorio%203/src/graficos.py). Se explica **qué se hace**, **cómo se calcula matemática y computacionalmente**, **para qué sirve en la etapa previa al modelado** y **cómo aplica directamente los aprendizajes de los Laboratorios 1 y 2**.

---

## 1. Propósito del EDA y la Inferencia Antes de Modelar

En ciencia de datos rigurosa, no se entrenan modelos predictivos a ciegas. El EDA y las pruebas de hipótesis cumplen cuatro funciones críticas antes de construir los pipelines de `scikit-learn`:

1. **Verificación de supuestos distribucionales:** Determinar si las variables numéricas presentan asimetría severa, colas pesadas o valores atípicos que sesguen la estimación por Mínimos Cuadrados Ordinarios (MCO / OLS) o distorsionen el escalamiento `StandardScaler`.
2. **Validación estadística de predictores candidatos:** Confirmar mediante pruebas formales ($p < \alpha = 0.05$) y **tamaños del efecto** ($\eta^2, \omega^2, \varepsilon^2, V \text{ de Cramér}, r, \rho_s$) que cada predictor candidato tiene una asociación genuina y no azarosa con `punt_global` y `supera_promedio`.
3. **Detección de redundancia y asociación entre predictores:** Identificar qué variables categóricas comparten alta información mutua (por ejemplo, `fami_estratovivienda` con `fami_tieneinternet` y `fami_educacionmadre`) para anticipar efectos de mediación o colinealidad en el modelo multivariable.
4. **Justificación de errores estándar robustos (HC3):** La prueba de homocedasticidad de Levene y Bartlett entre grupos socioeconómicos anticipa que la varianza de `punt_global` no es constante entre estratos ni entre colegios oficiales y privados, justificando el uso de errores estándar robustos de White-MacKinnon (`HC3`) en la regresión lineal.

---

## 2. Continuidad Metodológica con los Laboratorios 1 y 2

| Componente Analítico | Lo realizado en Lab 1 y Lab 2 | Aplicación y Evolución en el Laboratorio 3 |
| :--- | :--- | :--- |
| **Estadísticos descriptivos univariados** | En Lab 2 (`src/estadistica.py::tabla_resumen_numericas`) se calcularon media, mediana, moda, varianza, desviación estándar, IQR, CV, asimetría y curtosis para `punt_global` y `punt_ingles`. | En Lab 3 se extiende el cálculo a los predictores ordinales reconstruidos (`educ_madre_nivel` y `estrato_num`) sobre los $77,827$ estudiantes únicos, verificando que su distribución post-imputación preserve la estructura empírica. |
| **Detección de atípicos por IQR (Tukey)** | En Lab 2 se identificaron outliers en `punt_ingles` bajo la regla $1.5 \times \text{IQR}$ y se demostró que correspondían a puntajes reales ($0$ y $100$). | En Lab 3 (`evaluar_impacto_outliers`) se cuantifica el cambio porcentual en la media ($\Delta\%\mu$) y en la desviación estándar ($\Delta\%\sigma$) al filtrar outliers, demostrando por qué **no** deben eliminarse antes de entrenar los modelos. |
| **Pruebas de Normalidad y Homocedasticidad** | En Lab 2 se aplicaron Kolmogorov-Smirnov, Shapiro-Wilk, D'Agostino-Pearson, Levene y Bartlett. | En Lab 3 estas pruebas guían la interpretación de los residuos del modelo OLS y justifican acompañar el ANOVA clásico con **ANOVA de Welch** y **Kruskal-Wallis**. |
| **Correlación y Asociación Categórica** | En Lab 2 se calcularon Pearson, Spearman, Kendall, $\chi^2$, $V$ de Cramér e Información Mutua. | En Lab 3 estas métricas permiten comparar la capacidad explicativa del bloque socioeconómico puro (**Modelo A**) frente a la inclusión del desempeño en inglés (**Modelo B**, donde $r = 0.7852$). |

---

## 3. Análisis Exploratorio Univariado de Variables Numéricas

### 3.1 Fórmulas Estadísticas Implementadas

En [`src/estadistica.py::tabla_resumen_numericas`](file:///c:/Users/beltr/Documents/Universidad/Estadistica/Laboratorio%203/src/estadistica.py#L22-L64) se calculan para cada variable numérica $X = \{x_1, x_2, \dots, x_n\}$:

- **Media muestral ($\bar{x}$):**
  $$\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$$
- **Varianza muestral insesgada ($s^2$) y Desviación Estándar ($s$):**
  $$s^2 = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2, \quad s = \sqrt{s^2}$$
- **Rango Intercuartílico ($\text{IQR}$):**
  $$\text{IQR} = Q_3 - Q_1 = P_{75} - P_{25}$$
- **Coeficiente de Variación ($\text{CV}\%$):**
  $$\text{CV}(\%) = \left(\frac{s}{|\bar{x}|}\right) \times 100$$
- **Coeficiente de Asimetría de Fisher-Pearson ($g_1$):**
  $$g_1 = \frac{m_3}{m_2^{3/2}}, \quad m_k = \frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^k$$
- **Exceso de Curtosis de Fisher ($g_2$):**
  $$g_2 = \frac{m_4}{m_2^2} - 3$$

### 3.2 Resultados Empíricos sobre Estudiantes Únicos ($N = 77,827$)

| Variable | $N$ Válidos | Faltantes | Media | Mediana ($Q_2$) | Moda | Desv. Est. ($s$) | Varianza ($s^2$) | Mín | $Q_1$ | $Q_3$ | Máx | Rango | IQR | CV (%) | Asimetría ($g_1$) | Exceso Curtosis ($g_2$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`punt_global`** | $77,827$ | $0$ | $248.4631$ | $245.0000$ | $232.0000$ | $51.7019$ | $2673.0858$ | $0.0$ | $210.0$ | $284.0$ | $495.0$ | $495.0$ | $74.0$ | $20.8087\%$ | $0.2423$ | $-0.4150$ |
| **`punt_ingles`** | $77,545$ | $282$ | $50.1190$ | $48.0000$ | $45.0000$ | $13.0937$ | $171.4446$ | $0.0$ | $41.0$ | $58.0$ | $100.0$ | $100.0$ | $17.0$ | $26.1252\%$ | $0.5736$ | $0.3204$ |
| **`educ_madre_nivel`** | $71,599$ | $6,228$ | $4.5739$ | $5.0000$ | $5.0000$ | $2.3134$ | $5.3518$ | $0.0$ | $3.0$ | $6.0$ | $9.0$ | $9.0$ | $3.0$ | $50.5786\%$ | $-0.0709$ | $-0.7937$ |
| **`estrato_num`** | $70,224$ | $7,603$ | $2.0945$ | $2.0000$ | $2.0000$ | $1.0155$ | $1.0312$ | $1.0$ | $1.0$ | $3.0$ | $6.0$ | $5.0$ | $2.0$ | $48.4841\%$ | $0.8809$ | $0.5831$ |

### 3.3 Interpretación Sustantiva

1. **`punt_global`:** Presenta una distribución unimodal ligeramente asimétrica positiva ($g_1 = 0.2423$, media $248.46 > \text{mediana } 245.00$) y platicúrtica ($g_2 = -0.4150$), indicando mayor dispersión en el cuerpo central que una campana gaussiana de igual varianza. La cercanía de la media ($248.46$) y la mediana ($245.00$) al umbral oficial de $250$ puntos hace que la variable objetivo binaria `supera_promedio` ($\ge 250$) tenga un balance natural de clases ($48.13\%$ clase `1` vs. $51.87\%$ clase `0`).
2. **`punt_ingles`:** Muestra mayor asimetría hacia la derecha ($g_1 = 0.5736$) y leptocurtosis moderada ($g_2 = 0.3204$), producto de la cola superior de estudiantes bilingües o de colegios privados con puntajes entre $80$ y $100$.
3. **`educ_madre_nivel`:** Su mediana y moda se sitúan en $5.0$ (*Secundaria / Bachillerato completo*), con $Q_1 = 3.0$ (*Secundaria incompleta*) y $Q_3 = 6.0$ (*Técnica o tecnológica incompleta/completa*). Los $6,228$ valores faltantes corresponden a los $4,606$ sin reporte más los $1,622$ que respondieron `'No sabe'` o `'No Aplica'`.
4. **`estrato_num`:** Fuertemente concentrado en los estratos bajos ($Q_1 = 1.0$, mediana $= 2.0$, $Q_3 = 3.0$), lo que genera una asimetría positiva ($g_1 = 0.8809$). Los $7,603$ faltantes en la escala numérica 1–6 corresponden a los $4,957$ sin respuesta más los $2,646$ hogares clasificados como `'Sin Estrato'`.

---

## 4. Evaluación de Valores Atípicos (Criterio de Tukey $1.5 \times \text{IQR}$)

En [`src/estadistica.py::evaluar_impacto_outliers`](file:///c:/Users/beltr/Documents/Universidad/Estadistica/Laboratorio%203/src/estadistica.py#L67-L115), un valor $x_i$ se clasifica como atípico estadístico si cae fuera de las vallas de Tukey:

$$\text{Límite Inferior} = Q_1 - 1.5 \times \text{IQR}, \qquad \text{Límite Superior} = Q_3 + 1.5 \times \text{IQR}$$

### Resultados del Diagnóstico de Outliers

| Variable | Límite Inferior | Límite Superior | $N$ Outliers | % Outliers | Media Original | Media Sin Outliers | $\Delta\%$ Media | Desv. Est. Original | Desv. Est. Sin Outliers | $\Delta\%$ Desv. Est. |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`punt_global`** | $99.0000$ | $395.0000$ | $228$ | $0.2930\%$ | $248.4631$ | $248.0360$ | $-0.1719\%$ | $51.7019$ | $51.1474$ | $-1.0725\%$ |
| **`punt_ingles`** | $15.5000$ | $83.5000$ | $1,089$ | $1.4043\%$ | $50.1190$ | $49.6079$ | $-1.0198\%$ | $13.0937$ | $12.3168$ | $-5.9334\%$ |
| **`educ_madre_nivel`** | $-1.5000$ | $10.5000$ | $0$ | $0.0000\%$ | $4.5739$ | $4.5739$ | $0.0000\%$ | $2.3134$ | $2.3134$ | $0.0000\%$ |
| **`estrato_num`** | $-2.0000$ | $6.0000$ | $0$ | $0.0000\%$ | $2.0945$ | $2.0945$ | $0.0000\%$ | $1.0155$ | $1.0155$ | $0.0000\%$ |

### Decisión Metodológica para el Modelado

- En `punt_global`, apenas el **$0.2930\%$** ($228$ estudiantes) supera los $395$ puntos o cae por debajo de $99$ puntos. Su impacto en la media es de apenas $-0.1719\%$.
- En `punt_ingles`, el **$1.4043\%$** ($1,089$ estudiantes) obtiene más de $83.5$ puntos o menos de $15.5$ puntos.
- **No son errores de medición ni de digitación:** representan a los estudiantes de excelencia académica (becas *Generación E*, colegios bilingües) y a los estudiantes en vulnerabilidad extrema. Eliminar estos registros truncaría artificialmente la varianza real del examen y sesgaría los coeficientes de regresión hacia el centro (sesgo de selección por truncamiento en $Y$). Por tanto, **se conservan todos los $77,827$ estudiantes únicos**.

---

## 5. Análisis Exploratorio Categórico y Bivariado frente a `punt_global`

Utilizando [`src/estadistica.py::estadisticos_por_grupo`](file:///c:/Users/beltr/Documents/Universidad/Estadistica/Laboratorio%203/src/estadistica.py#L140-L162), se examina cómo varía el puntaje global (`punt_global`) y la tasa de éxito (`supera_promedio`) a través de cada predictor categórico.

### 5.1 Brechas Socioeconómicas e Institucionales Observadas

1. **Acceso a Internet en el Hogar (`fami_tieneinternet`):**
   - **Con Internet (`Si`, $n = 51,699$, $66.43\%$):** Media de `punt_global` $= \mathbf{260.41}$ puntos (Mediana $= 258.0$).
   - **Sin Internet (`No`, $n = 21,367$, $27.45\%$):** Media de `punt_global` $= \mathbf{226.43}$ puntos (Mediana $= 223.0$).
   - **Desconocido / Cuestionario Ausente ($n = 4,761$, $6.12\%$):** Media de `punt_global` $= \mathbf{217.59}$ puntos (Mediana $= 212.0$).
   - **Brecha bruta:** $+33.98$ puntos a favor de quienes tienen internet frente a quienes no tienen, y $+42.82$ puntos frente al grupo con cuestionario ausente (MNAR).

2. **Naturaleza del Colegio (`cole_naturaleza`):**
   - **No Oficial / Privado ($n = 18,495$, $23.76\%$):** Media $= \mathbf{273.68}$ puntos (Mediana $= 274.0$).
   - **Oficial / Público ($n = 59,332$, $76.24\%$):** Media $= \mathbf{240.60}$ puntos (Mediana $= 237.0$).
   - **Brecha bruta:** $+33.08$ puntos a favor de los colegios privados.

3. **Área de Ubicación del Colegio (`cole_area_ubicacion`):**
   - **Urbano ($n = 65,134$, $83.69\%$):** Media $= \mathbf{253.00}$ puntos (Mediana $= 250.0$).
   - **Rural ($n = 12,693$, $16.31\%$):** Media $= \mathbf{225.18}$ puntos (Mediana $= 221.0$).
   - **Brecha bruta:** $+27.82$ puntos a favor de la zona urbana.

4. **Género del Estudiante (`estu_genero`):**
   - **Masculino (`M`, $n = 35,436$, $45.53\%$):** Media $= \mathbf{251.81}$ puntos (Mediana $= 249.0$).
   - **Femenino (`F`, $n = 42,381$, $54.46\%$):** Media $= \mathbf{245.67}$ puntos (Mediana $= 242.0$).
   - **Brecha bruta:** $+6.14$ puntos a favor de los hombres.

5. **Estrato Socioeconómico (`fami_estratovivienda`):**
   - **Sin Estrato ($n = 2,646$):** Media $= \mathbf{208.53}$ puntos (la más baja de toda la distribución).
   - **Desconocido ($n = 4,957$):** Media $= \mathbf{220.47}$ puntos.
   - **Estrato 1 ($n = 23,040$):** Media $= \mathbf{234.48}$ puntos.
   - **Estrato 2 ($n = 26,862$):** Media $= \mathbf{252.87}$ puntos.
   - **Estrato 3 ($n = 15,075$):** Media $= \mathbf{266.22}$ puntos.
   - **Estrato 4 ($n = 3,663$):** Media $= \mathbf{281.36}$ puntos.
   - **Estrato 5 ($n = 1,071$):** Media $= \mathbf{291.77}$ puntos.
   - **Estrato 6 ($n = 513$):** Media $= \mathbf{298.81}$ puntos.
   - **Brecha bruta Estrato 6 vs. Estrato 1:** $+64.33$ puntos; **Estrato 6 vs. Sin Estrato:** $+90.28$ puntos.

6. **Educación de la Madre (`fami_educacionmadre`):**
   - Gradiente monotónico estricto desde **Ninguno** ($204.60$ puntos) y **Primaria incompleta** ($221.75$ puntos), pasando por **Secundaria completa** ($246.33$ puntos) y **Técnica/tecnológica completa** ($265.11$ puntos), hasta **Educación profesional completa** ($277.31$ puntos) y **Postgrado** ($\mathbf{296.44}$ puntos).
   - **Brecha bruta Postgrado vs. Ninguno:** $+91.84$ puntos.

---

## 6. Batería de Pruebas de Hipótesis e Interpretación Estadística

Todas las pruebas de hipótesis se ejecutan con nivel de significancia $\alpha = 0.05$ a través de las funciones de [`src/estadistica.py`](file:///c:/Users/beltr/Documents/Universidad/Estadistica/Laboratorio%203/src/estadistica.py).

### 6.1 Pruebas de Normalidad Univariada (`pruebas_normalidad`)

- **Hipótesis:**
  - $H_0$: La variable $X$ proviene de una distribución normal $\mathcal{N}(\mu, \sigma^2)$.
  - $H_1$: La variable $X$ **no** sigue una distribución normal.
- **Estadísticos aplicados:**
  1. **Kolmogorov-Smirnov ($D_{KS}$):** Mide la distancia suprema entre la función de distribución empírica $F_n(x)$ y la normal teórica $\Phi\left(\frac{x-\bar{x}}{s}\right)$ sobre los $N$ datos completos.
  2. **Shapiro-Wilk ($W$, submuestra aleatoria $n = 5,000$, `random_state=42`):** Contrasta la covarianza entre los estadísticos de orden y los cuantiles normales esperados (restringido a $5,000$ porque el algoritmo AS R94 de `scipy.stats.shapiro` pierde precisión de punto flotante para $N > 5000$).
  3. **D'Agostino-Pearson ($K^2 = Z_{g_1}^2 + Z_{g_2}^2$):** Combina las transformaciones normales de la asimetría ($g_1$) y la curtosis ($g_2$) en un estadístico $\chi^2$ con $2$ grados de libertad sobre toda la muestra.

| Variable | $D$ (KS) | $p$-valor KS | $W$ (Shapiro, $n=5000$) | $p$-valor Shapiro | $K^2$ (D'Agostino) | $p$-valor D'Agostino | Decisión ($\alpha=0.05$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`punt_global`** | $0.0294$ | $< 0.0001$ | $0.9923$ | $< 0.0001$ | $1497.5898$ | $< 0.0001$ | Rechaza $H_0$ |
| **`punt_ingles`** | $0.0589$ | $< 0.0001$ | $0.9781$ | $< 0.0001$ | $3991.8546$ | $< 0.0001$ | Rechaza $H_0$ |
| **`educ_madre_nivel`** | $0.1656$ | $< 0.0001$ | $0.9469$ | $< 0.0001$ | $5369.5105$ | $< 0.0001$ | Rechaza $H_0$ |
| **`estrato_num`** | $0.2337$ | $< 0.0001$ | $0.8482$ | $< 0.0001$ | $7941.3912$ | $< 0.0001$ | Rechaza $H_0$ |

> [!IMPORTANT]
> **Lección estadística del Laboratorio 2 aplicada al Laboratorio 3:** Con tamaños muestrales masivos ($N = 77,827$), el error estándar de los estadísticos de forma tiende a cero ($\text{SE} \propto 1/\sqrt{N}$), por lo que cualquier desviación mínima respecto a la campana gaussiana produce $p < 0.0001$. Sin embargo, el estadístico $W$ de Shapiro-Wilk para `punt_global` es **$0.9923$** (muy cercano a $1.0$) y la distancia máxima de Kolmogorov-Smirnov es apenas **$2.94\%$** ($D = 0.0294$), confirmando que `punt_global` es cuasi-normal en su forma práctica. Aun así, dado el rechazo formal de $H_0$, en todas las pruebas bivariadas acompañamos los métodos paramétricos con sus equivalentes **no paramétricos** (Spearman, Kendall, Kruskal-Wallis).

---

### 6.2 Pruebas de Correlación Numérica (`calcular_correlaciones`)

- **Hipótesis:**
  - $H_0$: No existe asociación lineal ($r = 0$) ni monótona ($\rho_s = 0, \tau_b = 0$) entre $X$ e $Y$.
  - $H_1$: Existe asociación estadísticamente significativa ($r \neq 0, \rho_s \neq 0, \tau_b \neq 0$).

| Par de Variables | $N$ Pares | Pearson ($r$) | $p$-valor Pearson | Spearman ($\rho_s$) | $p$-valor Spearman | Kendall ($\tau_b$) | $p$-valor Kendall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`punt_global` vs. `punt_ingles`** | $77,545$ | **$0.7852$** | $< 0.0001$ | **$0.7667$** | $< 0.0001$ | **$0.5835$** | $< 0.0001$ |
| **`punt_global` vs. `educ_madre_nivel`** | $71,599$ | **$0.3694$** | $< 0.0001$ | **$0.3692$** | $< 0.0001$ | **$0.2681$** | $< 0.0001$ |
| **`punt_global` vs. `estrato_num`** | $70,224$ | **$0.2651$** | $< 0.0001$ | **$0.2625$** | $< 0.0001$ | **$0.2007$** | $< 0.0001$ |
| **`punt_ingles` vs. `educ_madre_nivel`** | $71,440$ | **$0.4090$** | $< 0.0001$ | **$0.3976$** | $< 0.0001$ | **$0.2909$** | $< 0.0001$ |
| **`punt_ingles` vs. `estrato_num`** | $70,061$ | **$0.3510$** | $< 0.0001$ | **$0.3204$** | $< 0.0001$ | **$0.2462$** | $< 0.0001$ |
| **`educ_madre_nivel` vs. `estrato_num`** | $68,269$ | **$0.4272$** | $< 0.0001$ | **$0.4239$** | $< 0.0001$ | **$0.3396$** | $< 0.0001$ |

#### Hallazgos Clave para la Modelación:
1. **`punt_ingles` tiene una correlación positiva muy fuerte con `punt_global` ($r = 0.7852$, $r^2 = 0.6165$):** Por sí solo, el puntaje de inglés comparte más del $61.6\%$ de la varianza lineal del puntaje global.
2. **Dentro del bloque socioeconómico, `educ_madre_nivel` ($r = 0.3694$) supera en capacidad predictiva a `estrato_num` ($r = 0.2651$):** El capital cultural del hogar (escolaridad materna) está más estrechamente ligado al logro cognitivo que la estratificación tarifaria del inmueble.
3. **Correlación moderada entre predictores socioeconómicos ($r = 0.4272$ entre `educ_madre_nivel` y `estrato_num`):** Es suficientemente alta para generar **mediación estadística** cuando entran juntos al modelo OLS, pero suficientemente baja ($r < 0.70$) para no causar multicolinealidad severa ($\text{VIF} \approx 1.22$).
4. **Mediación de `punt_ingles`:** Obsérvese que `punt_ingles` correlaciona fuertemente con `educ_madre_nivel` ($r = 0.4090$) y con `estrato_num` ($r = 0.3510$). Esto anticipa que al incluir `punt_ingles` en el **Modelo B**, gran parte del efecto socioeconómico e institucional será absorbido por el desempeño en inglés.

---

### 6.3 Asociación entre Variables Categóricas (`asociacion_categoricas`)

- **Hipótesis de la prueba $\chi^2$ de Pearson:**
  - $H_0$: Las variables categóricas $A$ y $B$ son estocásticamente independientes ($P(A=i, B=j) = P(A=i)P(B=j)$).
  - $H_1$: Existe dependencia estadística entre $A$ y $B$.
- **Métricas de magnitud de asociación:**
  - **$V$ de Cramér:**
    $$V = \sqrt{\frac{\chi^2}{N \cdot \min(R-1, C-1)}} \in [0, 1]$$
  - **Información Mutua ($I(A; B)$ en nats):**
    $$I(A; B) = \sum_{i=1}^{R}\sum_{j=1}^{C} p(i, j) \ln\left(\frac{p(i, j)}{p(i)p(j)}\right)$$

#### Pares de Mayor Dependencia Estructural ($N = 77,827$):

| Variable 1 | Variable 2 | $\chi^2$ | gl | $p$-valor | $V$ de Cramér | Información Mutua (nats) | Interpretación |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `fami_estratovivienda` | `fami_tieneinternet` | $90,656.60$ | $14$ | $< 0.0001$ | **$0.7632$** | **$0.3478$** | Asociación muy fuerte (acceso a internet ligado al estrato y bloque MNAR compartido). |
| `fami_educacionmadre` | `fami_tieneinternet` | $73,294.77$ | $24$ | $< 0.0001$ | **$0.6862$** | **$0.2896$** | Asociación fuerte entre capital cultural y conectividad digital. |
| `fami_estratovivienda` | `fami_educacionmadre` | $91,173.49$ | $84$ | $< 0.0001$ | **$0.4085$** | **$0.3712$** | Mayor información mutua compartida entre variables del hogar. |
| `fami_tieneinternet` | `cole_area_ubicacion` | $10,331.49$ | $2$ | $< 0.0001$ | **$0.3643$** | **$0.0628$** | Brecha digital urbano-rural marcada. |
| `fami_estratovivienda` | `cole_naturaleza` | $10,275.34$ | $7$ | $< 0.0001$ | **$0.3634$** | **$0.0642$** | Segregación socioeconómica entre colegios oficiales y privados. |
| `fami_tieneinternet` | `supera_promedio` | $6,931.70$ | $2$ | $< 0.0001$ | **$0.2984$** | **$0.0455$** | Fuerte asociación directa con aprobar el umbral de 250 puntos. |
| `cole_naturaleza` | `supera_promedio` | $4,918.08$ | $1$ | $< 0.0001$ | **$0.2514$** | **$0.0319$** | Los colegios privados duplican la tasa de superación del promedio. |

---

### 6.4 Pruebas de Homocedasticidad (`pruebas_homocedasticidad`)

- **Hipótesis:**
  - $H_0$: $\sigma_1^2 = \sigma_2^2 = \dots = \sigma_k^2$ (la varianza de `punt_global` es constante entre todos los grupos de la variable categórica).
  - $H_1$: Al menos un grupo presenta varianza diferente ($\exists i \neq j : \sigma_i^2 \neq \sigma_j^2$).
- **Estadísticos:**
  - **Prueba de Levene (Brown-Forsythe, centrada en la mediana):** Robusta frente a desviaciones de la normalidad.
  - **Prueba de Bartlett:** Basada en la razón de verosimilitud normal (más sensible a la curtosis).

| Variable de Agrupación (sobre `punt_global`) | $k$ Grupos | Estadístico Levene ($W$) | $p$-valor Levene | Estadístico Bartlett ($T$) | $p$-valor Bartlett | Conclusión |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`fami_estratovivienda`** | $8$ | $152.0835$ | $< 0.0001$ | $1044.9189$ | $< 0.0001$ | Heterocedasticidad |
| **`fami_educacionmadre`** | $13$ | $104.3411$ | $< 0.0001$ | $1224.8967$ | $< 0.0001$ | Heterocedasticidad |
| **`fami_tieneinternet`** | $3$ | $389.3879$ | $< 0.0001$ | $742.5482$ | $< 0.0001$ | Heterocedasticidad |
| **`cole_naturaleza`** | $2$ | $356.5161$ | $< 0.0001$ | $353.0784$ | $< 0.0001$ | Heterocedasticidad |
| **`cole_area_ubicacion`** | $2$ | $375.3353$ | $< 0.0001$ | $386.6499$ | $< 0.0001$ | Heterocedasticidad |
| **`estu_genero`** | $2$ | $155.1278$ | $< 0.0001$ | $137.9892$ | $< 0.0001$ | Heterocedasticidad |

> [!NOTE]
> **Impacto directo en el Laboratorio 3:** El rechazo unánime de la homocedasticidad ($p < 0.0001$) demuestra que la dispersión del puntaje global cambia según el estrato, la naturaleza del colegio y el acceso a internet (los estratos altos y colegios privados tienen mayor dispersión por la coexistencia de colegios de élite y colegios privados de bajo costo). Por esta razón:
> 1. En el análisis bivariado usamos **ANOVA de Welch ($F_W$)**, que no asume varianzas iguales.
> 2. En la inferencia del modelo de Regresión Lineal Múltiple reportamos los **errores estándar robustos a la heterocedasticidad de White-MacKinnon (`HC3`)**.

---

### 6.5 Comparación de Medias y Medianas entre Grupos: ANOVA, Welch y Kruskal-Wallis (`pruebas_diferencia_grupos`)

- **Hipótesis:**
  - **ANOVA ($F$) y ANOVA de Welch ($F_W$):** $H_0: \mu_1 = \mu_2 = \dots = \mu_k$ vs. $H_1: \exists i \neq j \text{ tal que } \mu_i \neq \mu_j$.
  - **Kruskal-Wallis ($H$):** $H_0:$ Las $k$ poblaciones tienen la misma distribución/mediana de rangos vs. $H_1:$ Al menos un grupo domina estocásticamente a otro.
- **Tamaños del Efecto Calculados (indispensables cuando $N = 77,827$):**
  - **Eta cuadrado ($\eta^2$):** Proporción de la suma de cuadrados total explicada por el factor:
    $$\eta^2 = \frac{\text{SS}_{\text{entre}}}{\text{SS}_{\text{total}}}$$
  - **Omega cuadrado ($\omega^2$):** Estimador insesgado de la proporción de varianza poblacional explicada:
    $$\omega^2 = \frac{\text{SS}_{\text{entre}} - (k-1)\text{MS}_{\text{dentro}}}{\text{SS}_{\text{total}} + \text{MS}_{\text{dentro}}}$$
  - **Épsilon cuadrado ($\varepsilon^2$):** Tamaño del efecto no paramétrico basado en el estadístico $H$ de Kruskal-Wallis:
    $$\varepsilon^2 = \frac{H - k + 1}{N - k}$$

#### Resultados Consolidados sobre `punt_global`:

| Factor Categórico | $k$ | $F$ (ANOVA) | $F_W$ (Welch) | $H$ (Kruskal-Wallis) | $p$-valor (todos) | $\eta^2$ | $\omega^2$ | $\varepsilon^2$ | Magnitud del Efecto (Cohen) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`fami_educacionmadre`** | $13$ | $1159.1042$ | $1094.8519$ | $11,945.2891$ | $< 0.0001$ | **$0.1516$** | **$0.1514$** | **$0.1533$** | **Grande** ($\eta^2 > 0.14$) |
| **`fami_tieneinternet`** | $3$ | $4719.2248$ | $4981.3320$ | $8,842.1904$ | $< 0.0001$ | **$0.1082$** | **$0.1081$** | **$0.1136$** | **Moderado-Alto** ($\eta^2 \approx 0.11$) |
| **`fami_estratovivienda`** | $8$ | $1278.4901$ | $1251.0844$ | $8,391.4412$ | $< 0.0001$ | **$0.1031$** | **$0.1030$** | **$0.1077$** | **Moderado-Alto** ($\eta^2 \approx 0.10$) |
| **`cole_naturaleza`** | $2$ | $6291.8741$ | $5688.9102$ | $5,698.3129$ | $< 0.0001$ | **$0.0748$** | **$0.0748$** | **$0.0732$** | **Moderado** ($\eta^2 \approx 0.075$) |
| **`cole_area_ubicacion`** | $2$ | $3198.4125$ | $3564.1180$ | $3,218.7745$ | $< 0.0001$ | **$0.0395$** | **$0.0395$** | **$0.0413$** | **Pequeño-Moderado** ($\eta^2 \approx 0.04$) |
| **`estu_genero`** | $2$ | $271.4918$ | $269.8312$ | $264.1190$ | $< 0.0001$ | **$0.0035$** | **$0.0035$** | **$0.0034$** | **Pequeño** ($\eta^2 < 0.01$) |

---

## 7. Síntesis Inferencial: De las Pruebas de Hipótesis a los Modelos

La jerarquía de tamaños del efecto ($\eta^2, \omega^2, \varepsilon^2$) obtenida en la Sección 5 predice con exactitud matemática el comportamiento que observaremos en las Secciones 7 y 8 al entrenar los modelos de Regresión Lineal y Regresión Logística:

1. **Jerarquía de importancia dentro del bloque socioeconómico (Modelo A):**
   - `fami_educacionmadre` ($\eta^2 = 15.16\%$), `fami_tieneinternet` ($\eta^2 = 10.82\%$), `fami_estratovivienda` / `es_sin_estrato` ($\eta^2 = 10.31\%$) y `cole_naturaleza` ($\eta^2 = 7.48\%$) son los cuatro pilares del **Modelo A**.
   - Como estos cuatro factores comparten alta información mutua (Sección 6.3), la suma directa de sus $\eta^2$ individuales ($0.1516 + 0.1082 + 0.1031 + 0.0748 + 0.0395 + 0.0035 = 0.4807$) tiene una fuerte superposición, resultando en una varianza explicada conjunta neta de **$R^2 = 0.1888$ ($18.88\%$)** en el conjunto de prueba.
2. **Salto de poder predictivo al incluir `punt_ingles` (Modelo B):**
   - La correlación lineal de `punt_ingles` con `punt_global` ($r = 0.7852 \implies r^2 = 0.6165$) explica por qué al añadir esta única competencia al pipeline (**Modelo B**), el $R^2$ salta de $0.1888$ a **$0.6541$ ($65.41\%$)** en regresión lineal, y el $\text{ROC-AUC}$ sube de $0.7014$ a **$0.8957$** en regresión logística.

