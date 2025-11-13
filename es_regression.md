---
language: es
title: "Module 2: Machine Learning for Regression"
author: Daniel Neira
---
> Build a car price prediction model while learning linear regression, feature engineering, and regularization.
>
> Topics:
>
> - Linear regression (from scratch and with scikit-learn)
> - Exploratory data analysis
> - Feature engineering
> - Regularization techniques
> - Model validation

# Car price prediction project

## Material

- [Video](https://www.youtube.com/watch?v=vM3SqPNlStE) (5:36)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-21-car-price-prediction-project)
- [Cuaderno de Jupyter del video](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/chapter-02-car-price/02-carprice.ipynb)
- [Cuaderno de Jupyter adicional](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/02-regression/notebook.ipynb)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/02-regression/01-car-price-intro.md)

## Notas

- Abordaremos un problema práctico: predecir el precio de venta de un auto
- Utilizaremos un conjunto de datos alojados en [Kaggle](https://www.kaggle.com/CooperUnion/cardataset)
- Cubriremos los siguientes temas:
  - Preparar los datos y realizar un análisis exploratorio de ellos, EDA ("Exploratory Data Analysis")
  - Usar regresión lineal para predecir los precios
  - Entender cómo funciona la regresión lineal
  - Evaluar modelo usando la raíz del error cuadrático medio, RMSE ("Root Mean Square Error")
  - Realizar ingeniería de atributos ("feature engineering")
  - Aplicar regularización al modelo
  - Usar el modelo

# Data preparation

## Material

- [Video](https://www.youtube.com/watch?v=Kd74oR4QWGM) (9:26)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-2-slides)
- [Cuaderno de Jupyter del video](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/chapter-02-car-price/02-carprice.ipynb)
- [Cuaderno de Jupyter adicional](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/02-regression/notebook.ipynb)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/02-regression/02-data-preparation.md)

## Notas

- Los títulos de las columnas de los datos originales no tienen un formato común
  - Algunas tienen las palabras separadas por espacios
  - Otras por guiones bajos
  - Y no usan las mayúsculas y minúsculas de manera consistente
- Podemos formatear los títulos para usar guiones bajos en lugar de espacios y llevar todo el texto a minúsculas con la siguiente instrucción:
  - `df.columns.str.lower().str.replace(" ", "_")`
    - `df`: el _dataframe_ que contiene los datos
    - `df.columns`: el objeto que contiene los títulos de las columnas de `df`
    - `.str.lower()`: método asociado a series de Pandas para transformar el texto contenido en el objeto a minúsculas
    - `.str.replace(" ", "_")`: como antes, pero ahora reemplazando espacios por guiones bajos
  - Para hacer este cambio permanente, tenemos que guardar el resultado en el _dataframe_ original:
  - `df.columns = df.columns.str.lower().str.replace(" ", "_")`
  - Esta es la única forma pues estas operaciones no pueden aplicarse _in situ_
- Podemos hacer lo mismo con los datos operando sobre `df.dtypes[df.dtypes == 'object'].index`
  - Allí seleccionamos las columnas de tipo `object`, las que usualmente contienen _strings_
  - El resultado es una serie de Pandas, de la cual nos interesa su índice
  - Iteraremos sobre ese índice y aplicaremos las transformaciones a cada una de las columnas

# Exploratory data analysis

## Material

- [Video](https://www.youtube.com/watch?v=k6k8sQ0GhPM) (18:35)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-2-slides)
- [Cuaderno de Jupyter del video](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/chapter-02-car-price/02-carprice.ipynb)
- [Cuaderno de Jupyter adicional](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/02-regression/notebook.ipynb)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/02-regression/03-eda.md)

## Notas

- Métodos de utilidad:
  - `df[COLUMNA].unique`: retorna los valores únicos almacenados en `df[COLUMNA]`
  - `df[COLUMNA].nunique`: retorna el número de valores únicos almacenados en `df[COLUMNA]`
    - Algo así como `len(df[COLUMNA].unique)`
  - `df[COLUMN].isnull()`: retorna una serie _booleana_ con `False` para aquellos elementos que no son `null` y `True` en caso contrario
    - Podemos también aplicarla a un _dataframe_ completo: `df.isnull()`
    - Independiente de que usemos el método sobre una serie de Pandas o un _dataframe_, lo usual es encadenar el resultado a `sum`: `df[COLUMNA].isnull().sum()`
    - En esa operación, `True` es interpretado como 1 y `False` como 0
    - El resultado es, entonces, un conteo de los valores nulos
- Ejemplo de selección de filas de un _dataframe_:
  - Opción usual: `df[COLUMNA][df[COLUMNA] > VALOR]`
  - (_Bonus_) Opción de mejor desempeño: `df.loc[df[COLUMNA] > VALOR, COLUMNA]`
- Cómo importar `matplotlib`: `import matplotlib.pyplot as plt`
- Distribuciones de datos:
  - De cola larga (_long tail_): algunos de los datos están muy lejos del grueso de los demás
    - (_Bonus_) Estas distribuciones no se ajustan a modelos lineales
    - Podemos transformar los datos para que ellos se distribuyan con mayor simetría
    - Lo usual es transformarlos con una función logarítmica
    - Y, en la práctica, usaremos la transformación $y(x)=\log(x+1)$ para evitar problemas cuando $x^+ \to 0$
    - En NumPy, esta es la función `np.log1p`
    - (_Bonus_) Si la distribución original no es unimodal, la transformación no será necesariamente capaz de hacerla más simétrica

# Setting up the validation framework

## Material

- [Video](https://www.youtube.com/watch?v=ck0IfiPaQi0) (17:39)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-2-slides)
- [Cuaderno de Jupyter del video](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/chapter-02-car-price/02-carprice.ipynb)
- [Cuaderno de Jupyter adicional](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/02-regression/notebook.ipynb)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/02-regression/04-validation-framework.md)

## Notas

- Debemos particionar el conjunto de datos en datos de entrenamiento, validación y prueba
- Existen distintas alternativas para hacerlo
- La idea es sencilla: definir el tamaño de cada conjunto y luego extraer los datos al azar y sin reposición del conjunto de datos original
- Es importante que extraigamos los datos sin reposición: eso hace que los conjuntos de datos de la partición no se traslapen
- Y también es importante que seleccionemos los datos al azar
  - Los datos originales podrían estar ordenados de acuerdo a algún criterio
  - Extraer los datos sin azar de por medio podría hacer que cada partición no sea realmente representativa de todos los datos originales
  - Queremos que nuestras particiones de datos sean lo más heterogéneas posibles
- Algunas soluciones prácticas para realizar el particionamiento:
  - Usando la función `shuffle` de NumPy:
    ```python
    n = len(df)

    # partición 60 % entrenamiento, 20 % validación y 20 % prueba
    n_val = int(0.2 * n)
    n_test = int(0.2 * n)
    n_train = n - (n_val + n_test)
    idx = np.arange(n)

    # desordenar el índice `idx`
    np.random.seed(SEMILLA)  # si quisiéramos usar una semilla para generar los números pseudoaleatorios
    np.random.shuffle(idx)  # para desordenar el índice de forma pseudoaleatoria

    # alternativa moderna
    # rng = np.random.default_rng()  # le podemos pasar una semilla como argumento
    # rng.shuffle(idx)  # como `np.random.shuffle`, también es una operación in situ

    # crear las particiones
    df_train = df.iloc[idx[:n_train]].reset_index(drop=True)  # no nos interesa el índice
    df_val = df.iloc[idx[n_train:n_train+n_val]].reset_index(drop=True)
    df_test = df.iloc[idx[n_train+n_val:]].reset_index(drop=True)
    ```
  - Usando las funciones `split` de NumPy y `sample` de Pandas (60 %, 20 %, 20 %, nuevamente):
    ```python
    n = len(df)
    n_train = int(0.6 * n)
    n_val = int(0.2 * n)
    # los dataframes conservarán los índices revueltos
    # podemos pasarle un generador de números pseudoaleatorios `rng` a `df.sample` como parámetro `random_state`
    df_train, df_val, df_test = np.split(df.sample(n), [n_train, n_train + n_val])
    ```
  - Ver más alternativas: [How to split data into 3 sets (train, validation and test)?](https://stackoverflow.com/q/38250710) (Stack Overflow)
- Tendremos que volver a procesar los conjuntos de datos resultantes para formar la matriz de atributos $X$ y el vector de salida $y$
  - Por ejemplo:
    ```python
    target_column = 'OBJETIVO'
    X = df.drop(columns=[target_column]).to_numpy()  # `.drop` retorna una copia por omisión
    y = df[target_column].to_numpy(copy=True)  # nos aseguramos de trabajar con una copia
    ```
  - (_Bonus_) Dado que no modificaremos los datos, podríamos también trabajar con referencias a los datos
  - (_Bonus_) El modelo de memoria de Pandas es opaco y no se puede tener certeza de que estemos trabajando con copias a menos que copiemos los datos de forma explícita con `.copy()`
  - (_Bonus_) Pero no debería haber problema con, por ejemplo, crear `X` con `X = df[[COLUMNA_a, COLUMNA_b, ...]].to_numpy()`, excluyendo la columna de la variable objetivo de la _slice_

# Linear regression

## Material

- [Video](https://www.youtube.com/watch?v=Dn1eTQLsOdA) (19:10)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-2-slides)
- [Cuaderno de Jupyter del video](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/chapter-02-car-price/02-carprice.ipynb)
- [Cuaderno de Jupyter adicional](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/02-regression/notebook.ipynb)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/02-regression/05-linear-regression-simple.md)

## Notas

- Recordemos que el modelo es $g(X)\approx y$
- En el caso de la regresión lineal, $g$ será una combinación lineal de $X$
- Aterrizándolo: tomemos $x_i=(x_{i1}, \ldots, x_{in})$, la $i$-ésima observación almacenada en la matriz de atributos
- La combinación lineal será en nuestro caso una suma ponderada con un sesgo $w_0$:
  - $g(x_i)=w_0+w_1 x_{i1}+\ldots+w_n x_{in}=w_0+\sum_{j=1}^n w_j x_{ij}=w_0 + w_{1,\ldots,n}^\top x_i$
- El problema se reduce a encontrar los coeficientes $w_0, w_1, \ldots, w_n$ que generan la mejor predicción de $y_i$ bajo cierto criterio (como podría ser RMSE en el conjunto de validación)
- En caso de que hayamos transformado $y$ durante el modelamiento del problema (por ejemplo, usando `np.log1p`), tendremos que "deshacer" la transformación para los $y_p$ ($y$ predichos)
  - En el caso específico de la transformación `np.log1p`, usaremos `np.expm1` para llevar las predicciones al mismo espacio de la variable objetivo original

# Linear regression: vector form

## Material

- [Video](https://www.youtube.com/watch?v=YkyevnYyAww) (14:12)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-2-slides)
- [Cuaderno de Jupyter del video](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/chapter-02-car-price/02-carprice.ipynb)
- [Cuaderno de Jupyter adicional](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/02-regression/notebook.ipynb)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/02-regression/06-linear-regression-vector.md)

## Notas

- Podemos hacer $g(X)$ más compacto que lo que vimos en la sección anterior:
  - $g(x_i)=w_0 + w_{1,\ldots,n}^\top x_i=x_i^\top w$
    - $w=(w_0, w_1, \ldots, w_n)^\top$
    - $x_i=(1, x_1, x_2, \ldots, x_n)^\top$
- Y podemos generalizar esto a la matriz de atributos, a la que agregaremos una columa de 1 para poder incorporar $w_0$ en la fórmula
  - $g(X)=Xw = y_p$
    - $X=\begin{bmatrix} 1 & x_{11} & \ldots & x_{1n} \\ \vdots &  & \ddots \\ 1 & x_{m1} & \ldots & x_{mn} \end{bmatrix}$, con $X_{m \times (n+1)}$
    - $w=(w_0, w_1, \ldots, w_n)^\top$, con $w_{(n+1) \times 1}$
    - $y_p=(y_1, y_2, \ldots, y_m)^\top$, con $(y_{p})_{ m \times 1}$

# Training linear regression: Normal equation

## Material

- [Video](https://www.youtube.com/watch?v=hx6nak-Y11g) (16:25)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-2-slides)
- [Cuaderno de Jupyter del video](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/chapter-02-car-price/02-carprice.ipynb)
- [Cuaderno de Jupyter adicional](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/02-regression/notebook.ipynb)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/02-regression/07-linear-regression-training.md)

## Notas

- Queremos resolver $g(X)=Xw=y$ para $w$
- Si $X$ fuera cuadrada, podría tener inversa y el resultado en ese caso sería $w=X^{-1}y$
- Pero es raro que $X$ sea cuadrada, de modo que no tiene inversa
- Podemos construir una matriz cuadrada por el lado izquierdo de $w$ usando $X^\top$:
  - $X^\top X w = X^\top y$, donde $X^\top X$ (llamada matriz de Gram) siempre será cuadrada
- Asumiendo que $X^\top X$ es invertible, tenemos entonces una fórmula para $w$:
  - $w = (X^\top X)^{-1} X^\top y$
- Debemos recordar agregar un vector columna con el valor 1 a la matriz $X$ para obtener también un $w_0$ al aplicar la fórmula
  ```python
  # forma explícita: `np.column_stack`
  Xa = np.column_stack([np.ones(X.shape[0]), X])  # "Xa: X augmented"
  # (bonus) utilizando un atajo: `np.c_`
  Xa = np.c_[np.ones(X.shape[0]), X]  # mnemotecnia: "c" es por "columna"
  ```
- (_Bonus_) Cuando utilicemos una función de una librería externa para calcular los pesos $w$ tendremos que verificar antes acaso la librería se encarga de aumentar la matriz de atributos o si tenemos que hacerlo de manera manual como vimos recién
- (_Bonus_) La matriz de Gram podría ser invertible, pero aún así inestable
  - Para combatir esta posibilidad, lo usual es normalizar cada atributo de la matriz $X$
    - Restamos la media de la columna a cada uno de sus elementos y los dividimos por la desviación estándar
    - Esto también será útil cuando la magnitud de los atributos sea muy distinta
    - En general, siempre será preferible normalizar la matriz de atributos
    - Podemos normalizar los valores utilizando un código como el siguiente:
      ```python
      def scale_X(X: np.ndarray) -> np.ndarray:
        """
        Standardize each column of X to mean 0 and std 1.
        Columns with zero variance become columns of zeros.
        """
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0, ddof=0)
        eps = 1e-12
        std[std < eps] = 1.0  # prevent division by zero
        X_scaled = (X - mean) / std
        return X_scaled
      ```
    - Notemos que las columnas de valores constantes no aportan en nada (independiente de que las normalicemos o no) y oscurecen la interpretación de los resultados
      - Podríamos descartarlas en lugar de hacerlas cero con la normalización
      - Y definitivamente tenemos que hacer algo si tenemos más de 1 columna constante ya que eso hace que la matriz de Gram no sea invertible
        - Si usamos regularización (lo veremos más adelante), sí podremos invertir la matriz, pero eso es un parche
        - No tiene sentido usar atributos colineales
    - Finalmente, tendremos que prestar atención a cómo estimamos $w_0$
      - Si la agregáramos un vector de 1 a la matriz de atributos, esa columna no debe ser normalizada y tampoco debe ser penalizada si usáramos regularización
      - En la práctica podemos entrenar el modelo sin estimar $w_0$ de manera explícita ya que luego lo podemos calcular a partir de las predicciones
        - La fórmula y el razonamiento probablemente esté presente en [The Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/) (2017, Hastie et al.)
  - Otra estrategia para hacer frente a posibles inestabilidades es calcular la SVD (Singular Value Decomposition) de la matriz de atributos
    - Se trata de una descomposición tal que $X=U \Sigma V^\top$
    - Dada esta descomposición, la fórmula para los pesos se convierte en $\hat{w}=V\Sigma^{-1}U^\top y$, variante que tiene mejores propiedades numéricas, entre otras ventajas matemáticas

# Baseline model for car price prediction project

## Material

- [Video](https://www.youtube.com/watch?v=SvPpMMYtYbU) (9:32)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-2-slides)
- [Cuaderno de Jupyter del video](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/chapter-02-car-price/02-carprice.ipynb)
- [Cuaderno de Jupyter adicional](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/02-regression/notebook.ipynb)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/02-regression/08-baseline-model.md)

## Notas

- Los datos de ventas de autos que tenemos tienen atributos numéricos y también de tipo _string_
- La regresión lineal que hemos estado discutiendo solo funciona con atributos numéricos
- Modelo base: sin complicarnos la vida, descartamos todos los atributos que no sean numéricos y entrenamos un modelo lineal con los que no descartamos
- Debemos decidir qué hacer con los datos faltantes (`null`, `NaN`, _infinity_)
  - Podemos imputarlos con el valor 0:
    - Con ello le decimos al modelo lineal que no debe tratar de ajustar un peso a los $x_i=0$, pero sí al resto de la misma observación
      - No se puede saber a priori cómo influirá esto en los valores de los pesos en general
      - Es algo que tendremos que medir
    - Podría no tener sentido alguno físico, pero eso no necesariamente hará que nuestro modelo sea peor (se supone que contamos con un gran número de ejemplos que sí están completos)
      - Pero esto tendrá que ser evaluado bajo algún criterio
  - En otros casos se podrían imputar con el promedio de dicho atributo en el conjunto de entrenamiento
  - Otras alternativas serán más especializadas (no necesariamente complejas, pero menos populares)
- Tal y como mencionamos en una de las secciones anteriores, también debemos recordar revertir cualquier transformación que hayamos hecho a los valores de $y$
- Una vez ajustado el modelo no bastará con evaluarlo con una inspección visual
  - Una alternativa común para evaluar la calidad del ajuste es calcular su RMSE (próxima sección)

# Root mean square error

## Material

- [Video](https://www.youtube.com/watch?v=0LWoFtbzNUM) (7:30)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-2-slides)
- [Cuaderno de Jupyter del video](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/chapter-02-car-price/02-carprice.ipynb)
- [Cuaderno de Jupyter adicional](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/02-regression/notebook.ipynb)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/02-regression/09-rmse.md)

## Notas

- La fórmula para RMSE (Root Mean Square Error) es literal:
  - E: error. La diferencia entre las predicciones y los valores de las muestras
  - S: cuadrado. Elevamos cada uno de los errores al cuadrado
  - M: promedio. El promedio de los cuadrados del paso anterior
  - Root: raíz. Y terminamos calculando la raíz cuadrada del número del paso anterior
- Fórmula: $\mathrm{RMSE} = \sqrt{ \frac{1}{m} \sum_{i=1}^m (g(x_i) - y_i)^2 }$
  - Los $y_i$ serán del conjunto de validación o test, nunca del de entrenamiento
  - $m$ representa la cantidad de observaciones

# Using RMSE on validation data

## Material

- [Video](https://www.youtube.com/watch?v=0LWoFtbzNUM) (4:16)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-2-slides)
- [Cuaderno de Jupyter del video](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/chapter-02-car-price/02-carprice.ipynb)
- [Cuaderno de Jupyter adicional](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/02-regression/notebook.ipynb)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/02-regression/10-car-price-validation.md)

## Notas

- Estimamos los parámetros $w$ usando el conjunto de entrenamiento
- Evaluamos el desempeño del modelo ajustado calculando su RMSE con los datos de validación

# Feature engineering

## Material

- [Video](https://www.youtube.com/watch?v=-aEShw4ftB0) (5:29)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-2-slides)
- [Cuaderno de Jupyter del video](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/chapter-02-car-price/02-carprice.ipynb)
- [Cuaderno de Jupyter adicional](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/02-regression/notebook.ipynb)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/02-regression/11-feature-engineering.md)

## Notas

- No es obligación utilizar los atributos con los datos en bruto
- (_Bonus_) Mencionamos antes que podemos normalizarlos, pero esa no es la única transformación común que les podemos aplicar
- Será normal que creemos nuevos atributos a partir de los disponibles cuando nos parezca que ellos serán más informativos para el modelo
  - En el caso de los datos de venta de vehículos que estamos viendo en clases, la antigüedad de un auto debiera ser un buen atributo para predecir su precio
  - Contamos con un vector que contiene los años en los que se fabricó cada auto
  - No tiene mucho sentido usar ese dato en crudo: será mejor calcular la antigüedad restando el año actual a los años de fabricación de cada auto
  - Utilizaremos este nuevo atributo ("edad") para entrenar nuestro modelo
  - En el ejemplo de clases vimos que incluir este atributo reduce el valor del RMSE en el conjunto de validación
    - Es decir, el modelo se comporta mejor cuando aumentamos $X$ con este nuevo atributo comparado con la matriz $X$ que no lo contiene

# Categorical variables

## Material

- [Video](https://www.youtube.com/watch?v=sGLAToAAMa4) (16:06)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-2-slides)
- [Cuaderno de Jupyter del video](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/chapter-02-car-price/02-carprice.ipynb)
- [Cuaderno de Jupyter adicional](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/02-regression/notebook.ipynb)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/02-regression/12-categorical-variables.md)

## Notas

- En la regresión lineal solo podemos usar variables numéricas
- Algunas variables numéricas podrían en realidad representar categorías y tendrán que ser tratadas como tales
- Si quisiéramos incorporar variables categóricas a un modelo lineal, tenemos que transformarlas en numéricas
- Existen distintas alternativas para codificar una variable categórica como una numérica
- En clases vimos una que codifica una variable categórica usando $N$ variables numéricas, con $N$ la cantidad de categorías
  - (_Bonus_) A esta codificación se le denomina "one-hot encoding"
  - (_Bonus_) En la práctica, cada vez que usemos "one-hot encoding", lo haremos codificando una de las categorías usando, implícitamente, un vector de ceros que no agregaremos a la matriz de atributos
    - La idea es codificar $k$ categorías usando $k-1$ variables auxiliares ("dummy variables") en lugar de las $k$ que usamos en clases
  - (_Bonus_) Debemos ser juiciosos a la hora de codificar variables categóricas
  - (_Bonus_) El desempeño de los algoritmos caerá si incrementamos mucho la dimensión del problema (que es la dimensión de la matriz de atributos)
    - A medida que aumentamos los atributos, el espacio que ocupa cada uno de ellos en el nuevo volumen se va reduciendo
      - Esto ya es posible de visualizar al pasar de 2 a 3 dimensiones
      - Al pasar de dos vectores (2D, un cuadrado) a tres (3D, un cubo) estamos "creando" gran cantidad de espacio vacío
      - Podemos pensarlo así: en términos relativos, los 3 vectores del cubo ocupan menos espacio del total disponible que los 2 vectores del rectángulo
    - De hecho, el espacio que cada vector ocupa en el nuevo volumen se reduce de manera exponencial con cada nuevo vector agregado
    - De esta manera, los datos se vuelven cada vez más dispersos
    - Pasado cierto umbral de dispersión, los puntos están tan alejados unos de otros que la noción de distancia deja de tener importancia tangible (todos están simplemente lejos unos de otros)
    - Muchos algoritmos y modelos de aprendizaje de máquinas están construidos en base a distancias entre puntos
    - La eficacia de estos modelos y algoritmos se reduce y pueden incluso dejar de funcionar
      - El espacio de soluciones se vuelve demasiado grande y con ello se torna difícil poder generalizar a partir de los datos
    - Como el espacio que ocupa cada vector se reduce de manera exponencial con cada nuevo vector agregado, tendríamos que recolectar una cantidad exponencial adicional de datos con cada nuevo atributo para contrarrestar este efecto
    - Si bien este problema es grave (se le denomina la "maldición de la dimensionalidad" o "curse of dimensionality" por buenas razones), a veces necesitamos trabajar en espacios de datos dispersos
      - En estos casos podremos usar algoritmos especialmente diseñados para estos escenarios
      - También podemos agregar parámetros de regularización (veremos una variante de esto en la próxima sección) a nuestro modelo
        - En particular, podemos forzar a que los atributos irrelevantes se hagan cero durante la resolución algorítmica del problema
        - Esto disminuirá la dimensión del espacio de soluciones y mejorará el desempeño de los algoritmos
      - Pero no debemos olvidar que los algoritmos no pueden hacer magia y que cada nueva dimensión agregada a un problema tiene el potencial de echar a perder nuestro modelo al implementarlo
      - Así que si estamos trabajando con un problema complejo donde podemos disminuir las dimensiones eliminando, por ejemplo, los atributos menos informativos, es recomendable hacerlo
  - (_Bonus_) En el caso en que realmente necesitemos codificar una gran cantidad de variables categóricas, tendremos que buscar alternativas a "one-hot encoding" o simplemente replantearnos el modelamiento del problema
  - (_Bonus_) Finalmente, se recomienda normalizar la matriz de atributos previo a incluir las variables auxiliares asociadas a las categorías que estamos codificando
    - Esto mejorará la estabilidad de los algoritmos numéricos
    - En términos de lo recién discutido, la normalización hará que el espacio de soluciones sea comparativamente más denso que el original

# Regularization

## Material

- [Video](https://www.youtube.com/watch?v=91ve3EJlHBc) (12:03)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-2-slides)
- [Cuaderno de Jupyter del video](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/chapter-02-car-price/02-carprice.ipynb)
- [Cuaderno de Jupyter adicional](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/02-regression/notebook.ipynb)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/02-regression/13-regularization.md)

## Notas

- (_Bonus_) Lo que hemos estado discutiendo se puede formalizar como un problema de optimización para encontrar los pesos óptimos del modelo lineal (los que reducen el error cuadrático)
  - La función que hemos estado optimizando implícitamente al usar la fórmula $w=(X^\top X)^{-1} X^\top y$ puede ser modificada agregando términos de regularización
  - El tipo de regularización que usemos tendrá distintos efectos sobre los parámetros $w$ estimados
  - En clases vimos la regularización L2 ("ridge regression")
  - Este tipo de regularización penaliza la magnitud de los coeficientes y fuerza a que ellos tengan magnitudes de tamaños similares, disminuyendo su varianza
  - Esto será deseable cuando contemos con atributos de valores similares o que sean aproximadamente múltiplos de unos con otros
  - Cuando eso ocurre, la matriz de Gram deja de ser invertible o, cuando lo es, producirá grandes variaciones en los coeficientes que estamos estimando
  - No desearemos que los coeficientes tengan variaciones extremas pues eso indica que podríamos estar ajustando ruido en lugar de señal
  - Con la regularización L2 le indicamos al algoritmo que debe encontrar soluciones donde se las tiene que arreglar para encontrar soluciones que distribuyan la influencia de los atributos correlacionados en los valores de sus pesos, haciendo que ellos varíen de forma menos dramática
  - El precio que pagamos en la predicción es que introducimos un poco de sesgo, pero disminuimos la varianza
- Vimos en clases que podemos abordar el problema de atributos con valores similares usando regularización
- Cuando los atributos tienen valores similares se pueden producir problemas numéricos o derechamente impedir que podamos invertir la matriz de Gram
- Estos problemas se manifiestan en grandes variaciones de los valores de los coeficientes estimados
- Para evitar este problema sumaremos valores pequeños en la diagonal de la matriz de Gram
  - $X^\top X \to X^\top X + \alpha I$, con $I$ la matriz identidad y $\alpha \in \mathbb{R}$ un número pequeño
  - $X^\top X + \alpha I$ será invertible
  - Este cambio hace que los atributos dejen de estar tan correlaciones y dará como resultado coeficientes $w$ sin variaciones extremas
- El $\alpha$ de la regularización se transforma en un hiperparámetro del modelo
  - Un hiperparámetro será un parámetro adicional del modelo que en realidad no nos interesa estimar, pero que debemos ajustar para obtener los valores de los parámetros $w$ que sí nos interesa estimar
  - Como $\alpha$ no es intrínsecamente interesante, no hace falta ajustarlo de forma rigurosa, sino que basta con encontrar un valor $\alpha=\alpha^*$ que sea "suficientemente bueno"
  - Usualmente definiremos un conjunto de valores posibles para $\alpha$ y los probaremos todos
  - Escogeremos aquel valor $\alpha^*$ que nos ayude a encontrar los parámetros $w$ que produzcan el modelo de mejor desempeño con el conjunto de validación
- (_Bonus_) Finalmente, la estrategia mencionada abarca problemas donde genuinamente tenemos una matriz delicada de invertir
  - En los casos en que la dificultad de invertir la matriz de Gram provenga de estar trabajando con atributos similares (colineales), lo que debemos hacer es eliminar uno de esos atributos
  - No tiene sentido incluir atributos que no tan solo no aportan nueva información, sino que también dificultan trabajar con la que poseemos

# Tuning the model

## Material

- [Video](https://www.youtube.com/watch?v=lW-YVxPgzQw) (3:16)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-2-slides)
- [Cuaderno de Jupyter del video](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/chapter-02-car-price/02-carprice.ipynb)
- [Cuaderno de Jupyter adicional](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/02-regression/notebook.ipynb)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/02-regression/14-tuning-model.md)

## Notas

- El ajuste al que se hace mención en el título de esta sección es con respecto al valor del parámetro de regularización, llamado $r$ en el video de la clase
- Entrenaremos nuestro modelo con distintos valores de $r$
  - En el video se propone $r \in [0, 10^{-5}, 10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}, 10^0, 10^1]$
  - Después de cada i-ésima corrida contaremos con $(r_i, w_i)$, con $r_i$ alguno de los valores del conjunto anterior y $w=(w_{i1}, \ldots, w_{im})^\top$ los pesos obtenidos al entrenar el modelo con $r_i$ fijo
- Escogeremos aquel modelo cuya combinación de valores de $r^*=r_i$ y parámetros $w$ reduzcan el RMSE en el conjunto de validación

# Using the model

## Material

- [Video](https://www.youtube.com/watch?v=KT--uIJozes) (10:04)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-2-slides)
- [Cuaderno de Jupyter del video](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/chapter-02-car-price/02-carprice.ipynb)
- [Cuaderno de Jupyter adicional](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/02-regression/notebook.ipynb)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/02-regression/15-using-model.md)

## Notas

- En la sección anterior dijimos cómo encontrar el valor del coeficiente de regularización, el que llamamos $r^*$
- Una vez encontrado dicho valor volveremos a entrenar el modelo dejando $r^*$ fijo
- La diferencia es que ahora usaremos los conjuntos de entrenamiento y validación a la vez
- Entrenamos con más datos esperando que los valores de $w$ ajustados puedan generalizar mejor que antes
  - El modelo tendrá más ejemplos de los que aprender
  - La intuición es que, al tener más ejemplos con los que entrenar, el modelo tendrá más posibilidades de encontrar la señal en medio del ruido
- Una vez encontrados los $w$ óptimos, calculamos ahora el RMSE del modelo usando los datos del conjunto de prueba
- Lo que esperamos es que nuevo valor RMSE sea al menos tan bueno como aquel obtenido cuando entrenamos usando $r^*$ y nada más que los datos del conjunto de entrenamiento original
  - "Tan bueno" dependerá de los datos con los que estemos trabajando y nuestra experiencia
  - Por ejemplo, el nuevo RMSE podría ser 5 o 10 % peor que el anterior y todavía podríamos darnos por satisfechos en la medida que eso no sea raro cuando hayamos trabajado con datos y cantidades de observaciones similares
  - Pero si el nuevo RMSE es claramente peor que el anterior (digamos al menos un 20 % peor), solo cabe volver atrás y analizar qué fue lo que salió mal
    - Entre las causas se cuenta un posible sobreajuste a los datos
    - Adicionalmente, podría ser que nuestras particiones de datos no son representativas del fenómeno que queremos modelar
    - Quizás cometimos también un error al usar los datos y estamos filtrando información entre las distintas particiones ("data leakage")
    - Entre otras alternativas
- Si el resultado de todo este proceso fue exitoso (el RMSE en el conjunto de prueba es "bueno"), nos quedamos con $r^*$ y los valores de $w$ ajustados usando los conjuntos de entrenamiento y validación a la vez
- Podemos usar ahora estos parámetros y el modelo lineal para predecir el valor de cualquier otro auto cuyos atributos conozcamos

# Car price prediction project summary

## Material

- [Video](https://www.youtube.com/watch?v=_qI01YXbyro) (7:40)
- [Cuaderno de Jupyter del video](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/chapter-02-car-price/02-carprice.ipynb)
- [Cuaderno de Jupyter adicional](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/02-regression/notebook.ipynb)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/02-regression/16-summary.md)

## Notas

- N/A

# Explore more

## Material

- [Cuaderno de Jupyter adicional](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/02-regression/notebook.ipynb)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/02-regression/17-explore-more.md)

## Notas

- Otros conjuntos de datos con los que podemos practicar lo aprendido:
  - [California housing dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html) para predecir el precio de una casa
  - [Student Performance Data Set](https://archive.ics.uci.edu/ml/datasets/Student+Performance) para predecir el desempeño de un estudiante
  - [UCI ML Repository: Regression](https://archive.ics.uci.edu/datasets?Task=Regression): repositorio con varios datos adicionales

# Homework

## Material

- [Cuaderno de Jupyter del módulo](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/chapter-02-car-price/02-carprice.ipynb)
- [Cuaderno de Jupyter adicional](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/02-regression/notebook.ipynb)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/02-regression/homework.md)

## Notas

- [Solución](https://github.com/neira-daniel/machine-learning-zoomcamp-hw02)
