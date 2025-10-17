---
language: es
title: "Module 3: Machine Learning for Classification"
author: Daniel Neira
---
> Create a customer churn prediction system using logistic regression and learn about feature selection.
>
> Topics:
>
> - Logistic regression
> - Feature importance and selection
> - Categorical variable encoding
> - Model interpretation

# Churn prediction project

## Material

- [Video](https://www.youtube.com/watch?v=0Zw04wdeTQo) (9:26)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-3-machine-learning-for-classification)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/03-classification/01-churn-project.md)

## Notas

- Problema que discutiremos en este módulo: identificar los clientes de una empresa de telecomunicaciones que están cerca de dejar de usar el servicio ("churn")
- La idea es que, una vez que identifiquemos dichos clientes, la empresa les hará ofertas u ofrecerá descuentos para que no dejen de usar el servicio
- De ser aceptadas, las ofertas disminuirán las ganancias a corto plazo de la empresa, de modo que es importante identificar correctamente a los clientes que realmente están por dejar de usar el servicio
  - Hacerles ofertas a los clientes que era improbable que dejaran de usar el servicio será un mal negocio a corto plazo
- Abordaremos este problema desde el punto de vista de la clasificación
- En particular, modelaremos el problema de renovaciones de contrato de clientes como uno de clasificación binaria
  - Asignaremos 1 a los ejemplos que responden nuestra pregunta afirmativamente y 0 a los que responden negativamente
  - Por ejemplo, si la pregunta es "¿Dejó el cliente de usar el servicio?", entonces:
    - 1: sí, el cliente abandonó el servicio (el cliente no renovó el contrato)
    - 0: no, el cliente no abandonó el servicio (el cliente sí renovó el contrato)
  - (_Bonus_) El modelo es simétrico, así que da lo mismo en qué sentido planteemos la pregunta
    - Podríamos habernos preguntado "¿Sigue el cliente usando el servicio?"
    - En ese caso, invertiremos las etiquetas 1 y 0 del ejemplo anterior
  - (_Bonus_) Lo importante será ser consistentes en cómo tratamos los datos y cómo interpretamos los resultados de acuerdo a la pregunta exacta que nos formulamos
- La salida de este modelo será un número real entre 0 y 1
  - Dada una observación, este número representará la probabilidad de que esa observación responda afirmativamente a nuestra pregunta
    - Es decir, la probabilidad de que la variable objetivo tenga valor 1
- En este módulo trabajaremos con el conjunto de datos "Telco Customer Churn" [alojado en Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (también disponible en [GitHub](https://github.com/alexeygrigorev/mlbookcamp-code/blob/db3a803512b2c2ee51e818f2a9a12223ff2bf356/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv))

# Data preparation

## Material

- [Video](https://www.youtube.com/watch?v=VSGGU9gYvdg) (8:57)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-3-machine-learning-for-classification)
- [Cuaderno de Jupyter del video](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/03-classification/notebook.ipynb)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/03-classification/02-data-preparation.md)

## Notas

- Código sugerido en clases para normalizar los _strings_ contenidos en un _dataframe_:
  ```python
  # normalizar el texto de las columnas
  df.columns = df.columns.str.lower().str.replace(' ', '_')

  # normalizar el texto de los datos que no son numéricos
  # ~nota: `list` podría no ser necesario (probar)
  categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)
  for c in categorical_columns:
      df[c] = df[c].str.lower().str.replace(' ', '_')
  ```
  - Notemos que, en general, no podemos asumir que una columna con `dtype` `object` es lo mismo que el tipo `string`
  - Pero aquí estamos trabajando con datos que son conocidos para el analista del video y él sabe que todas esas columnas contienen cadenas de texto
  - (_Bonus_) Podemos inspeccionar una `COLUMNA` con `dtype` `object` para ver qué tipos de datos contiene realmente ejecutando `df[COLUMNA].map(type).value_counts()`
- Código sugerido en clases para transformar una columna con `dtype` `object` en numérica:
  ```python
  # `errors='coerce'` fuerza a que  los valores que no son numéricos sean reemplazamos por NaN
  df.COLUMNA = pd.to_numeric(df.COLUMNA, errors='coerce')
  # luego podemos imputar los valores faltantes reemplazándolos, por ejemplo, con ceros
  df.COLUMNA = df.COLUMNA.fillna(0)
  ```
  - Cuando Pandas le asigne `dtype` `object` a una columna que solo debiera contener datos numéricos es porque esta contiene valores inconsistentes
  - En el caso de los datos que se están explorando en clases, estos valores son espacios (si ejecutamos el código de normalización de texto antes presentado, serán caracteres `_`)
  - Siempre debemos echar un vistazo a los tipos de datos inferidos por Pandas para detectar este tipo de problemas antes de comenzar a usarlos en nuestro modelo
- Código sugerido en clases para transformar una columna de valores "yes/no" en una de 1 y 0:
  ```python
  df.COLUMNA = (df.COLUMNA == 'yes').astype(int)
  ```
  - El condicional retorna `True` o `False`, según sea el caso
  - Luego le pedimos a Python que transforme esos valores _booleanos_ en unos (para `True`) y ceros (para `False`)

# Setting up the validation framework

## Material

- [Video](https://www.youtube.com/watch?v=_lwz34sOnSE) (6:40)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-3-machine-learning-for-classification)
- [Cuaderno de Jupyter del video](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/03-classification/notebook.ipynb)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/03-classification/03-validation.md)

## Notas

- Particionar los datos en conjuntos de entrenamiento, validación y prueba (e %, v % y p %) usando scikit-learn (sklearn):
  ```python
  from sklearn.model_selection import train_test_split
  df_full_train, df_test = train_test_split(df, test_size=p/100, random_state=1)
  df_train, df_val = train_test_split(df_full_train, test_size=v/(100-p), random_state=1)
  ```
  - sklearn solo nos da una función para particionar el conjunto de entrenamiento en 2
  - Debemos ejecutarla dos veces (ajustando las proporciones la segunda vez) para obtener los 3 conjuntos deseados
- Debemos recordar construir las matrices $X$ y los vectores $y$ a partir de los datos particionados
  - Existen distintas alternativas para esto, siendo la siguiente la que recomiendan en clases:
    ```python
    y_train = df_train.churn.values
    y_val = df_val.churn.values
    y_test = df_test.churn.values

    del df_train['churn']
    del df_val['churn']
    del df_test['churn']
    ```

# EDA

## Material

- [Video](https://www.youtube.com/watch?v=BNF1wjBwTQA) (7:21)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-3-machine-learning-for-classification)
- [Cuaderno de Jupyter del video](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/03-classification/notebook.ipynb)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/03-classification/04-eda.md)

## Notas

- Algunas acciones relacionadas con el análisis exploratorio de datos (EDA: "exploratory data analysis")
  - Chequear si existen valores faltantes: `df.isnull().sum()`
  - Calcular la frecuencia de los valores de una columna:
    - En bruto: `df.COLUMNA.value_counts()`
    - Normalizado por el total de muestras: `df.COLUMNA.value_counts(normalize=True)`
  - Calcular la frecuencia de valores únicos por cada atributo categórico de un _dataframe_: `df.nunique()`

# Feature importance: Churn rate and risk ratio

## Material

- [Video](https://www.youtube.com/watch?v=fzdzPLlvs40) (18:08)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-3-machine-learning-for-classification)
- [Cuaderno de Jupyter del video](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/03-classification/notebook.ipynb)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/03-classification/05-risk.md)

## Notas

- "Churn rate": la fracción del total de observaciones que están etiquetadas como "churn"   (cancelación del servicio)
- Nos interesará cómo varía el _churn rate_ al interior de distintos grupos de individuos con respecto al valor de dicha tasa cuando consideramos todos los grupos a la vez
  - Por ejemplo, acaso el _churn rate_ depende del género de una persona
  - O si las parejas tienden a cancelar más que los solteros
  - Entre muchas otras comparaciones que dependerán de los datos específicos con los que estemos trabajando
- Relevancia de un atributo:
  - Informalmente, podemos conjeturar que aquellos grupos cuyo _churn rate_ difiere de manera clara con respecto al _churn rate_ total son _más importantes_ a la hora de predecir si un cliente cancelará el servicio
    - Le entregan más información al modelo para que pueda hacer predicciones
    - El atributo correspondiente debería ser, entonces, parte de nuestra matriz $X$
- Maneras de cuantificar la relevancia de un atributo en la predicción del _churn rate_
  - Definiciones:
    - Sea $c$ el _churn rate_
    - Sea $c_t$ el _churn rate_ total (considerando todos los grupos)
    - Sea $c_i$ el _churn rate_ al interior del i-ésimo grupo
  - Indicadores:
    - Diferencia: $c_t-c_i$
      - $c_t-c_i>0$: menos probable que los miembros del i-ésimo grupo cancelen
      - $c_t-c_i<0$: más probable que cancelen
    - "Risk ratio": $c_i/c_t$
      - $c_i/c_t>1$: alto riesgo; más probable que cancelen
      - $c_i/c_t<1$: bajo riesgo; menos probable que cancelen
  - (_Bonus_) Por supuesto que estos indicadores deben ser leídos con una tolerancia en mente
    - No basta con el _churn rate_ de un grupo sea distinto con respecto al total
    - Variaciones pequeñas serán parte natural de trabajar con datos reales y ruidosos
- En general, cuando estamos explorando los datos, nos interesará calcular algo similar a lo siguiente:
  ```sql
    SELECT
      attribute,
      AVG(churn),
      AVG(churn) - global_churn AS diff,
      AVG(churn) / global_churn AS risk
    FROM
        data
    GROUP BY
        attribute;
  ```
  - Podemos implementarlo de la siguiente forma:
  ```python
  from IPython.display import display  # necesario para imprimir múltiples dataframes
  for c in categorical:  # `categorical` es una lista de los atributos categóricos
    # `df_full_train` contiene tanto `df_train` como `df_val`
    # en el video no mencionan por qué ocupar este conjunto de datos
    # yo me inclinaría por usar todos los datos para así contar con más información para entenderlos
    df_group = df_full_train.groupby(c).churn.agg(['mean', 'count'])
    df_group['diff'] = df_group['mean'] - global_churn
    df_group['risk'] = df_group['mean'] / global_churn
    display(df_group)  # para imprimir `df_group` sin que haya interferencia entre iteraciones
    print()  # para dejar una línea en blanco entre dataframes
  ```

# Feature importance: Mutual information

## Material

- [Video](https://www.youtube.com/watch?v=_u2YaGT6RN0) (8:57)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-3-machine-learning-for-classification)
- [Cuaderno de Jupyter del video](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/03-classification/notebook.ipynb)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/03-classification/06-mutual-info.md)

## Notas

- Calcularemos la información mutua entre dos variables aleatorias para entender cuánta información una de ellas nos entrega sobre la otra
- En el caso de selección de atributos, nos interesará modelar el problema de clasificación usando los atributos que nos entregan más información sobre la variable objetivo
  - Calcularemos la información mutua entre los atributos disponibles en los datos y la variable objetivo
  - Si bien es difícil interpretar el valor de la información mutua por sí solo, dicha cifra nos permitirá rankear los distintos atributos que hayamos usado en los cálculos
  - Agregaremos al modelo aquellos atributos que nos entregan más información sobre el comportamiento de la variable objetivo
- En el caso del ejemplo de clases, calculamos la información mutua usando solo las variables categóricas
  - Código sugerido:
    ```python
    # función para calcular la información mutua entre dos variables discretas (etiquetas)
    from sklearn.metrics import mutual_info_score
    def mutual_info_churn_score(series):
      # lo ideal sería pasar el dataframe a la función, pero eso complica usar `apply` después
      return mutual_info_score(series, df_full_train.churn)
    # `apply` usa la función que le pasemos con cada una de las columnas del dataframe asociado
    df_full_train[categorical].apply(mutual_info_churn_score).sort_values(ascending=False)
    # ~nota: es probable que podamos usar funciones lambda, parciales o closures para poder pasarle
    # el dataframe a la función procesada por `apply` y así no depender de variables globales
    ```
- (_Bonus_) La "cantidad" de información mutua entre dos variables es difícil de interpretar
  - Son pocas las certezas que un valor de información mutua nos puede entregar por sí solo
    - En particular, el único valor que tiene significado intrínseco es 0: cuando la información mutua es cero estamos en presencia de variables independientes
    - Pero, por ejemplo, no hay ninguna cifra que, por sí misma, nos diga cuando dos variables están perfectamente correlacionadas
  - Las unidades en las que se expresa la información mutua no tienen nada que ver con las unidades de las variables de nuestro problema
  - Los valores tampoco están normalizados, de modo que solo sabemos que pueden ser mayores o iguales a 0
  - Cuando calculamos la información mutua entre $Y$ y $X$ estamos midiendo cuánto se reduce la incertidumbre en el valor de $Y$ al conocer el valor de $X$ (este valor no es simétrico)
    - Pero la implementación de `mutual_info_score` en Sklearn sí lo es
    - Nota: explorar [`mutual_info_regression`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_regression.html) y [`mutual_info_classif`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html), las que calculan la información mutua entre una matriz de atributos y una variable objetivo continua o discreta, respectivamente
  - Esta incertidumbre se mide en términos de entropía
    - Cuando $Y$ tiene una entropía alta, incluso pequeñas disminuciones de este valor gracias a $X$ pueden producir un valor de información mutua de gran magnitud
    - Es por eso que en estas aplicaciones no nos fijamos en los valores de información mutua en sí pues su interpretación es compleja y no nos entregan información práctica relevante
    - Pero estos valores sí son consistentes, de modo que funcionan perfecto para identificar cuáles de los atributos probados son más valiosos para explicar el comportamiento de la variable objetivo
    - Es por esto último que, a pesar de ser difícil de tratar, la información mutua termina teniendo una aplicación directa a la hora de rankear la importancia de cada uno de los atributos que pretendemos agregar al modelo

# Feature importance: Correlation

## Material

- [Video](https://www.youtube.com/watch?v=mz1707QVxiY) (12:58)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-3-machine-learning-for-classification)
- [Cuaderno de Jupyter del video](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/03-classification/notebook.ipynb)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/03-classification/07-correlation.md)

## Notas

- En el video se abordó el uso del coeficiente de correlación de Pearson para determinar cómo cambia el valor de la variable objetivo a medida que variamos el valor de un atributo
  - En particular, dado que estamos trabajando con un problema de clasificación binaria tipo "churn", usamos la correlación para ver cómo la variación del valor de cada atributo influye en la posibilidad de que un cliente no renueve su contrato
- El coeficiente de correlación de Pearson $r$ entre dos variables aleatorias:
  - Puede tomar valores entre -1 y 1, i.e., $|r| \leq 1$
  - $r=0$ significa que las variables son independientes
  - $r>0$ significa que cuando se incrementa el valor de una de las variables, el valor de la otra también aumenta
  - $r<0$ significa que las variables se mueven en sentidos opuestos
  - (_Bonus_) $r=1$ da cuenta de las variables son combinaciones lineales una de la otra
- Algunas reglas informales con respecto al valor de $r$:
  - $|r|<0.2$: la correlación es baja
  - $0.2 \leq |r| < 0.5$: correlación media
  - $0.5 \leq |r| \leq 1.0$: alta correlación
- Correlación en Pandas:
  - `df.corrwith(df2)`: correlación de las columnas de `df` con las columnas de `df2`
    - En el video vimos cuando `df2` es una serie (la variable objetivo)
  - (_Bonus_) `df.corr()`: correlación cruzada entre todas las columnas de `df`

# One-hot encoding

## Material

- [Video](https://www.youtube.com/watch?v=L-mjQFN5aR0) (15:34)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-3-machine-learning-for-classification)
- [Cuaderno de Jupyter del video](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/03-classification/notebook.ipynb)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/03-classification/08-ohe.md)

## Notas

- Podemos codificar una variable categórica en numérica usando "one-hot encoding"
- La idea es:
  - Identificar las $n_i$ categorías en cada atributo
  - Crear $n_i \cdot M$ vectores (variables) auxiliares, con $M$ la cantidad de atributos que estamos codificando
    - Por ejemplo, si queremos codificar un atributo que solo contiene los valores "masculino" y "femenino", necesitaremos crear 2 vectores auxiliares, uno para el valor "masculino" y otro para "femenino" ($M=1$ porque estamos usando un solo atributo)
  - Luego iteramos sobre las observaciones y llenamos los vectores auxiliares con valores 1 o 0
    - 1 cuando el vector auxiliar se "activa"
    - 0 cuando no lo hace
    - Por ejemplo, si la observación contiene el valor "femenino", pondremos un 1 en el vector auxiliar que representa "femenino" y un 0 en aquel que representa el valor "masculino"
  - (_Bonus_) Dado que este proceso puede crear gran cantidad de variables auxiliares y no es recomendable trabajar en espacios de muy alta dimensión, lo usual es descartar una de las variables auxiliares por cada atributo categórico
    - Por ejemplo, en el caso en que estamos codificando "masculino" y "femenino" nos bastará con una sola variable auxiliar para representar la información de manera unívoca
    - Digamos que conservamos el vector auxiliar asociado al valor "femenino" y descartamos el "masculino"
    - Entonces, cuando nos encontremos con el valor 1 sabremos que la observación contiene el valor "femenino" y si tuviera el valor 0 sabremos que se trata del valor "masculino"
    - Esto no solo se limita a atributos categóricos con 2 valores, sino que escala a cualquier cantidad de valores $n_i$
      - Siempre podemos codificar esa información usando $n_i-1$ vectores auxiliares
- Cómo usar "one-hot encoding" en Python
  - Opción del video: sklearn y `DictVectorizer`
    - Es una alternativa manual donde tenemos que realizar pasos intermedios para conseguir nuestro objetivo
    - En este caso terminamos codificando cada i-ésimo atributo usando $n_i$ vectores auxiliares y no $n_i -1$
      ```python
      from sklearn.feature_extraction import DictVectorizer

      # es probable que también sirva usando matrices _sparse_
      dv = DictVectorizer(sparse=False)

      # exportamos la información del df de entrenamiento en formato de diccionario de Python
      train_dict = df_train.to_dict(orient='records')
      # "ajustamos" los datos del diccionario: le pedimos a sklearn que extraiga las categorías disponibles en cada atributo
      # "transformamos" los datos usando one-hot encoding y las categorías antes identificadas
      # esta transformación conservará los atributos numéricos intactos
      X_train = dv.fit_transform(train_dict)

      # extraemos la información del df con los datos de validación
      val_dict = df_val.to_dict(orient='records')
      # y aquí solo transformamos los datos del diccionario recién exportado (no hace falta ajustar nada porque eso ya lo hicimos con los datos de entrenamiento)
      X_val = dv.transform(val_dict)

      # para ver cuáles fueron los atributos auxiliares creados con `dv`
      # ~nota: en el video usan `get_feature_names`, pero este método dejó de ser válido en sklearn 1.2
      dv.get_feature_names_out()
      ```
    - Algunas notas sobre el uso de `DictVectorizer`: "[Loading features from dicts](https://scikit-learn.org/stable/modules/feature_extraction.html#loading-features-from-dicts)" (documentación de scikit-learn)
  - (_Bonus_) Otras alternativas:
    - [`OneHotEncoder`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) de sklearn
      - `from sklearn.preprocessing import OneHotEncoder`
      - Nos permite usar $n_i-1$ vectores auxiliares por atributo
      - Su API es muy similar a la de `DictVectorizer`, así que no es difícil de usar
    - [`get_dummies`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.get_dummies.html) de Pandas
      - También nos permite usar $n_i-1$ vectores auxiliares por atributo
      - Su API es distinta de la de `DictVectorizer`, de modo que se tiene que estudiar en particular
      - Algunas notas sobre el uso de `get_dummies`: "[Reshaping and pivot tables: `get_dummies()` and `from_dummies()`](https://pandas.pydata.org/pandas-docs/stable/user_guide/reshaping.html#reshaping-dummies)" (documentación de Pandas)

# Logistic regression

## Material

- [Video](https://www.youtube.com/watch?v=7KFE2ltnBAg) (9:32)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-3-machine-learning-for-classification)
- [Cuaderno de Jupyter del video](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/03-classification/notebook.ipynb)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/03-classification/09-logistic-regression.md)

## Notas

- x

# Training logistic regression with scikit-Learn

## Material

- [Video]() (12:12)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-3-machine-learning-for-classification)
- [Cuaderno de Jupyter del video](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/03-classification/notebook.ipynb)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/03-classification/10-training-log-reg.md)

## Notas

- x

# Model interpretation

## Material

- [Video]() (16:17)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-3-machine-learning-for-classification)
- [Cuaderno de Jupyter del video](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/03-classification/notebook.ipynb)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/03-classification/11-log-reg-interpretation.md)

## Notas

- x

# Using the model

## Material

- [Video]() (10:15)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-3-machine-learning-for-classification)
- [Cuaderno de Jupyter del video](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/03-classification/notebook.ipynb)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/03-classification/12-using-log-reg.md)

## Notas

- x

# Summary

## Material

- [Video]() (6:06)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-3-machine-learning-for-classification)
- [Cuaderno de Jupyter del video](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/03-classification/notebook.ipynb)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/03-classification/13-summary.md)

## Notas

- x

# Explore more

## Material

- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/03-classification/14-explore-more.md)

## Notas

- x

# Homework

## Material

- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/03-classification/homework.md)

## Notas

- x
