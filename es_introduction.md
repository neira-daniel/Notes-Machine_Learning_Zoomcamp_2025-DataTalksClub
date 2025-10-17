---
language: es
title: "Module 1: Introduction to Machine Learning"
author: Daniel Neira
---
> Learn the fundamentals: what ML is, when to use it, and how to approach ML problems using the CRISP-DM framework.
>
> Topics:
>
> - ML vs rule-based systems
> - Supervised learning basics
> - CRISP-DM methodology
> - Model selection concepts
> - Environment setup

# Introduction to Machine Learning

## Material

- [Video](https://www.youtube.com/watch?v=Crm_5n4mvmg) (9:37)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-11-introduction-to-machine-learning)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/01-intro/01-what-is-ml.md)

## Notas

- Introducción al material usando el ejemplo de la venta de un auto usado
- El vendedor no quiere asignarle un precio al auto que sea:
  - Muy alto porque existen menos posibilidades de que alguien compre el auto
  - Muy bajo porque estaría ganando menos dinero del que podría
- Qué es un precio muy alto o muy bajo es algo que tenemos que determinar
- Personas con experiencia vendiendo autos usados lo pueden determinar usando unas cuantas variables
- Nosotros podemos automatizarlo usando ejemplos de autos vendidos y sus precios de venta
- El valor que queremos estimar (predecir): objetivo (_target_)
  - Ejemplo: el precio adecuado para el auto
- Las variables que determinarán el valor que queremos predecir: atributos (_features_)
  - Ejemplos: año de producción del auto, marca, modelo, kilometraje, etc.
- Nuestra tarea será extraer patrones de los datos que luego usaremos para predecir nuevos valores objetivo
  - Ejemplo: usar datos de ventas de autos usados para predecir los valores de venta de otros autos usados de acuerdo a sus características

# ML vs Rule-Based Systems

## Material

- [Video](https://www.youtube.com/watch?v=CeukwyUdaz) (17:27)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-12-ml-vs-rulebased-systems)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/01-intro/02-ml-vs-rules.md)
## Notas

- Sistemas basados en reglas:
  - Identificamos un problema y el objetivo que nos gustaría predecir
  - Usamos una serie de reglas para predecir el objetivo
  - Las reglas podrían tener que actualizarse con el tiempo
  - Cómo obtener las reglas variará dependiendo de la aplicación
  - La variable objetivo puede ser continua o discreta
- Ejemplo: clasificación de correo electrónico
  - Nos interesa identificar el correo _spam_
  - Podemos extraer patrones que caractericen los correos que son _spam_
  - Y podemos crear reglas en base a esos patrones
    - "Si el correo contiene la frase 'promoción', probablemente sea _spam_"
    - "Si el correo proviene de una dirección de correo electrónico desconocida, podría ser _spam_"
    - "Si el correo proviene de una dirección con la que hemos sostenido conversaciones en el pasado, es poco probable que se trate de _spam_, incluso cuando contenga la palabra 'promoción'"
    - Etc.
  - En este caso será necesario actualizar las reglas constantemente debido a que lo que caracteriza a un correo _spam_ va cambiando con el tiempo
  - Además lo que para un usuario es _spam_ podría no serlo para otro
- En algunos casos también podríamos entrenar un modelo de aprendizaje de máquinas para predecir la variable objetivo
  - En este caso haremos lo mismo que vimos antes:
    - Obtener datos de entrenamiento
    - Extraer patrones usando los datos
    - Entrenar el modelo con estos atributos
    - Usar el modelo entrenado para predecir la variable objetivo
  - En particular:
    - Podemos comenzar con las reglas definidas anteriormente
    - Dejar que el usuario etiquete los mensajes como _spam_ o no
    - Usamos esa etiqueta, los correos asociados y las reglas, que aquí usamos como atributos, para ajustar (entrenar) un modelo
    - Usaremos luego este modelo para predecir acaso un mensaje es _spam_ o no con cierta probabilidad
    - Podemos luego etiquetar como _spam_ todos aquellos correos cuya probabilidad de ser _spam_ es superior a un valor $p$ predeterminado
- En este caso, la diferencia entre un sistema basado en reglas y otro que utiliza aprendizaje de máquinas
  - Sistema basado en reglas:
    - Se trata de un programa con múltiples condiciones de tipo `if`-`then`-`else` con las que codificamos las reglas
    - La salida es determinista
      - En el ejemplo visto, un correo es o no es _spam_ sin mediar probabilidades
  - Aprendizaje de máquinas
    - Se trata de un modelo matemático
    - Ajustamos los parámetros del modelo usando atributos y ejemplos de entrenamiento
    - La salida del modelo es probabilística
      - En el ejemplo visto, calculamos la probabilidad de que un correo sea _spam_
      - Utilizamos esa probabilidad para etiquetar los correos

# Supervised Machine Learning

## Material

- [Video](https://www.youtube.com/watch?v=j9kcEuGcC2Y) (19:32)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-13-supervised-machine-learning)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/01-intro/03-supervised-ml.md)

## Notas

- Aprendizaje de máquinas supervisado:
  - Supervisado: reunimos un conjunto de ejemplos (datos) que contienen tanto las entradas como las salidas de un fenómeno a partir de las cuales la máquina debe _aprender_
  - Aprendizaje: entrenamos un modelo dejando que la "máquina" procese estos ejemplos y extraiga los patrones que están allí presentes
  - Utilizamos estos patrones para generalizar y poder así predecir a partir de entradas que no están presentes en los ejemplos
- Nomenclatura:
  - Atributos (_features_): variables que utilizamos para modelar la dinámica que está detrás del conjunto de datos
  - Observaciones (_observations_): muestras de los atributos
  - Matriz de atributos $X$: matriz que contiene múltiples observaciones de los atributos
    - Filas: observaciones
    - Columnas: atributos
  - Objetivo $y$ (_target_): vector que contiene las salidas
  - Modelo $g$: función cuyos parámetros deseamos ajustar a los datos
    - $g(X) \approx y$
  - Tipos de problemas:
    - Cuando $y \in \mathbb{R}$, problema de regresión
    - Cuando $y \in \mathbb{Z}$, problema de clasificación (entero será asociado a una etiqueta)
      - Clasificación binaria: un ítem pertenece o no a una categoría
      - Muticlase: un ítem pertenece a una categoría de las $N>2$ clases
    - Hablaremos de otros tipos de problemas cuando $y$ no se pueda describir en los términos anteriores
      - Ordenamiento (_ranking_): cuando el objetivo es ordenar un conjunto de ítems
        - Por ejemplo, en los sistemas de recomendación de las tiendas en línea
        - También es el problema que resuelven los motores de búsqueda web para ofrecernos el mejor resultado a una búsqueda _bajo cierto criterio definido por ellos_
      - (_Bonus_) Predicciones estructuradas (_structured predictions_): cuando la salida es un objeto estructurado con posibles dependencias internas
        - Por ejemplo, cuando deseamos predecir una estructura de árbol
        - Problemas de ese estilo se dan, por ejemplo, cuando queremos etiquetar las palabras de una oración de acuerdo a su función gramatical (sustantivos, verbos, adverbios, etc.)
        - El orden de las palabras crea una dependencia que nos da información sobre qué función cumple cada palabra dentro de la oración
        - Podemos imaginar este problema como uno en el que el codominio son todos los árboles de etiquetas posibles y que deseamos predecir aquel árbol que mejor describe la función gramatical de cada una de las palabras de una oración

# CRISP-DM

## Material

- [Video](https://www.youtube.com/watch?v=dCa3JvmJbr0) (20:57)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-14-crispdm)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/01-intro/04-crisp-dm.md)

## Notas

- CRISP-DM: Cross-industry standard process for data mining
- Se trata de un marco de trabajo estándar para la minería de datos
- Etapas:
  1. Comprensión del negocio (_business understanding_)
      - Identificar el problema
      - Entender cómo lo podemos resolver
        - ¿Es el aprendizaje de máquinas una solución viable?
        - ¿Es razonable usar aprendizaje de máquinas?
        - ¿Existen mejores alternativas?
      - Definir cómo evaluar las soluciones de una forma medible (qué significa tener éxito en este contexto)
        - Por ejemplo, una solución a un problema económico imaginario será exitosa si reduce los costos en al menos 10 %
  2. Comprensión de los datos (_data understanding_)
      - Identificar los datos disponibles
      - Evaluar acaso los datos disponibles serán suficientes para resolver el problema
        - ¿Son los datos confiables?
        - ¿Contamos con suficientes datos para entrenar un modelo de manera confiable?
      - Definir cómo recabar más datos en caso de que los disponibles no sean suficientes
  3. Preparación de los datos (_data preparation_)
      - Transformar los datos a un formato que pueda ser consumido por el algoritmo de entrenamiento
        - Implementar las etapas de transformación
        - Limpiar los datos
        - Tabularlos (armar la matriz de atributos)
  4. Modelamiento (_modeling_)
      - Usualmente, escoger un modelo dado y entrenarlo
      - Si las condiciones lo permiten, podemos escoger múltiples modelos y quedarnos con aquel que tenga mejor desempeño
      - Ejemplos de modelos:
        - Regresión lineal
        - Regresión logística
        - Árboles de decisión
        - Redes neuronales
        - Etc.
      - Podríamos tener que volver al paso anterior si descubrimos que debemos volver a procesar los datos y extraer otros patrones más explicativos de los datos
  5. Evaluación del modelo (_evaluation_)
      - Evaluamos el desempeño de un modelo usando la medida definida en el primer paso
        - ¿Alcanzamos la meta?
        - ¿Cuan por encima o debajo de ella estamos?
        - ¿Tienen sentido los resultados obtenidos?
        - ¿Era la meta razonable?
        - ¿Resolvimos el problema correcto o nuestra comprensión del negocio era errada?
      - Como resultado de la evaluación podríamos:
        - Tener que volver al primer paso y ajustar la meta o incluso repensar el problema usando todo lo que aprendimos en esta iteración
        - Concluir que no existe solución satisfactoria en caso de que no podamos alcanzar la meta, dejando el proyecto de lado
        - Continuar y desplegar la solución hallada
  6. Despliegue de la solución (_deployment_)
      - En modo de:
        - Prueba: se le hace llegar a una porción de los usuarios y se evalúa acaso funciona como esperábamos
        - Producción: hacemos llegar el modelo a todos los usuarios, usualmente luego de que la prueba haya sido exitosa
      - En ambos casos, el despliegue de la solución va de la mano con una evaluación constante del desempeño del modelo desplegado
        - El modelo podría comportarse peor de lo esperado en condiciones de uso reales
        - En este caso podríamos tener que abortar el despliegue a todos los usuarios
      - Independiente de que el modelo en producción sea exitoso, usarlo nos permite seguir reuniendo datos que podremos usar en entrenamientos futuros que lo mejoren
- Las etapas mencionadas son parte de un ciclo
  - Usualmente, iteraremos muchas veces sobre este ciclo
  - Los resultados obtenidos al final de cada iteración nos servirán para mejorar el modelo de la siguiente iteración
  - También nos permitirán refinar nuestra comprensión del negocio y del problema
  - Trabajar entendiendo que este es un proceso iterativo nos permite
    - Comenzar con modelos simples
    - Obtener retroalimentación práctica
    - Mejorar el modelo en cada iteración
      - Eventualmente, mejorar el modelo podría significar usar uno más complejo
      - Pero es mejor trabajar con el modelo más simple que nos permita cumplir nuestros objetivos
      - Solo complejizaremos un modelo cuando las condiciones así lo exijan

# Model Selection Process

## Material

- [Video](https://www.youtube.com/watch?v=OH_R0Sl9neM) (21:33)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-15-model-selection-process)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/01-intro/05-model-selection.md)

## Notas

- Podemos usar distintos modelos para intentar solucionar un problema
- Cuando tenemos varios modelos candidatos, tenemos que decidir con cuál quedarnos
- Alternativas de decisión:
  - Podríamos pensar en quedarnos con el modelo que mejor se ajusta a los datos de entrenamiento, pero eso es un error
    - Lo que nos interesa realmente es el poder predictivo del modelo
    - Ajustar muy bien a los datos de entrenamiento no nos asegura que el modelo será bueno prediciendo con datos con los que no entrenó
  - Escogeremos el modelo que tenga mejor poder de predicción
    - Es decir, el que tenga el menor error de predicción
    - Evaluaremos la calidad de la predicción con datos que no hayamos usado durante el entrenamiento
- Medimos el poder de predicción de un modelo usando datos distintos a los de entrenamiento
  - En la práctica, entonces, particionaremos los datos con los que contamos y usaremos una parte para entrenar y el resto para probar el modelo
  - Caso ideal: 3 partes
    - Conjunto de entrenamiento:
      - 70-90 % del total, dependiendo de si contamos con pocos o hartos datos
      - Será el conjunto que el modelo usará para extraer los patrones
    - Conjunto de validación:
      - 5-15 % del total
      - Estos datos nunca son utilizados en el entrenamiento
      - Será el conjunto de datos que usaremos para evaluar el poder de predicción del modelo
      - El modelo que dé mejores resultados con este conjunto será el modelo que seleccionaremos entre todos los candidatos
      - Sin embargo, debemos tener presente que, al estar usando múltiples modelos probabilísticos, existe la posibilidad de que uno de ellos tenga buen desempeño con el conjunto de validación solo de suerte (["multiple comparisons problem"](https://en.wikipedia.org/wiki/Multiple_comparisons_problem))
      - (_Bonus_) En el caso de que nuestro modelo tenga hiperparámetros (como los de regularización o tasa de aprendizaje), podemos ocupar este conjunto para afinar sus valores
    - Conjunto de prueba:
      - 5-15 % del total
      - Utilizaremos estos datos para evaluar cómo se comportará nuestro modelo frente a datos nunca vistos
      - La probabilidad de que un modelo tenga un buen desempeño con los datos de validación y también con los datos de entrenamiento solo de suerte es despreciable
      - Solo servirán para evaluar el modelo, no para mejorarlo
        - Si los usarámos para afinar el modelo, este conjunto de datos se tranformaría en uno de validación
      - (_Bonus_) Si el error de predicción en este conjunto fuese mayor al esperado, tendremos que volver a revisar las etapas anteriores
        - Podríamos haber sobreajustado los hiperparámetros durante la validación
        - Los datos con los que estamos trabajando podrían ser muy pocos o no ser representativos
        - Quizás nos equivocamos particionando los datos originales
  - (_Bonus_) En el caso no ideal podríamos tener muy pocos datos
    - Tan pocos datos que no podremos particionarlos en 3 sin correr el riesgo de que cada uno de los nuevos conjuntos no sea representativo del fenómeno que los generó
    - En este caso podemos particionar el conjunto en 2: conjuntos de entrenamiento y de validación
    - Y podemos generar múltiples particiones y trabajar con lo que se conoce como validación cruzada, que es una manera sintética de emular un conjunto de datos de gran tamaño usando pocos datos
- Flujo de trabajo:
  1. Particionar los datos en conjuntos de entrenamiento, validación y prueba
  2. Entrenar los modelos usando el conjunto de entrenamiento
  3. Validar los modelos (evaluar sus poderes de predicción) usando el conjunto de validación
  4. Seleccionar el modelo de mejor desempeño con los datos del conjunto de validación
  5. Corroborar el poder de predicción del modelo seleccionado usando los datos del conjunto de prueba
  6. En caso de que el desempeño del modelo escogido con los datos de prueba sea menor al esperado, se debe volver atrás y revisar qué pudo haber salido mal

# Setting up the Environment

## Material

- [Video](https://www.youtube.com/watch?v=pqQFlV3f9Bo) (8:53)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/01-intro/06-environment.md)

## Notas

- Alternativas en la nube
  - Máquinas de desarrollo remotas
    - GitHub Codespaces (gratuito hasta cierto punto)
    - GCP (contiene créditos de gracia)
    - AWS (pagado)
  - Servicios de Jupyter Notebooks
    - Google Colab (gratuito hasta cierto punto)
    - Kaggle (modelo de monetización no consultado)
- Alternativas locales
  - Conda y Miniconda (gratuito para uso personal)
  - (_Bonus_) Bastará con `pip` (o sus competidores, como `uv`) por lo menos durante los primeros 4 módulos del curso (gratuito para cualquier tipo de uso)
- Librerías que utilizaremos durante la primera parte del curso: `numpy pandas scikit-learn seaborn jupyter`

# Introduction to NumPy

## Material

- [Video](https://www.youtube.com/watch?v=Qa0-jYtRdbY) (21:48)
- [Cuaderno de Jupyter del video](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/01-intro/notebooks/07-numpy.ipynb)
- [Cuaderno de Jupyter adicional sobre NumPy](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/appendix-c-numpy.ipynb)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/01-intro/07-numpy.md)

## Notas

- Cosas que no tenía frescas en la memoria:
  - `np.full(DIMENSIONES, VALOR)`
    - Crea un arreglo de dimensión `DIMENSIONES` (escalar o tupla) y lo llena con el valor `VALOR` (escalar)
    - El tipo del arreglo es el tipo de `VALOR`
    - La alternativa que yo ocupaba: `VALOR * np.ones(DIMENSIONES)`
      - Cuyo tipo siempre es `float64` debido a que ese es el tipo que retorna `np.ones`
  - La forma moderna de trabajar con muestreo aleatorio en Numpy es usando `Generator` en lugar de `RandomState`
    - El flujo de trabajo mostrado en el cuaderno de Jupyter del video está [desactualizado](https://numpy.org/doc/stable/reference/random/legacy.html)
      - Instrucciones como `np.random.randint(2, high=33, size=10)`
      - Se deben reemplazar por instrucciones en la línea de `rng = np.random.default_rng(); rng.integers(2, high=33, size=10)`
      - Nota sobre los cambios: [link](https://numpy.org/doc/stable/reference/random/new-or-different.html)
    - El antiguo flujo de trabajo no será descontinuado en el corto plazo, pero los mantenedores de Numpy [recomiendan hacer la transición a `Generator` apenas sea posible](https://numpy.org/doc/stable/reference/random/index.html#:~:text=While%20there%20are%20no%20plans%20to%20remove%20them%20at%20this%20time,%20we%20do%20recommend%20transitioning%20to%20Generator%20as%20you%20can)
- Ver los cuadernos de Jupyter enlazados más arriba, así como el documento de Markdown, para ver los detalles de uso de Numpy

# Linear Algebra Refresher

## Material

- [Video](https://www.youtube.com/watch?v=zZyKUeOR4Gg) (27:36)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-18-linear-algebra-refresher)
- [Cuaderno de Jupyter del video](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/01-intro/notebooks/08-linear-algebra.ipynb)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/01-intro/08-linear-algebra.md)

## Notas

- Cosas que no tenía frescas en la memoria:
  - Producto punto entre vectores (arreglos de Numpy) $u$ y $v$: `u.dot(v)`
    - La alternativa que tenía presente era `np.dot(u, v)`
    - (_Bonus_) Debemos tener presente que el método `dot` de Numpy puede calcular más que productos puntos, cosa que dependerá de las dimensiones de $u$ y $v$ (ver [documentación](https://numpy.org/doc/stable/reference/generated/numpy.dot.html))
      - Dependiendo de las dimensiones de $u$ y $v$, el método `dot` puede incluso calcular el producto matricial
      - En el caso del producto matricial, si bien el resultado será el mismo que si utilizáramos `matmul`, es conceptualmente más claro usar `matmul` en lugar de apoyarnos en las convenciones de la implementación de `dot`
      - En la medida que el vector sea un arreglo sea 1D ($(m,)$ en lugar de $(1, m)$), `matmul` producirá exactamente el mismo resultado (tendrá las mismas dimensiones) que `dot` cuando `dot` esté calculando productos matriciales
  - Producto matricial entre una matriz $U_{m \times n}$ y un vector $v_{n \times 1}$: es el producto punto entre las filas de la matriz $U$ y el vector $v$
    - $U \cdot v = [u_0 v, u_1 v, \ldots, u_n v]^T_{1 \times m}$
  - Producto matricial entre una matriz $U_{m \times n}$ y una matriz $V_{n \times p}$: el el producto entre la matriz $U$ y las columnas de la matriz $V$ (que son vectores)
    - $U \cdot V = [U v_0, U v_1, \ldots, U v_p]_{m \times p}$
  - Inversa de una matriz: utilizaremos la función `np.linalg.inv`

# Introduction to Pandas

## Material

- [Video](https://www.youtube.com/watch?v=0j3XK5PsnxA) (28:34)
- [Cuaderno de Jupyter del video](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/01-intro/notebooks/09-pandas.ipynb)
- [Cuaderno de Jupyter adicional sobre Pandas](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/appendix-d-pandas.ipynb)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/01-intro/09-pandas.md)

## Notas

- Cosas que no tenía frescas en la memoria:
  - Usamos `del` para borrar una columna: `del df['column']`
  - Visualizar filas:
    - Usando los nombres de las filas:
      - `df.loc[FILAS]`, con `FILAS` un elemento único (escalar o _string_) o una lista
      - Esto será útil cuando el índice no sea el rango numérico que Pandas crea por omisión
    - Usando los índices de las filas
      - `df.iloc[FILAS]`, con `FILAS` un escalar o una lista de escalares
  - Resetear el índice (operación no muta el _dataframe_):
    - `df.reset_index()`: promueve el índice existente a una columna
    - `df.reset_index(drop=True)`: descarta el índice existente
  - Operaciones sobre _strings_:
    - Sea `df.my_strings` una columna del _dataframe_ que contiene _strings_:
      - `df.my_strings.str.lower()` convierte todos los _strings_ a minúsculas
      - `df.my_strings.str.replace(A, B)` reemplaza todas las ocurrencias del _string_ `A` por `B`
    - Estas operaciones se pueden encadenar una tras otra
    - No mutan el _dataframe_ original
  - Operaciones de resumen:
    - Están las clásicas, como `df.columna.min()`, `df.columna.max()` o `df.columna.mean()` entre otras
    - También existen otras:
      - `df.columna.describe()`: retorna cuentas, promedios, desviaciones estándar, etc., para los datos de `df.columna`
      - `df.describe()`: como el anterior, pero aplicado a todas las columnas de `df` que son de tipo numérico
  - Operaciones sobre categorías:
    - `df.columna.nunique()`: retorna la cantidad de valores únicos en `df.columna`
    - `df.nunique()`: retorna la cantidad de valores únicos en cada columna de `df`
  - Operaciones sobre valores faltantes:
    - `df.columna.isnull()`: retorna una serie _booleana_ que indica acaso un valor es nulo (Nan) o no
    - `df.isnull()`: como el anterior, pero aplicado a todo el `df`
    - Para conocer la cantidad de valores faltantes, usamos `sum`:
      - `df.columna.isnull().sum()` y `df.isnull().sum()`
      - Esto se apoya en que `True` es interpretado como 1 y `False` es interpretado como 0
  - Agrupaciones de datos
    - Usamos `df.groupby(COLUMNA)`
    - Por ejemplo, para un _dataframe_ con datos de autos:
      - Queremos agrupar los datos por el tipo de transmisión (automática o manual) y calcular el precio promedio de los autos al interior de cada grupo
      - En SQL:
        ```sql
        SELECT
          transmision_type
          AVG(MSRP)
        FROM
          cars
        GROUP BY
          transmission_type
        ```
      - En Pandas: `df.groupby('Transmission type').MSRP.mean()` con `df` el dataframe que contiene la información equivalente a `cars`
      - Podemos calcular cualquier operación de resumen sobre las columnas de datos, no tan solo el promedio
  - Extraer los arreglos de Numpy contenidos en un _dataframe_:
    - `df.columna.values` (sin paréntesis al final de `values` ya que no es una función)
    - También lo podemos hacer sobre todo el `df` o un subconjunto de columnas
  - Exportar datos:
    - Múltiples alternativas
    - En particular, para exportar los datos a un diccionario con cada fila un elemento del diccionario: `df.to_dict(orient='records')`

# Resumen

## Material

- [Video](https://www.youtube.com/watch?v=VRrEEVeJ440) (6:45)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-110-summary)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/01-intro/10-summary.md)

## Notas

- N/A

# Tarea

## Pandas version

```python
import numpy as np
import pandas as pd
print(pd.__version__)
```

Resultado: 2.3.2

## Obtener datos

```bash
wget https://raw.githubusercontent.com/alexeygrigorev/datasets/master/car_fuel_efficiency.csv
```

## Records count

> How many records are in the dataset?

```python
df = pd.read_csv('car_fuel_efficiency.csv')
print(len(df))  # 9704
```

Resultado: 9704

## Fuel types

> How many fuel types are presented in the dataset?

```python
# corroborar que las categorías están bien formateadas (podrían existir "typos")
print(set(df['fuel_type']))
# obtener lo pedido
df['fuel_type'].nunique()
```

Resultado: 2 ("Diesel" y "Gasoline")

## Missing values

> How many columns in the dataset have missing values?

```python
sum(df.isnull().sum() > 0)
```

Resultado: 4

## Max fuel efficiency

> What's the maximum fuel efficiency of cars from Asia?

```python
df[df['origin'] == 'Asia']['fuel_efficiency_mpg'].max()
```

Resultado: 23.759

## Median value of horsepower

> 1. Find the median value of horsepower column in the dataset.
> 2. Next, calculate the most frequent value of the same horsepower column.
> 3. Use fillna method to fill the missing values in horsepower column with the most frequent value from the previous step.
> 4. Now, calculate the median value of horsepower once again.
>
> Has it changed?
>
> - Yes, it increased
> - Yes, it decreased
> - No

```python
mediana = df['horsepower'].median()  # 149.0
moda = df['horsepower'].mode()  # 152.0
# `.iloc[0]` es necesario porque `.mode()` retorna una serie, no un escalar
nueva_mediana = df['horsepower'].fillna(moda.iloc[0]).median()  # 152.0
print(mediana > nueva_mediana)  # False
print(mediana < nueva_mediana)  # True
print(mediana == nueva_mediana)  # False
```

Resultado: sí, el valor aumentó

## Sum of weights

> 1. Select all the cars from Asia
> 2. Select only columns vehicle_weight and model_year
> 3. Select the first 7 values
> 4. Get the underlying NumPy array. Let's call it X.
> 5. Compute matrix-matrix multiplication between the transpose of X and X. To get the transpose, use X.T. Let's call the result XTX.
> 6. Invert XTX.
> 7. Create an array y with values [1100, 1300, 800, 900, 1000, 1100, 1200].
> 8. Multiply the inverse of XTX with the transpose of X, and then multiply the result by y. Call the result w.
> 9. What's the sum of all the elements of the result?

```python
from_asia = df[df['origin'] == 'Asia']
two_columns = from_asia[['vehicle_weight', 'model_year']]
first7values = two_columns.iloc[:7]
X = first7values.values
XTX = np.matmul(X.T, X)
invXTX = np.linalg.inv(XTX)
y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])
w = (invXTX @ X.T ) @ y  # `@` es sinónimo de `np.matmul` al trabajar con `ndarrays`
print(w.sum())  # 0.51877
```

Resultado: 0.51 (truncado)
