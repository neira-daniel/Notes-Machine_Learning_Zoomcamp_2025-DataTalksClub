---
language: es
title: "Module 6: Decision Trees and Ensemble Learning"
author: Daniel Neira
---
> Learn tree-based models and ensemble methods for better predictions.
>
> Topics:
>
> - Decision trees
> - Random Forest
> - Gradient boosting (XGBoost)
> - Hyperparameter tuning
> - Feature importance

# Credit risk scoring project

## Material

- [Video](https://www.youtube.com/watch?v=GJGmlfZoCoU) (6:32)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-6-decision-trees-and-ensemble-learning)
- [Cuaderno de Jupyter del video](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/06-trees/notebook.ipynb)
- [Conjunto de datos usado en este módulo](https://github.com/gastonstat/CreditScoring)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/06-trees/01-credit-risk.md)

## Notas

- Ajustaremos un modelo que evalúe el riesgo de prestar dinero a una persona
  - Tendremos que evaluar acaso la persona será capaz de pagar el préstamo
  - Lo haremos basándonos en sus antecedentes comerciales, profesionales, etc., y en el monto del préstamo
- Lo que hará el modelo es retornar la probabilidad de que el cliente entre en _default_
  - Es decir, la probabilidad de que sea incapaz de retornar el préstamo
  - Interpretaremos esta probabilidad de manera análoga a como lo hicimos con el problema de la probabilidad de _churning_ que vimos en los módulos de clasificación (ver [clasificación](./es_classification.md) y [métricas para evaluar la clasificación](./es_classification_metrics.md))
  - Asignaremos 1 cuando la probabilidad de _default_ sea mayor a cierto umbral y 0 en caso contrario
- El modelo asume que clientes con antecedentes parecidos tendrán probabilidades similares de _default_
  - Luego, lo que haremos será asignar al cliente nuevo a un grupo que contenga personas con antecedentes similares
  - Y evaluaremos su probabilidad de _default_ en base a los datos de _default_ de clientes similares
  - Esta probabilidad podría ser matizada en función de si tenemos datos de _default_ del cliente al que estamos evaluando
- Entrenaremos el modelo usando el conjunto de datos [CreditScoring](https://github.com/gastonstat/CreditScoring)

# Data cleaning and preparation

## Material

- [Video](https://www.youtube.com/watch?v=tfuQdI3YO2c) (11:51)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-6-decision-trees-and-ensemble-learning)
- [Cuaderno de Jupyter del video](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/06-trees/notebook.ipynb)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/06-trees/02-data-prep.md)

## Notas

- El conjunto de datos:
  - Codifica las variables categóricas [de manera numérica](https://github.com/gastonstat/CreditScoring/blob/78bec232d45e3b1c5ee2971a3611e3c1fafd1e0f/Part1_CredScoring_Processing.R#L118)
  - Codifica los valores faltantes en las variables numéricas con [la cifra 99999999](https://github.com/gastonstat/CreditScoring/blob/78bec232d45e3b1c5ee2971a3611e3c1fafd1e0f/Part1_CredScoring_Processing.R#L49)
- Dado que trabajaremos con los datos de forma interactiva, vale la pena dar nombres a las categorías
  - Porción de código utilizado durante la exposición para dar nombres a las categorías:
    ```python
    home_values = {
      1: 'rent',
      2: 'owner',
      3: 'private',
      4: 'ignore',
      5: 'parents',
      6: 'other',
      0: 'unk'
    }

    # podría ser recomendable agregar `.astype(str)` al final
    df.home = df.home.map(home_values)
    # alternativa con código más autoexplicativo
    df.home = df.home.replace(mapping).astype(str)
    ```
- Es importante también hacer algo con la codificación de los valores faltantes
  - Si los dejamos tal cual, no podremos confiar en los cálculos que hagamos con las columnas que tienen valores 99999999
  - Podemos marcar esos valores como `nan` para luego decidir cómo imputarlos
    ```python
    # iteramos sobre las columnas con problemas
    for c in ['income', 'assets', 'debt']:
      df[c] = df[c].replace(to_replace=99999999, value=np.nan)

    # comprobamos visualmente que las cosas están bien
    df.describe().round()
    ```
  - Descartamos también el registro de `status` que tiene el valor `unk`:
    ```python
    df = df[df.status != 'unk'].reset_index(drop=True)
    ```
- Finalmente, preparamos los conjuntos de entrenamiento, validación y prueba de la misma forma que hicimos en lecciones anteriores:
  ```python
  from sklearn.model_selection import train_test_split

  seed = 11  # puede ser `None` cuando no necesitamos reproducibilidad
  df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=seed)
  df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=seed)

  df_train = df_train.reset_index(drop=True)
  df_val = df_val.reset_index(drop=True)
  df_test = df_test.reset_index(drop=True)

  # codificamos el estado `default` como 1
  # como consecuencia, las probabilidades que retorne el modelo para el valor 1 serán las probabilidades de _default_
  y_train = (df_train.status == 'default').astype('int').values
  y_val = (df_val.status == 'default').astype('int').values
  y_test = (df_test.status == 'default').astype('int').values

  del df_train['status']
  del df_val['status']
  del df_test['status']
  ```

# Decision trees

## Material

- [Video](https://www.youtube.com/watch?v=YGiQvFbSIg8) (17:19)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-6-decision-trees-and-ensemble-learning)
- [Cuaderno de Jupyter del video](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/06-trees/notebook.ipynb)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/06-trees/03-decision-trees.md)

## Notas

- Los árboles de decisión son estructuras con forma de árbol
  - Sus nodos representan tests respecto de los atributos
  - Las ramas representan los resultados de los tests
  - Y las hojas representan una etiqueta
- Trabajaremos con árboles de decisión binarios
  - Solo existirán dos respuestas posibles para cada test
  - De modo que solo podrán salir dos ramas de cada nodo
- Podemos imaginar los árboles de decisión binarios como una seguidilla de tests de tipo `if-else`
  - Ejemplo programado a mano:
    ```python
    def assess_risk(client):
       if client['records'] == 'yes':
          if client['job'] == 'parttime':
            return 'default'
          else:
            return 'ok'
       else:
          if client['assets'] > 6000:
            return 'ok'
          else:
            return 'default'

    xi = df_train.iloc[0].to_dict()
    assess_risk(xi)
    ```
  - Lo que representa el siguiente árbol:
    ```mermaid
    flowchart TD
      start["Start: assess_risk(client)"] --> records{"client['records'] == 'yes'?"}

      records -->|False| assets{"client['assets'] > 6000?"}
      assets -->|False| H[Return 'default']
      assets -->|True| G[Return 'ok']

      records -->|True| job{"client['job'] == 'parttime'?"}
      job -->|False| E[Return 'ok']
      job -->|True| D[Return 'default']
    ```
- Los tests serán aprendidos de forma automática a partir de los datos
- Y si no le ponemos límite al algoritmo de aprendizaje, el modelo replicará los datos de entrenamiento a la perfección
- El siguiente ejemplo ajusta los pesos del modelo de tal forma que al calcular su ROC AUC usando los datos de entrenamiento da 1 y tan solo 0.65 cuando usamos los de validación
  ```python
  from sklearn.tree import DecisionTreeClassifier
  from sklearn.feature_extraction import DictVectorizer
  from sklearn.metrics import roc_auc_score
  from sklearn.tree import export_text

  train_dicts = df_train.fillna(0).to_dict(orient='records')

  dv = DictVectorizer(sparse=False)
  X_train = dv.fit_transform(train_dicts)

  dt = DecisionTreeClassifier()
  dt.fit(X_train, y_train)

  # imputamos los valores faltantes con 0 para que el algoritmo pueda correr y así ejemplificar lo
  # que queremos mostrar (AUC ROC = 1) y no porque 0 sea un buen valor para ocupar aquí
  val_dicts = df_val.fillna(0).to_dict(orient='records')
  X_val = dv.transform(val_dicts)

  y_pred = dt.predict_proba(X_val)[:, 1]
  roc_auc_score(y_val, y_pred)  # 0.6548400377806302

  y_pred = dt.predict_proba(X_train)[:, 1]
  roc_auc_score(y_train, y_pred)  # 1.0
  ```
- Un modelo como el del ejemplo no tiene capacidad de generalizar
  - Memoriza los datos y tiene un desempeño perfecto con ellos
  - Pero su desempeño es pobre cuando hacemos predicciones para datos que el modelo no vio durante su entrenamiento
- Decimos que el modelo está sobreajustado (_overfitted_) cuando tiene un buen desempeño con los datos de entrenamiento, pero es incapaz de generalizar a otros conjuntos de datos
- Será fácil identificar un árbol de decisión sobreajustado: tendrá gran profundidad
  - En el caso del modelo del ejemplo: `dt.get_depth()` retorna 19 o 20 dependiendo de la aleatoriedad del algoritmo
  - Esa es la máxima distancia entre la raíz del árbol y una hoja
- En el caso de sklearn, podemos imponer una altura máxima con el parámetro `max_depth`:
  ```python
  dt = DecisionTreeClassifier(max_depth=3)
  dt.fit(X_train, y_train)

  y_pred = dt.predict_proba(X_train)[:, 1]
  auc = roc_auc_score(y_train, y_pred)
  print('train:', auc)  # 0.78

  y_pred = dt.predict_proba(X_val)[:, 1]
  auc = roc_auc_score(y_val, y_pred)
  print('val:', auc)  # 0.74

  # sklearn nos permite visualizar la estructura del árbol ajustado en la terminal usando `export_text`
  tree_structure = export_text(dt, feature_names=list(dv.get_feature_names_out()))
  print(tree_structure)
  ```
  - El árbol del ejemplo (el contenido de `tree_structure`):
  ```
  |--- records=yes <= 0.50
  |   |--- job=partime <= 0.50
  |   |   |--- income <= 74.50
  |   |   |   |--- class: 0
  |   |   |--- income >  74.50
  |   |   |   |--- class: 0
  |   |--- job=partime >  0.50
  |   |   |--- assets <= 8750.00
  |   |   |   |--- class: 1
  |   |   |--- assets >  8750.00
  |   |   |   |--- class: 0
  |--- records=yes >  0.50
  |   |--- seniority <= 6.50
  |   |   |--- amount <= 862.50
  |   |   |   |--- class: 0
  |   |   |--- amount >  862.50
  |   |   |   |--- class: 1
  |   |--- seniority >  6.50
  |   |   |--- income <= 103.50
  |   |   |   |--- class: 1
  |   |   |--- income >  103.50
  |   |   |   |--- class: 0
  ```
  - El desempeño del modelo del ejemplo no es muy bueno, pero generaliza mejor que aquel sobreajustado
  - (_Bonus_) Podemos interpretar la profundidad como un hiperparámetro del modelo, de modo que podemos ajustarla usando el conjunto de validación
- "Decision stump": un árbol de decisión de profundidad 2
  - Solo contiene una condición
  - Su predicción se basa en solo un atributo
  - (_Bonus_) Por sí solo tendrá un desempeño _débil_ ("weak learner")
  - (_Bonus_) Pero podemos usar _decision stumps_ como base de modelos más complejos

# Decision tree learning algorithm

## Material

- [Video](https://www.youtube.com/watch?v=XODz6LwKY7g) (29:13)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-6-decision-trees-and-ensemble-learning)
- [Cuaderno de Jupyter del video](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/06-trees/notebook.ipynb)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/06-trees/04-decision-tree-learning.md)

## Notas

- Dado un atributo, el algoritmo de aprendizaje busca el umbral de corte que genere la mejor predicción
  - Las reglas serán del estilo `condición(atributo) > T`
- Al evaluar la calidad de la predicción del modelo hablaremos de su _impureza_ ("impurity")
- Existen distintas formas de evaluar la impureza del clasificador
  - Una alternativa sencilla es la llamada "misclassification rate" (útil para temas didácticos)
  - Otras más elaboradas son Gini y entropía, las que también serán más comunes de usar
  - En caso de que ocupemos árboles de decisión para problemas de regresión utilizaremos indicadores de impureza distintos a los mencionados
- "Misclassification rate" usando clasificadores binarios y _decision stumps_
  - Los _decision stumps_ tienen profundidad 1: la raíz conecta inmediatamente con las hojas
  - En el caso de árboles de decisión binarios tendremos un nodo raíz con una condición y dos hojas
  - Como las hojas contienen las etiquetas, cuando usamos _decision stumps_ tenemos que etiquetar cada rama inmediatamente
  - Una forma sencilla de etiquetar cada rama es por votación
    - Notemos que al aplicar la condición del nodo raíz al conjunto de datos lo estaremos particionando en dos
    - Cada partición tendrá su propia distribución de etiquetas `0` y `1`
      - En el ejemplo que estamos tratando en este módulo `0` es `ok` y `1` es `default`
    - Cuando usamos votaciones dejamos que los registros "voten" a favor de una etiqueta
    - Asignaremos a cada rama la etiqueta que tiene mayor "votación" en su partición de datos
  - Dadas las predicciones, ahora podemos evaluar el desempeño del modelo
  - Cuando lo hacemos usando "misclasification rate", esto se vuelve trivial
  - Bastará con calcular la razón de errores que cometimos al predecir las etiquetas: cantidad de predicciones erróneas / total de registros en la partición
  - Terminaremos con dos errores (uno por cada partición)
  - Los reduciremos de alguna manera ad-hoc
    - Promedio simple
    - Promedio ponderado por la cantidad de registros en cada partición
    - Otros
  - Evaluaremos el desempeño del clasificador para valores distintos del umbral `T` usando este último indicador
- Ejemplo sintético:
  ```python
  data = [
      [8000, 'default'],
      [2000, 'default'],
      [   0, 'default'],
      [5000, 'ok'],
      [5000, 'ok'],
      [4000, 'ok'],
      [9000, 'ok'],
      [3000, 'default'],
  ]

  df_example = pd.DataFrame(data, columns=['assets', 'status'])
  print(df_example)
  ```
  - El contenido de `df_example`:
    ```
       assets   status
    0    8000  default
    1    2000  default
    2       0  default
    3    5000       ok
    4    5000       ok
    5    4000       ok
    6    9000       ok
    7    3000  default
    ```
  - Prueba con umbral `T=4000`:
    ```python
    T = 4000

    df_left = df_example[df_example.assets <= T]
    df_right = df_example[df_example.assets > T]

    print(df_left.status.value_counts(normalize=True))
    print(df_left.status.value_counts(normalize=True))
    ```
  - La condición es `df_example.assets > T`
    - Esta genera dos ramas: una para `True` y otra para `False`
    - `False`: predecimos la etiqueta `default` debido a que son mayoría en el conjunto de datos resultantes
      ```
         assets   status
      1    2000  default
      2       0  default
      5    4000       ok
      7    3000  default
      ```
    - `True`: y aquí predecimos `ok` por razones análogas
      ```
         assets   status
      0    8000  default
      3    5000       ok
      4    5000       ok
      6    9000       ok
      ```
    - En ambos casos el _misclassification rate_ es de 25 % (4 registros en cada partición y en cada una de ellas etiquetamos 1 mal)
  - Si escogemos reducir el _misclassification rate_ usando un promedio simple, resultará que el _misclassification rate_ de este árbol de decisión binario para `T=4000` es de 25 %
  - Al probar distintos valores de `T` obtenemos lo siguiente:
    ```
    | T    | Decision Left | Impurity Left | Decision Right | Impurity Right | AVG |
    |------|---------------|---------------|----------------|----------------|-----|
    | 0    | DEFAULT       | 0%            | OK             | 43%            | 21% |
    | 2000 | DEFAULT       | 0%            | OK             | 33%            | 16% |
    | 3000 | DEFAULT       | 0%            | OK             | 20%            | 10% |
    | 4000 | DEFAULT       | 25%           | OK             | 25%            | 25% |
    | 5000 | DEFAULT       | 50%           | OK             | 50%            | 50% |
    | 8000 | DEFAULT       | 43%           | OK             | 0%             | 21% |
    ```
  - Escogeremos aquel `T` para el que la impureza del modelo sea mínima cuando usamos los datos de validación
    - Como siempre, no debemos hacer esta optimización con el conjunto de entrenamiento
    - Si lo hiciéramos, terminaríamos con un modelo sobreajustado
  - Pasando por alto lo anterior, en este ejercicio sintético vemos que `T=3000` minimiza la impureza promedio del clasificador
  - (_Bonus_) Notemos que simplemente calcular el promedio podría no ser lo ideal o lo realmente deseado
    - Quizás estamos particularmente interesados en no clasificar erróneamente como `ok` a un cliente que terminará cayendo en default
    - O podría ser importante incluir el monto de los préstamos como otro atributo del modelo y aceptar algunos default para clientes que no solicitaron mucho dinero si eso nos permite clasificar correctamente a los clientes que pidieron grandes préstamos y que no caerán en default
- Podemos actualizar el ejemplo para ver qué ocurre con 2 atributos
  ```python
  data = [
    [8000, 3000, 'default'],
    [2000, 1000, 'default'],
    [   0, 1000, 'default'],
    [5000, 1000, 'ok'],
    [5000, 1000, 'ok'],
    [4000, 1000, 'ok'],
    [9000,  500, 'ok'],
    [3000, 2000, 'default'],
  ]

  df_example = pd.DataFrame(data, columns=['assets', 'debt', 'status'])
  print(df_example)
  ```
  - El contenido de `df_example`:
    ```
       assets  debt   status
    0    8000  3000  default
    1    2000  1000  default
    2       0  1000  default
    3    5000  1000       ok
    4    5000  1000       ok
    5    4000  1000       ok
    6    9000   500       ok
    7    3000  2000  default
    ```
  - Y ahora podemos calcular la impureza con todas las combinaciones de los atributos y valores de `thresholds`:
    ```python
    thresholds = {
      'assets': [0, 2000, 3000, 4000, 5000, 8000],
      'debt': [500, 1000, 2000]
    }
    for feature, Ts in thresholds.items():
      print('#####################')
      print(f'Feature: {feature}')
      for T in Ts:
        print(f'T = {T}')
        df_left = df_example[df_example[feature] <= T]
        df_right = df_example[df_example[feature] > T]

        print(df_left.status.value_counts(normalize=True))
        print(df_right.status.value_counts(normalize=True))

        print()
    ```
  - Completamos la tabla anterior ahora con 3 nuevas filas correspondientes a los umbrales de `debt`:
    ```
    | T    | Decision Left | Impurity Left | Decision Right | Impurity Right | AVG |
    |------|---------------|---------------|----------------|----------------|-----|
    | 0    | DEFAULT       | 0%            | OK             | 43%            | 22% |
    | 2000 | DEFAULT       | 0%            | OK             | 33%            | 16% |
    | 3000 | DEFAULT       | 0%            | OK             | 20%            | 10% |
    | 4000 | DEFAULT       | 25%           | OK             | 25%            | 25% |
    | 5000 | DEFAULT       | 50%           | OK             | 50%            | 50% |
    | 8000 | DEFAULT       | 43%           | OK             | 0%             | 22% |
    |------|---------------|---------------|----------------|----------------|-----|
    | 500  | OK            | 0%            | DEFAULT        | 43%            | 22% |
    | 1000 | OK            | 33%           | DEFAULT        | 0%             | 16% |
    | 2000 | OK            | 43%           | DEFAULT        | 0%             | 22% |
    ```
  - Vemos que `T=3000` para el atributo `assets` sigue siendo el óptimo bajo este indicador de impureza
- Pseudocódigo del algoritmo que selecciona las particiones óptimas:
  ```
  for f in features:
    thresholds <- find all thresholds of f
    for t in thresholds:
      splits <- split the dataset using "f > t" condition
      impurities(f, t) <- compute the impurity of the split
  select (f, t) that gives the lowest impurities(f, t)
  ```
  - Podemos ejecutar esto de manera recursiva por cada partición
  - Pero recordemos que no podemos hacer que la recursión sea muy profunda o terminaremos sobreajustando
- Criterios de parada del algoritmo: cuándo debemos dejar de iterar en cada partición
  - No particionaremos datos que sean puros (o sea, donde todos tengan la misma etiqueta)
  - No particionaremos datos con menos registros que un mínimo predefinido
  - No sobrepasaremos la profundidad máxima predefinida
- El pseudocódigo del algoritmo de aprendizaje de los árboles de decisión:
  ```
  verificar que no hemos alcanzado la altura máxima
  left, right <- encontrar los (f, t) de la partición óptima
  if left es suficientemente grande e impuro:
    repetir los dos primeros pasos con el conjunto de datos left
  if right es suficientemente grande e impuro:
    repetir los dos primeros pasos con el conjunto de datos right
  ```
- Podemos encontrar más información sobre los algoritmos y sobre los arboles de decisión en sí en la [documentación de sklearn](https://scikit-learn.org/stable/modules/tree.html)

# Decision trees parameter tuning

## Material

- [Video](https://www.youtube.com/watch?v=XJaxwH50Qok) (14:06)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-6-decision-trees-and-ensemble-learning)
- [Cuaderno de Jupyter del video](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/06-trees/notebook.ipynb)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/06-trees/05-decision-tree-tuning.md)

## Notas

- En esta lección nos concentraremos en dos hiperparámetros del modelo:
  - Profundidad máxima: `max_depth` en sklearn
  - Tamaño mínimo de cada partición: `min_samples_leaf` en sklearn
- Método del instructor:
  - Este método se basa en que, con conjuntos de datos reales (gran cantidad de observaciones), puede ser muy caro optimizar `max_depth` y `min_samples_leaf` a la vez
  - Una estrategia pragmática frente a ese problema es primero optimizar `max_depth`, seleccionar unos cuantos valores de profundidad que sean promisorios bajo algún criterio y luego optimizar `min_samples_leaf` con ese subconjunto de valores de `max_depth`
  - El siguiente pseudocódigo describe esta técnica usando ROC AUC como criterio de desempeño del clasificador:
    ```
    D <- valores posibles de `max_depth`
    S <- valores posibles de `min_samples_leaf`
    for d in D:
      ROC_AUC(d) <- calcular ROC AUC para `DecisionTreeClassifier(max_depth=d)`
    d_candidatos <- escoger los N < length(D) mejores resultados de ROC_AUC(D)
    for d in d_candidatos:
      for s in S:
        ROC_AUC(d, s) <- calcular ROC AUC para `DecisionTreeClassifier(max_depth=d,   min_samples_leaf=s)`
    escoger el mejor `ROC_AUC(d_candidatos, S)` bajo cierto criterio
    ```
- Con respecto al ROC AUC o cualquier otro criterio de desempeño:
  - No necesariamente escogeremos el máximo absoluto de ROC AUC obtenido
  - Por ejemplo, si varias configuraciones tienen un ROC AUC similar, podríamos escoger aquella relacionada con el árbol menos profundo
- Visualización de los ROC AUC obtenidos usando un mapa de calor:
  - Si bien podemos buscar el ROC AUC máximo de forma programática, observar cómo se comportan los valores calculados también tiene valor
  - Podemos desplegar esos resultados usando un mapa de calor sobre una tabla pivote de la siguiente forma:
    - Seleccionar los valores de profundidad promisorios:
      ```python
      depths = [1, 2, 3, 4, 5, 6, 10, 15, 20, None]

      for depth in depths:
        dt = DecisionTreeClassifier(max_depth=depth)
        dt.fit(X_train, y_train)

        y_pred = dt.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)

        print('%4s -> %.3f' % (depth, auc))
      ```
    - Calcular ROC AUC para cada combinación de los valores promisorios de profundidad y el tamaño mínimo posible de las hojas:
      ```python
      scores = []

      for depth in [4, 5, 6]:
        for s in [1, 5, 10, 15, 20, 500, 100, 200]:
          dt = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=s)
          dt.fit(X_train, y_train)

          y_pred = dt.predict_proba(X_val)[:, 1]
          auc = roc_auc_score(y_val, y_pred)

          scores.append((depth, s, auc))

      columns = ['max_depth', 'min_samples_leaf', 'auc']
      df_scores = pd.DataFrame(scores, columns=columns)
      ```
    - Contenido de `df_scores.head()`:
      ```
         max_depth  min_samples_leaf       auc
      0          4                 1  0.761283
      1          4                 5  0.761283
      2          4                10  0.761283
      3          4                15  0.763726
      4          4                20  0.760910
      ```
    - Construir la tabla pivote:
      ```python
      df_scores_pivot = df_scores.pivot(index='min_samples_leaf', columns=['max_depth'], values=['auc'])
      ```
    - El contenido de `df_scores_pivot.round(3)` es:
      ```
                          auc
      max_depth             4      5      6
      min_samples_leaf
      1                 0.761  0.766  0.756
      5                 0.761  0.768  0.760
      10                0.761  0.762  0.778
      15                0.764  0.772  0.786
      20                0.761  0.774  0.774
      100               0.756  0.763  0.776
      200               0.747  0.759  0.768
      500               0.680  0.680  0.680
      ```
    - Podemos usar Seaborn para graficar la tabla pivote como un mapa de calor en una sola línea de código:
      ```python
      import seaborn as sns
      sns.heatmap(df_scores_pivot, annot=True, fmt=".3f", cmap="flare")
      ```
- Finalmente, entrenaremos el modelo una vez más con los valores de `max_depth` y `min_samples_leaf` seleccionados
  - Por ejemplo, con `max_depth=6` y `min_samples_leaf=15`:
    ```python
    dt = DecisionTreeClassifier(max_depth=6, min_samples_leaf=15)
    dt.fit(X_train, y_train)
    ```
- (_Bonus_) Impacto de `max_depth` y `min_samples_leaf` en el sesgo y la varianza de las predicciones
  - `min_samples_leaf` constante, `max_depth` variable:
    - Árboles profundos: bajo sesgo, alta varianza
      - Sobreajuste: estos árboles pueden llegar a aprender incluso el ruido del conjunto de datos
      - Su gran profundidad se traduce en un gran número de particiones
      - Y esto se traduce en gran varianza: un cambio pequeño en los atributos de la observación puede hacer que ella caiga en una rama completamente distinta del árbol, posiblemente variando la predicción
    - Árboles poco profundos: alto sesgo, baja varianza
      - Subajuste: estos árboles solo pueden capturar patrones superficiales de los datos
      - Su baja profundidad se traduce en un pequeño número de particiones
      - Y esto se traduce en baja varianza: las observaciones tienen que variar bastante antes de que sean asignadas a una hoja distinta de la original durante la predicción
  - `max_depth` infinito, `min_samples_leaf` variable:
    - Si no limitamos la profundidad del árbol, podemos explicar `min_samples_leaf` en función de lo que dijimos recién para `max_depth`:
      - Cuando `min_samples_leaf` es pequeño, los árboles son profundos (bajo sesgo, alta varianza)
      - Cuando `min_samples_leaf` es grande, los árboles son poco profundos (alto sesgo, baja varianza)
  - Casos extremos:
    - `max_depth` grande y `min_samples_leaf` pequeño: sobreajuste
    - `max_depth` pequeño y `min_samples_leaf` grande: subajuste
  - Tenemos que movernos entre estos extremos en función de nuestra tolerancia al sesgo y la varianza

# Ensemble learning and random forest

## Material

- [Video](https://www.youtube.com/watch?v=FZhcmOfNNZE) (26:05)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-6-decision-trees-and-ensemble-learning)
- [Cuaderno de Jupyter del video](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/06-trees/notebook.ipynb)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/06-trees/06-random-forest.md)

## Notas

- Vimos antes que, si no lo impedimos, los árboles de decisión tenderán a sobreajustar los datos
- (_Bonus_) Lo anterior se traduce en bajo sesgo y alta varianza
  - Nos gustaría tener tanto un sesgo bajo como una varianza baja
  - Pero eso no se puede forzar indefinidamente
  - Siempre existe un [compromiso entre ambas cantidades](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff)
- Podemos hacer frente a este problema introduciendo azar en el algoritmo
  - (_Bonus_) Si somos hábiles, podremos disminuir la varianza incrementando el sesgo ligeramente
- La idea aquí es combinar las predicciones de múltiples árboles de decisión
  - Funcionamiento básico:
    - Cada uno de los árboles recibe conjuntos de entrenamiento distintos (muestreo con reemplazo)
    - Y particiona los datos en cada paso usando un subconjunto de atributos escogidos al azar con reemplazo (escoge el óptimo dentro de ese grupo)
  - Si no introdujéramos azar, estaríamos básicamente entrenando modelos demasiado similares como para ser capaces de generalizar
- El modelo se compone así de $n$ árboles
  - Hacemos, entonces, $n$ predicciones para cada nueva observación
  - La etiqueta que retorna el modelo a partir de las $n$ predicciones puede ser calculada de al menos dos formas:
    - Votación: el modelo retorna la etiqueta predicha por la mayoría de los árboles
    - Promedio de probabilidades: el modelo retorna la etiqueta calculando $n^{-1}\sum_{i=1}^n p_i > u$, con $p_i$ la probabilidad retornada por el $i$-ésimo árbol y $u$ el umbral de decisión
      - Tal y como vimos en módulos anteriores, el modelo retorna la etiqueta 1 cuando la condición es verdadera y 0 en el caso contrario
- Entrenamos un _random forest_ con sklearn de manera análoga a los demás modelos que hemos visto en los módulos anteriores:
  ```python
  from sklearn.ensemble import RandomForestClassifier

  # podemos fijar hiperparámetros adicionales al número de árboles `n_estimators`
  rf = RandomForestClassifier(n_estimators=10)
  rf.fit(X_train, y_train)

  y_pred = rf.predict_proba(X_val)[:, 1]
  auc = roc_auc_score(y_val, y_pred)
  ```
- Podemos encontrar los hiperparámetros óptimos con las mismas estrategias ya vistas:
  - Para evaluar la profundidad:
    ```python
    scores = []

    for d in [5, 10, 15]:  # profundidad de cada árbol
      for n in range(10, 201, 10):  # número de árboles
        rf = RandomForestClassifier(n_estimators=n,
                                    max_depth=d)
        rf.fit(X_train, y_train)

        y_pred = rf.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)

        scores.append((d, n, auc))

    # ordenamiento de resultados
    columns = ['max_depth', 'n_estimators', 'auc']
    df_scores = pd.DataFrame(scores, columns=columns)

    for d in [5, 10, 15]:
      df_subset = df_scores[df_scores.max_depth == d]

      # gráfico ROC AUC vs número de árboles para una profundidad `d`
      plt.plot(df_subset.n_estimators, df_subset.auc,
              label='max_depth=%d' % d)

    # activamos la leyenda para diferenciar las curvas de acuerdo a la profundidad de los árboles
    plt.legend()
    ```
  - Para encontrar el tamaño mínimo de las hojas dada una altura fija (aquella encontrada en el paso anterior)
    ```python
    max_depth = 10

    scores = []

    for s in [1, 3, 5, 10, 50]:
      for n in range(10, 201, 10):
        rf = RandomForestClassifier(n_estimators=n,
                                    max_depth=max_depth,
                                    min_samples_leaf=s)
        rf.fit(X_train, y_train)

        y_pred = rf.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)

        scores.append((s, n, auc))

    columns = ['min_samples_leaf', 'n_estimators', 'auc']
    df_scores = pd.DataFrame(scores, columns=columns)

    colors = ['black', 'blue', 'orange', 'red', 'grey']
    values = [1, 3, 5, 10, 50]

    for s, col in zip(values, colors):
      df_subset = df_scores[df_scores.min_samples_leaf == s]

      plt.plot(df_subset.n_estimators, df_subset.auc,
               color=col,
               label='min_samples_leaf=%d' % s)

    plt.legend()
    ```
  - Y entrenamos finalmente el modelo con los hiperparámetros obtenidos:
    ```python
    min_samples_leaf = 3

    rf = RandomForestClassifier(n_estimators=100,
                                max_depth=max_depth,
                                min_samples_leaf=min_samples_leaf)
    rf.fit(X_train, y_train)
    ```
- En todos los ejemplos de clase se usó la semilla `random_state=1` al crear los clasificadores
  - Aquí la omitimos porque no necesitamos reproducibilidad de los resultados, sino entender los pasos
- Existen otros parámetros que podemos especificar al inicializar el modelo
  - Ver la [documentación de sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) para conocer más detalles
  - En particular, será útil modificar el argumento `n_jobs` para especificar la cantidad de procesos en paralelo que queremos ejecutar (el valor por omisión es `None`, el que se interpreta como 1 a menos que ejecutemos el código dentro de un [contexto ad-hoc](https://joblib.readthedocs.io/en/stable/generated/joblib.parallel_config.html))
  - Los _random forests_ son fácilmente paralelizables (no comparten estado), de modo que podríamos ganar mucho tiempo entrenando varios árboles a la vez

# Gradient boosting and XGBoost

## Material

- [Video](https://www.youtube.com/watch?v=xFarGClszEM) (19:05)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-6-decision-trees-and-ensemble-learning)
- [Cuaderno de Jupyter del video](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/06-trees/notebook.ipynb)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/06-trees/07-boosting.md)

## Notas

- _Random forests_:
  - Vimos que los árboles del algoritmo de _random forests_ no comparten estado
  - Como no comparten estado, son independientes y fácilmente paralelizables
- _Boosting_:
  - Corresponde a una estrategia donde se combinan modelos débiles (de bajo desempeño) para producir un modelo fuerte (de alto desempeño) de forma secuencial
  - Cada modelo débil de la cadena busca corregir los errores cometidos por el modelo anterior
- _Gradient boosting trees_:
  - Implementación específica de la idea de _boosting_
  - Se construye usando árboles de decisión sencillos como modelos débiles
  - Optimiza los pesos usando el algoritmo de descenso de gradiente en el error
- En esta sección utilizaremos [XGBoost](https://github.com/dmlc/xgboost) en lugar de sklearn (`uv add xgboost`) para entrenar el modelo
  - (_Bonus_) Podemos consultar la [galería de ejemplos](https://xgboost.readthedocs.io/en/stable/python/examples/index.html) oficiales de XGBoost para explorar las capacidades de esta librería (también disponible como un [repositorio de GitHub](https://github.com/dmlc/xgboost/tree/master/demo/guide-python))
- (_Bonus_) Si bien XGBoost tiene una [API compatible con sklearn](https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn), aquí trabajaremos con la [API propia de XGBoost](https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.training)
  - Es por eso que utilizaremos objetos `DMatrix` en los ejemplos de esta sección para interactuar con XGBoost y no otros propios del ecosistema de Python como NumPy o Pandas
  - La elección de la API propia de XGBoost también determina que usemos el método [`train`](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.train) en lugar de [`fit`](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRegressor.fit) al entrenar los modelos
- Código sugerido para entrenar un modelo de _gradient boosting trees_:
  ```python
  import xgboost as xgb

  features = list(dv.get_feature_names_out())
  dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
  dval = xgb.DMatrix(X_val, label=y_val, feature_names=features)

  xgb_params = {
    'eta': 0.3,
    'max_depth': 6,
    'min_child_weight': 1,

    'objective': 'binary:logistic',
    'nthread': 8,

    'seed': 1,
    'verbosity': 1,
  }

  model = xgb.train(xgb_params, dtrain, num_boost_round=10)
  ```
  - Las variables `dv`, `X_train`, etc, provienen de las [etapas anteriores de esta lección](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/06-trees/notebook.ipynb)
  - [`DMatrix`](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.DMatrix) es la estructura de datos interna utilizada por XGBoost
  - El diccionario `xgb_params` contiene los [parámetros](https://xgboost.readthedocs.io/en/stable/parameter.html#general-parameters) que XGBoost necesita para entrenar el modelo
    - Podemos especificar múltiples opciones
    - (_Bonus_) En particular, usaremos `booster` para especificar el tipo de modelo que queremos usar
      - Su valor por omisión es `gbtree`
        - [Utiliza árboles de decisión](https://xgboosting.com/configure-xgboost-tree-booster-gbtree/) en el rol de _weak learners_
      - [Alternativas](https://xgboosting.com/configure-xgboost-booster-parameter/):
        - `dart`: útil cuando `gbtree` está sobreajustando
        - `gblinear`: útil para datos en altas dimensions y _sparse_ cuyas relaciones se puedan aproximar con modelos lineales (produce modelos más sencillos que `gbtree`)
    - (_Bonus_) Significado de las opciones especificadas:
      - [`eta`](https://xgboosting.com/configure-xgboost-eta-parameter/): factor por el que multiplicar la salida de cada uno de los árboles usados durante cada etapa de _boosting_
        - En cada etapa $t$ de _boosting_: $\hat{y}^{(t)}=\hat{y}^{(t-1)}+\eta f_t(x)$, con $\eta \in [0, 1]$ y $f_t(x)$ la función que representa los árboles de la etapa $t$
        - Valores cercanos a 1 hacen que el algoritmo sea más rápido, pero su desempeño tenderá a ser subóptimo ya que las correcciones serán de gran magnitud y podrían intentar seguir el ruido en lugar de la señal, lo que dificulará la convergencia al óptimo (sufrirá de _overshooting_)
        - Valores cercanos a 0 fuerzan a que aumentemos la cantidad de etapas para lograr convergencia ya que cada paso es "pequeño", haciendo que el algoritmo sea más lento, pero las correcciones serán más precisas, minimizando el riesgo de sobreajustar y mejorando las posibilidades de llegar al óptimo
      - [`max_depth`](https://xgboosting.com/configure-xgboost-max_depth-parameter/): la profundidad máxima del árbol
        - Funciona de la misma forma que vimos en la sección "Decision trees parameter tuning"
      - [`min_child_weight`](https://xgboosting.com/configure-xgboost-min_child_weight-parameter/): nos permite controlar el tamaño mínimo de una partición
        - Dos casos:
          - En regresión es, efectivamente, el número [mínimo posible de instancias en un nodo](https://xgboost.readthedocs.io/en/stable/parameter.html#parameters-for-tree-booster)
          - En clasificación resulta tanto un _proxy_ del [nivel mínimo de pureza que aceptaremos](https://stats.stackexchange.com/a/323459) como de la cantidad de instancias mínimas que aceptaremos en un nodo
        - Se calcula a partir de la suma de los hessianos (segundas derivadas) de la función de error que estamos optimizando con el algoritmo
        - En particular, este parámetro fija el valor mínimo que debe tener esa suma en cada nodo del árbol
        - La intuición en el caso de la clasificación es que, cuando las particiones resultan demasiado puras, los [hessianos tendrán valores cercanos a 0](https://stats.stackexchange.com/a/323459)
        - También podemos ver que, para nodos con una pureza "baja" (suficientemente bien mezclados), esto controla el tamaño de las particiones:
          - Pocos valores que sumar: la suma será probablemente baja
          - Muchos valores que sumar: la suma será probablemente "alta"
        - Luego, a mayor valor de `min_child_weight`, el modelo será más conservador:
          - Los árboles serán más sencillos
          - Corremos el riesgo de subajustar el modelo
      - [`objective`](https://xgboost.readthedocs.io/en/stable/parameter.html#learning-task-parameters): determina la función de costo o error (función objetivo) que optimizará el algoritmo
        - El valor `binary:logistic` indica que queremos utilizar un modelo de regresión logística para clasificación binaria y que el modelo debe retornar probabilidades
          - El algoritmo usa un _ensemble_ de árboles de regresión para producir un valor real (no una etiqueta)
          - Luego suma todos esos valores para producir un _score_
          - Dicho _score_ es tranformado con una _sigmoide_ (función logística) para producir una probabilidad (tal y como vimos en el [módulo sobre clasificación](./es_classification.md))
          - Finalmente, el modelo evalúa el error que está cometiendo usando las probabilidades recién calculadas
            - Si no cambiamos el valor por omisión de [`eval_metric`](https://xgboosting.com/configure-xgboost-eval_metric-parameter/), el desempeño de este modelo será medido usando el logaritmo de la función de costo
        - Debemos [consultar la documentación](https://xgboost.readthedocs.io/en/stable/parameter.html#learning-task-parameters) para conocer todos los valores posibles de `objective` (información adicional en el sitio web [XGBoosting](https://xgboosting.com/configure-xgboost-objective-parameter/))
      - `nthread` y `seed` son autoexplicativas mientras que `verbosity` es [muy sencilla de entender](https://xgboosting.com/configure-xgboost-verbosity-parameter/)
- El argumento `num_boost_round` de `xgboost.train` fija la cantidad de etapas de _boosting_
  - Por omisión, el algoritmo ejecutará la totalidad de las etapas sin importar acaso el modelo está mejorando o si estamos sobreajustando a los datos
  - Usamos el parámetro [`early_stopping_rounds`](https://xgboost.readthedocs.io/en/stable/python/python_api.html#module-xgboost.training) (un número entero) para evitar que el algoritmo siga corriendo a pesar de que el desempeño del modelo no esté mejorando
  - El algoritmo se detendrá cuando pasen `early_stopping_rounds` etapas de _boosting_ sin que mejore la métrica especificada en `eval_metric`
  - Cuando ocupemos este parámetro tendremos también que especificar el conjunto donde queremos que se calcule la métrica, lo que hacemos a través del parámetro `evals`
- Ejemplo para visualizar cómo evoluciona el ROC AUC con cada etapa de _boosting_
  - Entrenamiento:
    ```python
    %%capture output

    xgb_params = {
      'eta': 0.3,
      'max_depth': 6,
      'min_child_weight': 1,

      'objective': 'binary:logistic',
      'eval_metric': 'auc',

      'nthread': 8,
      'seed': 1,
      'verbosity': 1,
    }

    model = xgb.train(xgb_params, dtrain, num_boost_round=200,
                      verbose_eval=5,
                      evals=watchlist)
    ```
    - Fijamos ROC AUC como métrica de evaluación del desempeño del modelo con `'eval_metric': 'auc'`
    - Este entrenamiento tiene 200 etapas de boosting y los resultados serán impresos en pantalla cada 5 etapas de _boosting_
    - `%%capture output`: esta es una "función mágica" de IPython que nos permite capturar lo que se estaría imprimiendo en pantalla y almacenarlo en una variable (en este caso, `output`)
      - Esto solo funcionará en Jupyter (y, probablemente, en IPython)
    - Existe una forma alternativa de acceder a estos resultados usando el argumento `evals_result` de la forma en que se muestra [en este ejemplo](https://github.com/dmlc/xgboost/blob/master/demo/guide-python/evals_result.py)
  - Visualización:
    ```python
    def parse_xgb_output(output):
      results = []

      for line in output.stdout.strip().split('\n'):
        it_line, train_line, val_line = line.split('\t')

        it = int(it_line.strip('[]'))
        train = float(train_line.split(':')[1])
        val = float(val_line.split(':')[1])

        results.append((it, train, val))

      columns = ['num_iter', 'train_auc', 'val_auc']
      df_results = pd.DataFrame(results, columns=columns)
      return df_results

    df_score = parse_xgb_output(output)
    plt.plot(df_score.num_iter, df_score.train_auc, label='train')
    plt.plot(df_score.num_iter, df_score.val_auc, label='val')
    plt.legend()
    ```
    - El resultado es que, a más tardar en la iteración 40 o 50, el modelo simplemente sobreajusta
      - ROC AUC es prácticamente 1 para el conjunto de entrenamiento
      - Y la tendencia del ROC AUC en el conjunto de validación cae
    - Si graficamos el ROC AUC del conjunto de validación por sí solo (`plt.plot(df_score.num_iter, df_score.val_auc, label='val'); plt.legend()`), veremos que bastaba con unas 30 etapas de _boosting_ para maximizarlo
    - Pero como no usamos el parámetro `early_stopping_rounds`, el algoritmo siguió ejecutándose durante las `num_boost_round` etapas de _boosting_ que le pedimos
  - Si bien lo lógico es usar `early_stopping_rounds` para ahorrarnos tiempo de cálculo de más y evitar sobreajuste, también tiene valor observar cómo se comportan las predicciones del modelo para entender su funcionamiento

# XGBoost parameter tuning

## Material

- [Video](https://www.youtube.com/watch?v=VX6ftRzYROM) (18:37)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-6-decision-trees-and-ensemble-learning)
- [Cuaderno de Jupyter del video](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/06-trees/notebook.ipynb)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/06-trees/08-xgb-tuning.md)

## Notas

- x

# Selecting the best model

## Material

- [Video](https://www.youtube.com/watch?v=lqdnyIVQq-M) (7:32)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-6-decision-trees-and-ensemble-learning)
- [Cuaderno de Jupyter del video](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/06-trees/notebook.ipynb)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/06-trees/09-final-model.md)

## Notas

- x

# Summary

## Material

- [Video](https://www.youtube.com/watch?v=JZ6sRZ_5j_c) (4:19)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-6-decision-trees-and-ensemble-learning)
- [Cuaderno de Jupyter del video](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/06-trees/notebook.ipynb)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/06-trees/10-summary.md)

## Notas

- x

# Explore more

## Material

- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/06-trees/11-explore-more.md)

## Notas

- x

# Homework

## Material

- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/06-trees/homework.md)

## Notas

- x
