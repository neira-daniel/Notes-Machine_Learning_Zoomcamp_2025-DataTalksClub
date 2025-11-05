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

- x

# Ensemble learning and random forest

## Material

- [Video](https://www.youtube.com/watch?v=FZhcmOfNNZE) (26:05)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-6-decision-trees-and-ensemble-learning)
- [Cuaderno de Jupyter del video](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/06-trees/notebook.ipynb)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/06-trees/06-random-forest.md)

## Notas

- x

# Gradient boosting and XGBoost

## Material

- [Video](https://www.youtube.com/watch?v=xFarGClszEM) (19:05)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-6-decision-trees-and-ensemble-learning)
- [Cuaderno de Jupyter del video](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/06-trees/notebook.ipynb)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/06-trees/07-boosting.md)

## Notas

- x

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
