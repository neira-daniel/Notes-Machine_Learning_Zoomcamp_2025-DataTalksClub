---
language: es
title: "Module 4: Evaluation Metrics for Classification"
author: Daniel Neira
---
> Learn how to properly evaluate classification models and handle imbalanced datasets.
>
> Topics:
>
> - Accuracy, precision, recall, F1-score
> - ROC curves and AUC
> - Cross-validation
> - Confusion matrices
> - Class imbalance handling

# Evaluation metrics: session overview

## Material

- [Video](https://www.youtube.com/watch?v=gmg5jw1bM8A) (3:33)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-4-evaluation-metrics-for-classification)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/01-overview.md)

## Notas

- Continuaremos abordando el problema tratado en el [módulo de clasificación](./es_classification.md): identificar los clientes de una empresa de telecomunicaciones que están cerca de dejar de usar el servicio ("churn")
- La idea es utilizar herramientas adicionales al cálculo de la exactitud (_accuracy_) del modelo para evaluar su desempeño

# Accuracy and dummy model

## Material

- [Video](https://www.youtube.com/watch?v=FW_l7lB0HUI) (13:21)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-4-evaluation-metrics-for-classification)
- [Cuaderno de Jupyter del video](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/04-evaluation/notebook.ipynb)
- [Cuaderno de Jupyter adicional](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/chapter-03-churn-prediction/04-metrics.ipynb)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/02-accuracy.md)

## Notas

- Exactitud (_accuracy_): cuantifica la fracción de predicciones correctas
  - Ratio entre la cantidad de predicciones correctas y el total de predicciones
- Podemos calcular la exactitud para distintos valores de umbral de clasificación
  - Hemos estado creando predicciones "duras" a partir de predicciones "suaves" (probabilidades) usando un umbral de 0.5
  - Podemos usar un grupo de umbrales candidatos y probarlos todos:
    ```python
    from sklearn.metrics import accuracy_score
    thresholds = np.linspace(0, 1, 21)
    scores = []
    for t in thresholds:
      score = accuracy_score(y_val, y_pred >= t)
      print('%.2f %.3f' % (t, score))
      scores.append(score)
    plt.plot(thresholds, scores)
    ```
  - Si ejecutamos ese código con los datos y las particiones del módulo anterior, veremos que, en este caso particular, el umbral 0.5 es el óptimo bajo el criterio de exactitud
  - En general, no deberíamos importar una librería de gran tamaño para hacer cosas sencillas como calcular esta exactitud, pero como ya veníamos trabajando con sklearn, no hay problema con importar paquetes o módulos adicionales de esa misma librería
- Desbalance de clases (_class imbalance_): cuando una clase tiene mucha mayor representación que otra en los datos
  - En el caso de los datos de la empresa de telecomunicaciones: la proporción de clientes que renuevan sus contratos con respecto a los que dejan de usar los servicios de la empresa es casi de 3:1
  - Cuando se dan situaciones tan desbalanceadas como esta, modelos sencillos que ni siquiera toman en cuenta los datos podrían tener buen desempeño
  - Este es el caso del modelo que predice que nadie dejará de usar los servicios de la empresa
    - Al ejecutar el código anterior vimos que si usamos un umbral 1.0 con la regresión logística, la exactitud del modelo es de 73 %
      - Esto es equivalente a usar un modelo que dice que nadie dejará de ocupar los servicios de la empresa
      - Y no hace falta mirar datos ni hacer ningún ajuste para construir dicho modelo
      - A este tipo de modelo le llamaron "dummy" en el video
    - El mejor resultado que obtuvimos con nuestro trabajo fue una exactitud de 80 %
    - Bajo el criterio de exactitud, el clasificador que ajustamos no está funcionando tan bien comparado con uno que ni siquiera mira los datos

# Confusion table

## Material

- [Video](https://www.youtube.com/watch?v=Jt2dDLSlBng) (19:54)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-4-evaluation-metrics-for-classification)
- [Cuaderno de Jupyter del video](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/04-evaluation/notebook.ipynb)
- [Cuaderno de Jupyter adicional](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/chapter-03-churn-prediction/04-metrics.ipynb)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/03-confusion-table.md)

## Notas

- Nos interesa cuantificar:
  - Los aciertos
    - Predijimos la etiqueta positiva correctamente: positivos verdaderos (_true positive_)
    - Predijimos negativo correctamente: negativos verdaderos (_true negative_)
  - Los errores
    - Predijimos la etiqueta positiva erróneamente: positivo falso (_false positive_)
    - Predijimos negativo erróneamente: negativo falso (_false negative_)
- En forma de diagrama (`t` es el umbral):
  ```mermaid
  flowchart TD
      A["g(xi)"] -->|" < t "| B["NEGATIVE:<br/>NO CHURN"]
      A -->|" ≥ t "| C["POSITIVE:<br/>CHURN"]

      B --> D["CUSTOMER DIDN'T CHURN<br/>(True Negative)"]
      B --> E["CUSTOMER CHURNED<br/>(False Negative)"]

      C --> F["CUSTOMER DIDN'T CHURN<br/>(False Positive)"]
      C --> G["CUSTOMER CHURNED<br/>(True Positive)"]

      style A fill:#ffffff,stroke:#000000,stroke-width:2px
      style B fill:#b0f3ff,stroke:#2596be,stroke-width:2px
      style C fill:#b0f3ff,stroke:#2596be,stroke-width:2px
      style D fill:#e6ffe6,stroke:#00aa00,stroke-width:2px
      style E fill:#ffe6e6,stroke:#ff0000,stroke-width:2px
      style F fill:#ffe6e6,stroke:#ff0000,stroke-width:2px
      style G fill:#e6ffe6,stroke:#00aa00,stroke-width:2px
  ```
- Organizamos estos 4 valores posibles en una tabla de 2x2 denominada "matriz de confusión" (_confusion matrix_, llamada ocasionalmente _confusion table_ en el video y en el título de esta sección):
  |                  | Prediction: negative | Prediction: positive |
  |------------------|----------------------|----------------------|
  | Actual: negative | True negative        | False positive       |
  | Actual: positive | False negative       | True positive        |
  - (_Bonus_) En [Wikipedia](https://en.wikipedia.org/wiki/Confusion_matrix#Example) orientan la matriz de otra forma, pero la información es la misma
    - Pero cabe tenerlo en consideración porque a veces se opera por columnas y filas y si no prestamos atención podríamos malinterpretar esos valores
- Código sugerido para construir esta matriz:
  ```python
  actual_positive = (y_val == 1)
  actual_negative = (y_val == 0)

  t = 0.5
  predict_positive = (y_pred >= t)
  predict_negative = (y_pred < t)

  tp = (predict_positive & actual_positive).sum()
  tn = (predict_negative & actual_negative).sum()

  fp = (predict_positive & actual_negative).sum()
  fn = (predict_negative & actual_positive).sum()

  confusion_matrix = np.array([
      [tn, fp],
      [fn, tp]
  ])

  # valores normalizados y desplegamos con 2 decimales
  print((confusion_matrix / confusion_matrix.sum()).round(2))
  ```
- Si utilizamos los valores normalizados, la exactitud será la suma de los valores de la primera columna del arreglo especificado en el código

# Precision and recall

## Material

- [Video](https://www.youtube.com/watch?v=gRLP_mlglMM) (14:40)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-4-evaluation-metrics-for-classification)
- [Cuaderno de Jupyter del video](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/04-evaluation/notebook.ipynb)
- [Cuaderno de Jupyter adicional](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/chapter-03-churn-prediction/04-metrics.ipynb)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/04-precision-recall.md)

## Notas

- Veremos indicadores relevantes para clasificadores binarios
- Nomenclatura:
  - tp: "true positive"
  - fp: "false positive"
  - tn: "true negative"
  - fn: "false negative"
- Precisión: fracción de predicciones positivas (etiqueta "1") correctas con respecto al total de predicciones positivas
  $$\frac{\mathrm{tp}}{\text{Nº positive predictions}}=\frac{\mathrm{tp}}{\mathrm{tp}+\mathrm{fp}}$$
- "Exhaustividad" (_recall_): fracción de predicciones positivas correctas con respecto al total de etiquetas positivas observadas
  $$\frac{\mathrm{tp}}{\text{Nº positive observations}}=\frac{\mathrm{tp}}{\mathrm{tp}+\mathrm{fn}}$$
- En el caso del clasificador de _churning_ ajustado en el video:
  - $1 - \mathrm{precisión} = 33 \%$ da cuenta de la fracción de personas a las que les ofrecimos descuentos a pesar de que era improbable de que no renovaran el contrato de servicio
  - $1 - \mathrm{recall} = 46 \%$ da cuenta de la fracción de personas que era probable que no renovaran el contrato de servicio y a las que no fuimos capaces de identificar como tales
  - Vemos que, a pesar de que la exactitud de este modelo era de $80 \%$, en realidad no está funcionando bien
- Regla mnemotécnica que aparece en las [notas comunitarias](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/08680929f3a1c9880f8bf26a6aa340502e18150d/04-evaluation/04-precision-recall.md):
  - Precision : From the `pre`dicted positives, how many we predicted right. See how the word `pre`cision is similar to the word `pre`diction?
  - Recall : From the `real` positives, how many we predicted right. See how the word `re`c`al`l is similar to the word `real`?

# ROC curves

## Material

- [Video 1](https://www.youtube.com/watch?v=dnBZLk53sQI) (34:45)
- [Video 2](https://www.youtube.com/watch?v=B5PATo1J6yw) (1:58)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-4-evaluation-metrics-for-classification)
- [Cuaderno de Jupyter del video](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/04-evaluation/notebook.ipynb)
- [Cuaderno de Jupyter adicional](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/chapter-03-churn-prediction/04-metrics.ipynb)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/05-roc.md)

## Notas

- ROC curve: "receiver operating characteristic curve"
- Nos permite ilustrar el desempeño de un clasificador binario para distintos umbrales de decisión
- La construimos a partir de dos indicadores que se pueden extraer de la matriz de confusión:
  - FPR: false positive rate
    - Razón entre los falsos positivos y todas las observaciones negativas
      $$\frac{\mathrm{fp}}{\text{Nº false observations}}=\frac{\mathrm{fp}}{\mathrm{tn}+\mathrm{fp}}$$
  - TPR: true positive rate
    - Razón entre los positivos verdaderos y todas las observaciones positivas
    - Es idéntico a _recall_
    $$\frac{\mathrm{tp}}{\text{Nº positive observations}}=\frac{\mathrm{tp}}{\mathrm{tp}+\mathrm{fn}}$$
- Naturalmente, nos gustaría que:
  - FPR sea lo menor posible: se minimice fp
  - TPR sea lo más grande posible: se minimice fn
- Código sugerido para crear este gráfico:
  ```python
  def tpr_fpr_dataframe(y_val, y_pred):
      scores = []

      thresholds = np.linspace(0, 1, 101)

      actual_positive = (y_val == 1)
      actual_negative = (y_val == 0)

      for t in thresholds:

          predict_positive = (y_pred >= t)
          predict_negative = (y_pred < t)

          tp = (predict_positive & actual_positive).sum()
          tn = (predict_negative & actual_negative).sum()

          fp = (predict_positive & actual_negative).sum()
          fn = (predict_negative & actual_positive).sum()

          scores.append((t, tp, fp, fn, tn))

      columns = ['threshold', 'tp', 'fp', 'fn', 'tn']
      df_scores = pd.DataFrame(scores, columns=columns)

      df_scores['tpr'] = df_scores.tp / (df_scores.tp + df_scores.fn)
      df_scores['fpr'] = df_scores.fp / (df_scores.fp + df_scores.tn)

      return df_scores

  df_scores = tpr_fpr_dataframe(y_val, y_pred)

  plt.plot(df_scores.threshold, df_scores['tpr'], label='TPR')
  plt.plot(df_scores.threshold, df_scores['fpr'], label='FPR')
  plt.legend()
  ```
- Nos interesará comparar curvas ROC con distintos modelos
- En particular, podemos visualizar la curva ROC de nuestro modelo comparado con un modelo ideal y con otro aleatorio
- El modelo ideal siempre acierta con su predicción
  - Podemos crearlo y visualizarlo con el siguiente código:
    ```python
    # identificamos el número de observaciones de cada etiqueta
    num_neg = (y_val == 0).sum()
    num_pos = (y_val == 1).sum()

    # creamos `y_ideal` con tantos 0 al inicio como `num_neg` y luego 1 como `num_pos`
    y_ideal = np.repeat([0, 1], [num_neg, num_pos])
    # esta es la probabilidad de las predicciones del modelo ideal
    # bastará con crear una malla de valores entre 0 y 1 y con el mismo largo que `y_val`
    # el umbral a partir del cual el modelo ideal predice 1 es `1 - y_val.mean()`
    y_ideal_pred = np.linspace(0, 1, len(y_val))

    df_ideal = tpr_fpr_dataframe(y_ideal, y_ideal_pred)

    plt.plot(df_ideal.threshold, df_ideal['tpr'], label='TPR')
    plt.plot(df_ideal.threshold, df_ideal['fpr'], label='FPR')
    plt.legend()
    ```
- El modelo aleatorio se construye a partir de una distribución uniforme entre 0 y 1
  - Podemos crearlo y visualizarlo con el siguiente código:
    ```python
    y_rand = np.random.uniform(0, 1, size=len(y_val))
    df_rand = tpr_fpr_dataframe(y_val, y_rand)

    plt.plot(df_rand.threshold, df_rand['tpr'], label='TPR')
    plt.plot(df_rand.threshold, df_rand['fpr'], label='FPR')
    plt.legend()
    ```
- Graficamos las curvas ROC de nuestro modelo y del modelo ideal con el siguiente código
  ```python
  plt.plot(df_scores.threshold, df_scores['tpr'], label='TPR', color='black')
  plt.plot(df_scores.threshold, df_scores['fpr'], label='FPR', color='blue')

  plt.plot(df_ideal.threshold, df_ideal['tpr'], label='TPR ideal')
  plt.plot(df_ideal.threshold, df_ideal['fpr'], label='FPR ideal')

  plt.legend()
  ```
  - Dejamos de lado el modelo aleatorio pues ya vimos que sus curvas son similares a rectas
  - Incluirlas en este gráfico simplemente dificulta su interpretación
- Lo que hacemos en la práctica es graficar la curva ROC en el espacio FPR vs TPR (el valor del umbral quedará implícito):
  ```python
  plt.figure(figsize=(5, 5))

  plt.plot(df_scores.fpr, df_scores.tpr, label='Model')
  # simplemente trazamos la recta en lugar del modelo aleatorio porque el comportamiento es similar
  plt.plot([0, 1], [0, 1], label='Random', linestyle='--')
  plt.plot(df_ideal.fpr, df_ideal.tpr, label='Ideal')

  plt.xlabel('FPR')
  plt.ylabel('TPR')

  plt.legend()
  ```
  - El comportamiento del modelo ideal es conocido en este espacio y siempre el mismo, de modo que rara vez lo graficaremos
    - Siempre va desde (0, 0) a (0, 1) y luego a (1, 1)
    - Nos indica que lo ideal sería que la curva ROC de nuestro modelo se acerque lo más posible al punto (0, 1)
  - Lo hacemos aquí para observarlo y estudiar su comportamiento nada más, pero luego lo omitiremos
  - En particular, si usamos sklearn para calcular los datos de la curva ROC, solo obtendremos la curva ROC de nuestro modelo (y la graficamos igual que antes):
    ```python
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_val, y_pred)

    plt.figure(figsize=(5, 5))

    plt.plot(fpr, tpr, label='Model')
    plt.plot([0, 1], [0, 1], label='Random', linestyle='--')

    plt.xlabel('FPR')
    plt.ylabel('TPR')

    plt.legend()
    ```

# ROC AUC

## Material

- [Video](https://www.youtube.com/watch?v=hvIQPAwkVZo) (15:42)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-4-evaluation-metrics-for-classification)
- [Cuaderno de Jupyter del video](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/04-evaluation/notebook.ipynb)
- [Cuaderno de Jupyter adicional](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/chapter-03-churn-prediction/04-metrics.ipynb)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/06-auc.md)

## Notas

- x

# Cross-validation

## Material

- [Video](https://www.youtube.com/watch?v=BIIZaVtUbf4) (17:23)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-4-evaluation-metrics-for-classification)
- [Cuaderno de Jupyter del video](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/04-evaluation/notebook.ipynb)
- [Cuaderno de Jupyter adicional](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/chapter-03-churn-prediction/04-metrics.ipynb)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/07-cross-validation.md)

## Notas

- x

# Summary

## Material

- [Video](https://www.youtube.com/watch?v=-v8XEQ2AHvQ) (6:42)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-4-evaluation-metrics-for-classification)
- [Cuaderno de Jupyter del video](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/04-evaluation/notebook.ipynb)
- [Cuaderno de Jupyter adicional](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/chapter-03-churn-prediction/04-metrics.ipynb)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/08-summary.md)

## Notas

- x

# Explore more

## Material

- [Video]() ()
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-4-evaluation-metrics-for-classification)
- [Cuaderno de Jupyter del video](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/04-evaluation/notebook.ipynb)
- [Cuaderno de Jupyter adicional](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/chapter-03-churn-prediction/04-metrics.ipynb)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/09-explore-more.md)

## Notas

- x

# Homework

## Material

- [Video]() ()
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-4-evaluation-metrics-for-classification)
- [Cuaderno de Jupyter del video](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/04-evaluation/notebook.ipynb)
- [Cuaderno de Jupyter adicional](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/chapter-03-churn-prediction/04-metrics.ipynb)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/homework.md)

## Notas

- x
