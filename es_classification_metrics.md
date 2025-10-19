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

- x

# Precision and recall

## Material

- [Video](https://www.youtube.com/watch?v=gRLP_mlglMM) (14:40)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-4-evaluation-metrics-for-classification)
- [Cuaderno de Jupyter del video](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/04-evaluation/notebook.ipynb)
- [Cuaderno de Jupyter adicional](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/chapter-03-churn-prediction/04-metrics.ipynb)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/04-precision-recall.md)

## Notas

- x

# ROC curves

## Material

- [Video 1](https://www.youtube.com/watch?v=dnBZLk53sQI) (34:45)
- [Video 2](https://www.youtube.com/watch?v=B5PATo1J6yw) (1:58)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-4-evaluation-metrics-for-classification)
- [Cuaderno de Jupyter del video](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/04-evaluation/notebook.ipynb)
- [Cuaderno de Jupyter adicional](https://github.com/alexeygrigorev/mlbookcamp-code/blob/master/chapter-03-churn-prediction/04-metrics.ipynb)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/05-roc.md)

## Notas

- x

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
