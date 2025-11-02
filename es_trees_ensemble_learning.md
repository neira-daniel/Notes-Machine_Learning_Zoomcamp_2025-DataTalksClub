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

- x

# Decision trees

## Material

- [Video](https://www.youtube.com/watch?v=YGiQvFbSIg8) (17:19)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-6-decision-trees-and-ensemble-learning)
- [Cuaderno de Jupyter del video](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/06-trees/notebook.ipynb)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/06-trees/03-decision-trees.md)

## Notas

- x

# Decision tree learning algorithm

## Material

- [Video](https://www.youtube.com/watch?v=XODz6LwKY7g) (29:13)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-6-decision-trees-and-ensemble-learning)
- [Cuaderno de Jupyter del video](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/06-trees/notebook.ipynb)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/06-trees/04-decision-tree-learning.md)

## Notas

- x

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
