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
