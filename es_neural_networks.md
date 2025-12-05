---
language: es
title: "Module 7: Neural Networks and Deep Learning"
author: Daniel Neira
---
> Introduction to neural networks using TensorFlow and Keras, including CNNs and transfer learning.
>
> Topics:
>
> - Neural network fundamentals
> - PyTorch
> - TensorFlow & Keras
> - Convolutional Neural Networks
> - Transfer learning
> - Model optimization

# Preamble: note from this module's maintainer

Note: in the module we use TensorFlow+Keras. These videos  were recorded a while ago, and while they are still relevant, PyTorch became the to-go framework for training neural networks.

That's why we also re-recorded the content of this module with PyTorch. You can find the materials in the [pytorch/](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/master/08-deep-learning/pytorch) folder.

We don't go over the theory in the PyTorch part. For that, refer to the main module (the one that still uses Keras).

How to watch it:

- If you want to learn PyTorch only, watch the module content for the theory only and then follow along the content of the PyTorch part.
- If you want to learn both (and have time), first do the module and then the PyTorch part.

# Fashion classification

## Material

- [Video](https://www.youtube.com/watch?v=it1Lu7NmMpw) (6:56)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-8-neural-networks-and-deep-learning-250592316)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/08-deep-learning/01-fashion-classification.md)

## Notas

- Hasta ahora solo hemos hablado de cómo trabajar con datos tabulares
- En este módulo liberaremos esa restricción y nos concentraremos en la clasificación de imágenes
- En particular:
    - Ajustaremos un clasificador multi-clase usando redes neuronales
    - Funcionamiento: dada una imagen, clasificaremos su contenido usando un modelo que escoge 1 de 10 categorías posibles
- Nos concentraremos en los aspectos prácticos de trabajar con redes neuronales
    - Podemos consultar [CS231n: Deep Learning for Computer Vision](https://cs231n.github.io/) (antiguo nombre en 2021: Convolutional neural networks for visual recognition) para estudiar la teoría que hay detrás
- Aplicación de este clasificador:
    - Imaginaremos que contamos con un sitio de ventas en línea
    - Le ofreceremos este servicio a los vendedores para que puedan clasificar sus productos en el sitio de ventas
    - Llamaremos a este servicio "Fashion classification service"
- Datos que utilizaremos: "[Clothing dataset (subset)](https://github.com/alexeygrigorev/clothing-dataset-small)"
    - Versión completa: "[Clothing dataset (full, high resolution)](https://www.kaggle.com/datasets/agrigorev/clothing-dataset-full)"

# TensorFlow and Keras

## Material

- [Video](https://www.youtube.com/watch?v=R6o_CUmoN9Q) (9:52)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-8-neural-networks-and-deep-learning-250592316)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/08-deep-learning/02-tensorflow-keras.md)

## Notas

- TensorFlow: librería para entrenar modelos de *deep learning*
- Keras: construida sobre TensorFlow, nos ofrece una API más sencilla para trabajar con redes neuronales
- Instrucciones oficiales [para instalar TensorFlow](https://www.tensorflow.org/install):
    - Para trabajar con la CPU: `pip install tensorflow`
    - Para trabajar con la GPU: `pip install tensorflow[and-cuda]`
- Como alternativa, podemos usar el contenedor de Docker oficial de TensorFlow:
    ```bash
    docker pull tensorflow/tensorflow:latest  # Download latest stable image
    docker run -it -p 8888:8888 tensorflow/tensorflow:latest-jupyter  # Start Jupyter server
    ```
- Importar TensorFlow y Keras:
    ```python
    import tensorflow as tf
    from tensorflow import keras
    ```
    - Es posible que TensorFlow nos haga algunas advertencias al importarlo
        - Serán usualmente nada más que mensajes informativos
    - Importar Keras no debería imprimir ninguna advertencia en pantalla
- Ejemplo para cargar una imagen de entrenamiento:
    ```python
    from tensorflow.keras.preprocessing.image import load_img
    path = './clothing-dataset-small-master/train/t-shirt/'
    name = '5f0a3fa0-6a3d-4b68-b213-72766a643de7.jpg'
    fullname = path + name
    load_img(fullname)
    ```
- Cuando trabajemos con redes neuronales e imágenes, será habitual tener que escalar las imágenes al tamaño esperado por la red
    - Ejemplos: 299x299, 224x224, 150x150
    - Esto será válido cuando nos apoyemos en redes neuronales pre-entrenadas
        - En este caso tendremos que consultar la documentación de dicha red para saber qué tipos de objetos ella necesita para funcionar correctamente
    - Escalamos la imagen al cargarla usando el parámetro `target_size`:
    - Por ejemplo, para llevar la imagen a una de dimensiones 299x299:
        ```python
        load_img(fullname, target_size=(299, 299))
        ```
- Representación de la imagen en memoria:
    - Es un arreglo de valores numéricos
    - En el caso de una imagen en color: 3 canales
        - R: rojo
        - G: verde
        - B: azul
    - Este arreglo tendrá así 3 dimensiones
        - Alto
        - Ancho
        - Canal
    - En el caso de la imagen de 299 por 299 píxeles del ejemplo, `np.array(img).shape` retorna `(299, 299, 3)`
    - Podemos ver también que `np.array(img).dtype` retorna `uint8`:
        - Significa que codifica la información usando enteros sin signo de 8 bits
        - Estos son 2^8=256 niveles, los que van de 0 a 255 (no cubre los números negativos dado que el tipo es sin signo)


# Pre-trained convolutional neural networks

## Material

- [Video](https://www.youtube.com/watch?v=qGDXEz-cr6M) (12:34)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-8-neural-networks-and-deep-learning-250592316)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/08-deep-learning/03-pretrained-models.md)

## Notas

- x

# Convolutional neural networks

## Material

- [Video](https://www.youtube.com/watch?v=BN-fnYzbdc8) (25:51)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-8-neural-networks-and-deep-learning-250592316)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/08-deep-learning/04-conv-neural-nets.md)

## Notas

- x

# Transfer learning

## Material

- [Video](https://www.youtube.com/watch?v=WKHylqfNmq4) (35:36)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-8-neural-networks-and-deep-learning-250592316)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/08-deep-learning/05-transfer-learning.md)

## Notas

- x

# Adjusting the learning rate

## Material

- [Video](https://www.youtube.com/watch?v=2gPmRRGz0Hc) (10:35)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-8-neural-networks-and-deep-learning-250592316)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/08-deep-learning/06-learning-rate.md)

## Notas

- x

# Checkpointing

## Material

- [Video](https://www.youtube.com/watch?v=NRpGUx0o3Ps) (10:15)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-8-neural-networks-and-deep-learning-250592316)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/08-deep-learning/07-checkpointing.md)

## Notas

- x

# Adding more layers

## Material

- [Video](https://www.youtube.com/watch?v=bSRRrorvAZs) (11:43)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-8-neural-networks-and-deep-learning-250592316)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/08-deep-learning/08-more-layers.md)

## Notas

- x

# Regularization and dropout

## Material

- [Video](https://www.youtube.com/watch?v=74YmhVM6FTM) (16:23)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-8-neural-networks-and-deep-learning-250592316)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/08-deep-learning/09-dropout.md)

## Notas

- x

# Data augmentation

## Material

- [Video](https://www.youtube.com/watch?v=aoPfVsS3BDE) (27:06)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-8-neural-networks-and-deep-learning-250592316)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/08-deep-learning/10-augmentation.md)

## Notas

- x

# Training a larger model

## Material

- [Video](https://www.youtube.com/watch?v=_QpDGJwFjYA) (7:28)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-8-neural-networks-and-deep-learning-250592316)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/08-deep-learning/11-large-model.md)

## Notas

- x

# Using the model

## Material

- [Video](https://www.youtube.com/watch?v=cM1WHKae1wo) (7:44)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-8-neural-networks-and-deep-learning-250592316)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/08-deep-learning/12-using-model.md)

## Notas

- x

# Summary

## Material

- [Video](https://www.youtube.com/watch?v=mn0BcXJlRFM) (6:01)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/08-deep-learning/13-summary.md)

## Notas

- x

# Explore more

## Material

- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/08-deep-learning/14-explore-more.md)

## Notas

- x

# Homework

## Material

- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/08-deep-learning/homework.md)
- [Enunciado (2025)](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/cohorts/2025/08-deep-learning/homework.md)

## Notas

- x
