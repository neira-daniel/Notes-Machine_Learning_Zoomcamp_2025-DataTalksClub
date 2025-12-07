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

- Podemos utilizar redes neuronales ya entrenadas como base para nuestros modelos
- Keras nos da acceso directo a [docenas de estos modelos](https://keras.io/api/applications/)
- En esta ocasión utilizaremos [Xception](https://keras.io/api/applications/xception/):
    ```python
    from tensorflow.keras.applications.xception import Xception
    from tensorflow.keras.applications.xception import preprocess_input
    from tensorflow.keras.applications.xception import decode_predictions
    model = Xception(weights='imagenet', input_shape=(299, 299, 3))
    X = preprocess_input(np.expand_dims(img, axis=0))
    pred = model.predict(X)
    decode_predictions(pred)
    ```
    - Debemos empaquetar la imagen para que el objeto tenga las dimensiones que la red neuronal de Xception espera
    - Podemos hacerlo en este caso con `np.expand_dims(img, axis=0)`
    - En otras situaciones, cuando estemos trabajando con más de 1 imagen, probablemente necesitaremos algo como `np.array([img_1, img_2, img_3, . . .])`
    - Debemos también preprocesar el nuevo objeto usando `preprocess_input`, una función ad-hoc de Xception para transformar los valores de la imagen al espacio donde trabaja esta red neuronal en particular
    - Cuando hacemos una predicción, obtenemos un arreglo de `n` x 1000 con:
        - `n` la cantidad de imágenes que le pasamos a la red neuronal
        - 1000 el número de etiquetas posibles (la cantidad de categorías con las que fue entrenada esta red neuronal)
    - Cada uno de esos 1000 valores es la probabilidad de que la imagen que le pasamos a la red pertenezca a la categoría correspondiente
    - Obtenemos una representación más amigable de los resultados usando la función `decode_predictions`
        - Obtenemos:
            ```
            [[('n03595614', 'jersey', np.float32(0.68196267)),
            ('n02916936', 'bulletproof_vest', np.float32(0.03814007)),
            ('n04370456', 'sweatshirt', np.float32(0.03432477)),
            ('n03710637', 'maillot', np.float32(0.011354245)),
            ('n04525038', 'velvet', np.float32(0.0018453626))]]
            ```
        - En este caso, la red neuronal tiene un 68 % de certeza de que el objeto de la imagen es un "jersey" ([ejemplo](https://github.com/EliSchwartz/imagenet-sample-images/blob/master/n03595614_jersey.JPEG), supuestamente extraído del conjunto de datos original)
- Cuando la red neuronal pre-entrenada que utilicemos no cumpla con nuestros requerimientos prácticos, podremos utilizarla como punto de partida para un nuevo entrenamiento, cosa que veremos más adelante

# Convolutional neural networks

## Material

- [Video](https://www.youtube.com/watch?v=BN-fnYzbdc8) (25:51)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-8-neural-networks-and-deep-learning-250592316)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/08-deep-learning/04-conv-neural-nets.md)

## Notas

- "Convolutional neural networks": también conocidas por su sigla CNN
- Las utilizamos en este caso para clasificar imágenes de dimensiones $(H, W, D)$
    - $H$: alto
    - $W$: ancho
    - $D$: profundidad, siendo $D=3$ en el caso de imágenes con 3 canales de colores
- Este tipo de redes neuronales contiene dos tipos de capas de interés: capas "convolucionales" y capas "densas"
    - Las capas convolucionales actúan de forma local sobre el objeto de entrada y extraen de él los atributos de utilidad, aprendiéndolos a partir de los datos apoyándose en el valor de la función de pérdida que optimiza la red completa
    - Las capas densas actúan de manera global sobre el objeto de entrada y se encargan de _mapear_ la información abstracta extraída por las capas convolucionales a las etiquetas, realizando así la clasificación
- Entre las capas convolucionales y las capas densas ubicamos una etapa de "flattening"
    - Esta se encarga de "aplanar" un tensor (arreglo multidimensional) de entrada, convirtiéndolo en un arreglo unidimensional
    - Hacemos esto porque las capas densas solo pueden operar con arreglos unidimensionales
- Además de las capas convolucionales y las densas y de la etapa de _aplanamiento_, podemos también agregar capas de "pooling":
    - Estas se ubican entre capas convolucionales sucesivas
    - Su rol es disminuir las dimensiones de los datos que están fluyendo a través de la red neuronal para reducir su número de parámetros y así alivianar la carga computacional del entrenamiento
- Encontraremos en la literatura que las CNN incorporan otras capas adicionales a las mencionadas, pero no las cubriremos en esta introducción al tema
- Capas convolucionales
    - El algoritmo se inicializa con $n_i$ imágenes pequeñas por cada capa convolucional
    - A cada una de estas imágenes pequeñas se les llama "filtro"
    - Cada filtro comienza el entrenamiento conteniendo ruido (ningún patrón específico)
    - Durante el entrenamiento, cada uno de los filtros de una capa es convolucionado con los datos de entrada
        - En el caso de la primera capa, los datos de entrada son la imagen de referencia que queremos clasificar (cada uno de los pixeles a través de los 3 canales de colores)
        - En las capas siguientes, los datos de entrada serán aquellos que la capa anterior haya producido
    - Cada convolución produce un arreglo 2D al que se le llama "feature map"
        - Calculamos la convolución sobre el objeto de entrada usando todas sus dimensiones a la vez: alto, ancho y profundidad
        - Luego, pensar que iteramos sobre el eje de la profundidad y que calculamos convoluciones bidimensionales es un error
    - Cada uno de los "feature maps" de una capa son apilados, creando un tensor 3D
        - Dijimos antes que la imagen original es un tensor de dimensiones $(H, W, D)$
        - Cada "feature map" tendrá dimensiones $(H_i, W_i)$ con $H_i \neq H$ y $W_i \neq W$
            - Los valores exactos de $H_i$ y $W_i$ dependerán del tamaño de los filtros usados en cada capa $i$
        - El tensor resultante tendrá dimensiones $(H_i, W_i, n_i)$
    - La capa siguiente de la red (convolucional o no) recibe este objeto 3D y lo transforma y propaga hacia la salida
    - En la salida se calcula el valor de la función de pérdida
    - Este valor es utilizado, entre otras cosas, para aprender los patrones óptimos que deben contener los filtros
- Capas "densas"
    - Hablamos en plural porque es técnicamente posible incluir más de una de estas capas, pero lo mejor es utilizar solo 1
    - Estas capas no trabajan con tensores, sino que con arreglos unidimensionales
        - Es por eso que debemos aplanar el tensor de entrada a ellas
    - Se les denomina densas porque mapean cada uno de los elementos del arreglo de entrada a las etiquetas de salida
        - El arreglo que reciben es de largo $H_N + W_N + n_N$, con $X_N$ representando el valor de la dimensión $X$ al final de todas las capas convolucionales
        - Luego, si el número de etiquetas es $L$, la cantidad de pesos con los que trabaja la capa densa en la entrada es $L \cdot (H_N + W_N + n_N)$
        - Las demás capas no están tan densamente conectadas como esta
    - El clasificador mismo de la última capa densa es clásico:
        - Una sigmoide en el caso de clasificación binaria
        - Una [función softmax](https://en.wikipedia.org/wiki/Softmax_function) en el caso de clasificación multiclase
    - Es por lo anterior que esta capa solo trabaja con vectores unidimensionales en la entrada
- Podemos leer más acerca del funcionamiento de las redes neuronales convolucionales en [Convolutional Neural Networks (CNNs / ConvNets)](https://cs231n.github.io/convolutional-networks/) del curso CS231n

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
