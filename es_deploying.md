---
language: es
title: "Module 5: Deploying Machine Learning Models"
author: Daniel Neira
---
> Turn your models into web services and deploy them with Docker and cloud platforms.
>
> Topics:
>
> - Model serialization with Pickle
> - FastAPI web services
> - Docker containerization
> - Cloud deployment

# Preamble: note from this module's maintainer

These materials are partly outdated, which is why we recorded a workshop that updates this module. You will find the materials in the [workshop/](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/05-deployment/workshop) folder.

The theory in the module is still relevant, which is why we suggest you to watch both the workshop and the module content. While watching the module content, focus on the theory part. The practical part is all covered in the workshop.

How to watch it:

- First, watch the units 5.1-5.6
  - From Session overview to Environment management with Docker
- Then watch the workshop "[Deploying ML Models with FastAPI and uv](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/05-deployment/workshop)" and do the practice
- Finally, do the homework

# Intro / Session overview

## Material

- [Video](https://www.youtube.com/watch?v=agIFak9A3m8) (4:24)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-5-model-deployment)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/05-deployment/01-intro.md)

## Notas

- Diagrama de la situación en torno a la que gira este módulo:
  ```diagram
  flowchart LR

  subgraph J[Jupyter Notebook]
      M1[model]
  end

  M1 --> F[model.bin]

  subgraph C[Churn Service]
      M2[model]
  end

  subgraph MK[Marketing Service]
      U[(Users)]
      E["Email Campaign (25%)"]
  end

  F --> C
  C --> MK
  MK --> C
  MK --> E
  E --> U
  ```
  - Contamos con nuestro clasificador ajustado en el [módulo anterior](./es_classification_metrics.md), el que se encuentra embebido en un cuaderno de Jupyter
  - Deseamos extraerlo y montarlo en un servidor web que lo pueda usar
  - La intención es que un tercero (por ejemplo, las personas del departamento de marketing) puedan enviar consultas al servicio web que está ejecutando el modelo
    - En este ejemplo, la gente de marketing le enviará descuentos promocionales a las personas que estarían por abandonar el servicio que está prestando la empresa
- En este módulo trabajaremos en cómo alojar un modelo en un servidor web y cómo hacerle consultas
- Abordaremos este problema por capas:
  ```
  +-------------------------------------------------------------+
  |                         Cloud                               |
  |                                                             |
  |   +-----------------------------------------------------+   |
  |   |                     Docker                          |   |
  |   |   (Environment — System Dependencies)               |   |
  |   |                                                     |   |
  |   |   +---------------------------------------------+   |   |
  |   |   |                 Pipenv                      |   |   |
  |   |   |   (Environment — Python Dependencies)       |   |   |
  |   |   |                                             |   |   |
  |   |   |   +-------------------------------------+   |   |   |
  |   |   |   |             Flask                   |   |   |   |
  |   |   |   |   (Web service)                     |   |   |   |
  |   |   |   |                                     |   |   |   |
  |   |   |   |   +-----------------------------+   |   |   |   |
  |   |   |   |   | CHURN PREDICTION MODEL      |   |   |   |   |
  |   |   |   |   +-----------------------------+   |   |   |   |
  |   |   |   +-------------------------------------+   |   |   |
  |   |   +---------------------------------------------+   |   |
  |   +-----------------------------------------------------+   |
  +-------------------------------------------------------------+
  ```
  - Núcleo: en el núcleo está el modelo de predicción de _churn_ que ajustamos en el módulo anterior
  - Servicio web: deseamos empaquetar el modelo en un servicio web
    - Flask en el caso del video
    - FastAPI en el caso de la tarea que tendremos que resolver
  - Entorno para dependencias de Python: tendremosq que crear un entorno de ejecución de Python que cuente con todas las dependencias de nuestro programa (las del servidor web y las de nuestro modelo)
    - Pipenv en el video
    - uv en la tarea
  - Entorno para dependencias del sistema: también tendremos que crear y configurar un entorno donde pueda correr el intérprete de Python
    - Utilizaremos Docker tanto en las clases como en la tarea
  - Nube: todo esto será alojado en la nube para que pueda ser accedido a través de internet

# Saving and loading the model

## Material

- [Video](https://www.youtube.com/watch?v=EJpqZ7OlwFU) (15:38)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-5-model-deployment)
- [Cuaderno de Jupyter del video](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/05-deployment/code/05-train-churn-model.ipynb)
- [Archivo final de entrenamiento: train.py](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/f60f128e85a0c7f3f9e4b1b0a1a7bd4ca2e76d72/05-deployment/code/train.py)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/05-deployment/02-pickle.md)

## Notas

- Código del modelo de clases:
  ```python
  import pandas as pd
  import numpy as np

  from sklearn.model_selection import train_test_split
  from sklearn.model_selection import KFold

  from sklearn.feature_extraction import DictVectorizer
  from sklearn.linear_model import LogisticRegression
  from sklearn.metrics import roc_auc_score

  df = pd.read_csv('data-week-3.csv')

  df.columns = df.columns.str.lower().str.replace(' ', '_')

  categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

  for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')

  df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
  df.totalcharges = df.totalcharges.fillna(0)

  df.churn = (df.churn == 'yes').astype(int)

  df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

  numerical = ['tenure', 'monthlycharges', 'totalcharges']

  categorical = [
      'gender',
      'seniorcitizen',
      'partner',
      'dependents',
      'phoneservice',
      'multiplelines',
      'internetservice',
      'onlinesecurity',
      'onlinebackup',
      'deviceprotection',
      'techsupport',
      'streamingtv',
      'streamingmovies',
      'contract',
      'paperlessbilling',
      'paymentmethod',
  ]

  def train(df_train, y_train, C=1.0):
    dicts = df_train[categorical + numerical].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y_train)

    return dv, model

  def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred

  C = 1.0

  dv, model = train(df_full_train, df_full_train.churn.values, C=C)
  y_pred = predict(df_test, dv, model)
  ```
- La función `predict` recibe `dv` y `model`
  - Tenemos que exportar esas variables
  - Luego cargaremos esas variables en el servidor web que estará corriendo el modelo
- Usaremos [pickle](https://docs.python.org/3/library/pickle.html) para serializar el contenido de `dv` y `model` y guardarlas en un archivo en disco:
  ```python
  # pickle es parte de la librería estándar de Python
  import pickle

  output_file = f'model_C={C}.bin'
  with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)
  ```
- Ejecutaremos este código en la máquina de destino para cargar el modelo y dejarlo listo para funcionar:
  ```python
  import pickle

  input_file = 'model_C=1.0.bin'
  with open(input_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)
  ```
  - Debemos tener sklearn instalado para que Python sepa qué es lo que está cargando en memoria cuando está cargando el contenido de `input_file`
- Podemos probar que el modelo funciona haciendo una predicción:
  ```python
  customer = {
      'gender': 'female',
      'seniorcitizen': 0,
      'partner': 'yes',
      'dependents': 'no',
      'phoneservice': 'no',
      'multiplelines': 'no_phone_service',
      'internetservice': 'dsl',
      'onlinesecurity': 'no',
      'onlinebackup': 'yes',
      'deviceprotection': 'no',
      'techsupport': 'no',
      'streamingtv': 'no',
      'streamingmovies': 'no',
      'contract': 'month-to-month',
      'paperlessbilling': 'yes',
      'paymentmethod': 'electronic_check',
      'tenure': 1,
      'monthlycharges': 29.85,
      'totalcharges': 29.85
  }

  X = dv.transform([customer])
  model.predict_proba(X)[0, 1]
  ```
- En el video muestran cómo exportar el contenido de un cuaderno de Jupyter a un script de Python
  - En resultado, luego de algunas modificaciones cosméticas, es [train.py](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/f60f128e85a0c7f3f9e4b1b0a1a7bd4ca2e76d72/05-deployment/code/train.py)

# Web services: introduction to Flask

## Material

- [Video](https://www.youtube.com/watch?v=W7ubna1Rfv8) (6:40)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-5-model-deployment)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/05-deployment/03-flask-intro.md)

## Notas

- Un servicio web es un servicio con el que nos comunicamos a través de una red
- Usaremos Flask para crear un servicio web
- Queremos que nuestro servicio web responda a una consulta GET a la dirección `localhost:9696/ping` con el mensaje `PONG`
- Código:
  ```python
  # /// script
  # requires-python = ">=3.13"
  # dependencies = [
  #     "flask",
  # ]
  # ///

  from flask import Flask

  # give an identity to your web service
  app = Flask('ping')

  # use decorator to add Flask's functionality to our function
  @app.route('/ping', methods=['GET'])
  def ping():
      return 'PONG'

  if __name__ == '__main__':
      # run the code in local machine with the debugging mode true and port 9696
      app.run(host='127.0.0.1', port=9696, debug=True)
  ```
  - Podemos verificar que el servidor está funcionando abriendo `http://localhost:9696/ping` en un navegador que se encuentre en la misma máquina donde estamos ejecutando el script
    - También podemos usar `curl`: `curl localhost:9696/ping`
  - (_Bonus_) Dado que las dependencias están declaradas [de forma estándar en el código](https://packaging.python.org/en/latest/specifications/inline-script-metadata/#inline-script-metadata), podemos ejecutarlo directamente con uv: `uv run example.py` (asumiendo que `example.py` es el nombre del script)
    - uv instalará las dependencias y creará un entorno virtual donde ejecutar el script de forma automática
    - Notas:
      - El [código original](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/f60f128e85a0c7f3f9e4b1b0a1a7bd4ca2e76d72/05-deployment/03-flask-intro.md) no incluye la directiva de dependencias
      - Es fácil agregarlas a un script existente [usando uv](https://docs.astral.sh/uv/guides/scripts/#declaring-script-dependencies)
      - En este caso: `uv add --script example.py 'Flask'`

# Serving the churn model with Flask

## Material

- [Video](https://www.youtube.com/watch?v=Q7ZWPgPnRz8) (16:37)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-5-model-deployment)
- [Archivo `predict.py`](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/a4519a916e1fdc050f39d1d51ef6945a05332cee/05-deployment/code/predict.py)
- [Archivo `predict-test.py`](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/a4519a916e1fdc050f39d1d51ef6945a05332cee/05-deployment/code/predict-test.py)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/05-deployment/04-flask-deployment.md)

## Notas

- En esta lección se mostró cómo montar un _endpoint_ llamado `predict` que responde a consultas a través de la red
- Tal como antes, se utilizó Flask para implementar la lógica
  - Este es un servidor de desarrollo
- Luego se mostró cómo levantar un servidor de producción compatible con [WSGI](https://en.wikipedia.org/wiki/Web_Server_Gateway_Interface) (estándar de comunicación para servidor escritos en Python)
  - Se mencionaron dos alternativas:
    - Linux: `gunicorn`:
      - `gunicorn --bind 127.0.0.1:9696 predict:app`
    - Windows: `waitress`
      - `waitress-serve --listen 127.0.0.1:9696 predict:app`
    - En ambos casos, `predict:app` dice que el servidor debe trabajar con el objeto `app` que se encuentra en `predict.py`
- Detalles sobre el código:
  - `predict.py`:
    ```python
    import pickle

    from flask import Flask
    from flask import request
    from flask import jsonify

    model_file = 'model_C=1.0.bin'

    with open(model_file, 'rb') as f_in:
        dv, model = pickle.load(f_in)

    app = Flask('churn')

    @app.route('/predict', methods=['POST'])
    def predict():
        customer = request.get_json()

        X = dv.transform([customer])
        y_pred = model.predict_proba(X)[0, 1]
        churn = y_pred >= 0.5

        result = {
            'churn_probability': float(y_pred),
            'churn': bool(churn)
        }

        return jsonify(result)

    if __name__ == "__main__":
        app.run(debug=True, host='127.0.0.1', port=9696)
    ```
    - `@app.route('/predict', methods=['POST'])`
      - El cliente está enviando información al servidor (la información de la persona para la que quiere hacer una predicción), de modo que usamos `POST` en lugar de `GET`
    - `from flask import request` y `customer = request.get_json()`
      - Declaramos que la información que recibe el servidor debe estar en formato JSON
      - En este caso usamos un diccionario de Python que es compatible con ese formato, de modo que no hicimos ninguna transformación, pero no siempre podemos suponer que esto funcionará
    - `float(y_pred)` y `bool(churn)`
      - Los tipos de `y_pred` y `churn` son tipos de NumPy
      - Debemos transformalos a tipos estándar de Python para retornarlos al cliente
    - `from flask import jsonify` y `return jsonify(result)`
      - El servidor retorna el resultado en formato JSON
    - `app.run(debug=True, host='127.0.0.1', port=9696)`
      - El servidor correrá en modo `debug` cada vez que lo invoquemos como script
        - Esto es, cada vez que ejecutemos `python3 predict.py`
      - Cuando corramos el servidor usando `gunicorn`, `waitress` o similar, no estaremos ejecutando `predict.py` como script
        - De modo que no correrá en modo _debug_

# Python virtual environment: Pipenv

## Material

- [Video](https://www.youtube.com/watch?v=BMXh8JGROHM) (15:43)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-5-model-deployment)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/05-deployment/05-pipenv.md)

## Notas

- [`pipenv`](https://pipenv.pypa.io/en/latest/) nos permite administrar entornos virtuales de Python
  - Cada entorno virtual es independiente de los demás
  - Podemos entonces instalar paquetes en un entorno virtual sin que estos interfieran con los paquetes de otros entornos
  - Esto facilita trabajar en proyectos con distintos requerimientos
  - Ejecutaremos cada proyecto dentro de su propio entorno virtual
  - De esta manera nos ahorramos problemas de incompatibilidad entre paquetes, además de que nos permite instalar distintas versiones de un mismo paquete por proyecto
- Además de lo anterior, `pipenv` también nos permite administrar `pip` y `Pipfile` (la referencia de qué es lo que está instalado en cada entorno virtual)
  - Estos archivos serán fundamentales para poder recrear un entorno virtual en otra máquina
  - De esta forma podremos, por ejemplo, instalar las mismas versiones de los paquetes de Python usados durante el desarrollo del programa en el servidor web que alojará dicha aplicación
- `pipenv` es mantenido por [la misma organización](https://packaging.python.org/en/latest/key_projects/#pypa-projects) que mantiene `pip`, `pipx`, `setuptools`, `wheel`, etc.
- Instalación: `pip install --user pipenv`
  - Ejecutamos `pipenv --version` para verificar que el programa se instaló correctamente
  - En caso de existir un problema, consultar la [documentación](https://pipenv.pypa.io/en/latest/installation.html)
- Uso:
  - Inicializar un proyecto en el directorio actual: `pipenv install`
  - Fijar la versión de Python: `pipenv --python VERSION`
  - Instalar paquetes en el proyecto: `pipenv install PACKAGE1 PACKAGE2 . . .`
    - Podemos especificar las versiones
    - Por ejemplo, `PACKAGE1==0.1.1` o `PACKAGE2>=1.0`
  - Instalar los paquetes especificados en `Pipfile.lock`: `pipenv sync`
  - Activar el entorno virtual: `pipenv shell` (`exit` para salir)
  - Ejecutar un programa o script dentro del entorno virtual sin activarlo: `pipenv run COMANDO`
    - Por ejemplo, `pipenv run gunicorn --bind 127.0.0.1:9696 predict:app`
  - Entre [otros comandos](https://pipenv.pypa.io/en/latest/quick_start.html)
- A esta altura ya podemos trabajar en distintos proyectos de Python sin que se produzcan conflictos entre ellos
  - Podemos instalar las dependencias de Python de cada proyecto en entornos virtuales independientes
- Pero todavía nos falta ver cómo manejar dependencias del sistema
  - Es decir, programas propios del sistema operativo
  - Para administrar entornos a nivel de sistema operativo, usaremos Docker (próxima sección)

# Environment management: Docker

## Material

- [Video](https://www.youtube.com/watch?v=wAtyYZ6zvAs) (19:09)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-5-model-deployment)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/05-deployment/06-docker.md)

## Notas

- Usaremos Docker para crear [contenedores](https://www.docker.com/resources/what-container/)
- Podremos alojar nuestras aplicaciones en contenedores
- Los contenedores serán parecidos a máquinas virtuales
  - Las máquinas virtuales proveen una abstracción de hardware
  - Docker no virtualiza hardware, sino sistemas operativos
    - Utiliza un mismo kernel para los distintos contenedores manteniendo independencia entre ellos
    - Podremos montar múltiples contenedores en una misma máquina sin crear conflictos
    - Los contenedores serán más livianos y eficientes que montar máquinas virtuales dedicadas
- Podremos crear contenedores basados en sistemas operativos
  - Por ejemplo, un contenedor que use Ubuntu y otro que use Alpine o Amazon Linux
  - Y en cada contenedor instalaremos solo las dependencias que sean necesarias para correr la aplicación que planeamos alojar allí
- Cada contenedor será independiente de los demás
- Creamos un contenedor para nuestro clasificador con el siguiente código:
  ```dockerfile
  # `python:3.8.12-slim`: la imagen sobre la que queremos crear el contenedor
  # en este caso se trata de una imagen de Python 3.8.12 corriendo en una versión de Debian que solo contiene los paquetes necesarios para ejecutar Python
  # esto es suficiente cuando queremos ejecutar aplicaciones de Python simples
  FROM python:3.8.12-slim

  # usaremos `RUN` para especificar las instrucciones que deseamos ejecutar en el intérprete de comandos que se encuentra dentro del contenedor (en este caso, Bash en Debian)
  RUN pip install pipenv

  # `WORKDIR DIR` es, básicamente, `mkdir DIR; cd DIR`
  WORKDIR /app

  # necesitaremos los Pipfiles en el contenedor para poder recrear en entorno virtual que usamos durante el desarrollo del clasificador
  COPY ["Pipfile", "Pipfile.lock", "./"]

  # instalamos las dependencias de nuestra aplicación de Python
  RUN pipenv install --system --deploy

  # también necesitaremos almacenar el script de predicción y el modelo en el contenedor
  COPY ["predict.py", "model_C=1.0.bin", "./"]

  # puerto del contenedor que deseamos exponer al exterior
  EXPOSE 9696

  # instrucción que deseamos ejecutar al montar el contenedor
  # en este caso, levantamos el servidor que corre nuestra aplicación de predicción
  ENTRYPOINT ["gunicorn", "--bind=127.0.0.1:9696", "predict:app"]
  ```
  - La versión `slim` podría no ser siempre la adecuada ([más información](https://hub.docker.com/_/python/#image-variants))
  - `RUN pipenv install --system --deploy` es la forma de instalar los paquetes en un contenedor de Docker [recomendada por los desarrolladores de pipenv](https://pipenv.pypa.io/en/latest/installation.html#docker-installation)
- Creamos la imagen con `docker build -t zoomcamp-test .`
  - `-t` nos permite darle un nombre al contenedor
  - `.` le indica a Docker que debe crear la imagen a partir del Dockerfile que se encuentra en el directorio actual
- Montamos la imagen con `docker run -it -rm -p 9696:9696 zoomcamp-test`
  - `it` le indica a Docker que deseamos abrir una sesión interactiva en el contenedor (TTY)
  - `rm` borrará el contenedor una vez que deje de correr
  - `-p 9696:9696` mapea el puerto 9696 del _host_ al puerto 9696 del contenedor
    - Sigue la convención `HOST_PORT:CONTAINER_PORT`
- Podemos verificar que el contenedor está funcionando enviando una consulta ejecutando el código contenido en [`predict-test.py`](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/94329a0489ea7f934e279ef3334ef98a5988c65e/05-deployment/code/predict-test.py):
  ```python
  #!/usr/bin/env python
  # coding: utf-8

  import requests

  url = 'http://localhost:9696/predict'

  customer_id = 'xyz-123'
  customer = {
      "gender": "female",
      "seniorcitizen": 0,
      "partner": "yes",
      "dependents": "no",
      "phoneservice": "no",
      "multiplelines": "no_phone_service",
      "internetservice": "dsl",
      "onlinesecurity": "no",
      "onlinebackup": "yes",
      "deviceprotection": "no",
      "techsupport": "no",
      "streamingtv": "no",
      "streamingmovies": "no",
      "contract": "month-to-month",
      "paperlessbilling": "yes",
      "paymentmethod": "electronic_check",
      "tenure": 24,
      "monthlycharges": 29.85,
      "totalcharges": (24 * 29.85)
  }

  response = requests.post(url, json=customer).json()
  print(response)

  if response['churn'] == True:
      print('sending promo email to %s' % customer_id)
  else:
      print('not sending promo email to %s' % customer_id)
  ```

# Deployment to the cloud: AWS Elastic Beanstalk (optional)

## Material

- [Video](https://www.youtube.com/watch?v=HGPJ4ekhcLg) (16:35)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-5-model-deployment)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/05-deployment/07-aws-eb.md)

## Notas

- Razones del instructor para cubrir AWS Elastic Beanstalk
  - Tiene experiencia con AWS
  - AWS es, probablemente, el proveedor de servicios en la nube más popular
  - Y, en su opinión, Elastic Beanstalk es "relativamente sencillo"
    - "Basta con unos cuantos comandos para desplegar algo en la nube"
- Alternativas mencionadas en el video:
  - Heroku
  - Google Cloud
  - Azure
  - Python Anywhere
- Instrucciones para [crear una cuenta AWS](https://mlbookcamp.com/article/aws)
- Forma de imaginar la arquitectura:
  - Contamos con una instancia de AWS EB en la nube
  - Almacenamos nuestro contenedor en su interior
  - Los clientes (en el sentido del modelo cliente-servidor) dirigen sus consultas a AWS EB
  - AWS EB reenvía (_forwards_) la consulta al contenedor a través del puerto configurado
  - El programa que está corriendo dentro del contenedor responde y envía la respuesta de vuelta a AWS EB
  - AWS EB reenvía la respuesta al cliente
- Balance de carga
  - AWS EB agrega o disminuye el número de instancias activas de nuestro contenedor dependiendo de la demanda
    - _Scale up_: agrega instancias para hacer frente a un tráfico alto
    - _Scale down_: disminuye el número de instancias cuando cae el tráfico
  - Se encarga también de distribuir las consultas entre las distintas imágenes activas para que el desempeño de nuestro servicio no decaiga
  - (_Bonus_) Nada de esto saldrá gratis, claramente, pero podría ser deseable si es que estamos prestando un servicio que no puede quedar fuera de línea y que debe tener un alto desempeño
- Asumiendo que ya contratamos AWS EB, ahora podremos usar la utilidad `eb` en nuestro computador para interactuar con el servicio
  - Instalación: `pipenv install awsebcli --dev && pipenv shell`
    - Usamos `--dev` para indicar que es una dependencia de desarrollo
    - No será instalada en producción
    - Activamos el entorno para poder usar `eb`, el ejecutable de AWS EB
  - Registro de una aplicación con EB: `eb init -p docker -r eu-north-1 churn-serving`
    - `-p docker`: la plataforma para la aplicación es Docker
    - `-r eu-north-1`: la región donde se debe ubicar el servidor es `eu-north-1`
    - `churn-serving`: el nombre de la aplicación a registrar con EB
  - Verificación de que todo salió bien:
    - Primero corroboramos que se creó el archivo de configuración `./elasticbeanstalk/config.yml`
    - Y luego probamos el servicio de forma local: `eb local run --port 9696`
    - Se montará el contenedor y comenzará a funcionar
    - Consultamos el servicio usando el script `predict-test.py`, debiendo recibir una respuesta de la imagen que está corriendo de forma local
  - Despliegue del servicio: `eb create churn-service-env`
    - `churn-service-env`: el nombre del entorno que queremos crear (es diferente del nombre de la aplicación)
    - Este procedimiento puede tomar varios minutos en completarse
    - Al finalizar, imprimirá entre los logs la dirección web donde nuestra aplicación se encuentra activa y escuchando ("INFO Application available at...")
      - Esta dirección estará disponible también en la consola de AWS de nuestra cuenta
      - Podemos acceder a esa consola a través de la página web de AWS
  - Terminar el servicio: `eb terminate churn-service-env`
    - `churn-service-env`: naturalmente, el nombre del entorno que deseamos dejar fuera de línea
- Consideraciones importantes:
  - La aplicación que desplegamos con AWS EB estará escuchando en el puerto 80 y no en el puerto que especificamos en nuestra aplicación (en el caso del ejemplo, el 9696)
    - AWS EB se ocupará de reenviar las consultas desde el puerto 80 del servidor al puerto 9696 del contenedor
    - Como el puerto 80 es el puerto típico de los servicios web, no hace falta explicitarlo en la URL (la dirección web `example.com:80` es equivalente a `example.com`)
  - Durante la configuración del servicio no tomamos medidas de seguridad
    - Si seguimos los pasos de esta guía tal cual, nuestro servicio estará abierto a toda internet sin filtro
    - Cualquier persona que conozca la URL donde está corriendo nuestro programa de Python será capaz de enviarle consultas
    - Esto se puede prestar para abusos, donde el eventual exceso de conexiones o consultas al servicio se verán reflejados en nuestra boleta de AWS EB
    - También supone un riesgo de filtración de información privada
      - No es el caso de esta aplicación de Python, pero en otros casos podría quedar, por ejemplo, una base de datos abierta a internet, cuya información podría ser extraída por terceros
    - En la práctica buscaremos limitar el acceso a la aplicación a un grupo determinado de personas (esto está fuera del alcance del curso)

# Workshop: Deploying ML Models with FastAPI and uv

## Material

- [Video](https://www.youtube.com/watch?v=jzGzw98Eikk) (1:41:23)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/05-deployment/workshop/README.md)

## Notas

- x

# Summary

## Material

- [Video](https://www.youtube.com/watch?v=sSAqYSk7Br4) (1:50)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-5-model-deployment)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/05-deployment/08-summary.md)

## Notas

- N/A

# Explore more

## Material

- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/05-deployment/09-explore-more.md)

## Notas

- Probar otros _frameworks_ para crear servicios web
  - Flask y FastAPI no son los únicos
- Probar alternativas para la administración de entornos de Python
  - [venv](https://docs.python.org/3/library/venv.html)
  - [uv](https://docs.astral.sh/uv/)
  - [pixi](https://prefix.dev/tools/pixi)
  - Anaconda ([conda](https://docs.conda.io/en/latest/), [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main) o [miniforge](https://github.com/conda-forge/miniforge), la alternativa que prefiero entre estas 3)
  - [Poetry](https://python-poetry.org/)
- Probar otros servicios de alojamiento de servicios web:
  - AWS
  - GCP
  - Azure
  - Heroku
  - Python Anywhere
  - Etc.

# Homework

## Material

- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/05-deployment/homework.md)

## Notas

- x
