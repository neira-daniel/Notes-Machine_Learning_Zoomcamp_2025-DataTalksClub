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

- x

# Saving and loading the model

## Material

- [Video](https://www.youtube.com/watch?v=EJpqZ7OlwFU) (15:38)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-5-model-deployment)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/05-deployment/02-pickle.md)

## Notas

- x

# Web services: introduction to Flask

## Material

- [Video](https://www.youtube.com/watch?v=W7ubna1Rfv8) (6:40)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-5-model-deployment)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/05-deployment/03-flask-intro.md)

## Notas

- x

# Serving the churn model with Flask

## Material

- [Video](https://www.youtube.com/watch?v=Q7ZWPgPnRz8) (16:37)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-5-model-deployment)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/05-deployment/04-flask-deployment.md)

## Notas

- x

# Python virtual environment: Pipenv

## Material

- [Video](https://www.youtube.com/watch?v=BMXh8JGROHM) (15:43)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-5-model-deployment)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/05-deployment/05-pipenv.md)

## Notas

- x

# Environment management: Docker

## Material

- [Video](https://www.youtube.com/watch?v=wAtyYZ6zvAs) (19:09)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-5-model-deployment)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/05-deployment/06-docker.md)

## Notas

- x

# Workshop: Deploying ML Models with FastAPI and uv

## Material

- [Video](https://www.youtube.com/watch?v=jzGzw98Eikk) (1:41:23)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/05-deployment/workshop/README.md)

## Notas

- x

# Deployment to the cloud: AWS Elastic Beanstalk (optional)

## Material

- [Video](https://www.youtube.com/watch?v=HGPJ4ekhcLg) (16:35)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-5-model-deployment)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/05-deployment/07-aws-eb.md)

## Notas

- x

# Summary

## Material

- [Video](https://www.youtube.com/watch?v=sSAqYSk7Br4) (1:50)
- [Diapositivas](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-5-model-deployment)
- [Página de la lección en GitHub](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/05-deployment/08-summary.md)

## Notas

- x

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
