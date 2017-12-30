FROM jupyter/datascience-notebook

RUN mkdir /home/jovyan/work/data

ADD . /home/jovyan/work

RUN pip install quandl plotly

ENV QUANDL_API_KEY $QUANDL_API_KEY