FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime

RUN apt-get update

COPY requirements.txt /

RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install -r /requirements.txt

WORKDIR /workspace/
