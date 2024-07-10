FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel

MAINTAINER yu

WORKDIR /home/

RUN pip install tensorboard
RUN pip install matplotlib

COPY ./ ./
