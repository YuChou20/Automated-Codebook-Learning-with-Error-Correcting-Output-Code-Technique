FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu20.04

MAINTAINER yu

WORKDIR /home/

RUN set -xe \
    && apt-get update \
    && apt-get install python3-pip
RUN pip install --upgrade pip
RUN pip install tensorboard
RUN pip install matplotlib
RUN pip install torchvision