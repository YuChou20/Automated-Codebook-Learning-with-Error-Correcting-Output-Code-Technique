FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu20.04

MAINTAINER yu

WORKDIR /home/

RUN set -xe \
    && apt-get update \
    && apt-get install python3-pip -y
RUN pip install --upgrade pip
RUN pip install tensorboard
RUN pip install matplotlib
RUN pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1

COPY ./ ./
