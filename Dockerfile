FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu20.04

MAINTAINER yu

WORKDIR /home/

RUN pip install tensorboard
RUN pip install matplotlib
RUN pip install torchvision