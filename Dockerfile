FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y --no-install-recommends \
	git \
	curl \
	wget \
    less \
 && rm -rf /var/lib/apt/lists/*

# http://bugs.python.org/issue19846
ENV LANG C.UTF-8
# https://github.com/docker-library/python/issues/147
ENV PYTHONIOENCODING UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
	python3.6 \
	python3.6-dev \
	python3-pip \
	python3-setuptools \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    libxrender1 \
 && rm -rf /var/lib/apt/lists/*

RUN apt update && pip3 install --upgrade pip

COPY requirements.txt ./

RUN pip3 install --no-cache-dir -r requirements.txt; rm requirements.txt

WORKDIR /workspace