FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
ARG PYTHON_VERSION=3.10

# Set timezone, necessary
ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /root
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda-12.4
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
# setup pyenv
ENV HOME /root
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
ENV PYTHONPATH ${HOME}/src
ENV PIP_NO_CACHE_DIR=on

RUN apt-get update && apt-get install -y \
    software-properties-common \
    && apt-get -y upgrade \
    && apt-get clean

RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get -y update
RUN apt-get -y install --no-install-recommends \
            git \
            make \
            cmake \
            build-essential \
            python${PYTHON_VERSION}-dev \
            python3-pip \
            python${PYTHON_VERSION}-distutils \
            libssl-dev \
            zlib1g-dev \
            libbz2-dev \
            libreadline-dev \
            libsqlite3-dev \
            liblzma-dev \
            libffi-dev \
            curl \
            ffmpeg \
            libglm-dev \
            libopencv-dev \
            libegl1-mesa \
            libgles2-mesa-dev \
            xvfb \
            curl \
            gh

RUN pip install --upgrade setuptools
RUN pip install --upgrade pip
RUN pip install torch torchvision torchaudio  --index-url https://download.pytorch.org/whl/cu124
RUN apt-get remove -y python3-blinker