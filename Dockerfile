FROM docker.io/ubuntu:latest
# update aptitude
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=US/Eastern
RUN apt-get update && apt-get -y --fix-missing upgrade

RUN apt-get -y install \
    graphviz python3-pip \
    python3-numpy \
    git

WORKDIR /app
COPY . .
RUN pip install .[all]
