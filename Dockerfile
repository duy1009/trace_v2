FROM python:3.8

RUN pip install --upgrade pip

MAINTAINER Duy<duytb1009@gmail.com>

WORKDIR /code

RUN pip install torch --index-url https://download.pytorch.org/whl/cu117

COPY ./requirements.txt ./requirements.txt 

RUN pip install -r ./requirements.txt

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install torchvision --index-url https://download.pytorch.org/whl/cu117
COPY . .

ENV PYTHONUNBUFFERED=True \
    PORT=8002 \
    HOST=0.0.0.0

EXPOSE 8002

