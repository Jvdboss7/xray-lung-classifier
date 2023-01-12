FROM python:3.8-slim-buster

COPY . /xray_classifier

WORKDIR /xray_classifier

RUN pip install --upgrade pip && apt-get update && apt-get install ffmpeg libsm6 libxext6 -y && pip install -r requirements.txt && pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu

CMD ["python","train.py"]