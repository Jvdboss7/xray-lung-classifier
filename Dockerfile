FROM python:3.8-slim-buster
COPY . /xray_classifier
WORKDIR /xray_classifier
RUN pip install --upgrade pip
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install -r requirements.txt
RUN pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install -e .
CMD ["python","app.py"]