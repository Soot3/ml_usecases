FROM python:3.8
WORKDIR /train_keras
RUN pip install -U pandas joblib numpy tensorflow keras
COPY train.py /train_keras
ENTRYPOINT ["python", "train.py"]