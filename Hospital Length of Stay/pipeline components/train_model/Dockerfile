FROM python:3.8
WORKDIR /train_model
RUN pip install -U catboost numpy pandas joblib
COPY train.py /train_model
ENTRYPOINT [ "python", "train.py" ]