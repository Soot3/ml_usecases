FROM python:3.8
WORKDIR /catboost
RUN pip install -U catboost scikit-learn numpy pandas joblib
COPY cat.py /catboost
ENTRYPOINT [ "python", "cat.py" ]
