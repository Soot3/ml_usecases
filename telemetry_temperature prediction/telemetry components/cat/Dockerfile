FROM python:3.8
WORKDIR /cat
RUN pip install -U scikit-learn numpy pandas joblib catboost
COPY cat.py /cat
ENTRYPOINT [ "python", "cat.py" ]