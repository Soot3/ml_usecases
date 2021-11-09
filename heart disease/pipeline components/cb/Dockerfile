FROM python:3.8
WORKDIR /cb
RUN pip install -U scikit-learn numpy pandas joblib catboost
COPY cb.py /cb
ENTRYPOINT [ "python", "cb.py" ]