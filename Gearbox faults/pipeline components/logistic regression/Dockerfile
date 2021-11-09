FROM python:3.8
WORKDIR /lr
RUN pip install -U scikit-learn numpy pandas joblib
COPY logistic.py /lr
ENTRYPOINT [ "python", "logistic.py" ]