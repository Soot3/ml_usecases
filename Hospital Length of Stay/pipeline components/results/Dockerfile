FROM python:3.8
WORKDIR /metrics
RUN pip install -U joblib numpy pandas
COPY metrics.py /metrics
ENTRYPOINT [ "python", "metrics.py" ]