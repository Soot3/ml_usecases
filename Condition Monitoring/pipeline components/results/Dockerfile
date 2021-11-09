FROM python:3.8
WORKDIR /results
RUN pip install -U joblib pandas
COPY results.py /results
ENTRYPOINT [ "python", "results.py" ]