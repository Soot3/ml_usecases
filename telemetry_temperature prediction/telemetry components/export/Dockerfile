FROM python:3.8
WORKDIR /export
RUN pip install -U joblib boto3 sklearn
COPY export.py /export
ENTRYPOINT [ "python", "export.py" ]