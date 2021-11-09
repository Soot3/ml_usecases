FROM python:3.8
WORKDIR /export
RUN pip install -U joblib requests google-cloud-storage google-auth
COPY export.py /export
ENTRYPOINT [ "python", "export.py" ]