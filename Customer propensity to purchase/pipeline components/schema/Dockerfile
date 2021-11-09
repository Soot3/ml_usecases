FROM python:3.8
WORKDIR /schema
RUN pip install -U joblib pandas tensorflow-data-validation ipython tensorflow 
COPY schemag.py /schema
ENTRYPOINT ["python", "schemag.py"]