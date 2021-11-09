FROM python:3.8
WORKDIR /statgen
RUN pip install -U joblib pandas tensorflow-data-validation ipython tensorflow 
COPY statg.py /statgen
ENTRYPOINT ["python", "statg.py"]