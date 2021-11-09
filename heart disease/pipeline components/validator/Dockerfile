FROM python:3.8
WORKDIR /validator
RUN pip install -U joblib pandas tensorflow-data-validation ipython tensorflow 
COPY validator.py /validator
ENTRYPOINT ["python", "validator.py"]