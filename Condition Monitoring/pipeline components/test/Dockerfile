FROM python:3.8
WORKDIR /test
RUN pip install -U pandas joblib numpy tensorflow keras 
COPY test.py /test
ENTRYPOINT ["python", "test.py"]