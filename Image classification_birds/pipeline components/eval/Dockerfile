FROM python:3.8
WORKDIR /eval
RUN pip install -U joblib tensorflow 
COPY eval.py /eval
ENTRYPOINT [ "python", "eval.py" ]