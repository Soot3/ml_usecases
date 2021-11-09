FROM python:3.8
WORKDIR /preprocessing
RUN pip install -U scikit-learn numpy pandas
COPY preprocess.py /preprocessing
ENTRYPOINT [ "python", "preprocess.py" ]