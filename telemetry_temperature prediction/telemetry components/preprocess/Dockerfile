FROM python:3.8
WORKDIR /preprocess
RUN pip install -U scikit-learn numpy pandas joblib datetime
COPY preprocess.py /preprocess
ENTRYPOINT [ "python", "preprocess.py" ]