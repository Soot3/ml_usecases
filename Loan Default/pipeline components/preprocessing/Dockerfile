FROM python:3.8
WORKDIR /preprocessing
RUN pip install scikit-learn numpy pandas regex imbalanced-learn -U
COPY preprocess.py /preprocessing
ENTRYPOINT [ "python", "preprocess.py" ]