FROM python:3.8
WORKDIR /knn
RUN pip install -U scikit-learn numpy pandas joblib
COPY knn.py /knn
ENTRYPOINT [ "python", "knn.py" ]
