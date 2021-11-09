FROM python:3.8
WORKDIR /pca
RUN pip install -U scikit-learn numpy pandas
COPY pca.py /pca
ENTRYPOINT [ "python", "pca.py" ]