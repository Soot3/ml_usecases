FROM python:3.8
WORKDIR /keras_model
RUN pip install -U scikit-learn numpy pandas joblib tensorflow
COPY km.py /keras_model
ENTRYPOINT [ "python", "km.py" ]