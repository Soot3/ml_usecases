FROM python:3.8
WORKDIR /keras_model
RUN pip install -U numpy joblib tensorflow
COPY km.py /keras_model
ENTRYPOINT [ "python", "km.py" ]