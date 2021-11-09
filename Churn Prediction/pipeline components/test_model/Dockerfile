FROM python:3.8
WORKDIR /test_model
RUN pip install -U scikit-learn numpy pandas
COPY test_model.py /test_model
ENTRYPOINT [ "python", "test_model.py" ]