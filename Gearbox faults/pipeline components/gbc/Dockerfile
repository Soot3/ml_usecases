FROM python:3.8
WORKDIR /gbc
RUN pip install -U scikit-learn numpy pandas joblib
COPY gradient.py /gbc
ENTRYPOINT [ "python", "gradient.py" ]
