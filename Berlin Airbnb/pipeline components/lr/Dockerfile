FROM python:3.8
WORKDIR /lr
RUN pip install -U scikit-learn numpy pandas joblib
COPY linear.py /lr
ENTRYPOINT [ "python", "linear.py" ]
