FROM python:3.8
WORKDIR /random
RUN pip install -U scikit-learn numpy pandas joblib
COPY rf.py /random
ENTRYPOINT [ "python", "rf.py" ]
