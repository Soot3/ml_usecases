FROM python:3.8
WORKDIR /logistic
RUN pip install -U scikit-learn numpy pandas joblib
COPY logistic.py /logistic
ENTRYPOINT [ "python", "logistic.py" ]