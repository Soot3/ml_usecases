FROM python:3.8
WORKDIR /randomforest
RUN pip install -U scikit-learn numpy pandas joblib
COPY rf.py /randomforest
ENTRYPOINT [ "python", "rf.py" ]