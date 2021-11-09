FROM python:3.8
WORKDIR /xgb
RUN pip install -U scikit-learn numpy pandas joblib xgboost
COPY xg.py /xgb
ENTRYPOINT [ "python", "xg.py" ]
