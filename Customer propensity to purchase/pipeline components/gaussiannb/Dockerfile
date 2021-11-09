FROM python:3.8
WORKDIR /gaussianNB
RUN pip install -U scikit-learn numpy pandas joblib
COPY gnb.py /gaussianNB
ENTRYPOINT [ "python", "gnb.py" ]