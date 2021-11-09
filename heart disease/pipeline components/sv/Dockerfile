FROM python:3.8
WORKDIR /sv
RUN pip install -U scikit-learn numpy pandas joblib
COPY sv.py /sv
ENTRYPOINT [ "python", "sv.py" ]
