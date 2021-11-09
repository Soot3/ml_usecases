FROM python:3.8
WORKDIR /load_data
RUN pip install -U joblib pandas
COPY load_data.py /load_data
ENTRYPOINT ["python", "load_data.py"]
