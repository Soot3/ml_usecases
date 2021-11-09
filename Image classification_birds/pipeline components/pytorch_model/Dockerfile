FROM python:3.8
WORKDIR /pyt_model
RUN pip install -U torch joblib torchvision tqdm
COPY pyt.py /pyt_model
ENTRYPOINT [ "python", "pyt.py" ]