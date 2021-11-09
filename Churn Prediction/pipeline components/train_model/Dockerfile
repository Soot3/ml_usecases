FROM python:3.8
WORKDIR /train_model
RUN pip install -U scikit-learn numpy pandas
COPY train_model.py /train_model
ENTRYPOINT [ "python", "train_model.py" ]