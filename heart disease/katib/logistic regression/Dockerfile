FROM python:3.8
RUN pip install pandas scikit-learn
COPY lrjobheart.py /
ENTRYPOINT ["python", "/lrjobheart.py", "--max_iter", "100", "--penalty", "l2", "--solver", "liblinear"]