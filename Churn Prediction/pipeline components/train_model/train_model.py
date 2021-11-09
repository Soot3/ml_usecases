import argparse
def train_model(clean_data):
  import joblib
  import numpy as np
  import pandas as pd  
  from sklearn.ensemble import RandomForestClassifier
  data = joblib.load(clean_data)
  rfcla = RandomForestClassifier(n_estimators=78,random_state=9,n_jobs=27,max_features= 'auto')
  rfcla.fit(data["train"]["X"], data["train"]["y"])
  joblib.dump(rfcla,'fit_model')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--clean_data')
  args = parser.parse_args()
  train_model(args.clean_data)