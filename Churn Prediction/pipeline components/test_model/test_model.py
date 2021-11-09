import argparse
def test_model(clean_data, fit_model):
  import numpy as np
  import pandas as pd
  import joblib
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.metrics import confusion_matrix,classification_report
  rfcla = joblib.load(fit_model)
  data = joblib.load(clean_data)
  Y_pred = rfcla.predict(data["test"]["X"])
  score = rfcla.score(data["test"]["X"], data["test"]["y"])
  rfcla_cm = confusion_matrix(data["test"]["y"], Y_pred)
  cr = classification_report(data["test"]["y"], Y_pred, target_names=['No', 'Yes'])
  data = {"score":score, "cm":rfcla_cm,"cr":cr}
  joblib.dump(data,'metrics')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--clean_data')
  parser.add_argument('--fit_model')
  args = parser.parse_args()
  test_model(args.clean_data,args.fit_model)