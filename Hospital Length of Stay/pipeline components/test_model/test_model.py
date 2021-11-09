import argparse
def test_model(clean_data, fit_model):
  from catboost import CatBoostClassifier
  from sklearn.metrics import roc_auc_score,accuracy_score,precision_score,recall_score,f1_score
  from sklearn.metrics import confusion_matrix, roc_curve, auc
  import numpy as np
  import pandas as pd
  import joblib  
  model = joblib.load(fit_model)
  #model.load_model(fit_model)
  data = joblib.load(clean_data)

  x_test = data['X_test']
  y_test = data['Y_test']

  pred = model.predict(x_test)
  #Model accuracy
  acc = accuracy_score(y_test, pred)*100

  precision = precision_score(y_test, pred,average='weighted')*100
  recall = recall_score(y_test, pred,average='weighted')*100
  f1 = f1_score(y_test, pred,average='weighted')*100


  # Confusion matrix
  confusion = pd.DataFrame(confusion_matrix(y_test, pred))

  data = {'accuracy':acc, 'precision':precision,'recall':recall,'f1':f1,'confusion':confusion}
  joblib.dump(data,'metrics')
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--clean_data')
  parser.add_argument('--fit_model')
  args = parser.parse_args()
  test_model(args.clean_data,args.fit_model)