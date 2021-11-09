import argparse
def cb(clean_data):
  import joblib
  import numpy as np
  import pandas as pd
  from sklearn import metrics
  from catboost import CatBoostClassifier

  data = joblib.load(clean_data)
  X_train = data['X_train']
  y_train = data['Y_train']
  X_test = data['X_test']
  y_test = data['Y_test']

  cb_model = CatBoostClassifier(verbose=0)

  cb_model.fit(X_train, y_train)

  y_pred = cb_model.predict(X_test)

  # Test score
  test = cb_model.score(X_test, y_test)
  train = cb_model.score(X_train, y_train)
  print('test accuracy:')
  print(test)
  print('train accuracy:')
  print(train)

  #Classification Report
  report = metrics.classification_report(y_test, y_pred, output_dict=True)
  df_classification_report = pd.DataFrame(report).transpose()
  print(df_classification_report)

  cb_metrics = {'train':train, 'test':test, 'report':df_classification_report,'model':cb_model}
  joblib.dump(cb_metrics,'cb_metrics')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--clean_data')
  args = parser.parse_args()
  cb(args.clean_data)