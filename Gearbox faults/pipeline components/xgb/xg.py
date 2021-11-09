import argparse
def xgb(clean_data):
  import joblib
  import numpy as np
  import pandas as pd
  from sklearn import metrics
  from xgboost import XGBClassifer
  
  data = joblib.load(clean_data)
  X_train = data['X_train']
  y_train = data['Y_train']
  X_test = data['X_test']
  y_test = data['Y_test']  

  xgb_model = XGBClassifier(objective="binary:logistic", learning_rate = 0.08, random_state=42, n_estimators=600)
  xgb_model.fit(X_train, y_train)

  y_pred = xgb_model.predict(X_test)

  train = xgb_model.score(X_train, y_train)
  test = xgb_model.score(X_test, y_test)
  print('test accuracy:')
  print(test)
  print('train accuracy')
  print(train)

  #Classification Report
  report = metrics.classification_report(y_test, y_pred, output_dict=True)
  df_classification_report = pd.DataFrame(report).transpose()
  print(df_classification_report)

  xgb_metrics = {'train': train, 'test':test, 'report':df_classification_report}
  joblib.dump(xgb_metrics,'xgb_metrics')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--clean_data')
  args = parser.parse_args()
  xgb(args.clean_data)
