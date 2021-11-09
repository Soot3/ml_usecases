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
  cat = CatBoostClassifier(n_estimators = 50,random_state = 2020, learning_rate = 0.08, bagging_temperature=0.3, verbose = 0)

  cat.fit(X_train,y_train)

  y_pred = cat.predict(X_test)
  
  train = cat.score(X_train, y_train)
  test = cat.score(X_test, y_test)
  print('test accuracy:')
  print(test)
  print('train accuracy')
  print(train)

  #Classification Report
  report = metrics.classification_report(y_test, y_pred, output_dict=True)
  df_classification_report = pd.DataFrame(report).transpose()
  print(df_classification_report)

  cb_metrics = {'train': train, 'test':test, 'report':df_classification_report}
  joblib.dump(cb_metrics,'cb_metrics')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--clean_data')
  args = parser.parse_args()
  cb(args.clean_data)
