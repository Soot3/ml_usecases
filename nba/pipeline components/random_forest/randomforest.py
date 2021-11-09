import argparse
def rf(clean_data):
  import joblib
  import numpy as np
  import pandas as pd
  from sklearn import metrics
  from sklearn.ensemble import RandomForestClassifier
  data = joblib.load(clean_data)
  X_train = data['X_train']
  y_train = data['Y_train']
  X_test = data['X_test']
  y_test = data['Y_test']

  rfcla = RandomForestClassifier(max_features='auto',
                       n_estimators=10, random_state=42, max_depth=5, min_samples_leaf=100) 

  rfcla.fit(X_train, y_train)

  y_pred = rfcla.predict(X_test)

  # Test score
  test = rfcla.score(X_test, y_test)
  train = rfcla.score(X_train, y_train)
  print('test accuracy:')
  print(test)
  print('train accuracy:')
  print(train)

  #Classification Report
  report = metrics.classification_report(y_test, y_pred, output_dict=True)
  df_classification_report = pd.DataFrame(report).transpose()
  print(df_classification_report)

  rf_metrics = {'train':train, 'test':test, 'report':df_classification_report}
  joblib.dump(rf_metrics,'rf_metrics')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--clean_data')
  args = parser.parse_args()
  rf(args.clean_data)