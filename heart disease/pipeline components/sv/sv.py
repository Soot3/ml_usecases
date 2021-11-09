import argparse
def sv(clean_data):
  import joblib
  import numpy as np
  import pandas as pd
  from sklearn import metrics
  from sklearn.svm import SVC

  data = joblib.load(clean_data)
  X_train = data['X_train']
  y_train = data['Y_train']
  X_test = data['X_test']
  y_test = data['Y_test']

  sv_model = SVC()

  sv_model.fit(X_train, y_train)

  y_pred = sv_model.predict(X_test)

  # Test score
  test = sv_model.score(X_test, y_test)
  train = sv_model.score(X_train, y_train)
  print('test accuracy:')
  print(test)
  print('train accuracy:')
  print(train)

  #Classification Report
  report = metrics.classification_report(y_test, y_pred, output_dict=True)
  df_classification_report = pd.DataFrame(report).transpose()
  print(df_classification_report)

  sv_metrics = {'train':train, 'test':test, 'report':df_classification_report,'model':sv_model}
  joblib.dump(sv_metrics,'sv_metrics')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--clean_data')
  args = parser.parse_args()
  sv(args.clean_data)