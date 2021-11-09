import argparse
def lr(clean_data):
  import joblib
  import numpy as np
  import pandas as pd
  from sklearn import metrics
  from sklearn.linear_model import LogisticRegression

  data = joblib.load(clean_data)
  X_train = data['X_train']
  y_train = data['Y_train']
  X_test = data['X_test']
  y_test = data['Y_test']

  logistic = LogisticRegression()

  logistic.fit(X_train, y_train)

  prediction_lr = logistic.predict(X_test)

  # Test score
  test = logistic.score(X_test, y_test)
  print('test accuracy:')
  print(test)
  train=logistic.score(X_train, y_train)
  print('train accuracy:')
  print(train)

  report = metrics.classification_report(y_test, prediction_lr, output_dict=True)
  df_classification_report = pd.DataFrame(report).transpose()
  print(df_classification_report)

  lr_metrics = {'train': train, 'test':test, 'report':df_classification_report}
  joblib.dump(lr_metrics,'lr_metrics')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--clean_data')
  args = parser.parse_args()
  lr(args.clean_data)