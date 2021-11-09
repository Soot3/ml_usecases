import argparse
def knn(clean_data):
  import joblib
  import numpy as np
  import pandas as pd
  from sklearn import metrics
  from sklearn.neighbors import KNeighborsClassifier

  data = joblib.load(clean_data)
  X_train = data['X_train']
  y_train = data['Y_train']
  X_test = data['X_test']
  y_test = data['Y_test']

  knn_model = KNeighborsClassifier()

  knn_model.fit(X_train, y_train)

  y_pred = knn_model.predict(X_test)

  # Test score
  test = knn_model.score(X_test, y_test)
  train = knn_model.score(X_train, y_train)
  print('test accuracy:')
  print(test)
  print('train accuracy:')
  print(train)

  #Classification Report
  report = metrics.classification_report(y_test, y_pred, output_dict=True)
  df_classification_report = pd.DataFrame(report).transpose()
  print(df_classification_report)

  knn_metrics = {'train':train, 'test':test, 'report':df_classification_report,'model':knn_model}
  joblib.dump(knn_metrics,'knn_metrics')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--clean_data')
  args = parser.parse_args()
  knn(args.clean_data)