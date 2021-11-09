import argparse
def gbc(clean_data):
  import joblib
  import numpy as np
  import pandas as pd
  from sklearn import metrics
  from sklearn.ensemble import GradientBoostingClassifier
  
  data = joblib.load(clean_data)
  X_train = data['X_train']
  y_train = data['Y_train']
  X_test = data['X_test']
  y_test = data['Y_test']  

  gbcla = GradientBoostingClassifier(max_features='auto',
                       n_estimators=100, random_state=42, max_depth=5, min_samples_leaf=100, learning_rate = 0.08) 

  gbcla.fit(X_train, y_train)

  y_pred = gbcla.predict(X_test)

  # Test score
  test = gbcla.score(X_test, y_test)
  print('test accuracy:')
  print(test)
  train = gbcla.score(X_train, y_train)
  print('train accuracy:')
  print(train)

  #Classification Report
  report = metrics.classification_report(y_test, y_pred, output_dict=True)
  df_classification_report = pd.DataFrame(report).transpose()
  print(df_classification_report)
  gbc_metrics = {'train': train, 'test':test, 'report':df_classification_report}
  joblib.dump(gbc_metrics,'gbc_metrics')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--clean_data')
  args = parser.parse_args()
  gbc(args.clean_data)