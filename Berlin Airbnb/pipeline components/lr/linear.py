import argparse
def lr(clean_data):
  import joblib
  import numpy as np
  import pandas as pd
  from sklearn.linear_model import LinearRegression
  from sklearn.metrics import r2_score, mean_squared_error

  data = joblib.load(clean_data)
  X_train = data['X_train']
  y_train = data['Y_train']
  X_test = data['X_test']
  y_test = data['Y_test']

  model = LinearRegression()

  model.fit(X_train, y_train)

  y_test_pred = model.predict(X_test)
  # Test score
  score_lr = r2_score(y_test, y_test_pred)
  print("R^2 score is: {0: .4f}".format(r2_score(y_test, y_test_pred)))

  rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
  print("RMSE is: {0: .4f}".format(rmse))

  lr_metrics = {'score': score_lr, 'rmse':rmse, 'model':model}
  joblib.dump(lr_metrics,'lr_metrics')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--clean_data')
  args = parser.parse_args()
  lr(args.clean_data)
