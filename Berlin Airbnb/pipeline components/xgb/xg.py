import argparse
def xgb(clean_data):
  import joblib
  import numpy as np
  import pandas as pd
  from sklearn.metrics import r2_score, mean_squared_error
  import xgboost as xgb
  
  data = joblib.load(clean_data)
  X_train = data['X_train']
  y_train = data['Y_train']
  X_test = data['X_test']
  y_test = data['Y_test']  

  xgb_clf = xgb.XGBRegressor(n_estimators=100, max_depth=5)

  xgb_clf.fit(X_train, y_train)

  y_test_pred = xgb_clf.predict(X_test)


  # Test score
  score_xg = r2_score(y_test, y_test_pred)
  print("R^2 score is: {0: .4f}".format(score_xg))

  rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
  print("RMSE is: {0: .4f}".format(rmse))

  xgb_metrics = {'score': score_xg, 'rmse':rmse, 'model':xgb_clf}
  joblib.dump(xgb_metrics,'xgb_metrics')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--clean_data')
  args = parser.parse_args()
  xgb(args.clean_data)
