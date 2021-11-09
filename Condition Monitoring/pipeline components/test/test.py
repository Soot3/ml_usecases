import argparse
def test_model(clean_data, fit_model):
  import numpy as np
  import pandas as pd
  import joblib
  from tensorflow import keras
  from keras.models import load_model
  model = load_model(fit_model)
  data = joblib.load(clean_data)
  X_train = data['X_train']
  X_test = data['X_test']
  X_pred_train = model.predict(np.array(X_train))
  X_pred_train = pd.DataFrame(X_pred_train, 
                        columns=X_train.columns)
  X_pred_train.index = X_train.index

  scored_train = pd.DataFrame(index=X_train.index)
  scored_train['Loss_mae'] = np.mean(np.abs(X_pred_train-X_train), axis = 1)
  thress = np.max(scored_train['Loss_mae'])
  scored_train['Threshold'] = thress
  scored_train['Anomaly'] = scored_train['Loss_mae'] > scored_train['Threshold']
  X_pred = model.predict(np.array(X_test))
  X_pred = pd.DataFrame(X_pred, 
                        columns=X_test.columns)
  X_pred.index = X_test.index

  scored = pd.DataFrame(index=X_test.index)
  scored['Loss_mae'] = np.mean(np.abs(X_pred-X_test), axis = 1)
  scored['Threshold'] = thress
  scored['Anomaly'] = scored['Loss_mae'] > scored['Threshold']
  scored = pd.concat([scored_train, scored])
  data = scored[scored['Anomaly']==True]    
  joblib.dump(data,'metrics')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--clean_data')
  parser.add_argument('--fit_model')
  args = parser.parse_args()
  test_model(args.clean_data,args.fit_model)