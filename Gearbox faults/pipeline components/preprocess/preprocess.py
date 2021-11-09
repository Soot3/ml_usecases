import argparse
def preprocessing(data):
  import joblib
  import numpy as np
  import pandas as pd
  from sklearn import preprocessing
  all_df = joblib.load(data)
  X = all_df.drop(['failure'], axis=1)
  y = all_df['failure']
  train_percentage = 0.70
  train_size = int(len(X.index)*train_percentage)
  X_train = X[:train_size]
  y_train = y[:train_size]
  X_test = X[train_size:20203]
  y_test = y[train_size:20203]

  scaler = preprocessing.MinMaxScaler()

  X_train = pd.DataFrame(scaler.fit_transform(X_train), 
                                columns=X_train.columns, 
                                index=X_train.index)


  X_test = pd.DataFrame(scaler.transform(X_test), 
                              columns=X_test.columns, 
                              index=X_test.index)
  
  data_dic = {"X_train": X_train,"X_test": X_test, "Y_train": y_train, "Y_test": y_test}

  joblib.dump(data_dic,'clean_data')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data')
  args = parser.parse_args()
  preprocessing(args.data)