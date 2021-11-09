import argparse
def preprocessing(data):
  import joblib
  import numpy as np
  import pandas as pd
  from sklearn import preprocessing
  df = joblib.load(data)
  df1 = df.drop(['Date', 'Hours', 'Sample Number', 'Month', 'timestamp', 'Minutes','Seconds',"Mode"], axis=1)
  train_percentage = 0.30
  train_size = int(len(df1.index)*train_percentage)
  x_train = df1[:train_size]
  x_test = df1[train_size:490000]
  scaler = preprocessing.MinMaxScaler()

  X_train = pd.DataFrame(scaler.fit_transform(x_train), 
                                columns=x_train.columns, 
                                index=x_train.index)
  # Random shuffle training data
  X_train.sample(frac=1)

  X_test = pd.DataFrame(scaler.transform(x_test), 
                              columns=x_test.columns, 
                              index=x_test.index)
  data = {"X_train": X_train,"X_test": X_test}
  joblib.dump(data,'clean_data')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data')
  args = parser.parse_args()
  preprocessing(args.data)