import argparse
def preprocessing(data):
  import joblib
  import numpy as np
  import pandas as pd
  from sklearn import preprocessing
  from sklearn.model_selection import train_test_split
  data = joblib.load(data)
  data.dropna(inplace=True)
  objects = data.select_dtypes('object').columns.to_list()
  le = preprocessing.LabelEncoder()
  data[objects] = data[objects].apply(le.fit_transform)
  X = data.drop(columns=['Stay_Days','case_id','patientid'])
  
  y = data['Stay_Days']
  scaler = preprocessing.RobustScaler()
  X = scaler.fit_transform(X.astype(float))

  X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=9)

  data_dic = {"X_train": X_train,"X_test": X_test, "Y_train": Y_train, "Y_test": Y_test}

  joblib.dump(data_dic,'clean_data')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data')
  args = parser.parse_args()
  preprocessing(args.data)