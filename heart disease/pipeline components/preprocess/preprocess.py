import argparse
def preprocessing(data):
  import joblib
  import numpy as np
  import pandas as pd
  from sklearn.model_selection import train_test_split
  from sklearn.utils import resample
  from sklearn.preprocessing import StandardScaler
  df = joblib.load(data)
  def removeOutlier(att, df):

      lowerbound = att.mean() - 3 * att.std()
      upperbound = att.mean() + 3 * att.std()

      df1 = df[(att > lowerbound) & (att < upperbound)]

      df = df1.copy()

      return df
  df = removeOutlier(df.trtbps, df)
  df = removeOutlier(df.chol, df)   
  
    # Separate Target Classes
  df_1 = df[df.output==1]
  df_2 = df[df.output==0]
  
  # Upsample minority class
  df_upsample_1 = resample(df_2, 
                                  replace=True,     # sample with replacement
                                  n_samples=163,    # to match majority class
                                  random_state=123) # reproducible results

  # Combine majority class with upsampled minority class
  df_upsampled = pd.concat([df_1, df_upsample_1])
  x = df_upsampled.drop('output', axis = 1)
  y = df_upsampled['output']  

  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 111)
  scaler = StandardScaler()

  x_train = scaler.fit_transform(x_train)
  x_test = scaler.fit_transform(x_test)

  data_dic = {"X_train": x_train,"X_test": x_test, "Y_train": y_train, "Y_test": y_test}

  joblib.dump(data_dic,'clean_data')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data')
  args = parser.parse_args()
  preprocessing(args.data)