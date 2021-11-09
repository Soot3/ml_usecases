import argparse
import os
from sklearn.ensemble import RandomForestClassifier as rfc

import pandas as pd

#from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
#from timeit import default_timer as timestamp

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_estimators',
                        type = int,
                        default = 100,
                        help = 'The number of trees in the forest.')
    parser.add_argument('--min_samples_leaf',
                        type = int,
                        default = 2,
                        help = 'The minimum number of samples required to be at a leaf node.')
    parser.add_argument('--min_samples_split',
                        type = int,
                        default = 2,
                        help = 'The minimum number of samples required to split an internal node.')
    args = parser.parse_args()
    
    df=pd.read_csv('https://raw.githubusercontent.com/Soot3/testing/master/heart.csv')

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


    params = {
        'n_estimators': args.n_estimators,
        'min_samples_leaf': args.min_samples_leaf,
        'min_samples_split': args.min_samples_split
    }
    

    #start = timestamp()
    model = rfc()
    model.fit(x_train, y_train)
    #stop = timestamp()

#print('time=%.3f' % (stop - start))

    predictions = model.predict(x_test)

    print('accuracy=%.3f' % accuracy_score(y_test, predictions))
